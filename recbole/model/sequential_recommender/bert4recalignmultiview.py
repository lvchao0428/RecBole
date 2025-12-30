"""
BERT4RecAlignMultiView - Multi-View Text Features for BERT4Rec
Based on SASRecAlignMultiViewV2 architecture
"""

import json
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.sequential_recommender.bert4recalign import BERT4RecAlign, DCNV2Cross
from recbole.model.layers import MLPLayers


class BERT4RecAlignMultiView(BERT4RecAlign):
    """
    BERT4RecAlign variant with per-view multi-view embeddings and alignment.
    
    Architecture:
    1. Load per-view embeddings from split directory
    2. Per-view SENet enhancement for feature refinement
    3. Per-view L2 normalization
    4. Per-view alignment loss with ID embeddings (with multiview_align_scale)
    5. Gated view fusion via concat → projection
    6. Cross Network fusion with ID embeddings
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_text_view_split = bool(config["use_text_view_split"]) if "use_text_view_split" in config else False
        self.use_multiview_text_cross = bool(config["use_multiview_text_cross"]) if "use_multiview_text_cross" in config else False
        
        if "item_text_emb_split_dir" in config and config["item_text_emb_split_dir"]:
            self.text_view_split_dir = os.path.abspath(os.path.expanduser(config["item_text_emb_split_dir"]))
        else:
            self.text_view_split_dir = None
        
        self.text_view_senet_ratio = int(config["text_view_senet_ratio"]) if "text_view_senet_ratio" in config else 2
        self.use_text_view_senet = bool(config["use_text_view_senet"]) if "use_text_view_senet" in config else True
        self.text_view_half_precision = bool(config["text_view_half_precision"]) if "text_view_half_precision" in config else True
        self.text_view_storage_dtype = torch.float16 if self.text_view_half_precision else torch.float32

        self.multiview_align_scale = float(config["multiview_align_scale"]) if "multiview_align_scale" in config else 1.0
        self.per_view_l2_norm = bool(config["per_view_l2_norm"]) if "per_view_l2_norm" in config else True

        self.text_view_buffer_names = []
        self.text_view_proj = nn.ModuleList()
        self.text_view_senet = nn.ModuleList()
        self.text_view_gate_params = None
        self.text_view_align_weights = None
        self.multiview_concat_proj = None

        if self.use_text_view_split:
            if not self.text_view_split_dir:
                raise ValueError("use_text_view_split=True but item_text_emb_split_dir is missing.")
            meta_path = os.path.join(self.text_view_split_dir, "views.json")
            if not os.path.exists(meta_path):
                raise ValueError(f"views.json not found in {self.text_view_split_dir}")
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            self.text_view_meta = meta
            prompts = meta.get("prompts", [])
            if len(prompts) == 0:
                raise ValueError("views.json contains no prompt metadata.")

            self.num_text_views = len(prompts)
            
            self.text_view_gate_params = nn.Parameter(torch.zeros(self.num_text_views, dtype=torch.float32))
            self.text_view_align_weights = nn.Parameter(torch.ones(self.num_text_views, dtype=torch.float32))

            for view in prompts:
                idx = view["index"]
                file_name = view["file"]
                file_path = os.path.join(self.text_view_split_dir, file_name)
                if not os.path.exists(file_path):
                    raise ValueError(f"Split embedding file not found: {file_path}")
                npy = np.load(file_path)
                tensor = torch.from_numpy(npy).to(self.text_view_storage_dtype)
                buffer_name = f"text_view_emb_{idx}"
                self.register_buffer(buffer_name, tensor)
                self.text_view_buffer_names.append(buffer_name)

                view_dim = int(view["vector_dim"])
                self.text_view_proj.append(nn.Linear(view_dim, self.hidden_size))

                reduction = max(1, self.hidden_size // self.text_view_senet_ratio)
                self.text_view_senet.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_size, reduction),
                        nn.ReLU(),
                        nn.Linear(reduction, self.hidden_size),
                        nn.Sigmoid(),
                    )
                )

            multiview_input_dim = self.num_text_views * self.hidden_size
            if self.item_text_emb_base is not None:
                base_dim = self.item_text_emb_base.shape[1]
                multiview_input_dim += base_dim
            
            self.multiview_concat_proj = nn.Linear(multiview_input_dim, self.hidden_size)
            
            self.multiview_text_cross = None
            self.multiview_text_deep = None
            self.multiview_text_predictor = None
            self.multiview_text_cross_dropout = None
            
            fusion_input_dim = self.hidden_size + self.hidden_size
            
            if self.use_cross:
                if self.use_multiview_text_cross:
                    self.multiview_text_cross = DCNV2Cross(self.hidden_size, num_layers=self.text_cross_layer_num)
                    self.multiview_text_deep = MLPLayers(
                        [self.hidden_size, self.hidden_size], 
                        dropout=0.0, 
                        bn=self.text_mlp_bn
                    )
                    self.multiview_text_predictor = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
                    if self.cross_dropout_prob > 0.0:
                        self.multiview_text_cross_dropout = nn.Dropout(self.cross_dropout_prob)
                
                self.item_fusion_cross = DCNV2Cross(fusion_input_dim, num_layers=self.text_cross_layer_num)
                self.item_fusion_deep = MLPLayers(
                    [fusion_input_dim, self.hidden_size], 
                    dropout=0.0, 
                    bn=self.text_mlp_bn
                )
                self.item_fusion_predictor = nn.Linear(
                    fusion_input_dim + self.hidden_size,
                    self.hidden_size
                )
                
                if self.cross_dropout_prob > 0.0:
                    self.item_fusion_cross_dropout = nn.Dropout(self.cross_dropout_prob)
            
            if self.item_emb_norm is None and self.fused_item_norm_flag:
                self.item_emb_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            if self.fused_item_norm is None and self.fused_item_norm_flag:
                self.fused_item_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            
            self.logger.info(
                "BERT4RecAlignMultiView initialized: %d views, SENet=%s (ratio=%d), per-view L2 norm=%s",
                self.num_text_views, 
                "enabled" if self.use_text_view_senet else "disabled",
                self.text_view_senet_ratio, 
                self.per_view_l2_norm
            )
            self.logger.info(
                "Multiview enhancements: multiview_align_scale=%.2f, per_view_l2_norm=%s",
                self.multiview_align_scale, self.per_view_l2_norm
            )

    def _get_view_buffer(self, idx: int) -> torch.Tensor:
        """Get the embedding buffer for a specific view."""
        name = self.text_view_buffer_names[idx]
        return getattr(self, name)

    def _gather_text_views(self, ids_flat: torch.Tensor) -> torch.Tensor:
        """
        Gather and refine per-view features using SENet.
        
        Args:
            ids_flat: Item IDs [B]
            
        Returns:
            Stacked view features [B, num_views, hidden_size]
        """
        if not self.use_text_view_split:
            raise RuntimeError("text view split not enabled.")

        view_features = []
        for idx, proj in enumerate(self.text_view_proj):
            view_emb_table = self._get_view_buffer(idx)
            if view_emb_table.device != ids_flat.device:
                view_emb_table = view_emb_table.to(ids_flat.device)
            gathered = view_emb_table[ids_flat]
            
            if gathered.dtype != proj.weight.dtype:
                gathered = gathered.to(proj.weight.dtype)
            projected = proj(gathered)
            
            if self.use_text_view_senet:
                excitation = self.text_view_senet[idx](projected)
                refined = projected * excitation
            else:
                refined = projected
            
            if self.per_view_l2_norm:
                refined = F.normalize(refined, p=2, dim=-1)
            
            view_features.append(refined)
        
        stacked = torch.stack(view_features, dim=1)
        return stacked

    def _get_fused_item_embeddings(self, item_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Get fused item embeddings with multi-view text features.
        Supports chunking to prevent CUDA timeout for large item sets.
        """
        if not self.use_text_view_split:
            return super()._get_fused_item_embeddings(item_ids)

        # Handle chunking for large item sets to prevent CUDA timeout
        if item_ids is None:
            all_ids = torch.arange(self.n_items, device=self.item_embedding.weight.device)
            if (
                isinstance(self.fusion_chunk_size, int)
                and self.fusion_chunk_size > 0
                and self.fusion_chunk_size < all_ids.numel()
            ):
                chunks = []
                for chunk_ids in torch.split(all_ids, self.fusion_chunk_size):
                    chunks.append(self._get_fused_item_embeddings(chunk_ids))
                return torch.cat(chunks, dim=0)
            item_emb = self.item_embedding.weight[:self.n_items]
        else:
            all_ids = item_ids
            item_emb = self.item_embedding(item_ids)

        if not self.fuse_text_feature or len(self.text_view_buffer_names) == 0:
            return item_emb

        view_stack = self._gather_text_views(all_ids)
        
        if self.detach_text_emb:
            view_stack = view_stack.detach()

        view_weights = torch.sigmoid(self.text_view_gate_params).to(view_stack.device)
        
        weighted_views = []
        for idx in range(self.num_text_views):
            view_feat = view_stack[:, idx, :]
            weighted = view_weights[idx] * view_feat
            weighted_views.append(weighted)
        
        text_concat = torch.cat(weighted_views, dim=-1)
        
        if self.item_text_emb_base is not None:
            base_feat = self.item_text_emb_base[all_ids]
            if self.detach_text_emb:
                base_feat = base_feat.detach()
            text_concat = torch.cat([base_feat, text_concat], dim=-1)
        
        text_proj = self.multiview_concat_proj(text_concat)
        
        if (
            self.use_cross 
            and self.multiview_text_cross is not None 
            and self.multiview_text_deep is not None 
            and self.multiview_text_predictor is not None
        ):
            cross_out = self.multiview_text_cross(text_proj)
            if self.multiview_text_cross_dropout is not None:
                cross_out = self.multiview_text_cross_dropout(cross_out)
            deep_out = self.multiview_text_deep(text_proj)
            text_fused = torch.cat([cross_out, deep_out], dim=1)
            text_proj = self.multiview_text_predictor(text_fused)
            
            if self.text_proj_norm is not None:
                text_proj = self.text_proj_norm(text_proj)
        
        return self._fuse_with_cross_network(item_emb, text_proj, all_ids)
    
    def _fuse_with_cross_network(
        self, 
        item_emb: torch.Tensor, 
        text_raw: torch.Tensor, 
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse item embeddings with text features using cross network.
        """
        alpha = torch.sigmoid(self.text_gate_param) if hasattr(self, 'text_gate_param') else torch.tensor(1.0)
        
        if self.text_item_gate_all is not None:
            gate = self.text_item_gate_all[item_ids]
            gate = gate.to(item_emb.device).unsqueeze(1)
            alpha = alpha * gate
        
        temp_scale = 1.0
        effective_text_weight = alpha * self.text_weight * temp_scale
        
        if self.use_cross and self.item_fusion_predictor is not None:
            if effective_text_weight.dim() == 0:
                scaled_text = effective_text_weight * text_raw
            else:
                scaled_text = effective_text_weight * text_raw
            
            item_emb_for_fusion = item_emb
            if self.item_emb_norm is not None:
                item_emb_for_fusion = self.item_emb_norm(item_emb_for_fusion)
            
            fusion_input = torch.cat([item_emb_for_fusion, scaled_text], dim=1)
            cross_out = self.item_fusion_cross(fusion_input)
            if self.item_fusion_cross_dropout is not None:
                cross_out = self.item_fusion_cross_dropout(cross_out)
            deep_out = self.item_fusion_deep(fusion_input)
            fused = torch.cat([cross_out, deep_out], dim=1)
            fused_emb = self.item_fusion_predictor(fused)
        else:
            item_emb_for_fusion = item_emb
            if self.item_emb_norm is not None:
                item_emb_for_fusion = self.item_emb_norm(item_emb_for_fusion)
            
            scaled_text = effective_text_weight * text_raw
            fused_emb = item_emb_for_fusion + scaled_text
        
        if self.fused_item_norm is not None:
            fused_emb = self.fused_item_norm(fused_emb)
        
        return fused_emb

    def calculate_loss(self, interaction):
        """
        Calculate loss with per-view alignment losses.
        """
        loss = super().calculate_loss(interaction)
        
        if (
            self.use_text_view_split 
            and self.use_align 
            and self.alignment_weight > 0.0
            and len(self.text_view_buffer_names) > 0
        ):
            pos_items = interaction[self.POS_ITEMS]
            valid_mask = (interaction[self.MASK_INDEX] > 0).view(-1)
            if valid_mask.any():
                pos_ids_flat = pos_items.view(-1)[valid_mask]
                
                id_item_emb = self.item_embedding(pos_ids_flat)
                view_stack = self._gather_text_views(pos_ids_flat)
                
                if self.detach_text_emb:
                    view_stack = view_stack.detach()
                
                cold_start_weights = self._compute_cold_start_weights(pos_ids_flat)
                use_weighted_align = self.cold_start_align_boost > 0
                
                per_view_align_losses = []
                for idx in range(self.num_text_views):
                    view_feat = view_stack[:, idx, :]
                    
                    if use_weighted_align:
                        align_loss_i = self._info_nce_align_weighted(id_item_emb, view_feat, cold_start_weights)
                    else:
                        align_loss_i = self._info_nce_align(id_item_emb, view_feat)
                    per_view_align_losses.append(align_loss_i)
                
                align_weights = F.softmax(self.text_view_align_weights, dim=0)
                
                total_align_loss = sum(
                    align_weights[idx] * per_view_align_losses[idx] 
                    for idx in range(self.num_text_views)
                )
                
                scaled_align_loss = self.multiview_align_scale * total_align_loss
                loss = loss + self.alignment_weight * scaled_align_loss
                
                if not getattr(self, '_multiview_align_debug_logged', False):
                    try:
                        self.logger.info(
                            "BERT4RecAlignMultiView: per-view alignment enabled | "
                            "total_align_loss=%.6f | multiview_align_scale=%.2f | scaled_loss=%.6f",
                            total_align_loss.item(), self.multiview_align_scale, scaled_align_loss.item()
                        )
                    except Exception:
                        pass
                    self._multiview_align_debug_logged = True
        
        return loss


# V2 is the same as base MultiView with additional enhancements
class BERT4RecAlignMultiViewV2(BERT4RecAlignMultiView):
    """
    Enhanced version of BERT4RecAlignMultiView with same features as SASRecAlignMultiViewV2.
    """
    pass


# Aliases
BERT4Rec_Align_MultiView = BERT4RecAlignMultiView
BERT4Rec_Align_MultiView_V2 = BERT4RecAlignMultiViewV2

__all__ = [
    "BERT4RecAlignMultiView", 
    "BERT4RecAlignMultiViewV2",
    "BERT4Rec_Align_MultiView",
    "BERT4Rec_Align_MultiView_V2"
]

