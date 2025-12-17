"""
SASRecAlignMultiView V2 - Enhanced Multi-View Text Features

改动点汇总（相比 sasrecalignmultiview.py）：
============================================

【改动1】每个view独立做L2归一化
    - 位置: _gather_text_views() 方法
    - 原因: 让每个view的特征在自己的空间内归一化，而不是跨view归一化权重
    - 搜索标记: # [CHANGE-1]

【改动2】移除跨view权重归一化
    - 位置: _get_fused_item_embeddings() 方法
    - 原因: 原先 view_weights.sum() 归一化导致4个view平均只有0.25权重，削弱了multi-view贡献
    - 搜索标记: # [CHANGE-2]

【改动3】增加 multiview_align_scale 参数
    - 位置: __init__() 和 calculate_loss() 方法
    - 原因: Multi-View专属对齐损失放大系数，增强multi-view的对齐信号
    - 配置项: multiview_align_scale (默认 1.0，建议设为 2.0-3.0)
    - 搜索标记: # [CHANGE-3]

【改动4】SENet ratio 默认值调整
    - 位置: __init__() 方法
    - 原因: 减少信息压缩，保留更多multi-view信息
    - 配置项: text_view_senet_ratio (默认从4改为2)
    - 搜索标记: # [CHANGE-4]

【改动5】增加 per_view_l2_norm 开关
    - 位置: __init__() 和 _gather_text_views() 方法
    - 原因: 可配置是否对每个view做独立L2归一化
    - 配置项: per_view_l2_norm (默认 True)
    - 搜索标记: # [CHANGE-5]

配置示例 (yaml):
===============
multiview_align_scale: 2.0      # Multi-View专属对齐损失放大
text_view_senet_ratio: 2        # SENet压缩比（原默认4）
per_view_l2_norm: true          # 每个view独立L2归一化
"""

import json
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.sequential_recommender.sasrec_align import SASRecAlign


class SASRecAlignMultiViewV2(SASRecAlign):
    """
    SASRecAlign variant with ENHANCED per-view multi-view embeddings and alignment.
    
    Key improvements over V1:
    1. Per-view L2 normalization (instead of cross-view weight normalization)
    2. Configurable multiview_align_scale for stronger alignment signal
    3. Reduced SENet compression ratio (default 2 instead of 4)
    4. Independent view weights without forced normalization
    
    Architecture:
    1. Load per-view embeddings from split directory
    2. Per-view SENet enhancement for feature refinement
    3. Per-view L2 normalization (NEW in V2)
    4. Per-view alignment loss with ID embeddings (with multiview_align_scale)
    5. Gated view fusion via concat → projection (without weight normalization)
    6. Cross Network fusion with ID embeddings (inherited from parent)
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_text_view_split = bool(config["use_text_view_split"]) if "use_text_view_split" in config else False
        # Switch to enable/disable DCN-V2 text_cross for multiview
        self.use_multiview_text_cross = bool(config["use_multiview_text_cross"]) if "use_multiview_text_cross" in config else False
        if "item_text_emb_split_dir" in config and config["item_text_emb_split_dir"]:
            self.text_view_split_dir = os.path.abspath(os.path.expanduser(config["item_text_emb_split_dir"]))
        else:
            self.text_view_split_dir = None
        
        # [CHANGE-4] SENet ratio 默认值从4改为2，减少信息压缩
        self.text_view_senet_ratio = int(config["text_view_senet_ratio"]) if "text_view_senet_ratio" in config else 2
        
        self.text_view_half_precision = (
            bool(config["text_view_half_precision"]) if "text_view_half_precision" in config else True
        )
        self.text_view_storage_dtype = torch.float16 if self.text_view_half_precision else torch.float32

        # [CHANGE-3] Multi-View专属对齐损失放大系数
        self.multiview_align_scale = float(config["multiview_align_scale"]) if "multiview_align_scale" in config else 1.0
        
        # [CHANGE-5] 每个view独立L2归一化开关
        self.per_view_l2_norm = bool(config["per_view_l2_norm"]) if "per_view_l2_norm" in config else True

        self.text_view_buffer_names = []
        self.text_view_proj = nn.ModuleList()
        self.text_view_senet = nn.ModuleList()
        self.text_view_gate_params = None
        self.text_view_align_weights = None  # Learnable per-view alignment weights
        self.multiview_concat_proj = None  # Projection after concat

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
            
            # Learnable per-view gate parameters (for weighted fusion)
            self.text_view_gate_params = nn.Parameter(torch.zeros(self.num_text_views, dtype=torch.float32))
            
            # Learnable per-view alignment weights (for multi-view alignment loss)
            self.text_view_align_weights = nn.Parameter(torch.ones(self.num_text_views, dtype=torch.float32))

            # Load per-view embeddings and create SENet modules
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
                # Projection: view_dim → hidden_size
                self.text_view_proj.append(nn.Linear(view_dim, self.hidden_size))

                # SENet: hidden_size → reduction → hidden_size (attention weights)
                # [CHANGE-4] 使用更小的压缩比，保留更多信息
                reduction = max(1, self.hidden_size // self.text_view_senet_ratio)
                self.text_view_senet.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_size, reduction),
                        nn.ReLU(),
                        nn.Linear(reduction, self.hidden_size),
                        nn.Sigmoid(),
                    )
                )

            # Multi-view concat projection
            multiview_input_dim = self.num_text_views * self.hidden_size  # 4×256 = 1024
            if self.item_text_emb_base is not None:
                base_dim = self.item_text_emb_base.shape[1]  # 256
                multiview_input_dim += base_dim  # 1024 + 256 = 1280
            
            # Project to hidden_size (256) for fair comparison with TF-IDF+LLM baseline
            self.multiview_concat_proj = nn.Linear(multiview_input_dim, self.hidden_size)
            
            # ===== Optional: Add text_cross DCN-V2 for multiview features =====
            self.multiview_text_cross = None
            self.multiview_text_deep = None
            self.multiview_text_predictor = None
            self.multiview_text_cross_dropout = None
            
            # Reinitialize item fusion networks with correct dimensions
            if self.use_cross:
                from recbole.model.sequential_recommender.sasrec_align import DCNV2Cross
                from recbole.model.layers import MLPLayers
                
                # DCN-V2 #1: multiview text cross (optional)
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
                
                # DCN-V2 #2: item fusion cross (always enabled when use_cross=True)
                fusion_input_dim = self.hidden_size + self.hidden_size  # 256 + 256 = 512
                
                self.item_fusion_cross = DCNV2Cross(fusion_input_dim, num_layers=self.text_cross_layer_num)
                self.item_fusion_deep = MLPLayers(
                    [fusion_input_dim, self.hidden_size], 
                    dropout=0.0, 
                    bn=self.text_mlp_bn
                )
                self.item_fusion_predictor = nn.Linear(
                    fusion_input_dim + self.hidden_size,  # 512 + 256 = 768
                    self.hidden_size
                )
                
                if self.cross_dropout_prob > 0.0:
                    self.item_fusion_cross_dropout = nn.Dropout(self.cross_dropout_prob)
            
            # Ensure normalization layers exist
            if self.item_emb_norm is None and self.fused_item_norm_flag:
                self.item_emb_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            if self.fused_item_norm is None and self.fused_item_norm_flag:
                self.fused_item_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            
            # Log configuration with V2 specific info
            has_base = self.item_text_emb_base is not None
            proj_input_dim = multiview_input_dim
            proj_output_dim = self.hidden_size
            has_text_cross = self.multiview_text_cross is not None
            
            self.logger.info(
                "SASRecAlignMultiViewV2 initialized: %d views, SENet ratio=%d, per-view L2 norm=%s",
                self.num_text_views, self.text_view_senet_ratio, self.per_view_l2_norm
            )
            # [CHANGE-3] 记录multiview_align_scale
            self.logger.info(
                "V2 enhancements: multiview_align_scale=%.2f, per_view_l2_norm=%s, no cross-view weight normalization",
                self.multiview_align_scale, self.per_view_l2_norm
            )
            self.logger.info(
                "Multi-view projection: [%d → %d] | Base features: %s | Fusion input dim: %d",
                proj_input_dim, proj_output_dim, 
                "enabled" if has_base else "disabled",
                fusion_input_dim
            )
            self.logger.info(
                "DCN-V2 layers: multiview_text_cross=%s | item_fusion_cross (W: %dx%d × %d layers)",
                "enabled" if has_text_cross else "disabled",
                fusion_input_dim, fusion_input_dim, self.text_cross_layer_num
            )

    def _get_view_buffer(self, idx: int) -> torch.Tensor:
        """Get the embedding buffer for a specific view."""
        name = self.text_view_buffer_names[idx]
        return getattr(self, name)

    def _gather_text_views(self, ids_flat: torch.Tensor) -> torch.Tensor:
        """
        Gather and refine per-view features using SENet.
        
        [CHANGE-1] 每个view独立做L2归一化
        
        Args:
            ids_flat: Item IDs [B]
            
        Returns:
            Stacked view features [B, num_views, hidden_size]
        """
        if not self.use_text_view_split:
            raise RuntimeError("text view split not enabled.")

        view_features = []
        for idx, proj in enumerate(self.text_view_proj):
            # 1. Load raw view embeddings
            view_emb_table = self._get_view_buffer(idx)
            if view_emb_table.device != ids_flat.device:
                view_emb_table = view_emb_table.to(ids_flat.device)
            gathered = view_emb_table[ids_flat]  # [B, view_dim]
            
            # 2. Project to hidden_size
            if gathered.dtype != proj.weight.dtype:
                gathered = gathered.to(proj.weight.dtype)
            projected = proj(gathered)  # [B, hidden_size]
            
            # 3. SENet enhancement
            excitation = self.text_view_senet[idx](projected)  # [B, hidden_size]
            refined = projected * excitation  # [B, hidden_size]
            
            # [CHANGE-1] 每个view独立做L2归一化
            # 原因: 让每个view的特征在自己的空间内归一化，保持各view的相对重要性
            if self.per_view_l2_norm:
                refined = F.normalize(refined, p=2, dim=-1)  # L2 normalize each view independently
            
            view_features.append(refined)
        
        # Stack all views
        stacked = torch.stack(view_features, dim=1)  # [B, num_views, hidden_size]
        return stacked

    def _get_fused_item_embeddings(self, item_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Get fused item embeddings with multi-view text features.
        
        [CHANGE-2] 移除跨view权重归一化，让每个view独立贡献
        
        Flow: SENet → L2Norm → Gate → Concat → Projection → Cross Network Fusion
        
        Args:
            item_ids: Specific item IDs or None for all items
            
        Returns:
            Fused item embeddings [B, hidden_size] or [n_items, hidden_size]
        """
        if not self.use_text_view_split:
            return super()._get_fused_item_embeddings(item_ids)

        # Handle chunking for large item sets
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
            item_emb = self.item_embedding.weight
        else:
            all_ids = item_ids
            item_emb = self.item_embedding(item_ids)

        # Early return if text fusion is disabled
        if not self.fuse_text_feature or len(self.text_view_buffer_names) == 0:
            return item_emb

        # Step 1: Gather SENet-enhanced + L2-normalized multi-view features [B, num_views, hidden_size]
        view_stack = self._gather_text_views(all_ids)
        
        if self.detach_text_emb:
            view_stack = view_stack.detach()

        # Step 2: Apply per-view gates (learnable importance weights)
        # [CHANGE-2] 移除跨view权重归一化
        # 原先: view_weights = sigmoid(params) / sum → 每个view平均只有0.25权重
        # 现在: view_weights = sigmoid(params) → 每个view可以独立学习0~1的权重
        view_weights = torch.sigmoid(self.text_view_gate_params).to(view_stack.device)
        # 不再做归一化: view_weights = view_weights / view_weights.sum().clamp_min(1e-6)
        
        # Step 3: Weight each view and concat
        weighted_views = []
        for idx in range(self.num_text_views):
            view_feat = view_stack[:, idx, :]  # [B, hidden_size]
            weighted = view_weights[idx] * view_feat
            weighted_views.append(weighted)
        
        # Concatenate all weighted views [B, num_views * hidden_size]
        text_concat = torch.cat(weighted_views, dim=-1)  # [B, 1024]
        
        # Step 3.5: Concat with base text features if available
        if self.item_text_emb_base is not None:
            base_feat = self.item_text_emb_base[all_ids]  # [B, 256]
            if self.detach_text_emb:
                base_feat = base_feat.detach()
            text_concat = torch.cat([base_feat, text_concat], dim=-1)
        
        # Step 4: Project to hidden_size (256)
        text_proj = self.multiview_concat_proj(text_concat)
        
        # Step 4.5: Apply DCN-V2 #1 (multiview_text_cross) if enabled
        if (
            self.use_cross 
            and self.multiview_text_cross is not None 
            and self.multiview_text_deep is not None 
            and self.multiview_text_predictor is not None
        ):
            cross_out = self.multiview_text_cross(text_proj)  # [B, 256]
            if self.multiview_text_cross_dropout is not None:
                cross_out = self.multiview_text_cross_dropout(cross_out)
            deep_out = self.multiview_text_deep(text_proj)  # [B, 256]
            text_fused = torch.cat([cross_out, deep_out], dim=1)  # [B, 512]
            text_proj = self.multiview_text_predictor(text_fused)  # [B, 256]
            
            if self.text_proj_norm is not None:
                text_proj = self.text_proj_norm(text_proj)
        
        # Step 5: Use cross network fusion logic (DCN-V2 #2: item_fusion_cross)
        return self._fuse_with_cross_network(item_emb, text_proj, all_ids)
    
    def _fuse_with_cross_network(
        self, 
        item_emb: torch.Tensor, 
        text_raw: torch.Tensor, 
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse item embeddings with text features using cross network.
        
        Args:
            item_emb: Item embeddings [B, hidden_size=256]
            text_raw: Projected multi-view (+ base) text features [B, hidden_size=256]
            item_ids: Item IDs [B]
            
        Returns:
            Fused embeddings [B, hidden_size=256]
        """
        # Calculate effective text weight
        alpha = torch.sigmoid(self.text_gate_param) if hasattr(self, 'text_gate_param') else torch.tensor(1.0)
        
        # Per-item gating
        if self.text_item_gate_all is not None:
            gate = self.text_item_gate_all[item_ids]
            gate = gate.to(item_emb.device).unsqueeze(1)
            alpha = alpha * gate
        
        # Temperature and alignment scaling
        if hasattr(self, 'alignment_temp_scale_flag') and self.alignment_temp_scale_flag:
            temp_scale = 1.0 / self.temperature if self.temperature > 0 else 1.0
        else:
            temp_scale = 1.0
            
        effective_text_weight = alpha * self.text_weight * temp_scale
        
        # Use cross network fusion if enabled
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
        
        [CHANGE-3] 增加 multiview_align_scale 参数，放大multi-view对齐损失
        
        In addition to the base CE/BPR loss, this method adds alignment losses
        for each view separately, weighted by learnable parameters and scaled
        by multiview_align_scale.
        """
        # Call parent's loss calculation (CE/BPR loss)
        loss = super().calculate_loss(interaction)
        
        # Add per-view alignment losses if enabled
        if (
            self.use_text_view_split 
            and self.use_align 
            and self.alignment_weight > 0.0
            and len(self.text_view_buffer_names) > 0
        ):
            # Get positive items from interaction
            pos_items = interaction[self.POS_ITEM_ID]
            
            # Get ID embeddings for positive items
            id_item_emb = self.item_embedding(pos_items)  # [B, hidden_size]
            
            # Gather all view features for positive items
            view_stack = self._gather_text_views(pos_items)  # [B, num_views, hidden_size]
            
            if self.detach_text_emb:
                view_stack = view_stack.detach()
            
            # Calculate alignment loss for each view separately
            per_view_align_losses = []
            for idx in range(self.num_text_views):
                view_feat = view_stack[:, idx, :]  # [B, hidden_size]
                
                # Use parent's InfoNCE alignment loss
                align_loss_i = self._info_nce_align(id_item_emb, view_feat)
                per_view_align_losses.append(align_loss_i)
            
            # Apply learnable per-view alignment weights
            align_weights = F.softmax(self.text_view_align_weights, dim=0)  # Normalize weights
            
            # Weighted sum of per-view alignment losses
            total_align_loss = sum(
                align_weights[idx] * per_view_align_losses[idx] 
                for idx in range(self.num_text_views)
            )
            
            # [CHANGE-3] 应用 multiview_align_scale 放大multi-view对齐损失
            # 原先: loss = loss + alignment_weight * total_align_loss
            # 现在: loss = loss + alignment_weight * multiview_align_scale * total_align_loss
            scaled_align_loss = self.multiview_align_scale * total_align_loss
            loss = loss + self.alignment_weight * scaled_align_loss
            
            # Debug logging (first step only)
            if not getattr(self, '_multiview_align_debug_logged', False):
                try:
                    align_weights_str = ", ".join([f"w{i}={align_weights[i].item():.4f}" for i in range(self.num_text_views)])
                    losses_str = ", ".join([f"L{i}={per_view_align_losses[i].item():.6f}" for i in range(self.num_text_views)])
                    view_gate_str = ", ".join([f"g{i}={torch.sigmoid(self.text_view_gate_params[i]).item():.4f}" for i in range(self.num_text_views)])
                    self.logger.info(
                        "SASRecAlignMultiViewV2: per-view alignment enabled | "
                        "total_align_loss=%.6f | multiview_align_scale=%.2f | scaled_loss=%.6f",
                        total_align_loss.item(), self.multiview_align_scale, scaled_align_loss.item()
                    )
                    self.logger.info(
                        "  align_weights=[%s] | losses=[%s]",
                        align_weights_str, losses_str
                    )
                    self.logger.info(
                        "  view_gates (no normalization)=[%s]",
                        view_gate_str
                    )
                except Exception:
                    pass
                self._multiview_align_debug_logged = True
        
        return loss


# Aliases for different naming conventions
SASRec_Align_MultiView_V2 = SASRecAlignMultiViewV2
SASRecAlignMultiView_V2 = SASRecAlignMultiViewV2


__all__ = ["SASRecAlignMultiViewV2", "SASRec_Align_MultiView_V2", "SASRecAlignMultiView_V2"]
