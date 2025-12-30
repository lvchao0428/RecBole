# -*- coding: utf-8 -*-
# Align-only variant of BERT4Rec with optional text embedding alignment
# Enhanced version with full TF-IDF + LLM + DCN-V2 support (similar to SASRecAlign)

import random
import os
import json

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, MLPLayers
from recbole.model.loss import BPRLoss


class DCNV2Cross(nn.Module):
    """DCN-V2 cross network (non-mix) over dense features.

    Follows the original implementation in recbole.model.context_aware_recommender.dcnv2
    with the update rule: x_{l+1} = x_l + x_0 ⊙ (W_l x_l + b_l).
    """

    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_layers = int(max(0, num_layers))
        # W: (in_feature_num, in_feature_num) per layer
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(torch.randn(self.input_dim, self.input_dim))
            for _ in range(self.num_layers)
        )
        # b: (in_feature_num, 1) per layer
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.input_dim, 1))
            for _ in range(self.num_layers)
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 0:
            return x0
        # x0: [batch, in_feature_num]
        x0_u = x0.unsqueeze(dim=2)  # [B, D, 1]
        xl = x0_u
        for i in range(self.num_layers):
            xl_w = torch.matmul(self.cross_layer_w[i], xl)  # [B, D, 1]
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x0_u, xl_w)
            xl = xl_dot + xl
        xl = xl.squeeze(dim=2)  # [B, D]
        return xl


class BERT4RecAlign(SequentialRecommender):
    def __init__(self, config, dataset):
        super(BERT4RecAlign, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.mask_ratio = config["mask_ratio"]

        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = config["POS_ITEMS"]
        self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        self.loss_type = config["loss_type"]
        self.initializer_range = config["initializer_range"]

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # additional regularization / scoring configs
        self.label_smoothing = float(config["label_smoothing"]) if "label_smoothing" in config else 0.0
        self.cosine_score = bool(config["cosine_score"]) if "cosine_score" in config else False
        self.cosine_scale = float(config["cosine_scale"]) if "cosine_scale" in config else 10.0

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_size, padding_idx=0
        )  # mask token add 1
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items))

        # --- text-alignment settings & feature fusion ---
        self.alignment_weight = config["alignment_weight"] if "alignment_weight" in config else 0.0
        self.temperature = config["temperature"] if "temperature" in config else 0.07
        self.normalize_text = config["normalize_text"] if "normalize_text" in config else True
        self.detach_text_emb = config["detach_text_emb"] if "detach_text_emb" in config else True
        self.use_llm = config["use_llm"] if "use_llm" in config else False
        self.use_cross = config["use_cross"] if "use_cross" in config else False
        self.use_align = config["use_align"] if "use_align" in config else True
        self.text_cross_layer_num = config["text_cross_layer_num"] if "text_cross_layer_num" in config else 3
        
        # Cross-output dropout and learnable text gate configs
        self.cross_dropout_prob = float(config["cross_dropout_prob"]) if "cross_dropout_prob" in config else 0.0
        self.text_gate_init = float(config["text_gate_init"]) if "text_gate_init" in config else 0.5
        self.text_gate_reg_l2 = float(config["text_gate_reg_l2"]) if "text_gate_reg_l2" in config else 0.0
        self.text_gate_reg_entropy = float(config["text_gate_reg_entropy"]) if "text_gate_reg_entropy" in config else 0.0
        # learnable global gate alpha in [0,1] via sigmoid
        self.text_gate_param = nn.Parameter(torch.tensor(self.text_gate_init, dtype=torch.float32))
        
        # Explicit switch to fully disable text features and mimic pure BERT4Rec
        self.disable_text_feature = bool(config["disable_text_feature"]) if "disable_text_feature" in config else False
        # Training utilities
        self.freeze_backbone = bool(config["freeze_backbone"]) if "freeze_backbone" in config else False
        
        # [Cold-Start Alignment Boost] 冷启动对齐权重增强
        self.cold_start_align_boost = float(config["cold_start_align_boost"]) if "cold_start_align_boost" in config else 0.0
        self.cold_start_align_threshold = int(config["cold_start_align_threshold"]) if "cold_start_align_threshold" in config else 10
        
        # New: simple non-cross enhancements
        self.text_weight = float(config["text_weight"]) if "text_weight" in config else 1.0
        self.text_tail_threshold = int(config["text_tail_threshold"]) if "text_tail_threshold" in config else 0
        # Control whether text features participate in item embedding fusion
        self.fuse_text_feature = bool(config["fuse_text_feature"]) if "fuse_text_feature" in config else True
        # Text MLP normalization
        self.text_mlp_bn = bool(config["text_mlp_bn"]) if "text_mlp_bn" in config else False
        # Normalization toggles for projections and fused item embeddings
        self.text_proj_norm_flag = bool(config["text_proj_norm"]) if "text_proj_norm" in config else True
        self.fused_item_norm_flag = bool(config["fused_item_norm"]) if "fused_item_norm" in config else True
        
        # Chunk size for memory-friendly full item fusion (0 disables chunking)
        # Prevents CUDA timeout when computing fused embeddings for all items
        self.fusion_chunk_size = int(config["fusion_chunk_size"]) if "fusion_chunk_size" in config else 0

        # For backward compatibility: accept single path as base
        item_text_emb_path_base = config["item_text_emb_path_base"] if "item_text_emb_path_base" in config else None
        item_text_emb_path_llm = config["item_text_emb_path_llm"] if "item_text_emb_path_llm" in config else None
        if item_text_emb_path_base is None and item_text_emb_path_llm is None:
            # Fallback to legacy key
            item_text_emb_path_base = config["item_text_emb_path"] if "item_text_emb_path" in config else None

        if self.disable_text_feature:
            emb_base = None
            emb_llm = None
        else:
            emb_base = self._load_text_embeddings(item_text_emb_path_base, self.n_items)
            emb_llm = self._load_text_embeddings(item_text_emb_path_llm, self.n_items)

        if self.normalize_text:
            with torch.no_grad():
                if emb_base is not None:
                    norms = torch.norm(emb_base, p=2, dim=1, keepdim=True)
                    emb_base = emb_base / norms.clamp_min(1e-8)
                    emb_base[torch.isnan(emb_base)] = 0.0
                if emb_llm is not None:
                    norms = torch.norm(emb_llm, p=2, dim=1, keepdim=True)
                    emb_llm = emb_llm / norms.clamp_min(1e-8)
                    emb_llm[torch.isnan(emb_llm)] = 0.0

        self.register_buffer("item_text_emb_base", emb_base if emb_base is not None else None)
        self.register_buffer("item_text_emb_llm", emb_llm if emb_llm is not None else None)

        if self.disable_text_feature:
            self.fuse_text_feature = False
        else:
            if self.use_llm:
                if self.item_text_emb_llm is None:
                    raise ValueError(
                        "BERT4RecAlign: use_llm=True but item_text_emb_path_llm is missing."
                    )
            else:
                if self.item_text_emb_base is None:
                    raise ValueError(
                        "BERT4RecAlign: text features are enabled but item_text_emb_path_base is missing."
                    )

        # Precompute item popularity for optional tail gating
        pop_counts = None
        try:
            inter_iids = dataset.inter_feat[dataset.iid_field].numpy()
            pop_counts = np.bincount(inter_iids, minlength=self.n_items)
        except Exception:
            pop_counts = np.zeros((self.n_items,), dtype=np.int64)
        self.register_buffer("item_popularity", torch.from_numpy(pop_counts).long())
        if self.text_tail_threshold > 0:
            gate = (self.item_popularity <= int(self.text_tail_threshold)).float()
        else:
            gate = None
        self.register_buffer("text_item_gate_all", gate)

        # Determine text input dimension
        base_dim = int(self.item_text_emb_base.shape[1]) if self.item_text_emb_base is not None else 0
        llm_dim = int(self.item_text_emb_llm.shape[1]) if self.item_text_emb_llm is not None else 0
        self._text_mode = "none"
        if self.use_llm:
            if base_dim > 0 and llm_dim > 0:
                self._text_mode = "both"
                text_in_dim = base_dim + llm_dim
            elif llm_dim > 0:
                self._text_mode = "llm"
                text_in_dim = llm_dim
            elif base_dim > 0:
                self._text_mode = "base"
                text_in_dim = base_dim
            else:
                text_in_dim = 0
        else:
            if base_dim > 0:
                self._text_mode = "base"
                text_in_dim = base_dim
            else:
                text_in_dim = 0

        self._text_input_dim = text_in_dim

        # --- Text Amplifier / SENet configuration ---
        self.num_text_views = int(config["num_text_views"]) if "num_text_views" in config else 1
        self.text_use_senet = bool(config["text_use_senet"]) if "text_use_senet" in config else False
        self.text_amplifier = None
        
        if text_in_dim > 0 and self.text_use_senet:
            try:
                from recbole.model.sequential_recommender.text_amplifier import TextFeatureAmplifier
                self.logger.info(f"BERT4RecAlign: initializing TextFeatureAmplifier with {self.num_text_views} views, SENet=True.")
                self.text_amplifier = TextFeatureAmplifier(
                    input_dim=text_in_dim,
                    output_dim=self.hidden_size,
                    num_views=self.num_text_views,
                    apply_senet=True
                )
            except ImportError:
                self.logger.warning("BERT4RecAlign: TextFeatureAmplifier not found. SENet disabled.")
                self.text_use_senet = False

        # Build text projection/fusion modules
        self.text_cross = None
        self.text_deep = None
        self.text_predictor = None
        self.item_text_proj = None
        self.item_concat_predictor = None
        self.text_proj_norm = None
        self.fused_item_norm = None
        self.text_cross_dropout = None
        self.item_fusion_cross_dropout = None
        
        # Item-side fusion modules
        self.item_fusion_cross = None
        self.item_fusion_deep = None
        self.item_fusion_predictor = None
        self.item_emb_norm = None
        
        if text_in_dim > 0:
            if self.fused_item_norm_flag:
                self.item_emb_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

            current_text_dim = text_in_dim
            if self.text_amplifier is not None:
                current_text_dim = self.hidden_size

            if self.use_cross:
                self.text_cross = DCNV2Cross(current_text_dim, num_layers=self.text_cross_layer_num)
                self.text_deep = MLPLayers([current_text_dim, self.hidden_size], dropout=0.0, bn=self.text_mlp_bn)
                self.text_predictor = nn.Linear(current_text_dim + self.hidden_size, self.hidden_size)
                
                fusion_input_dim = self.hidden_size + current_text_dim
                self.item_fusion_cross = DCNV2Cross(fusion_input_dim, num_layers=self.text_cross_layer_num)
                self.item_fusion_deep = MLPLayers([fusion_input_dim, self.hidden_size], dropout=0.0, bn=self.text_mlp_bn)
                self.item_fusion_predictor = nn.Linear(fusion_input_dim + self.hidden_size, self.hidden_size)
                
                if self.cross_dropout_prob > 0.0:
                    self.text_cross_dropout = nn.Dropout(self.cross_dropout_prob)
                    self.item_fusion_cross_dropout = nn.Dropout(self.cross_dropout_prob)
            else:
                if self.text_amplifier is not None:
                    self.item_text_proj = self.text_amplifier
                else:
                    self.item_text_proj = nn.Linear(current_text_dim, self.hidden_size)
                self.item_concat_predictor = nn.Linear(self.hidden_size * 2, self.hidden_size)

            if self.text_proj_norm_flag:
                self.text_proj_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

            if self.fused_item_norm_flag:
                self.fused_item_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self._align_debug_logged = False
        self._gate_debug_logged = False

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ["BPR", "CE"]
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.set_freeze(self.freeze_backbone)

    def set_freeze(self, freeze: bool) -> None:
        """Freeze or unfreeze backbone (ID/position embeddings + transformer encoder + layer norms)."""
        self.item_embedding.weight.requires_grad_(not freeze)
        self.position_embedding.weight.requires_grad_(not freeze)
        for p in self.trm_encoder.parameters():
            p.requires_grad_(not freeze)
        for p in self.LayerNorm.parameters():
            p.requires_grad_(not freeze)
        for p in self.output_ffn.parameters():
            p.requires_grad_(not freeze)
        for p in self.output_ln.parameters():
            p.requires_grad_(not freeze)
        self.output_bias.requires_grad_(not freeze)

    def get_optimizer_grouped_parameters(self, config):
        """Build optimizer param groups for lightweight, per-module learning rates."""
        base_lr = float(config["learning_rate"])
        base_wd = float(config["weight_decay"])
        lr_text_head = float(config["lr_text_head"]) if "lr_text_head" in config else base_lr
        lr_dnn_cross = float(config["lr_dnn_cross"]) if "lr_dnn_cross" in config else base_lr
        lr_backbone = float(config["lr_backbone"]) if "lr_backbone" in config else base_lr
        wd_text_head = float(config["wd_text_head"]) if "wd_text_head" in config else base_wd
        wd_dnn_cross = float(config["wd_dnn_cross"]) if "wd_dnn_cross" in config else base_wd
        wd_backbone = float(config["wd_backbone"]) if "wd_backbone" in config else base_wd

        def collect_params(modules, extra_params=None):
            params = []
            for m in modules:
                if m is None:
                    continue
                for p in m.parameters(recurse=True):
                    if p.requires_grad:
                        params.append(p)
            if extra_params:
                for p in extra_params:
                    if p is not None and getattr(p, "requires_grad", False):
                        params.append(p)
            return params

        text_head_modules = []
        if self.text_amplifier is not None:
            text_head_modules.append(self.text_amplifier)

        if self.use_cross:
            text_head_modules.extend([self.text_cross, self.text_deep, self.text_predictor])
        else:
            if self.item_text_proj is not self.text_amplifier:
                text_head_modules.append(self.item_text_proj)
        if self.text_proj_norm is not None:
            text_head_modules.append(self.text_proj_norm)

        dnn_cross_modules = []
        if self.use_cross:
            dnn_cross_modules.extend([self.item_fusion_cross, self.item_fusion_deep, self.item_fusion_predictor])
        else:
            dnn_cross_modules.append(self.item_concat_predictor)
        if self.fused_item_norm is not None:
            dnn_cross_modules.append(self.fused_item_norm)
        dnn_extra_params = [self.text_gate_param]

        backbone_modules = [self.item_embedding, self.position_embedding, self.trm_encoder, 
                           self.LayerNorm, self.output_ffn, self.output_ln]

        g_text = collect_params(text_head_modules)
        g_dnn = collect_params(dnn_cross_modules, extra_params=dnn_extra_params)
        g_backbone = collect_params(backbone_modules, extra_params=[self.output_bias])

        groups = []
        if len(g_text) > 0:
            groups.append({"params": g_text, "lr": lr_text_head, "weight_decay": wd_text_head})
        if len(g_dnn) > 0:
            groups.append({"params": g_dnn, "lr": lr_dnn_cross, "weight_decay": wd_dnn_cross})
        if len(g_backbone) > 0:
            groups.append({"params": g_backbone, "lr": lr_backbone, "weight_decay": wd_backbone})

        covered = {id(p) for g in groups for p in g["params"]}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in covered]
        if len(rest) > 0:
            groups.append({"params": rest, "lr": base_lr, "weight_decay": base_wd})
        return groups

    def _load_text_embeddings(self, path, expected_rows):
        if path is None or (isinstance(path, str) and path.strip() == ""):
            return None
        if not isinstance(path, str) or not os.path.exists(path):
            return None
        emb = None
        try:
            if path.endswith(".npy"):
                emb_np = np.load(path)
                emb = torch.from_numpy(emb_np).float()
            else:
                loaded = torch.load(path, map_location="cpu")
                if isinstance(loaded, torch.Tensor):
                    emb = loaded.float()
                elif isinstance(loaded, np.ndarray):
                    emb = torch.from_numpy(loaded).float()
                elif isinstance(loaded, dict) and "emb" in loaded:
                    emb = loaded["emb"].float()
        except Exception:
            emb = None
        if emb is None or emb.dim() != 2:
            return None
        if emb.size(0) != expected_rows:
            return None
        return emb

    def _has_item_text(self) -> bool:
        return (
            hasattr(self, "item_text_emb_base")
            and hasattr(self, "item_text_emb_llm")
            and (self.item_text_emb_base is not None or self.item_text_emb_llm is not None)
        )

    def _gather_text_raw(self, ids_flat: torch.Tensor) -> torch.Tensor:
        parts = []
        if self._text_mode in ("base", "both") and self.item_text_emb_base is not None:
            parts.append(self.item_text_emb_base[ids_flat])
        if self._text_mode in ("llm", "both") and self.item_text_emb_llm is not None:
            parts.append(self.item_text_emb_llm[ids_flat])
        if len(parts) == 0:
            return torch.zeros((ids_flat.size(0), 0), device=ids_flat.device)
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def _project_text(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.size(1) == 0:
            return torch.zeros((raw.size(0), self.hidden_size), device=raw.device)
        if self.use_cross and self.text_cross is not None and self.text_deep is not None and self.text_predictor is not None:
            if self.text_amplifier is not None:
                raw = self.text_amplifier(raw)
            
            cross_out = self.text_cross(raw)
            if self.text_cross_dropout is not None:
                cross_out = self.text_cross_dropout(cross_out)
            deep_out = self.text_deep(raw)
            fused = torch.cat([cross_out, deep_out], dim=1)
            proj = self.text_predictor(fused)
        elif (not self.use_cross) and self.item_text_proj is not None:
            proj = self.item_text_proj(raw)
        else:
            proj = torch.zeros((raw.size(0), self.hidden_size), device=raw.device)
        if self.text_proj_norm is not None:
            proj = self.text_proj_norm(proj)
        return proj

    def _compute_cold_start_weights(self, item_ids: torch.Tensor) -> torch.Tensor:
        """计算冷启动商品的对齐损失权重。"""
        if self.cold_start_align_boost <= 0:
            return torch.ones(item_ids.size(0), device=item_ids.device)
        
        item_pop = self.item_popularity[item_ids].float()
        threshold = float(self.cold_start_align_threshold)
        cold_factor = torch.clamp(threshold - item_pop, min=0) / threshold
        weights = 1.0 + self.cold_start_align_boost * cold_factor
        return weights

    def _info_nce_align_weighted(self, a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """带权重的InfoNCE对齐损失。"""
        if a.size(0) == 0 or b.size(0) == 0:
            return torch.zeros(1, device=a.device)
        
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        sim = torch.matmul(a, b.t())
        sim_scaled = sim / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        per_sample_loss = F.cross_entropy(sim_scaled, labels, reduction='none')
        weighted_loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        return weighted_loss

    def _info_nce_align(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.size(0) == 0 or b.size(0) == 0:
            return torch.zeros(1, device=a.device)
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        logits = torch.matmul(a, b.t()) / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        return nn.CrossEntropyLoss()(logits, labels)

    def _get_fused_item_embeddings(self, item_ids: torch.Tensor = None) -> torch.Tensor:
        """Get item embeddings fused with text features.
        Supports chunking to prevent CUDA timeout for large item sets.
        """
        if item_ids is None:
            all_ids = torch.arange(self.n_items, device=self.item_embedding.weight.device)
            # Chunking support to prevent CUDA timeout
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
        
        if not self.fuse_text_feature:
            return item_emb

        if (
            (not self._has_item_text())
            or (self.use_cross and self.item_fusion_predictor is None)
            or ((not self.use_cross) and (self.item_text_proj is None or self.item_concat_predictor is None))
        ):
            return item_emb

        text_raw = self._gather_text_raw(all_ids)
        if self.detach_text_emb:
            text_raw = text_raw.detach()
        
        alpha = torch.sigmoid(self.text_gate_param)
        align_scale = (1.0 + self.alignment_weight) if self.alignment_weight > 0 else 1.0
        temp_scale = (0.07 / self.temperature) if self.temperature > 0 else 1.0
        effective_text_weight = alpha * self.text_weight * align_scale * temp_scale
        
        if self.use_cross and self.item_fusion_predictor is not None:
            if self.text_amplifier is not None:
                text_raw = self.text_amplifier(text_raw)

            if self.text_item_gate_all is not None:
                if item_ids is None:
                    gate = self.text_item_gate_all
                else:
                    gate = self.text_item_gate_all[all_ids]
                gate = gate.to(item_emb.device).unsqueeze(1)
                scaled_text = (effective_text_weight * gate) * text_raw
            else:
                scaled_text = effective_text_weight * text_raw

            item_emb_for_fusion = item_emb
            if self.item_emb_norm is not None:
                item_emb_for_fusion = self.item_emb_norm(item_emb)

            fusion_input = torch.cat([item_emb_for_fusion, scaled_text], dim=1)
            cross_out = self.item_fusion_cross(fusion_input)
            if self.item_fusion_cross_dropout is not None:
                cross_out = self.item_fusion_cross_dropout(cross_out)
            deep_out = self.item_fusion_deep(fusion_input)
            fused = torch.cat([cross_out, deep_out], dim=1)
            fused_emb = self.item_fusion_predictor(fused)
        else:
            text_proj = self._project_text(text_raw)
            if self.text_item_gate_all is not None:
                if item_ids is None:
                    gate = self.text_item_gate_all
                else:
                    gate = self.text_item_gate_all[all_ids]
                gate = gate.to(item_emb.device).unsqueeze(1)
                scaled_text = (effective_text_weight * gate) * text_proj
            else:
                scaled_text = effective_text_weight * text_proj
            
            item_emb_for_fusion = item_emb
            if self.item_emb_norm is not None:
                item_emb_for_fusion = self.item_emb_norm(item_emb)
                
            concat = torch.cat([item_emb_for_fusion, scaled_text], dim=1)
            fused_emb = self.item_concat_predictor(concat)

        if self.fused_item_norm is not None:
            fused_emb = self.fused_item_norm(fused_emb)
        return fused_emb

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reconstruct_test_data(self, item_seq, item_seq_len):
        padding = torch.zeros(
            item_seq.size(0), dtype=torch.long, device=item_seq.device
        )
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        item_seq = item_seq[:, 1:]
        return item_seq

    def forward(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)
        return output

    def multi_hot_embed(self, masked_index, max_length):
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        pos_items = interaction[self.POS_ITEMS]
        neg_items = interaction[self.NEG_ITEMS]
        masked_index = interaction[self.MASK_INDEX]

        seq_output = self.forward(masked_item_seq)
        pred_index_map = self.multi_hot_embed(
            masked_index, masked_item_seq.size(-1)
        )
        pred_index_map = pred_index_map.view(
            masked_index.size(0), masked_index.size(1), -1
        )
        seq_output = torch.bmm(pred_index_map, seq_output)

        if self.loss_type == "BPR":
            pos_items_emb = self._get_fused_item_embeddings(pos_items)
            neg_items_emb = self._get_fused_item_embeddings(neg_items)
            pos_score = (
                torch.sum(seq_output * pos_items_emb, dim=-1) + self.output_bias[pos_items]
            )
            neg_score = (
                torch.sum(seq_output * neg_items_emb, dim=-1) + self.output_bias[neg_items]
            )
            targets_mask = (masked_index > 0).float()
            loss = -torch.sum(
                torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets_mask
            ) / torch.sum(targets_mask)
        elif self.loss_type == "CE":
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            test_item_emb = self._get_fused_item_embeddings()
            if self.cosine_score:
                seq_n = F.normalize(seq_output, dim=-1)
                item_n = F.normalize(test_item_emb, dim=-1)
                logits = self.cosine_scale * torch.matmul(seq_n, item_n.transpose(0, 1)) + self.output_bias
            else:
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) + self.output_bias
            targets_mask = (masked_index > 0).float().view(-1)
            loss = torch.sum(
                loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1))
                * targets_mask
            ) / torch.sum(targets_mask)
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Alignment loss
        if (
            self.use_align
            and self.alignment_weight > 0.0
            and self._has_item_text()
            and ((self.use_cross and self.text_predictor is not None) or ((not self.use_cross) and self.item_text_proj is not None))
        ):
            valid_mask = (masked_index > 0).view(-1)
            if valid_mask.any():
                pos_ids_flat = pos_items.view(-1)[valid_mask]
                id_item_e = self.item_embedding(pos_ids_flat)
                txt_raw = self._gather_text_raw(pos_ids_flat)
                if self.detach_text_emb:
                    txt_raw = txt_raw.detach()
                txt_item_e = self._project_text(txt_raw)
                
                cold_start_weights = self._compute_cold_start_weights(pos_ids_flat)
                use_weighted_align = self.cold_start_align_boost > 0
                
                if use_weighted_align:
                    align_loss = self._info_nce_align_weighted(id_item_e, txt_item_e, cold_start_weights)
                else:
                    align_loss = self._info_nce_align(id_item_e, txt_item_e)
                loss = loss + self.alignment_weight * align_loss

                if not self._align_debug_logged:
                    try:
                        self.logger.info(
                            "BERT4RecAlign: first-step align_loss=%.6f, masked_pos=%d, proj_norm=%.6f",
                            align_loss.item(), int(pos_ids_flat.numel()), float(
                                (self.text_predictor.weight if (self.use_cross and self.text_predictor is not None) else self.item_text_proj.weight).norm().item()
                            ),
                        )
                    except Exception:
                        pass
                    self._align_debug_logged = True

        # Regularization on text gate alpha
        if self.text_gate_reg_l2 > 0.0 or self.text_gate_reg_entropy > 0.0:
            alpha = torch.sigmoid(self.text_gate_param)
            if self.text_gate_reg_l2 > 0.0:
                loss = loss + self.text_gate_reg_l2 * (alpha ** 2)
            if self.text_gate_reg_entropy > 0.0:
                eps = 1e-8
                entropy = -(alpha * torch.log(alpha + eps) + (1.0 - alpha) * torch.log(1.0 - alpha + eps))
                loss = loss + self.text_gate_reg_entropy * entropy

        # Log gate alpha on first training step
        if (not self._gate_debug_logged) and self.training:
            try:
                alpha_val = float(torch.sigmoid(self.text_gate_param).detach().cpu().item())
                self.logger.info(
                    "BERT4RecAlign: first-step text_gate_alpha=%.6f (use_llm=%s, use_cross=%s, use_align=%s, text_mode=%s)",
                    alpha_val, str(self.use_llm), str(self.use_cross), str(self.use_align), getattr(self, "_text_mode", "unknown")
                )
            except Exception:
                pass
            self._gate_debug_logged = True

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self._get_fused_item_embeddings(test_item)
        if self.cosine_score:
            seq_n = F.normalize(seq_output, dim=1)
            item_n = F.normalize(test_item_emb, dim=1)
            scores = self.cosine_scale * torch.mul(seq_n, item_n).sum(dim=1) + self.output_bias[test_item]
        else:
            scores = (torch.mul(seq_output, test_item_emb)).sum(dim=1) + self.output_bias[test_item]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self._get_fused_item_embeddings()
        if self.cosine_score:
            seq_n = F.normalize(seq_output, dim=1)
            item_n = F.normalize(test_items_emb, dim=1)
            scores = self.cosine_scale * torch.matmul(seq_n, item_n.transpose(0, 1)) + self.output_bias
        else:
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.output_bias
        return scores


# Alias to enable model name 'BERT4Rec_Align' to load this module
BERT4Rec_Align = BERT4RecAlign
