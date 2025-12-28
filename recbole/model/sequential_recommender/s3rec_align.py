# -*- coding: utf-8 -*-
# Align-only variant of S3Rec with optional text embedding alignment
# Enhanced version with full TF-IDF + LLM + DCN-V2 support (similar to SASRecAlign)

import random
import os

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, MLPLayers
from recbole.model.loss import BPRLoss


class DCNV2Cross(nn.Module):
    """DCN-V2 cross network (non-mix) over dense features."""

    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_layers = int(max(0, num_layers))
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(torch.randn(self.input_dim, self.input_dim))
            for _ in range(self.num_layers)
        )
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.input_dim, 1))
            for _ in range(self.num_layers)
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 0:
            return x0
        x0_u = x0.unsqueeze(dim=2)
        xl = x0_u
        for i in range(self.num_layers):
            xl_w = torch.matmul(self.cross_layer_w[i], xl)
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x0_u, xl_w)
            xl = xl_dot + xl
        xl = xl.squeeze(dim=2)
        return xl


class S3RecAlign(SequentialRecommender):
    """
    S3RecAlign - Enhanced S3Rec with text alignment support.
    
    NOTE: S3Rec has two stages: pretrain and finetune.
    Text alignment is only applied during the finetune stage.
    """

    def __init__(self, config, dataset):
        super(S3RecAlign, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.FEATURE_FIELD = config["item_attribute"]
        self.FEATURE_LIST = self.FEATURE_FIELD + config["LIST_SUFFIX"]
        self.train_stage = config["train_stage"]  # pretrain or finetune
        self.pre_model_path = config["pre_model_path"]
        self.mask_ratio = config["mask_ratio"]
        self.aap_weight = config["aap_weight"]
        self.mip_weight = config["mip_weight"]
        self.map_weight = config["map_weight"]
        self.sp_weight = config["sp_weight"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # load dataset info
        self.n_items = dataset.item_num + 1  # for mask token
        self.mask_token = self.n_items - 1
        self.n_features = dataset.num(self.FEATURE_FIELD) - 1
        self.item_feat = dataset.get_item_feature()

        # Additional regularization / scoring configs
        self.label_smoothing = float(config["label_smoothing"]) if "label_smoothing" in config else 0.0
        self.cosine_score = bool(config["cosine_score"]) if "cosine_score" in config else False
        self.cosine_scale = float(config["cosine_scale"]) if "cosine_scale" in config else 10.0

        # define layers
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.feature_embedding = nn.Embedding(self.n_features, self.hidden_size, padding_idx=0)

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

        # modules for pretrain
        self.aap_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.mip_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.map_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.sp_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.pretrain_loss_fct = nn.BCEWithLogitsLoss(reduction="none")

        # --- text-alignment settings & feature fusion (finetune only) ---
        self.alignment_weight = config["alignment_weight"] if "alignment_weight" in config else 0.0
        self.temperature = config["temperature"] if "temperature" in config else 0.07
        self.normalize_text = config["normalize_text"] if "normalize_text" in config else True
        self.detach_text_emb = config["detach_text_emb"] if "detach_text_emb" in config else True
        self.use_llm = config["use_llm"] if "use_llm" in config else False
        self.use_cross = config["use_cross"] if "use_cross" in config else False
        self.use_align = config["use_align"] if "use_align" in config else True
        self.text_cross_layer_num = config["text_cross_layer_num"] if "text_cross_layer_num" in config else 3
        
        self.cross_dropout_prob = float(config["cross_dropout_prob"]) if "cross_dropout_prob" in config else 0.0
        self.text_gate_init = float(config["text_gate_init"]) if "text_gate_init" in config else 0.5
        self.text_gate_reg_l2 = float(config["text_gate_reg_l2"]) if "text_gate_reg_l2" in config else 0.0
        self.text_gate_reg_entropy = float(config["text_gate_reg_entropy"]) if "text_gate_reg_entropy" in config else 0.0
        self.text_gate_param = nn.Parameter(torch.tensor(self.text_gate_init, dtype=torch.float32))
        
        self.disable_text_feature = bool(config["disable_text_feature"]) if "disable_text_feature" in config else False
        self.freeze_backbone = bool(config["freeze_backbone"]) if "freeze_backbone" in config else False
        
        self.cold_start_align_boost = float(config["cold_start_align_boost"]) if "cold_start_align_boost" in config else 0.0
        self.cold_start_align_threshold = int(config["cold_start_align_threshold"]) if "cold_start_align_threshold" in config else 10
        
        self.text_weight = float(config["text_weight"]) if "text_weight" in config else 1.0
        self.text_tail_threshold = int(config["text_tail_threshold"]) if "text_tail_threshold" in config else 0
        self.fuse_text_feature = bool(config["fuse_text_feature"]) if "fuse_text_feature" in config else True
        self.text_mlp_bn = bool(config["text_mlp_bn"]) if "text_mlp_bn" in config else False
        self.text_proj_norm_flag = bool(config["text_proj_norm"]) if "text_proj_norm" in config else True
        self.fused_item_norm_flag = bool(config["fused_item_norm"]) if "fused_item_norm" in config else True

        # Load text embeddings (for finetune stage)
        item_text_emb_path_base = config["item_text_emb_path_base"] if "item_text_emb_path_base" in config else None
        item_text_emb_path_llm = config["item_text_emb_path_llm"] if "item_text_emb_path_llm" in config else None
        if item_text_emb_path_base is None and item_text_emb_path_llm is None:
            item_text_emb_path_base = config["item_text_emb_path"] if "item_text_emb_path" in config else None

        # Note: n_items includes mask token, but text embeddings don't have mask token
        actual_n_items = self.n_items - 1
        
        if self.disable_text_feature or self.train_stage == "pretrain":
            emb_base = None
            emb_llm = None
        else:
            emb_base = self._load_text_embeddings(item_text_emb_path_base, actual_n_items)
            emb_llm = self._load_text_embeddings(item_text_emb_path_llm, actual_n_items)

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

        self.register_buffer("item_text_emb_base", emb_base)
        self.register_buffer("item_text_emb_llm", emb_llm)

        if self.disable_text_feature or self.train_stage == "pretrain":
            self.fuse_text_feature = False

        # Precompute item popularity
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

        # SENet configuration
        self.num_text_views = int(config["num_text_views"]) if "num_text_views" in config else 1
        self.text_use_senet = bool(config["text_use_senet"]) if "text_use_senet" in config else False
        self.text_amplifier = None
        
        if text_in_dim > 0 and self.text_use_senet and self.train_stage == "finetune":
            try:
                from recbole.model.sequential_recommender.text_amplifier import TextFeatureAmplifier
                self.text_amplifier = TextFeatureAmplifier(
                    input_dim=text_in_dim,
                    output_dim=self.hidden_size,
                    num_views=self.num_text_views,
                    apply_senet=True
                )
            except ImportError:
                self.text_use_senet = False

        # Build text projection/fusion modules (finetune only)
        self.text_cross = None
        self.text_deep = None
        self.text_predictor = None
        self.item_text_proj = None
        self.item_concat_predictor = None
        self.text_proj_norm = None
        self.fused_item_norm = None
        self.text_cross_dropout = None
        self.item_fusion_cross_dropout = None
        self.item_fusion_cross = None
        self.item_fusion_deep = None
        self.item_fusion_predictor = None
        self.item_emb_norm = None
        
        if text_in_dim > 0 and self.train_stage == "finetune":
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

        # Loss function for finetune
        if self.loss_type == "BPR" and self.train_stage == "finetune":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE" and self.train_stage == "finetune":
            try:
                self.loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            except TypeError:
                self.loss_fct = nn.CrossEntropyLoss()
        elif self.train_stage == "finetune":
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        assert self.train_stage in ["pretrain", "finetune"]
        if self.train_stage == "pretrain":
            self.apply(self._init_weights)
        else:
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            self.logger.info(f"Load pretrained model from {self.pre_model_path}")
            self.load_state_dict(pretrained["state_dict"], strict=False)
            self.set_freeze(self.freeze_backbone)

    def set_freeze(self, freeze: bool) -> None:
        """Freeze or unfreeze backbone."""
        self.item_embedding.weight.requires_grad_(not freeze)
        self.position_embedding.weight.requires_grad_(not freeze)
        self.feature_embedding.weight.requires_grad_(not freeze)
        for p in self.trm_encoder.parameters():
            p.requires_grad_(not freeze)
        for p in self.LayerNorm.parameters():
            p.requires_grad_(not freeze)

    def get_optimizer_grouped_parameters(self, config):
        """Build optimizer param groups for per-module learning rates."""
        base_lr = float(config["learning_rate"])
        base_wd = float(config["weight_decay"])
        lr_text_head = float(config["lr_text_head"]) if "lr_text_head" in config else base_lr
        lr_dnn_cross = float(config["lr_dnn_cross"]) if "lr_dnn_cross" in config else base_lr
        lr_backbone = float(config["lr_backbone"]) if "lr_backbone" in config else base_lr

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

        backbone_modules = [self.item_embedding, self.position_embedding, self.feature_embedding, 
                           self.trm_encoder, self.LayerNorm]

        g_text = collect_params(text_head_modules)
        g_dnn = collect_params(dnn_cross_modules, extra_params=[self.text_gate_param])
        g_backbone = collect_params(backbone_modules)

        groups = []
        if len(g_text) > 0:
            groups.append({"params": g_text, "lr": lr_text_head, "weight_decay": base_wd})
        if len(g_dnn) > 0:
            groups.append({"params": g_dnn, "lr": lr_dnn_cross, "weight_decay": base_wd})
        if len(g_backbone) > 0:
            groups.append({"params": g_backbone, "lr": lr_backbone, "weight_decay": base_wd})

        covered = {id(p) for g in groups for p in g["params"]}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in covered]
        if len(rest) > 0:
            groups.append({"params": rest, "lr": base_lr, "weight_decay": base_wd})
        return groups

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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
        # Clamp ids to valid range (exclude mask token)
        ids_clamped = torch.clamp(ids_flat, 0, self.n_items - 2)
        parts = []
        if self._text_mode in ("base", "both") and self.item_text_emb_base is not None:
            parts.append(self.item_text_emb_base[ids_clamped])
        if self._text_mode in ("llm", "both") and self.item_text_emb_llm is not None:
            parts.append(self.item_text_emb_llm[ids_clamped])
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
        if self.cold_start_align_boost <= 0:
            return torch.ones(item_ids.size(0), device=item_ids.device)
        
        ids_clamped = torch.clamp(item_ids, 0, self.n_items - 1)
        item_pop = self.item_popularity[ids_clamped].float()
        threshold = float(self.cold_start_align_threshold)
        cold_factor = torch.clamp(threshold - item_pop, min=0) / threshold
        weights = 1.0 + self.cold_start_align_boost * cold_factor
        return weights

    def _info_nce_align_weighted(self, a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
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
        """Get item embeddings fused with text features (finetune only)."""
        if item_ids is None:
            all_ids = torch.arange(self.n_items - 1, device=self.item_embedding.weight.device)
            item_emb = self.item_embedding.weight[:self.n_items - 1]
        else:
            all_ids = item_ids
            item_emb = self.item_embedding(item_ids)
        
        if not self.fuse_text_feature or self.train_stage == "pretrain":
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

            ids_clamped = torch.clamp(all_ids, 0, self.item_popularity.size(0) - 1)
            if self.text_item_gate_all is not None:
                gate = self.text_item_gate_all[ids_clamped]
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
            ids_clamped = torch.clamp(all_ids, 0, self.item_popularity.size(0) - 1)
            if self.text_item_gate_all is not None:
                gate = self.text_item_gate_all[ids_clamped]
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

    # ========== S3Rec pretrain methods ==========
    def _associated_attribute_prediction(self, sequence_output, feature_embedding):
        sequence_output = self.aap_norm(sequence_output)
        sequence_output = sequence_output.view([-1, sequence_output.size(-1), 1])
        score = torch.matmul(feature_embedding, sequence_output)
        return score.squeeze(-1)

    def _masked_item_prediction(self, sequence_output, target_item_emb):
        sequence_output = self.mip_norm(sequence_output.view([-1, sequence_output.size(-1)]))
        target_item_emb = target_item_emb.view([-1, sequence_output.size(-1)])
        score = torch.mul(sequence_output, target_item_emb)
        return torch.sigmoid(torch.sum(score, -1))

    def _masked_attribute_prediction(self, sequence_output, feature_embedding):
        sequence_output = self.map_norm(sequence_output)
        sequence_output = sequence_output.view([-1, sequence_output.size(-1), 1])
        score = torch.matmul(feature_embedding, sequence_output)
        return score.squeeze(-1)

    def _segment_prediction(self, context, segment_emb):
        context = self.sp_norm(context)
        score = torch.mul(context, segment_emb)
        return torch.sigmoid(torch.sum(score, dim=-1))

    def forward(self, item_seq, bidirectional=True):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        attention_mask = self.get_attention_mask(item_seq, bidirectional=bidirectional)
        trm_output = self.trm_encoder(input_emb, attention_mask, output_all_encoded_layers=True)
        seq_output = trm_output[-1]
        return seq_output

    def pretrain(self, features, masked_item_sequence, pos_items, neg_items,
                 masked_segment_sequence, pos_segment, neg_segment):
        """Pretrain model using four pre-training tasks."""
        sequence_output = self.forward(masked_item_sequence)
        feature_embedding = self.feature_embedding.weight

        # AAP
        aap_score = self._associated_attribute_prediction(sequence_output, feature_embedding)
        aap_loss = self.pretrain_loss_fct(aap_score, features.view(-1, self.n_features).float())
        aap_mask = (masked_item_sequence != self.mask_token).float() * (masked_item_sequence != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embedding(pos_items)
        neg_item_embs = self.item_embedding(neg_items)
        pos_score = self._masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self._masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = pos_score - neg_score
        mip_loss = self.pretrain_loss_fct(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_item_sequence == self.mask_token).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self._masked_attribute_prediction(sequence_output, feature_embedding)
        map_loss = self.pretrain_loss_fct(map_score, features.view(-1, self.n_features).float())
        map_mask = (masked_item_sequence == self.mask_token).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        segment_context = self.forward(masked_segment_sequence)[:, -1, :]
        pos_segment_emb = self.forward(pos_segment)[:, -1, :]
        neg_segment_emb = self.forward(neg_segment)[:, -1, :]
        pos_segment_score = self._segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self._segment_prediction(segment_context, neg_segment_emb)
        sp_distance = pos_segment_score - neg_segment_score
        sp_loss = torch.sum(self.pretrain_loss_fct(sp_distance, torch.ones_like(sp_distance, dtype=torch.float32)))

        pretrain_loss = (
            self.aap_weight * aap_loss
            + self.mip_weight * mip_loss
            + self.map_weight * map_loss
            + self.sp_weight * sp_loss
        )
        return pretrain_loss

    def _neg_sample(self, item_set):
        item = random.randint(1, self.n_items - 2)  # Exclude mask token
        while item in item_set:
            item = random.randint(1, self.n_items - 2)
        return item

    def _padding_zero_at_left(self, sequence):
        pad_len = self.max_seq_length - len(sequence)
        sequence = [0] * pad_len + sequence
        return sequence

    def reconstruct_pretrain_data(self, item_seq, item_seq_len):
        """Generate pre-training data for the pre-training stage."""
        device = item_seq.device
        batch_size = item_seq.size(0)

        item_feature_seq = self.item_feat[self.FEATURE_FIELD][item_seq.cpu()] - 1
        end_index = item_seq_len.cpu().numpy().tolist()
        item_seq = item_seq.cpu().numpy().tolist()
        item_feature_seq = item_feature_seq.cpu().numpy().tolist()

        sequence_instances = []
        associated_features = []
        long_sequence = []
        for i, end_i in enumerate(end_index):
            sequence_instances.append(item_seq[i][:end_i])
            long_sequence.extend(item_seq[i][:end_i])
            associated_features.extend([[0] * self.n_features] * (self.max_seq_length - end_i))
            for indexes in item_feature_seq[i][:end_i]:
                features = [0] * self.n_features
                try:
                    for index in indexes:
                        if index >= 0:
                            features[index] = 1
                except:
                    features[indexes] = 1
                associated_features.append(features)

        masked_item_sequence = []
        pos_items = []
        neg_items = []
        for instance in sequence_instances:
            masked_sequence = instance.copy()
            pos_item = instance.copy()
            neg_item = instance.copy()
            for index_id, item in enumerate(instance):
                prob = random.random()
                if prob < self.mask_ratio:
                    masked_sequence[index_id] = self.mask_token
                    neg_item[index_id] = self._neg_sample(instance)
            masked_item_sequence.append(self._padding_zero_at_left(masked_sequence))
            pos_items.append(self._padding_zero_at_left(pos_item))
            neg_items.append(self._padding_zero_at_left(neg_item))

        masked_segment_list = []
        pos_segment_list = []
        neg_segment_list = []
        for instance in sequence_instances:
            if len(instance) < 2:
                masked_segment = instance.copy()
                pos_segment = instance.copy()
                neg_segment = instance.copy()
            else:
                sample_length = random.randint(1, len(instance) // 2)
                start_id = random.randint(0, len(instance) - sample_length)
                neg_start_id = random.randint(0, len(long_sequence) - sample_length)
                pos_segment = instance[start_id : start_id + sample_length]
                neg_segment = long_sequence[neg_start_id : neg_start_id + sample_length]
                masked_segment = (
                    instance[:start_id]
                    + [self.mask_token] * sample_length
                    + instance[start_id + sample_length :]
                )
                pos_segment = (
                    [self.mask_token] * start_id
                    + pos_segment
                    + [self.mask_token] * (len(instance) - (start_id + sample_length))
                )
                neg_segment = (
                    [self.mask_token] * start_id
                    + neg_segment
                    + [self.mask_token] * (len(instance) - (start_id + sample_length))
                )
            masked_segment_list.append(self._padding_zero_at_left(masked_segment))
            pos_segment_list.append(self._padding_zero_at_left(pos_segment))
            neg_segment_list.append(self._padding_zero_at_left(neg_segment))

        associated_features = torch.tensor(associated_features, dtype=torch.long, device=device)
        associated_features = associated_features.view(-1, self.max_seq_length, self.n_features)

        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        masked_segment_list = torch.tensor(masked_segment_list, dtype=torch.long, device=device).view(batch_size, -1)
        pos_segment_list = torch.tensor(pos_segment_list, dtype=torch.long, device=device).view(batch_size, -1)
        neg_segment_list = torch.tensor(neg_segment_list, dtype=torch.long, device=device).view(batch_size, -1)

        return (
            associated_features,
            masked_item_sequence,
            pos_items,
            neg_items,
            masked_segment_list,
            pos_segment_list,
            neg_segment_list,
        )

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        if self.train_stage == "pretrain":
            (
                features,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            ) = self.reconstruct_pretrain_data(item_seq, item_seq_len)

            loss = self.pretrain(
                features,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            )
        else:
            # Finetune stage with text alignment
            pos_items = interaction[self.POS_ITEM_ID]
            seq_output = self.forward(item_seq, bidirectional=False)
            seq_output = self.gather_indexes(seq_output, item_seq_len - 1)

            if self.loss_type == "BPR":
                neg_items = interaction[self.NEG_ITEM_ID]
                pos_items_emb = self._get_fused_item_embeddings(pos_items)
                neg_items_emb = self._get_fused_item_embeddings(neg_items)
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
                loss = self.loss_fct(pos_score, neg_score)
            else:  # CE
                test_item_emb = self._get_fused_item_embeddings()
                if self.cosine_score:
                    seq_n = F.normalize(seq_output, dim=1)
                    item_n = F.normalize(test_item_emb, dim=1)
                    logits = self.cosine_scale * torch.matmul(seq_n, item_n.transpose(0, 1))
                else:
                    logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = self.loss_fct(logits, pos_items)

            # Alignment loss
            if (
                self.use_align
                and self.alignment_weight > 0.0
                and self._has_item_text()
                and ((self.use_cross and self.text_predictor is not None) or 
                     ((not self.use_cross) and self.item_text_proj is not None))
            ):
                pos_ids_flat = pos_items.view(-1)
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
                            "S3RecAlign: first-step align_loss=%.6f, batch_pos=%d",
                            align_loss.item(), int(pos_ids_flat.numel())
                        )
                    except Exception:
                        pass
                    self._align_debug_logged = True

            # Regularization on text gate
            if self.text_gate_reg_l2 > 0.0 or self.text_gate_reg_entropy > 0.0:
                alpha = torch.sigmoid(self.text_gate_param)
                if self.text_gate_reg_l2 > 0.0:
                    loss = loss + self.text_gate_reg_l2 * (alpha ** 2)
                if self.text_gate_reg_entropy > 0.0:
                    eps = 1e-8
                    entropy = -(alpha * torch.log(alpha + eps) + (1.0 - alpha) * torch.log(1.0 - alpha + eps))
                    loss = loss + self.text_gate_reg_entropy * entropy

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, bidirectional=False)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self._get_fused_item_embeddings(test_item)
        if self.cosine_score:
            seq_n = F.normalize(seq_output, dim=1)
            item_n = F.normalize(test_item_emb, dim=1)
            scores = self.cosine_scale * torch.mul(seq_n, item_n).sum(dim=1)
        else:
            scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, bidirectional=False)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self._get_fused_item_embeddings()
        if self.cosine_score:
            seq_n = F.normalize(seq_output, dim=1)
            item_n = F.normalize(test_items_emb, dim=1)
            scores = self.cosine_scale * torch.matmul(seq_n, item_n.transpose(0, 1))
        else:
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


# Alias
S3Rec_Align = S3RecAlign

