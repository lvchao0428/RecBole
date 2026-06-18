# -*- coding: utf-8 -*-
"""
FDSAAlign V3 - FDSA backbone with V3 text pipeline

FDSA backbone (dual transformer + concat) with scoring-only text fusion:
forward() is unchanged; text fusion applies only in calculate_loss / predict / full_sort_predict.

V3 text pipeline: DCN-V2 cross, InfoNCE alignment, cold-start reweighting.
"""

import os
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer, VanillaAttention, MLPLayers
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


class FDSAAlignV3(SequentialRecommender):
    """
    FDSAAlign V3 - FDSA backbone with V3 text pipeline (scoring-only text fusion).

    Core weight parameters:
    - align_weight: alignment loss weight (training)
    - cold_text_boost: cold-start boost for alignment loss (training)
    - infer_boost: inference boost for text signal on cold items
    - cold_threshold: popularity threshold for cold-start reweighting
    """

    def __init__(self, config, dataset):
        super(FDSAAlignV3, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.selected_features = config["selected_features"]
        self.pooling_mode = config["pooling_mode"]
        self.device = config["device"]
        self.num_feature_field = len(config["selected_features"])

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset,
            self.hidden_size,
            self.selected_features,
            self.pooling_mode,
            self.device,
        )

        self.item_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.feature_att_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        # For simplicity, we use same architecture for item_trm and feature_trm
        self.feature_trm_encoder = TransformerEncoder(
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
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.other_parameter_name = ["feature_embed_layer"]

        # additional regularization / scoring configs
        self.label_smoothing = float(config["label_smoothing"]) if "label_smoothing" in config else 0.0
        self.cosine_score = bool(config["cosine_score"]) if "cosine_score" in config else False
        self.cosine_scale = float(config["cosine_scale"]) if "cosine_scale" in config else 10.0

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            try:
                self.loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            except TypeError:
                self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # ============ V3 simplified weight config ============
        self.align_weight = float(config["align_weight"]) if "align_weight" in config else 0.1
        self.cold_text_boost = float(config["cold_text_boost"]) if "cold_text_boost" in config else 0.0
        self.infer_boost = float(config["infer_boost"]) if "infer_boost" in config else 0.0
        self.cold_threshold = int(config["cold_threshold"]) if "cold_threshold" in config else 10

        self.temperature = float(config["temperature"]) if "temperature" in config else 0.07
        self.text_weight = float(config["text_weight"]) if "text_weight" in config else 1.0
        self.normalize_text = bool(config["normalize_text"]) if "normalize_text" in config else True
        self.detach_text_emb = bool(config["detach_text_emb"]) if "detach_text_emb" in config else True
        self.use_llm = bool(config["use_llm"]) if "use_llm" in config else False
        self.use_cross = bool(config["use_cross"]) if "use_cross" in config else False
        self.use_align = bool(config["use_align"]) if "use_align" in config else True
        self.text_cross_layer_num = int(config["text_cross_layer_num"]) if "text_cross_layer_num" in config else 3
        self.cross_dropout_prob = float(config["cross_dropout_prob"]) if "cross_dropout_prob" in config else 0.0
        self.text_gate_init = float(config["text_gate_init"]) if "text_gate_init" in config else 0.5

        # learnable global gate alpha in [0,1] via sigmoid
        self.text_gate_param = nn.Parameter(torch.tensor(self.text_gate_init, dtype=torch.float32))

        self.disable_text_feature = bool(config["disable_text_feature"]) if "disable_text_feature" in config else False
        self.freeze_backbone = bool(config["freeze_backbone"]) if "freeze_backbone" in config else False

        self.fuse_text_feature = bool(config["fuse_text_feature"]) if "fuse_text_feature" in config else True
        self.text_mlp_bn = bool(config["text_mlp_bn"]) if "text_mlp_bn" in config else False
        self.text_proj_norm_flag = bool(config["text_proj_norm"]) if "text_proj_norm" in config else True
        self.fused_item_norm_flag = bool(config["fused_item_norm"]) if "fused_item_norm" in config else True
        self.fusion_chunk_size = int(config["fusion_chunk_size"]) if "fusion_chunk_size" in config else 0

        # Load text embeddings
        item_text_emb_path_base = config["item_text_emb_path_base"] if "item_text_emb_path_base" in config else None
        item_text_emb_path_llm = config["item_text_emb_path_llm"] if "item_text_emb_path_llm" in config else None
        if item_text_emb_path_base is None and item_text_emb_path_llm is None:
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
                        "FDSAAlignV3: use_llm=True but item_text_emb_path_llm is missing."
                    )
            else:
                if self.item_text_emb_base is None:
                    raise ValueError(
                        "FDSAAlignV3: text features are enabled but item_text_emb_path_base is missing."
                    )

        # Precompute item popularity for cold-start weighting
        pop_counts = None
        try:
            inter_iids = dataset.inter_feat[dataset.iid_field].numpy()
            pop_counts = np.bincount(inter_iids, minlength=self.n_items)
        except Exception:
            pop_counts = np.zeros((self.n_items,), dtype=np.int64)
        self.register_buffer("item_popularity", torch.from_numpy(pop_counts).long())

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
        self.item_fusion_cross = None
        self.item_fusion_deep = None
        self.item_fusion_predictor = None
        self.item_emb_norm = None

        if text_in_dim > 0:
            if self.fused_item_norm_flag:
                self.item_emb_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

            if self.use_cross:
                self.text_cross = DCNV2Cross(text_in_dim, num_layers=self.text_cross_layer_num)
                self.text_deep = MLPLayers([text_in_dim, self.hidden_size], dropout=0.0, bn=self.text_mlp_bn)
                self.text_predictor = nn.Linear(text_in_dim + self.hidden_size, self.hidden_size)

                fusion_input_dim = self.hidden_size + text_in_dim
                self.item_fusion_cross = DCNV2Cross(fusion_input_dim, num_layers=self.text_cross_layer_num)
                self.item_fusion_deep = MLPLayers([fusion_input_dim, self.hidden_size], dropout=0.0, bn=self.text_mlp_bn)
                self.item_fusion_predictor = nn.Linear(fusion_input_dim + self.hidden_size, self.hidden_size)

                if self.cross_dropout_prob > 0.0:
                    self.text_cross_dropout = nn.Dropout(self.cross_dropout_prob)
                    self.item_fusion_cross_dropout = nn.Dropout(self.cross_dropout_prob)
            else:
                self.item_text_proj = nn.Linear(text_in_dim, self.hidden_size)
                self.item_concat_predictor = nn.Linear(self.hidden_size * 2, self.hidden_size)

            if self.text_proj_norm_flag:
                self.text_proj_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

            if self.fused_item_norm_flag:
                self.fused_item_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self._align_debug_logged = False
        self._gate_debug_logged = False
        self._fusion_chunk_warned = False

        if self.align_weight > 0.0 and self._has_item_text():
            self.logger.info(
                "FDSAAlignV3: text_mode=%s, align_weight=%.3f, cold_text_boost=%.2f, infer_boost=%.2f, cold_threshold=%d",
                self._text_mode, self.align_weight, self.cold_text_boost, self.infer_boost, self.cold_threshold
            )

        self.apply(self._init_weights)
        self.set_freeze(self.freeze_backbone)

    def set_freeze(self, freeze: bool) -> None:
        """Freeze or unfreeze backbone."""
        self.item_embedding.weight.requires_grad_(not freeze)
        self.position_embedding.weight.requires_grad_(not freeze)
        for p in self.item_trm_encoder.parameters():
            p.requires_grad_(not freeze)
        for p in self.feature_trm_encoder.parameters():
            p.requires_grad_(not freeze)
        for p in self.feature_embed_layer.parameters():
            p.requires_grad_(not freeze)
        for p in self.feature_att_layer.parameters():
            p.requires_grad_(not freeze)
        for p in self.LayerNorm.parameters():
            p.requires_grad_(not freeze)
        for p in self.concat_layer.parameters():
            p.requires_grad_(not freeze)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
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
        if self.use_cross and self.text_cross is not None:
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

    def _compute_cold_weights(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute cold-start weights (used for training and inference).

        weight = 1.0 + boost * max(0, threshold - pop) / threshold
        """
        item_pop = self.item_popularity[item_ids].float()
        threshold = float(self.cold_threshold)
        cold_factor = torch.clamp(threshold - item_pop, min=0) / max(threshold, 1.0)
        return cold_factor  # [0, 1]

    def _info_nce_align(self, a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """InfoNCE alignment loss with optional sample weights."""
        if a.size(0) == 0 or b.size(0) == 0:
            return torch.zeros(1, device=a.device)

        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        sim = torch.matmul(a, b.t())
        sim_scaled = sim / self.temperature
        labels = torch.arange(a.size(0), device=a.device)

        if weights is not None and self.cold_text_boost > 0:
            per_sample_loss = F.cross_entropy(sim_scaled, labels, reduction='none')
            sample_weights = 1.0 + self.cold_text_boost * weights
            weighted_loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
            return weighted_loss
        else:
            return F.cross_entropy(sim_scaled, labels)

    def _get_fused_item_embeddings(self, item_ids: torch.Tensor = None) -> torch.Tensor:
        """Get item embeddings fused with text features."""
        if item_ids is None:
            all_ids = torch.arange(self.n_items, device=self.item_embedding.weight.device)
            item_emb = self.item_embedding.weight
        else:
            all_ids = item_ids
            item_emb = self.item_embedding(item_ids)

        if not self.fuse_text_feature:
            return item_emb

        if not self._has_item_text():
            return item_emb

        if (self.use_cross and self.item_fusion_predictor is None) or \
           ((not self.use_cross) and (self.item_text_proj is None or self.item_concat_predictor is None)):
            return item_emb

        text_raw = self._gather_text_raw(all_ids)
        if self.detach_text_emb:
            text_raw = text_raw.detach()

        alpha = torch.sigmoid(self.text_gate_param)
        effective_text_weight = alpha * self.text_weight

        if self.infer_boost > 0 and not self.training:
            cold_factor = self._compute_cold_weights(all_ids)
            cold_boost = 1.0 + self.infer_boost * cold_factor
            cold_boost = cold_boost.unsqueeze(1).to(item_emb.device)
            effective_text_weight = effective_text_weight * cold_boost

        if self.use_cross and self.item_fusion_predictor is not None:
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
            scaled_text = effective_text_weight * text_proj

            item_emb_for_fusion = item_emb
            if self.item_emb_norm is not None:
                item_emb_for_fusion = self.item_emb_norm(item_emb)

            concat = torch.cat([item_emb_for_fusion, scaled_text], dim=1)
            fused_emb = self.item_concat_predictor(concat)

        if self.fused_item_norm is not None:
            fused_emb = self.fused_item_norm(fused_emb)
        return fused_emb

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # get item_trm_input
        # item position add position embedding
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)

        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding["item"]
        dense_embedding = dense_embedding["item"]

        # concat the sparse embedding and float embedding
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)

        # [batch len num_features hidden_size]
        feature_table = torch.cat(feature_table, dim=-2)

        # feature_emb [batch len hidden]
        # weight [batch len num_features]
        # if only one feature, the weight would be 1.0
        feature_emb, attn_weight = self.feature_att_layer(feature_table)
        # feature position add position embedding
        feature_emb = feature_emb + position_embedding
        feature_emb = self.LayerNorm(feature_emb)
        feature_trm_input = self.dropout(feature_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        item_trm_output = self.item_trm_encoder(
            item_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )
        item_output = item_trm_output[-1]

        feature_trm_output = self.feature_trm_encoder(
            feature_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )  # [B Len H]
        feature_output = feature_trm_output[-1]

        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.gather_indexes(feature_output, item_seq_len - 1)  # [B H]

        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)
        return seq_output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

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

        if (
            self.use_align
            and self.align_weight > 0.0
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

            cold_weights = self._compute_cold_weights(pos_ids_flat) if self.cold_text_boost > 0 else None
            align_loss = self._info_nce_align(id_item_e, txt_item_e, cold_weights)
            loss = loss + self.align_weight * align_loss

            if not self._align_debug_logged:
                try:
                    self.logger.info(
                        "FDSAAlignV3: first-step align_loss=%.6f, batch_pos=%d",
                        align_loss.item(), int(pos_ids_flat.numel())
                    )
                    if self.cold_text_boost > 0 and cold_weights is not None:
                        sample_weights = 1.0 + self.cold_text_boost * cold_weights
                        self.logger.info(
                            "  cold_text_boost=%.2f, weights=[min=%.2f, max=%.2f, mean=%.2f]",
                            self.cold_text_boost,
                            sample_weights.min().item(),
                            sample_weights.max().item(),
                            sample_weights.mean().item()
                        )
                except Exception:
                    pass
                self._align_debug_logged = True

        if (not self._gate_debug_logged) and self.training:
            try:
                alpha_val = float(torch.sigmoid(self.text_gate_param).detach().cpu().item())
                self.logger.info(
                    "FDSAAlignV3: text_gate_alpha=%.4f, text_mode=%s, infer_boost=%.2f",
                    alpha_val, self._text_mode, self.infer_boost
                )
            except Exception:
                pass
            self._gate_debug_logged = True

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
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
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self._get_fused_item_embeddings()
        if self.cosine_score:
            seq_n = F.normalize(seq_output, dim=1)
            item_n = F.normalize(test_items_emb, dim=1)
            scores = self.cosine_scale * torch.matmul(seq_n, item_n.transpose(0, 1))
        else:
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


# Alias
FDSA_Align_V3 = FDSAAlignV3
