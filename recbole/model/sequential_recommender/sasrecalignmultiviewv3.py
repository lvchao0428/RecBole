"""
SASRecAlignMultiViewV3 - Multi-View text features aligned with V3 LLM fusion.

Design (SENet removed):
1. Concat selected raw views (+ optional TF base) along feature dim → fusion tower.
2. Project with a single Linear(concat_dim → hidden), same role as V3 item_text_proj.
3. Fuse with ID emb via concat+predictor (no-Cross) or DCN (Cross) — same as V3.

Per-source align (pre-concat, consistent with V3):
  Each text source is independently aligned with ID via InfoNCE before concat.
  - TF-IDF → align_proj_base (inherited from V3, Linear(base_dim→H))
  - Each view → shared_view_align_proj (Linear(view_dim→H), shared across views)
  Total: (1 + N_views) × InfoNCE, averaged. align_proj_base is reused from parent.

Extensibility for single-view ablation:
- text_view_indices: null | [0,1,2,3] | [0] | [1,2] …
- mv_include_base: default True — prepend TF-IDF like V3 LLM text_mode=both.

Weights from SASRecAlignV3: align_weight / cold_text_boost / infer_boost / cold_threshold.
SENet is permanently removed (no modules, no config hooks).
"""

import json
import os
import numpy as np
import torch
from torch import nn

from recbole.model.sequential_recommender.sasrecalignv3 import SASRecAlignV3, DCNV2Cross
from recbole.model.layers import MLPLayers


class SASRecAlignMultiViewV3(SASRecAlignV3):
    """
    Multi-view text tower that mirrors V3 LLM fusion:

        raw_views_concat [B, Σ d_v]  →  mv_text_proj  →  [B, H]
        fuse(ID, text) via concat+predictor or DCN-V2
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_text_view_split = bool(config["use_text_view_split"]) if "use_text_view_split" in config else False
        self.use_multiview_text_cross = (
            bool(config["use_multiview_text_cross"]) if "use_multiview_text_cross" in config else False
        )

        if "item_text_emb_split_dir" in config and config["item_text_emb_split_dir"]:
            self.text_view_split_dir = os.path.abspath(os.path.expanduser(config["item_text_emb_split_dir"]))
        else:
            self.text_view_split_dir = None

        # Default on: match V3 LLM text_mode=both (TF ∥ semantic). Off = views-only ablation.
        self.mv_include_base = bool(config["mv_include_base"]) if "mv_include_base" in config else True

        self.text_view_half_precision = (
            bool(config["text_view_half_precision"]) if "text_view_half_precision" in config else True
        )
        self.text_view_storage_dtype = torch.float16 if self.text_view_half_precision else torch.float32

        raw_indices = config["text_view_indices"] if "text_view_indices" in config else None
        self._text_view_indices_cfg = raw_indices

        self.text_view_buffer_names = []
        self.text_view_dims = []
        self.active_view_indices = []
        self.active_buffer_positions = []
        self.mv_text_proj = None
        self.mv_item_concat_predictor = None
        self.mv_text_input_dim = 0

        self.multiview_text_cross = None
        self.multiview_text_deep = None
        self.multiview_text_predictor = None
        self.multiview_text_cross_dropout = None

        # Per-view align: each view independently aligned with ID via shared proj
        self.use_per_view_align = bool(config["use_per_view_align"]) if "use_per_view_align" in config else True
        self.shared_view_align_proj = None

        if self.use_text_view_split:
            self._init_multiview_tower(config)

        self._multiview_align_debug_logged = False

    def _init_multiview_tower(self, config):
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

        index_to_buffer_pos = {}
        for view in prompts:
            idx = int(view["index"])
            file_name = view["file"]
            file_path = os.path.join(self.text_view_split_dir, file_name)
            if not os.path.exists(file_path):
                raise ValueError(f"Split embedding file not found: {file_path}")

            npy = np.load(file_path)
            tensor = torch.from_numpy(npy).to(self.text_view_storage_dtype)
            buffer_name = f"text_view_emb_{idx}"
            self.register_buffer(buffer_name, tensor)
            index_to_buffer_pos[idx] = len(self.text_view_buffer_names)
            self.text_view_buffer_names.append(buffer_name)
            self.text_view_dims.append(int(view["vector_dim"]))

        if self._text_view_indices_cfg is None or self._text_view_indices_cfg == "all":
            selected = [int(v["index"]) for v in prompts]
        else:
            selected = [int(i) for i in list(self._text_view_indices_cfg)]
            for i in selected:
                if i not in index_to_buffer_pos:
                    raise ValueError(f"text_view_indices contains unknown view index {i}")

        self.active_view_indices = selected
        self.active_buffer_positions = [index_to_buffer_pos[i] for i in selected]

        view_concat_dim = sum(self.text_view_dims[pos] for pos in self.active_buffer_positions)
        base_dim = 0
        if self.mv_include_base and self.item_text_emb_base is not None:
            base_dim = int(self.item_text_emb_base.shape[1])

        self.mv_text_input_dim = base_dim + view_concat_dim
        if self.mv_text_input_dim <= 0:
            raise ValueError("MV text input dim is 0; check views / mv_include_base.")

        if self.use_cross:
            # Text-side DCN on raw concat (analogous to V3 text_cross on LLM raw)
            self.multiview_text_cross = DCNV2Cross(self.mv_text_input_dim, num_layers=self.text_cross_layer_num)
            self.multiview_text_deep = MLPLayers(
                [self.mv_text_input_dim, self.hidden_size], dropout=0.0, bn=self.text_mlp_bn
            )
            self.multiview_text_predictor = nn.Linear(
                self.mv_text_input_dim + self.hidden_size, self.hidden_size
            )
            if self.cross_dropout_prob > 0.0:
                self.multiview_text_cross_dropout = nn.Dropout(self.cross_dropout_prob)

            fusion_input_dim = self.hidden_size + self.hidden_size
            self.item_fusion_cross = DCNV2Cross(fusion_input_dim, num_layers=self.text_cross_layer_num)
            self.item_fusion_deep = MLPLayers(
                [fusion_input_dim, self.hidden_size], dropout=0.0, bn=self.text_mlp_bn
            )
            self.item_fusion_predictor = nn.Linear(fusion_input_dim + self.hidden_size, self.hidden_size)
            if self.cross_dropout_prob > 0.0:
                self.item_fusion_cross_dropout = nn.Dropout(self.cross_dropout_prob)
        else:
            # LLM-parity: Linear(concat_dim → H) then ID∥text concat predictor
            self.mv_text_proj = nn.Linear(self.mv_text_input_dim, self.hidden_size)
            self.mv_item_concat_predictor = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if self.item_emb_norm is None and self.fused_item_norm_flag:
            self.item_emb_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        if self.fused_item_norm is None and self.fused_item_norm_flag:
            self.fused_item_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        if self.text_proj_norm is None and self.text_proj_norm_flag:
            self.text_proj_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # Per-view align: shared lightweight projection (view_dim → H)
        # All active views must share the same dim for shared proj; if mixed, fall back to concat align.
        active_dims = [self.text_view_dims[pos] for pos in self.active_buffer_positions]
        self._uniform_view_dim = active_dims[0] if len(set(active_dims)) == 1 else None

        if self.use_per_view_align and self._uniform_view_dim is not None:
            self.shared_view_align_proj = nn.Linear(self._uniform_view_dim, self.hidden_size)
            view_align_str = f"per-view (shared {self._uniform_view_dim}→{self.hidden_size}, {len(self.active_buffer_positions)} views)"
        elif self.use_per_view_align and self._uniform_view_dim is None:
            self.logger.warning(
                "use_per_view_align=True but views have mixed dims %s; falling back to concat align.",
                active_dims,
            )
            self.use_per_view_align = False
            view_align_str = "concat (fallback, mixed view dims)"
        else:
            view_align_str = "concat"

        align_sources = []
        if self.mv_include_base and self.align_proj_base is not None:
            align_sources.append(f"base({self._base_dim}→{self.hidden_size})")
        align_sources.append(view_align_str)

        self.logger.info(
            "SASRecAlignMultiViewV3: views=%s (active=%s), raw_concat_dim=%d "
            "(base=%d + views=%d), mv_include_base=%s, SENet=removed, fusion=%s, "
            "per-source align=[%s]",
            self.num_text_views,
            self.active_view_indices,
            self.mv_text_input_dim,
            base_dim,
            view_concat_dim,
            self.mv_include_base,
            "cross" if self.use_cross else "concat+proj",
            " + ".join(align_sources),
        )
        self.logger.info(
            "V3 weights: align_weight=%.3f, cold_text_boost=%.2f, infer_boost=%.2f, cold_threshold=%d",
            self.align_weight,
            self.cold_text_boost,
            self.infer_boost,
            self.cold_threshold,
        )

    def _get_view_buffer(self, buffer_pos: int) -> torch.Tensor:
        name = self.text_view_buffer_names[buffer_pos]
        return getattr(self, name)

    def _gather_mv_text_raw(self, ids_flat: torch.Tensor) -> torch.Tensor:
        """Concat raw active views (+ optional base). Mirrors V3._gather_text_raw for LLM."""
        parts = []
        if self.mv_include_base and self.item_text_emb_base is not None:
            parts.append(self.item_text_emb_base[ids_flat])

        for pos in self.active_buffer_positions:
            table = self._get_view_buffer(pos)
            if table.device != ids_flat.device:
                table = table.to(ids_flat.device)
            parts.append(table[ids_flat])

        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def _project_mv_text(self, raw: torch.Tensor) -> torch.Tensor:
        """Project concat raw → hidden (same role as V3._project_text)."""
        if self.use_cross and self.multiview_text_predictor is not None:
            if raw.dtype != next(self.multiview_text_cross.parameters()).dtype:
                raw = raw.float()
            cross_out = self.multiview_text_cross(raw)
            if self.multiview_text_cross_dropout is not None:
                cross_out = self.multiview_text_cross_dropout(cross_out)
            deep_out = self.multiview_text_deep(raw)
            fused = torch.cat([cross_out, deep_out], dim=1)
            proj = self.multiview_text_predictor(fused)
        elif self.mv_text_proj is not None:
            if raw.dtype != self.mv_text_proj.weight.dtype:
                raw = raw.to(self.mv_text_proj.weight.dtype)
            proj = self.mv_text_proj(raw)
        else:
            proj = torch.zeros((raw.size(0), self.hidden_size), device=raw.device)

        if self.text_proj_norm is not None:
            proj = self.text_proj_norm(proj)
        return proj

    def _get_fused_item_embeddings(self, item_ids: torch.Tensor = None) -> torch.Tensor:
        if not self.use_text_view_split:
            return super()._get_fused_item_embeddings(item_ids)

        if item_ids is None:
            all_ids = torch.arange(self.n_items, device=self.item_embedding.weight.device)
            if (
                isinstance(self.fusion_chunk_size, int)
                and self.fusion_chunk_size > 0
                and self.fusion_chunk_size < all_ids.numel()
            ):
                chunks = [
                    self._get_fused_item_embeddings(chunk_ids)
                    for chunk_ids in torch.split(all_ids, self.fusion_chunk_size)
                ]
                return torch.cat(chunks, dim=0)
            item_emb = self.item_embedding.weight
        else:
            all_ids = item_ids
            item_emb = self.item_embedding(item_ids)

        if not self.fuse_text_feature or len(self.text_view_buffer_names) == 0:
            return item_emb

        text_raw = self._gather_mv_text_raw(all_ids)
        if self.detach_text_emb:
            text_raw = text_raw.detach()

        text_proj = self._project_mv_text(text_raw)
        return self._fuse_id_and_text(item_emb, text_proj, all_ids)

    def _fuse_id_and_text(
        self, item_emb: torch.Tensor, text_proj: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        alpha = torch.sigmoid(self.text_gate_param) if hasattr(self, "text_gate_param") else torch.tensor(1.0)
        effective_text_weight = alpha * self.text_weight

        if self.infer_boost > 0 and item_ids is not None and not self.training:
            cold_factor = self._compute_cold_weights(item_ids)
            cold_boost = (1.0 + self.infer_boost * cold_factor).unsqueeze(1).to(item_emb.device)
            effective_text_weight = effective_text_weight * cold_boost

        item_emb_for_fusion = item_emb
        if self.item_emb_norm is not None:
            item_emb_for_fusion = self.item_emb_norm(item_emb_for_fusion)

        scaled_text = effective_text_weight * text_proj

        if self.use_cross and self.item_fusion_predictor is not None:
            fusion_input = torch.cat([item_emb_for_fusion, scaled_text], dim=1)
            cross_out = self.item_fusion_cross(fusion_input)
            if self.item_fusion_cross_dropout is not None:
                cross_out = self.item_fusion_cross_dropout(cross_out)
            deep_out = self.item_fusion_deep(fusion_input)
            fused = torch.cat([cross_out, deep_out], dim=1)
            fused_emb = self.item_fusion_predictor(fused)
        elif self.mv_item_concat_predictor is not None:
            fused_emb = self.mv_item_concat_predictor(
                torch.cat([item_emb_for_fusion, scaled_text], dim=-1)
            )
        else:
            fused_emb = item_emb_for_fusion + scaled_text

        if self.fused_item_norm is not None:
            fused_emb = self.fused_item_norm(fused_emb)
        return fused_emb

    def calculate_loss(self, interaction):
        # Parent CE/BPR. Disable parent align — we handle all align here.
        saved_align = self.use_align
        if self.use_text_view_split:
            self.use_align = False
        try:
            loss = SASRecAlignV3.calculate_loss(self, interaction)
        finally:
            self.use_align = saved_align

        if not (
            self.use_text_view_split
            and saved_align
            and self.align_weight > 0.0
            and len(self.active_buffer_positions) > 0
        ):
            return loss

        pos_items = interaction[self.POS_ITEM_ID]
        id_item_emb = self.item_embedding(pos_items)
        cold_weights = self._compute_cold_weights(pos_items) if self.cold_text_boost > 0 else None

        align_losses = []
        align_labels = []

        # TF-IDF align via parent's align_proj_base (consistent with V3 per-source design)
        if (self.mv_include_base
                and self.align_proj_base is not None
                and self.item_text_emb_base is not None):
            base_raw = self.item_text_emb_base[pos_items]
            if self.detach_text_emb:
                base_raw = base_raw.detach()
            base_proj = self.align_proj_base(base_raw.to(self.align_proj_base.weight.dtype))
            align_losses.append(self._info_nce_align(id_item_emb, base_proj, cold_weights))
            align_labels.append("base")

        # Per-view align: each view independently via shared projection
        if self.use_per_view_align and self.shared_view_align_proj is not None:
            for i, pos in enumerate(self.active_buffer_positions):
                table = self._get_view_buffer(pos)
                if table.device != pos_items.device:
                    table = table.to(pos_items.device)
                view_raw = table[pos_items]
                if self.detach_text_emb:
                    view_raw = view_raw.detach()
                if view_raw.dtype != self.shared_view_align_proj.weight.dtype:
                    view_raw = view_raw.to(self.shared_view_align_proj.weight.dtype)
                view_proj = self.shared_view_align_proj(view_raw)
                align_losses.append(self._info_nce_align(id_item_emb, view_proj, cold_weights))
                align_labels.append(f"v{self.active_view_indices[i]}")

        if len(align_losses) > 0:
            mv_align = sum(align_losses) / len(align_losses)
            loss = loss + self.align_weight * mv_align

        if not self._multiview_align_debug_logged:
            try:
                parts = ", ".join(
                    f"{lbl}={l.item():.4f}" for lbl, l in zip(align_labels, align_losses)
                )
                self.logger.info(
                    "SASRecAlignMultiViewV3: per-source align | avg=%.6f | [%s]",
                    mv_align.item() if align_losses else 0.0, parts,
                )
                if self.cold_text_boost > 0 and cold_weights is not None:
                    sample_weights = 1.0 + self.cold_text_boost * cold_weights
                    self.logger.info(
                        "  cold_text_boost=%.2f, weights=[min=%.2f, max=%.2f, mean=%.2f]",
                        self.cold_text_boost,
                        sample_weights.min().item(),
                        sample_weights.max().item(),
                        sample_weights.mean().item(),
                    )
            except Exception:
                pass
            self._multiview_align_debug_logged = True

        return loss


SASRec_Align_MultiView_V3 = SASRecAlignMultiViewV3
SASRecAlignMultiView_V3 = SASRecAlignMultiViewV3

__all__ = ["SASRecAlignMultiViewV3", "SASRec_Align_MultiView_V3", "SASRecAlignMultiView_V3"]
