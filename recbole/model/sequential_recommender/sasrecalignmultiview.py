import json
import os
import numpy as np
import torch
from torch import nn

from recbole.model.sequential_recommender.sasrec_align import SASRecAlign


class SASRecAlignMultiView(SASRecAlign):
    """
    SASRecAlign variant tailored for per-prompt multi-view embeddings.
    Stage 1+2 (already completed): load per-view vectors + SENet-style refinement.
    Stage 3 (implemented here): FiBiNET-inspired residual cross between each view and ID embedding.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_text_view_split = bool(config["use_text_view_split"]) if "use_text_view_split" in config else False
        if "item_text_emb_split_dir" in config and config["item_text_emb_split_dir"]:
            self.text_view_split_dir = os.path.abspath(os.path.expanduser(config["item_text_emb_split_dir"]))
        else:
            self.text_view_split_dir = None
        self.text_view_senet_ratio = int(config["text_view_senet_ratio"]) if "text_view_senet_ratio" in config else 4
        self.text_view_cross_dropout = float(config["text_view_cross_dropout"]) if "text_view_cross_dropout" in config else 0.1
        self.text_view_residual_init = float(config["text_view_residual_init"]) if "text_view_residual_init" in config else 0.5
        self.text_view_cross_layers = int(config["text_view_cross_layers"]) if "text_view_cross_layers" in config else 1

        self.text_view_buffer_names = []
        self.text_view_proj = nn.ModuleList()
        self.text_view_senet = nn.ModuleList()
        self.text_view_bilinear = nn.ModuleList()
        self.text_view_residual_gates = nn.ParameterList()
        self.text_view_gate_params = None
        self.text_view_dropout = (
            nn.Dropout(self.text_view_cross_dropout) if self.text_view_cross_dropout > 0.0 else None
        )

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

            for view in prompts:
                idx = view["index"]
                file_name = view["file"]
                file_path = os.path.join(self.text_view_split_dir, file_name)
                if not os.path.exists(file_path):
                    raise ValueError(f"Split embedding file not found: {file_path}")
                npy = np.load(file_path)
                tensor = torch.from_numpy(npy).float()
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

                # FiBiNET-style bilinear interaction between ID emb and each view
                self.text_view_bilinear.append(nn.Bilinear(self.hidden_size, self.hidden_size, self.hidden_size))
                self.text_view_residual_gates.append(
                    nn.Parameter(torch.tensor(self.text_view_residual_init, dtype=torch.float32))
                )

            # Ensure normalization layers exist for residual fusion even if base class skipped them
            if self.item_emb_norm is None and self.fused_item_norm_flag:
                self.item_emb_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            if self.fused_item_norm is None and self.fused_item_norm_flag:
                self.fused_item_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def _get_view_buffer(self, idx: int) -> torch.Tensor:
        name = self.text_view_buffer_names[idx]
        return getattr(self, name)

    def _gather_text_views(self, ids_flat: torch.Tensor) -> torch.Tensor:
        """Gather SENet-refined per-view features for given item ids."""
        if not self.use_text_view_split:
            raise RuntimeError("text view split not enabled.")

        view_features = []
        for idx, proj in enumerate(self.text_view_proj):
            view_emb_table = self._get_view_buffer(idx)
            if view_emb_table.device != ids_flat.device:
                view_emb_table = view_emb_table.to(ids_flat.device)
            gathered = view_emb_table[ids_flat]  # [B, view_dim]
            projected = proj(gathered)  # [B, hidden]
            squeeze = projected  # treat embedding as 1D field; no spatial dims to pool
            excitation = self.text_view_senet[idx](squeeze)
            refined = projected * excitation
            view_features.append(refined)
        stacked = torch.stack(view_features, dim=1)  # [B, num_views, hidden]
        return stacked

    def _gather_text_raw(self, ids_flat: torch.Tensor) -> torch.Tensor:
        if not self.use_text_view_split:
            return super()._gather_text_raw(ids_flat)

        view_stack = self._gather_text_views(ids_flat)
        fused = view_stack.mean(dim=1)
        return fused

    def _project_text(self, raw: torch.Tensor) -> torch.Tensor:
        if self.use_text_view_split:
            return raw
        return super()._project_text(raw)

    def _get_fused_item_embeddings(self, item_ids: torch.Tensor = None) -> torch.Tensor:
        """Override item fusion to leverage per-view residual cross."""
        if not self.use_text_view_split:
            return super()._get_fused_item_embeddings(item_ids)

        if item_ids is None:
            all_ids = torch.arange(self.n_items, device=self.item_embedding.weight.device)
            item_emb = self.item_embedding.weight
        else:
            all_ids = item_ids
            item_emb = self.item_embedding(item_ids)

        if not self.fuse_text_feature or len(self.text_view_buffer_names) == 0:
            return item_emb

        view_stack = self._gather_text_views(all_ids)
        if self.detach_text_emb:
            view_stack = view_stack.detach()

        item_emb_for_fusion = item_emb
        if self.item_emb_norm is not None:
            item_emb_for_fusion = self.item_emb_norm(item_emb_for_fusion)

        alpha = torch.sigmoid(self.text_gate_param) * self.text_weight
        if self.text_item_gate_all is not None:
            gate = self.text_item_gate_all if item_ids is None else self.text_item_gate_all[all_ids]
            gate = gate.to(item_emb.device).unsqueeze(1)
            alpha = alpha * gate
        if alpha.dim() == 0:
            alpha = alpha.view(1, 1).expand(item_emb.size(0), -1)
        elif alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        view_weights = torch.sigmoid(self.text_view_gate_params)
        weight_norm = view_weights / view_weights.sum().clamp_min(1e-4)

        residual = torch.zeros_like(item_emb_for_fusion)
        for idx in range(len(self.text_view_buffer_names)):
            view_feat = view_stack[:, idx, :]
            interaction = self.text_view_bilinear[idx](item_emb_for_fusion, view_feat)
            if self.text_view_dropout is not None:
                interaction = self.text_view_dropout(interaction)
            residual_gate = torch.sigmoid(self.text_view_residual_gates[idx])
            residual = residual + weight_norm[idx] * residual_gate * interaction

        fused_emb = item_emb_for_fusion + alpha * residual
        if self.fused_item_norm is not None:
            fused_emb = self.fused_item_norm(fused_emb)
        return fused_emb


# Alias so --model SASRec_Align_MultiView works
SASRec_Align_MultiView = SASRecAlignMultiView


__all__ = ["SASRecAlignMultiView", "SASRec_Align_MultiView"]

