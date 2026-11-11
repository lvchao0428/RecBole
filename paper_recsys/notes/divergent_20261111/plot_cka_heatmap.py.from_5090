#!/usr/bin/env python3
"""CKA heatmap + effective-rank bar chart (Beauty)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "cka_subsample_beauty.json") as f:
    data = json.load(f)

labels = ["Desc", "Func", "Aud", "Style", "TF-IDF"]
# Build 5x5 CKA matrix (Beauty means)
cka = np.eye(5)
# View pairs
view_map = {
    ("Desc", "Func"): data["view_pairs"]["Description vs Function"]["mean"],
    ("Desc", "Aud"): data["view_pairs"]["Description vs Audience"]["mean"],
    ("Desc", "Style"): data["view_pairs"]["Description vs Style"]["mean"],
    ("Func", "Aud"): data["view_pairs"]["Function vs Audience"]["mean"],
    ("Func", "Style"): data["view_pairs"]["Function vs Style"]["mean"],
    ("Aud", "Style"): data["view_pairs"]["Audience vs Style"]["mean"],
}
tf_map = {
    "Desc": data["tfidf_pairs"]["TF-IDF vs Description"]["mean"],
    "Func": data["tfidf_pairs"]["TF-IDF vs Function"]["mean"],
    "Aud": data["tfidf_pairs"]["TF-IDF vs Audience"]["mean"],
    "Style": data["tfidf_pairs"]["TF-IDF vs Style"]["mean"],
}
idx = {n: i for i, n in enumerate(labels)}
for (a, b), v in view_map.items():
    cka[idx[a], idx[b]] = v
    cka[idx[b], idx[a]] = v
for name, v in tf_map.items():
    cka[idx[name], idx["TF-IDF"]] = v
    cka[idx["TF-IDF"], idx[name]] = v

# Effective rank
er_labels = ["Desc", "Func", "Aud", "Style", "Concat-4", "TF-IDF"]
er_vals = [
    data["effective_rank"]["Description"]["mean"],
    data["effective_rank"]["Function"]["mean"],
    data["effective_rank"]["Audience"]["mean"],
    data["effective_rank"]["Style"]["mean"],
    data["effective_rank"]["concat_4views"]["mean"],
    data["effective_rank"]["tfidf"]["mean"],
]
er_dims = [64, 64, 64, 64, 256, 256]
er_ratio = [v / d for v, d in zip(er_vals, er_dims)]

fig = plt.figure(figsize=(7.2, 3.0))
gs = GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.35)

ax0 = fig.add_subplot(gs[0])
im = ax0.imshow(cka, cmap="YlOrRd", vmin=0.0, vmax=1.0)
ax0.set_xticks(range(5))
ax0.set_yticks(range(5))
ax0.set_xticklabels(labels, rotation=35, ha="right")
ax0.set_yticklabels(labels)
ax0.set_title("(a) Linear CKA (Beauty)")
for i in range(5):
    for j in range(5):
        ax0.text(j, i, f"{cka[i, j]:.2f}", ha="center", va="center",
                 fontsize=7, color="black" if cka[i, j] < 0.7 else "white")
cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)

ax1 = fig.add_subplot(gs[1])
colors = ["#4C72B0"] * 4 + ["#C44E52", "#55A868"]
bars = ax1.bar(range(len(er_labels)), er_ratio, color=colors, edgecolor="black", linewidth=0.4)
ax1.set_xticks(range(len(er_labels)))
ax1.set_xticklabels(er_labels, rotation=30, ha="right")
ax1.set_ylabel("Effective rank / dim")
ax1.set_ylim(0, 1.15)
ax1.set_title("(b) Effective rank ratio")
ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
for i, (r, v, d) in enumerate(zip(er_ratio, er_vals, er_dims)):
    ax1.text(i, r + 0.02, f"{v:.0f}/{d}", ha="center", va="bottom", fontsize=6.5)

fig.suptitle("Prompt views are highly redundant; TF-IDF is complementary", y=1.02, fontsize=10)
out = Path(__file__).with_name("fig_cka_heatmap")
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"))
print(f"Wrote {out}.pdf / .png")
