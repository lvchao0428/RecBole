#!/usr/bin/env python3
"""Protocol contrast: GTS-Last vs GTS-Random vs LOO ranking flip (MRR@10 warm / LOO)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# 3-seed mean MRR@10 (GTS warm / Clean LOO)
methods = ["ID", "TF", "TF+LLM", "TF+MV"]
# Beauty (GTS-Random = 3 target-sampling seeds, then 3 training seeds)
beauty_last = [0.0032, 0.0053, 0.0060, 0.0064]
beauty_rand = [0.0074, 0.0124, 0.0120, 0.0124]
beauty_loo = [0.0175, 0.0251, 0.0254, 0.0253]
# Toys
toys_last = [0.0031, 0.0040, 0.0035, 0.0042]
toys_rand = [0.0076, 0.0113, 0.0108, 0.0108]
toys_loo = [0.0152, 0.0247, 0.0258, 0.0258]

x = np.arange(len(methods))
w = 0.25

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=False)
colors = ["#4C72B0", "#55A868", "#DD8452"]
labels = ["GTS-Last", "GTS-Random", "Clean LOO"]

for ax, title, last, rand, loo in [
    (axes[0], "Amazon Beauty", beauty_last, beauty_rand, beauty_loo),
    (axes[1], "Amazon Toys", toys_last, toys_rand, toys_loo),
]:
    series = [last, rand, loo]
    for i, (vals, c, lab) in enumerate(zip(series, colors, labels)):
        ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c,
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title(title)
    ax.set_ylabel("MRR@10")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Point-estimate orderings; ≈ when the gap is within one seed-level std
    # (see tab:protocol). Not a unique-winner claim.
    orders = {
        "Amazon Beauty": (
            "Last: TF+MV ≈ TF+LLM > TF > ID",
            "Rand: TF ≈ TF+MV ≈ TF+LLM > ID",
            "LOO: TF+LLM ≈ TF+MV ≈ TF > ID",
        ),
        "Amazon Toys": (
            "Last: TF+MV ≈ TF > TF+LLM > ID",
            "Rand: TF ≈ TF+MV ≈ TF+LLM > ID",
            "LOO: TF+LLM ≈ TF+MV ≈ TF > ID",
        ),
    }[title]
    ax.text(
        0.03, 0.98,
        "\n".join(orders),
        transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="gray", alpha=0.92),
    )

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, ncol=3, loc="upper center",
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
fig.suptitle("Evaluation protocol changes text-representation rankings "
             "(point estimates; ≈ = within 1 std)",
             y=1.12, fontsize=9.5)

out = Path(__file__).with_name("fig_protocol_contrast")
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"))
print(f"Wrote {out}.pdf / .png")
