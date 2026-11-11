#!/usr/bin/env python3
"""Protocol contrast v2: ranking annotation below the bars so it does not cover LOO."""
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
    "mathtext.fontset": "stix",
})

methods = ["ID", "TF", "TF+LLM", "TF+MV"]
beauty_last = [0.0032, 0.0053, 0.0060, 0.0064]
beauty_rand = [0.0074, 0.0124, 0.0120, 0.0124]
beauty_loo = [0.0175, 0.0251, 0.0254, 0.0253]
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

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, ncol=3, loc="upper center",
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
fig.suptitle("Evaluation protocol changes text-representation rankings",
             y=1.10, fontsize=10)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])

out = Path(__file__).with_name("fig_protocol_contrast_v2")
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"))
print(f"Wrote {out}.pdf / .png")
