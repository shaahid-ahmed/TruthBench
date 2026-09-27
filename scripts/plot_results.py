"""Plot accuracy with confidence intervals per category"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

MODELS = [("qwen", "Qwen2.5-7B-Instruct", "#2a78d6", "o"),
          ("mistral", "Mistral-7B-Instruct-v0.3", "#eb6834", "s")]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
OUTPUT_PATH = "../figures/mc1_by_category.png"

mc1 = pd.read_csv("../data/full_mc1_results_combined.csv")
tests = pd.read_csv("../data/significance_test_results.csv")

primary = tests[tests["scope"] == "primary"].iloc[0]
cats = tests[(tests["scope"] == "category") & (tests["e9_status"] == "pass")]
# Testable categories first
cats = cats.assign(_k=(cats["t6_status"] != "testable")).sort_values(["_k", "n"], ascending=[True, False])

rows = [("Overall (all 817)", mc1,
         f"McNemar p = {primary['p_raw']:.3f} (primary, uncorrected)")]
for _, c in cats.iterrows():
    if c["t6_status"] == "testable":
        note = f"Holm p = {c['p_holm_corrected']:.3f}" + (" *" if c["significant_after_correction"] else "")
    else:
        note = f"untestable ({int(c['n_discordant'])} discordant)"
    rows.append((f"{c['category']} (n={int(c['n'])})", mc1[mc1["category"] == c["category"]], note))

fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

y_positions = []
for i, (label, sub, note) in enumerate(rows):
    y = -i - (0.6 if i > 0 else 0)  # Gap under the primary row
    y_positions.append(y)
    for j, (key, name, color, marker) in enumerate(MODELS):
        s = sub[sub["model"] == key]["correct"]
        k, n = int(s.sum()), len(s)
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        yy = y + (0.14 if j == 0 else -0.14)
        ax.plot([lo, hi], [yy, yy], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(k / n, yy, marker=marker, ms=6.5, color=color, mec=SURFACE, mew=1.5, zorder=3,
                label=name if i == 0 else None)
    ax.text(1.03, y, note, transform=ax.get_yaxis_transform(), va="center", ha="left",
            fontsize=8.5, color=INK if i == 0 else INK_2)

ax.axhline(-0.8, color=GRID, lw=1)
ax.set_yticks(y_positions)
ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_xlim(-0.02, 1.0)
ax.set_ylim(y_positions[-1] - 0.6, 0.6)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_xlabel("MC1 accuracy (dot) with 95% Wilson CI", fontsize=9, color=INK_2)

n_tested = int((cats["t6_status"] == "testable").sum())
n_sig = int(cats["significant_after_correction"].fillna(False).astype(bool).sum())
fig.suptitle("Mistral edges Qwen overall on TruthfulQA MC1; no category survives Holm correction",
             x=0.02, y=0.99, va="top", ha="left", fontsize=11.5, color=INK, fontweight="bold")
fig.text(0.02, 0.962, f"{n_sig} of {n_tested} testable categories significant after Holm; "
         f"{len(cats) - n_tested} categories too sparse to test (<6 discordant pairs)",
         ha="left", va="top", fontsize=9, color=INK_2)
fig.legend(loc="upper left", bbox_to_anchor=(0.015, 0.94), ncol=2, frameon=False, fontsize=9,
           handletextpad=0.4, columnspacing=1.6)

fig.tight_layout(rect=(0, 0, 0.82, 0.915))
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, facecolor=SURFACE)
print(f"Saved {OUTPUT_PATH}")
