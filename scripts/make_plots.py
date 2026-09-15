"""Строит графики сравнения конфигураций для README."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Данные из README
configs = [
    "Pretrained\n(no FT)",
    "Baseline\n1 epoch",
    "Baseline\n3 epochs",
    "Baseline +\nfrozen enc.",
    "ZeRO-3\noffload 1 ep",
    "ZeRO-3 +\nfrozen enc.",
]
vram = [None, 5.93, 5.94, 3.98, 2.87, 1.99]
wer = [28.14, 38.40, 31.18, 30.80, 42.97, 25.10]
time_s = [None, 19, 50, 32, 42, 94]

# Цвета: лучший результат — зелёный, худший — красный
def color_for_wer(w):
    if w == min(wer):
        return "#2ecc71"  # зелёный
    if w == max(wer):
        return "#e74c3c"  # красный
    return "#3498db"      # синий

def color_for_vram(v):
    if v is None:
        return "#95a5a6"
    if v == min([x for x in vram if x is not None]):
        return "#2ecc71"
    return "#3498db"


# График 1: WER по конфигурациям
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(configs))
colors = [color_for_wer(w) for w in wer]
bars = ax.bar(x, wer, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(y=28.14, color="gray", linestyle="--", linewidth=1,
           label="Pretrained (28.14%)")
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel("WER (%)", fontsize=11)
ax.set_title("Word Error Rate by configuration (lower is better)", fontsize=12)
ax.legend()
for bar, w in zip(bars, wer):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{w:.1f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "wer_comparison.png", dpi=150)
plt.close()

# График 2: VRAM по конфигурациям
fig, ax = plt.subplots(figsize=(10, 5))
vram_vals = [v if v is not None else 0 for v in vram]
colors = [color_for_vram(v) for v in vram]
bars = ax.bar(x, vram_vals, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel("Max VRAM (GB)", fontsize=11)
ax.set_title("Peak VRAM by configuration (lower is better)", fontsize=12)
for bar, v in zip(bars, vram):
    if v is not None:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, 0.1,
                "n/a", ha="center", va="bottom", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(OUT_DIR / "vram_comparison.png", dpi=150)
plt.close()

# График 3: Tradeoff VRAM vs WER
fig, ax = plt.subplots(figsize=(8, 6))
for i, (v, w, c) in enumerate(zip(vram, wer, configs)):
    if v is None:
        continue
    ax.scatter(v, w, s=120, color=color_for_wer(w), edgecolor="black", zorder=3)
    ax.annotate(c.replace("\n", " "), (v, w),
                textcoords="offset points", xytext=(8, 5), fontsize=8)
ax.set_xlabel("Max VRAM (GB)", fontsize=11)
ax.set_ylabel("WER (%)", fontsize=11)
ax.set_title("Tradeoff: VRAM vs WER (lower-left is better)", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "tradeoff.png", dpi=150)
plt.close()

# График 4: Время обучения
fig, ax = plt.subplots(figsize=(10, 5))
time_vals = [t if t is not None else 0 for t in time_s]
bars = ax.bar(x, time_vals, color="#9b59b6", edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel("Training time (s)", fontsize=11)
ax.set_title("Training time by configuration (lower is better)", fontsize=12)
for bar, t in zip(bars, time_s):
    if t is not None:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{t}s", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, 1,
                "n/a", ha="center", va="bottom", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(OUT_DIR / "time_comparison.png", dpi=150)
plt.close()

print(f"Saved 4 plots to {OUT_DIR}")