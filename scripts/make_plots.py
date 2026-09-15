"""Строит графики сравнения конфигураций для README."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Данные из финального README ---
# Только конфигурации с полными данными по всем метрикам.

configs = [
    "Baseline +\nfrozen 3 ep",
    "ZeRO-1\n3 ep",
    "ZeRO-2 default\n3 ep",
    "ZeRO-2 tuned\n3 ep",
    "ZeRO-3 default\n3 ep",
    "ZeRO-3 tuned\n3 ep",
]

vram =  [3.98, 2.17, 3.04, 1.21, 1.99, 0.85]
ram =   [9.07, 12.32, 12.57, 12.60, 13.29, 13.30]
time_s =[32, 41, 58, 50, 102, 86]
wer =   [30.80, 29.66, 27.38, 25.86, 25.10, 28.52]


# --- Вспомогательные функции ---

def color_for(value, values, best="min"):
    """Зелёный для лучшего, красный для худшего, синий для остальных."""
    target = min(values) if best == "min" else max(values)
    worst = max(values) if best == "min" else min(values)
    if value == target:
        return "#2ecc71"
    if value == worst:
        return "#e74c3c"
    return "#3498db"


def annotate_bars(ax, bars, values, fmt="{:.2f}", offset=0.02):
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                fmt.format(v), ha="center", va="bottom", fontsize=8)


# ============================================================
# График 1: WER
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(configs))
colors = [color_for(w, wer, best="min") for w in wer]
bars = ax.bar(x, wer, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(y=28.14, color="gray", linestyle="--", linewidth=1,
           label="Pretrained (28.14%)")
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylabel("WER (%)", fontsize=11)
ax.set_title("Word Error Rate by configuration (lower is better)", fontsize=12)
ax.legend()
annotate_bars(ax, bars, wer, fmt="{:.2f}%", offset=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "wer_comparison.png", dpi=150)
plt.close()


# ============================================================
# График 2: VRAM
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))
colors = [color_for(v, vram, best="min") for v in vram]
bars = ax.bar(x, vram, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylabel("Max VRAM (GB)", fontsize=11)
ax.set_title("Peak VRAM by configuration (lower is better)", fontsize=12)
annotate_bars(ax, bars, vram, fmt="{:.2f}", offset=0.05)
plt.tight_layout()
plt.savefig(OUT_DIR / "vram_comparison.png", dpi=150)
plt.close()


# ============================================================
# График 3: CPU RAM
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))
colors = [color_for(r, ram, best="min") for r in ram]
bars = ax.bar(x, ram, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylabel("Peak CPU RAM (GB)", fontsize=11)
ax.set_title("Peak CPU RAM by configuration (lower is better)", fontsize=12)
annotate_bars(ax, bars, ram, fmt="{:.2f}", offset=0.1)
plt.tight_layout()
plt.savefig(OUT_DIR / "ram_comparison.png", dpi=150)
plt.close()


# ============================================================
# График 4: Время обучения
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(x, time_s, color="#9b59b6", edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylabel("Training time (s)", fontsize=11)
ax.set_title("Training time by configuration (lower is better)", fontsize=12)
annotate_bars(ax, bars, time_s, fmt="{:.0f}s", offset=1)
plt.tight_layout()
plt.savefig(OUT_DIR / "time_comparison.png", dpi=150)
plt.close()


# ============================================================
# График 5: Tradeoff VRAM vs WER
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))
for label, v, w in zip(configs, vram, wer):
    color = color_for(w, wer, best="min")
    ax.scatter(v, w, s=140, color=color, edgecolor="black", zorder=3)
    ax.annotate(label.replace("\n", " "), (v, w),
                textcoords="offset points", xytext=(8, 5), fontsize=8)
ax.set_xlabel("Max VRAM (GB)", fontsize=11)
ax.set_ylabel("WER (%)", fontsize=11)
ax.set_title("Tradeoff: VRAM vs WER (lower-left is better)", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "tradeoff.png", dpi=150)
plt.close()


# ============================================================
# График 6: Tradeoff VRAM vs CPU RAM
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))
for label, v, r in zip(configs, vram, ram):
    ax.scatter(v, r, s=140, color="#3498db", edgecolor="black", zorder=3)
    ax.annotate(label.replace("\n", " "), (v, r),
                textcoords="offset points", xytext=(8, 5), fontsize=8)
ax.set_xlabel("Max VRAM (GB)", fontsize=11)
ax.set_ylabel("Peak CPU RAM (GB)", fontsize=11)
ax.set_title("Memory shift: VRAM vs CPU RAM (offload trades one for the other)",
             fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "tradeoff_memory.png", dpi=150)
plt.close()


# ============================================================
# График 7: ZeRO stage comparison (grouped bars)
# ============================================================

zero_labels = ["ZeRO-1", "ZeRO-2\ndefault", "ZeRO-2\ntuned",
               "ZeRO-3\ndefault", "ZeRO-3\ntuned"]
zero_vram = [2.17, 3.04, 1.21, 1.99, 0.85]
zero_ram = [12.32, 12.57, 12.60, 13.29, 13.30]
zero_time = [41, 58, 50, 102, 86]
zero_wer = [29.66, 27.38, 25.86, 25.10, 28.52]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
x = np.arange(len(zero_labels))

# VRAM
axes[0].bar(x, zero_vram, color="#3498db", edgecolor="black", linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(zero_labels, fontsize=9)
axes[0].set_ylabel("Max VRAM (GB)")
axes[0].set_title("VRAM")
for i, v in enumerate(zero_vram):
    axes[0].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

# CPU RAM
axes[1].bar(x, zero_ram, color="#e67e22", edgecolor="black", linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(zero_labels, fontsize=9)
axes[1].set_ylabel("Peak CPU RAM (GB)")
axes[1].set_title("CPU RAM")
for i, v in enumerate(zero_ram):
    axes[1].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)

# Time
axes[2].bar(x, zero_time, color="#9b59b6", edgecolor="black", linewidth=0.5)
axes[2].set_xticks(x)
axes[2].set_xticklabels(zero_labels, fontsize=9)
axes[2].set_ylabel("Training time (s)")
axes[2].set_title("Time")
for i, v in enumerate(zero_time):
    axes[2].text(i, v + 2, f"{v}s", ha="center", fontsize=9)

# WER
axes[3].bar(x, zero_wer, color="#2ecc71", edgecolor="black", linewidth=0.5)
axes[3].set_xticks(x)
axes[3].set_xticklabels(zero_labels, fontsize=9)
axes[3].set_ylabel("WER (%)")
axes[3].set_title("WER")
axes[3].axhline(y=28.14, color="gray", linestyle="--", linewidth=1)
for i, v in enumerate(zero_wer):
    axes[3].text(i, v + 0.3, f"{v:.2f}%", ha="center", fontsize=9)

plt.suptitle("ZeRO stage comparison (Whisper-small, 3 epochs)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "zero_stages.png", dpi=150)
plt.close()


# ============================================================
# График 8: Buffer tuning effect (ZeRO-2 vs ZeRO-3)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
width = 0.35

# ZeRO-2: default vs tuned
labels_2 = ["ZeRO-2\ndefault", "ZeRO-2\ntuned"]
vram_2 = [3.04, 1.21]
time_2 = [58, 50]
x2 = np.arange(len(labels_2))

ax1 = axes[0]
ax1.bar(x2 - width/2, vram_2, width, label="VRAM (GB)",
        color="#3498db", edgecolor="black", linewidth=0.5)
ax1.bar(x2 + width/2, [t/10 for t in time_2], width, label="Time (s / 10)",
        color="#9b59b6", edgecolor="black", linewidth=0.5)
ax1.set_xticks(x2)
ax1.set_xticklabels(labels_2)
ax1.set_title("ZeRO-2: buffer tuning")
ax1.legend()
for i, (v, t) in enumerate(zip(vram_2, time_2)):
    ax1.text(i - width/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    ax1.text(i + width/2, t/10 + 0.05, f"{t}s", ha="center", fontsize=9)

# ZeRO-3: default vs tuned
labels_3 = ["ZeRO-3\ndefault", "ZeRO-3\ntuned"]
vram_3 = [1.99, 0.85]
time_3 = [102, 86]
x3 = np.arange(len(labels_3))

ax2 = axes[1]
ax2.bar(x3 - width/2, vram_3, width, label="VRAM (GB)",
        color="#3498db", edgecolor="black", linewidth=0.5)
ax2.bar(x3 + width/2, [t/10 for t in time_3], width, label="Time (s / 10)",
        color="#9b59b6", edgecolor="black", linewidth=0.5)
ax2.set_xticks(x3)
ax2.set_xticklabels(labels_3)
ax2.set_title("ZeRO-3: buffer tuning")
ax2.legend()
for i, (v, t) in enumerate(zip(vram_3, time_3)):
    ax2.text(i - width/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    ax2.text(i + width/2, t/10 + 0.05, f"{t}s", ha="center", fontsize=9)

plt.suptitle("Buffer tuning effect: VRAM vs Time", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "buffer_tuning.png", dpi=150)
plt.close()


print(f"Saved 8 plots to {OUT_DIR}")