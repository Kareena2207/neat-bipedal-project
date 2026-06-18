"""
Generates a single combined plot comparing all experiments.
Run this after all training is complete.
"""
import os
import pickle
import numpy as np
import neat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.expanduser("~/Desktop/neat_bipedal_project")

# Official results from evaluate_all.py
results = {
    "Baseline (speed)": {
        "avg": 235.43, "std": 78.74,
        "best": 257.18, "worst": -107.59,
        "failure_rate": 5, "nodes": 13,
        "connections": 103, "generations": 280,
        "color": "#2196F3"
    },
    "Efficiency": {
        "avg": 217.86, "std": 97.05,
        "best": 270.03, "worst": -44.02,
        "failure_rate": 20, "nodes": 36,
        "connections": 148, "generations": 400,
        "color": "#FF9800"
    },
    "Stability": {
        "avg": 272.92, "std": 6.89,
        "best": 279.60, "worst": 253.17,
        "failure_rate": 0, "nodes": 19,
        "connections": 129, "generations": 233,
        "color": "#4CAF50"
    },
    "Ablation (fixed topology)": {
        "avg": 183.45, "std": 60.12,
        "best": 211.64, "worst": -13.31,
        "failure_rate": 10, "nodes": 4,
        "connections": 50, "generations": 500,
        "color": "#F44336"
    },
}

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(
    "NEAT BipedalWalker — Complete Experiment Comparison",
    fontsize=16, fontweight='bold', y=1.02
)

names   = list(results.keys())
colors  = [results[n]["color"] for n in names]
avgs    = [results[n]["avg"]   for n in names]
stds    = [results[n]["std"]   for n in names]
bests   = [results[n]["best"]  for n in names]
fails   = [results[n]["failure_rate"] for n in names]
nodes   = [results[n]["nodes"] for n in names]
short_names = ["Baseline", "Efficiency", "Stability", "Ablation"]

# ── Plot 1: Average reward with std dev ───────────────────────────────────
ax1 = axes[0]
bars = ax1.bar(short_names, avgs, color=colors, alpha=0.85,
               edgecolor='white', linewidth=1.5)
ax1.errorbar(short_names, avgs, yerr=stds, fmt='none',
             color='black', capsize=6, linewidth=2)

# Add value labels on bars
for bar, avg, std in zip(bars, avgs, stds):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + std + 3,
             f'{avg:.1f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

ax1.set_title('Average Reward (20 episodes)', fontsize=13)
ax1.set_ylabel('Reward', fontsize=12)
ax1.set_ylim(-50, 320)
ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax1.grid(True, alpha=0.3, axis='y')
ax1.tick_params(axis='x', rotation=15)

# ── Plot 2: Failure rate ──────────────────────────────────────────────────
ax2 = axes[1]
bars2 = ax2.bar(short_names, fails, color=colors, alpha=0.85,
                edgecolor='white', linewidth=1.5)

for bar, fail in zip(bars2, fails):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{fail}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

ax2.set_title('Failure Rate (20 episodes)', fontsize=13)
ax2.set_ylabel('Failure Rate (%)', fontsize=12)
ax2.set_ylim(0, 30)
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=15)

# ── Plot 3: Network complexity ────────────────────────────────────────────
ax3 = axes[2]
x     = np.arange(len(short_names))
width = 0.4

bars3a = ax3.bar(x - width/2, nodes,
                 width, label='Hidden nodes',
                 color=colors, alpha=0.85,
                 edgecolor='white', linewidth=1.5)
bars3b = ax3.bar(x + width/2,
                 [results[n]["connections"] for n in names],
                 width, label='Connections',
                 color=colors, alpha=0.4,
                 edgecolor='white', linewidth=1.5,
                 hatch='//')

for bar, val in zip(bars3a, nodes):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             str(val), ha='center', va='bottom', fontsize=9)

ax3.set_title('Evolved Network Complexity', fontsize=13)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(short_names, rotation=15)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_path = os.path.join(BASE_DIR, "results", "comparison_all_experiments.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Comparison plot saved to {save_path}")
