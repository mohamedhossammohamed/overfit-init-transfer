import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.mplconfig')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 13,
    'axes.edgecolor': '#DEDAD0',
    'axes.linewidth': 1.0,
    'grid.color': '#EFECE6',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'axes.facecolor': '#FAF9F6',
    'figure.facecolor': '#FAF9F6',
})

RUNS_FILE = 'results/runs.jsonl'
OUTPUT_DIR = 'docs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load data ---
runs = {}
with open(RUNS_FILE, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        if 'train_loss' not in record:
            continue
        run_id = record['run_id']
        if run_id not in runs:
            runs[run_id] = []
        runs[run_id].append(record)

# --- Color palette ---
COLORS = {
    'A':    '#2A2822',
    'B_0.5': '#C86D3B',
    'B_0.9': '#8B3A19',
    'C_0.5': '#5B8C69',
    'C_0.9': '#2E5A3C',
}
STYLES = {'42': '-', '1337': '--'}

def cond(run_id):
    if run_id.startswith('a_'): return 'A'
    if 'from_quran_frac0.5' in run_id: return 'B_0.5'
    if 'from_quran_frac0.9' in run_id: return 'B_0.9'
    if 'finetune_frac0.5' in run_id: return 'C_0.5'
    if 'finetune_frac0.9' in run_id: return 'C_0.9'
    return None

def seed(run_id):
    return '42' if 'seed42' in run_id else '1337'

def scale(run_id):
    return '100k' if '_100k_' in run_id else '2M'

# Use train_loss throughout — val_loss diverges at 2M due to extreme overfitting
# on the tiny Shakespeare corpus, making train_loss the informative comparison
METRIC = 'train_loss'
METRIC_LABEL = 'Training Loss'

# ================================================================
# Figure 1: Learning Curve Overlays
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for sc, ax in zip(['100k', '2M'], axes):
    ax.set_title(f'{METRIC_LABEL} Trajectory — {sc}', pad=10)
    ax.set_xlabel('Tokens Seen (finetuning phase)')
    ax.set_ylabel(METRIC_LABEL)
    ax.grid(True)

    for rid, recs in sorted(runs.items()):
        if scale(rid) != sc or 'pretrain' in rid:
            continue
        c = cond(rid)
        if not c:
            continue
        s = seed(rid)
        x = [r['tokens_seen'] for r in recs]
        y = [r[METRIC] for r in recs]
        ax.plot(x, y, color=COLORS[c], linestyle=STYLES[s],
                alpha=0.85, linewidth=1.5)

legend_elements = [
    Line2D([0], [0], color=COLORS['A'],    lw=1.8, label='A (baseline)'),
    Line2D([0], [0], color=COLORS['B_0.5'], lw=1.8, label='B corpus (frac 0.5)'),
    Line2D([0], [0], color=COLORS['B_0.9'], lw=1.8, label='B corpus (frac 0.9)'),
    Line2D([0], [0], color=COLORS['C_0.5'], lw=1.8, label='C noise (frac 0.5)'),
    Line2D([0], [0], color=COLORS['C_0.9'], lw=1.8, label='C noise (frac 0.9)'),
    Line2D([0], [0], color='#999', lw=1.5, ls='-',  label='Seed 42 (solid)'),
    Line2D([0], [0], color='#999', lw=1.5, ls='--', label='Seed 1337 (dashed)'),
]
axes[1].legend(handles=legend_elements, loc='upper right',
               frameon=True, facecolor='#FAF9F6', edgecolor='#DEDAD0')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=200, bbox_inches='tight')
plt.close()

# ================================================================
# Figure 2: Relative Delta Bar Chart
# ================================================================
final = {}
for rid, recs in runs.items():
    if 'pretrain' in rid: continue
    c = cond(rid)
    if not c: continue
    final[(scale(rid), seed(rid), c)] = recs[-1][METRIC]

conditions = ['B_0.5', 'B_0.9', 'C_0.5', 'C_0.9']
cond_labels = ['B corpus\n(frac 0.5)', 'B corpus\n(frac 0.9)', 'C noise\n(frac 0.5)', 'C noise\n(frac 0.9)']

deltas_100k, deltas_2M = [], []
for c in conditions:
    d42 = final[('100k', '42', c)] - final[('100k', '42', 'A')]
    d1337 = final[('100k', '1337', c)] - final[('100k', '1337', 'A')]
    deltas_100k.append((d42 + d1337) / 2)
    deltas_2M.append(final[('2M', '42', c)] - final[('2M', '42', 'A')])

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(conditions))
w = 0.35
r1 = ax.bar(x - w/2, deltas_100k, w, label='100k (mean of seeds 42 & 1337)', color='#6B6860', alpha=0.85)
r2 = ax.bar(x + w/2, deltas_2M, w, label='2M (seed 42 only)', color='#B0522D', alpha=0.85)
ax.axhline(0, color='#2A2822', lw=0.9)
ax.set_ylabel(f'Δ Final {METRIC_LABEL} vs. Baseline (A)')
ax.set_title('Performance Relative to Random-Init Baseline', pad=12)
ax.set_xticks(x); ax.set_xticklabels(cond_labels)
ax.legend(frameon=True, facecolor='#FAF9F6', edgecolor='#DEDAD0')
ax.grid(True, axis='y')

for rects in [r1, r2]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:+.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -10), textcoords='offset points',
                    ha='center', va='bottom' if h >= 0 else 'top', fontsize=7.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'relative_delta.png'), dpi=200, bbox_inches='tight')
plt.close()

# ================================================================
# Figure 3: Recovery Dynamics (early finetuning steps)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

for idx, (sc, max_step) in enumerate([('100k', 300), ('2M', 2000)]):
    ax = axes[idx]
    ax.set_title(f'Early Recovery — {sc} (seed 42)', pad=10)
    ax.set_xlabel('Steps')
    ax.set_ylabel(METRIC_LABEL)
    ax.grid(True)
    for c in ['A', 'B_0.5', 'B_0.9', 'C_0.5', 'C_0.9']:
        for rid in runs:
            if scale(rid) == sc and seed(rid) == '42' and cond(rid) == c and 'pretrain' not in rid:
                recs = [r for r in runs[rid] if r['step'] <= max_step]
                ax.plot([r['step'] for r in recs], [r[METRIC] for r in recs],
                        label=c.replace('_', ' '), color=COLORS[c], lw=1.6)
                break
axes[0].legend(frameon=True, facecolor='#FAF9F6', edgecolor='#DEDAD0')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'recovery_dynamics.png'), dpi=200, bbox_inches='tight')
plt.close()

# ================================================================
# Figure 4: Weight Update Norm Trajectories
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for sc, ax in zip(['100k', '2M'], axes):
    ax.set_title(f'Weight Update Norm — {sc} (seed 42)', pad=10)
    ax.set_xlabel('Steps')
    ax.set_ylabel('‖ΔW‖')
    ax.grid(True)
    for c in ['A', 'B_0.5', 'B_0.9', 'C_0.5', 'C_0.9']:
        for rid in runs:
            if scale(rid) == sc and seed(rid) == '42' and cond(rid) == c and 'pretrain' not in rid:
                recs = runs[rid]
                ax.plot([r['step'] for r in recs], [r['weight_update_norm'] for r in recs],
                        label=c.replace('_', ' '), color=COLORS[c], lw=1.3, alpha=0.85)
                break
axes[1].legend(frameon=True, facecolor='#FAF9F6', edgecolor='#DEDAD0')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'weight_update_norms.png'), dpi=200, bbox_inches='tight')
plt.close()

# ================================================================
# Figure 5: Final Loss Heatmap
# ================================================================
cond_map = {'A': 0, 'B_0.5': 1, 'B_0.9': 2, 'C_0.5': 3, 'C_0.9': 4}
col_map  = {('100k','42'):0, ('100k','1337'):1, ('2M','42'):2, ('2M','1337'):3}
row_labels = ['A (baseline)', 'B corpus (0.5)', 'B corpus (0.9)', 'C noise (0.5)', 'C noise (0.9)']
col_labels = ['100k s42', '100k s1337', '2M s42', '2M s1337']

matrix = np.full((5, 4), np.nan)
for (sc, sd, c), loss in final.items():
    matrix[cond_map[c], col_map[(sc, sd)]] = loss

fig, ax = plt.subplots(figsize=(6.5, 4.2))
im = ax.imshow(matrix, cmap='YlOrRd_r', aspect='auto')
ax.set_xticks(range(4)); ax.set_xticklabels(col_labels)
ax.set_yticks(range(5)); ax.set_yticklabels(row_labels)

for i in range(5):
    for j in range(4):
        v = matrix[i, j]
        if np.isnan(v):
            ax.text(j, i, 'pending', ha='center', va='center', color='#6B6860', style='italic', fontsize=8)
        else:
            ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                    color='#2A2822' if v > 1.5 else '#FFFFFF', fontsize=9)

ax.set_title(f'Final {METRIC_LABEL} — All Conditions', pad=12)
plt.colorbar(im, ax=ax, label=METRIC_LABEL)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()

print('All 5 figures generated in docs/figures/')
