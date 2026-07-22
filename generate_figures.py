import os
import json
import matplotlib
import pandas as pd
import numpy as np

os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.mplconfig')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_run_id(run_id):
    if 'pretrain' in run_id:
        return None
    
    scale = '2M' if '2M' in run_id else '100k'
    seed = '42' if 'seed42' in run_id else '1337'
    
    if run_id.startswith('a_'):
        condition = 'A'
        frac = None
    elif run_id.startswith('b_'):
        condition = 'B'
        frac = '0.5' if 'frac0.5' in run_id else '0.9'
    elif run_id.startswith('c_'):
        condition = 'C'
        frac = '0.5' if 'frac0.5' in run_id else '0.9'
    else:
        return None
        
    return {
        'run_id': run_id,
        'scale': scale,
        'seed': seed,
        'condition': condition,
        'frac': frac,
        'label': f"{condition} {frac}" if frac else condition
    }

def get_style(meta):
    colors = {
        'A': '#2A2822',
        'B 0.5': '#C86D3B',
        'B 0.9': '#8B3A19',
        'C 0.5': '#5B8C69',
        'C 0.9': '#2E5A3C'
    }
    color = colors.get(meta['label'], '#000000')
    linestyle = '-' if meta['seed'] == '42' else '--'
    return color, linestyle

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            if 'train_loss' not in row or row['train_loss'] is None:
                continue
            if 'run_id' not in row:
                continue
                
            run_id = row['run_id']
            meta = parse_run_id(run_id)
            if meta is None:
                continue
            
            data.append({
                'run_id': run_id,
                'scale': meta['scale'],
                'seed': meta['seed'],
                'condition': meta['condition'],
                'frac': meta['frac'],
                'label': meta['label'],
                'step': row.get('step', 0),
                'tokens_seen': row.get('tokens_seen', 0),
                'train_loss': row['train_loss'],
                'weight_update_norm': row.get('weight_update_norm', np.nan)
            })
    return pd.DataFrame(data)

def main():
    os.makedirs('docs/figures', exist_ok=True)
    df = load_data('results/runs.jsonl')
    
    if df.empty:
        print("No data loaded. Check runs.jsonl")
        return
        
    # Pre-compute final losses
    final_losses = df.loc[df.groupby('run_id')['step'].idxmax()]
    
    # 1. learning_curves.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, scale in enumerate(['100k', '2M']):
        ax = axes[i]
        subset = df[df['scale'] == scale]
        for run_id, group in subset.groupby('run_id'):
            meta = parse_run_id(run_id)
            color, ls = get_style(meta)
            ax.plot(group['tokens_seen'], group['train_loss'], color=color, linestyle=ls, label=f"{meta['label']} (s{meta['seed']})")
        ax.set_title(f"Learning Curves ({scale})")
        ax.set_xlabel("Tokens Seen")
        ax.set_ylabel("Train Loss")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize='small', loc='upper right')
    plt.tight_layout()
    plt.savefig('docs/figures/learning_curves.png', dpi=200)
    plt.close()

    # 2. relative_delta.png
    deltas = []
    for scale in ['100k', '2M']:
        scale_final = final_losses[final_losses['scale'] == scale]
        baseline_mean = scale_final[scale_final['label'] == 'A']['train_loss'].mean()
        for label in ['B 0.5', 'B 0.9', 'C 0.5', 'C 0.9']:
            cond_mean = scale_final[scale_final['label'] == label]['train_loss'].mean()
            deltas.append({
                'scale': scale,
                'label': label,
                'delta': cond_mean - baseline_mean
            })
    delta_df = pd.DataFrame(deltas)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, scale in enumerate(['100k', '2M']):
        ax = axes[i]
        subset = delta_df[delta_df['scale'] == scale]
        colors = [get_style({'label': l, 'seed': '42'})[0] for l in subset['label']]
        bars = ax.bar(subset['label'], subset['delta'], color=colors)
        ax.set_title(f"Delta vs Baseline A ({scale})")
        ax.set_ylabel("Delta Final Train Loss")
        ax.axhline(0, color='black', linewidth=0.8)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), ha='center', va='bottom' if yval > 0 else 'top')
    plt.tight_layout()
    plt.savefig('docs/figures/relative_delta.png', dpi=200)
    plt.close()

    # 3. recovery_dynamics.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (scale, max_steps) in enumerate([('100k', 300), ('2M', 2000)]):
        ax = axes[i]
        subset = df[(df['scale'] == scale) & (df['seed'] == '42') & (df['step'] <= max_steps)]
        for run_id, group in subset.groupby('run_id'):
            meta = parse_run_id(run_id)
            color, ls = get_style(meta)
            ax.plot(group['step'], group['train_loss'], color=color, linestyle=ls, label=meta['label'])
        ax.set_title(f"Recovery Dynamics ({scale}, Seed 42)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Train Loss")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig('docs/figures/recovery_dynamics.png', dpi=200)
    plt.close()

    # 4. weight_update_norms.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, scale in enumerate(['100k', '2M']):
        ax = axes[i]
        subset = df[(df['scale'] == scale) & (df['seed'] == '42')]
        for run_id, group in subset.groupby('run_id'):
            meta = parse_run_id(run_id)
            color, ls = get_style(meta)
            ax.plot(group['step'], group['weight_update_norm'], color=color, linestyle=ls, label=meta['label'], alpha=0.7)
        ax.set_title(f"Weight Update Norms ({scale}, Seed 42)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Weight Update Norm")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig('docs/figures/weight_update_norms.png', dpi=200)
    plt.close()

    # 5. loss_heatmap.png
    heatmap_data = np.zeros((5, 4))
    rows = ['A', 'B 0.5', 'B 0.9', 'C 0.5', 'C 0.9']
    cols = [('100k', '42'), ('100k', '1337'), ('2M', '42'), ('2M', '1337')]
    
    for r_idx, label in enumerate(rows):
        for c_idx, (scale, seed) in enumerate(cols):
            val = final_losses[(final_losses['label'] == label) & (final_losses['scale'] == scale) & (final_losses['seed'] == seed)]
            if len(val) > 0:
                heatmap_data[r_idx, c_idx] = val.iloc[0]['train_loss']
            else:
                heatmap_data[r_idx, c_idx] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap_data, cmap='YlGnBu')
    
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels([f"{s} s{sd}" for s, sd in cols])
    ax.set_yticklabels(rows)
    
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = heatmap_data[i, j]
            text = ax.text(j, i, f"{v:.4f}" if not np.isnan(v) else "NaN", ha="center", va="center", color="black" if np.isnan(v) or v > np.nanmean(heatmap_data) else "white")
            
    ax.set_title("Final Train Loss Heatmap")
    plt.tight_layout()
    plt.savefig('docs/figures/loss_heatmap.png', dpi=200)
    plt.close()

    # 6. scale_interaction.png
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.35
    x = np.arange(2)
    
    a_100k = final_losses[(final_losses['scale'] == '100k') & (final_losses['label'] == 'A')]['train_loss'].mean()
    best_pre_100k_data = final_losses[(final_losses['scale'] == '100k') & (final_losses['label'] != 'A')].groupby('label')['train_loss'].mean()
    best_pre_100k = best_pre_100k_data.min() if not best_pre_100k_data.empty else np.nan
    best_pre_100k_label = best_pre_100k_data.idxmin() if not best_pre_100k_data.empty else 'N/A'
    
    a_2m = final_losses[(final_losses['scale'] == '2M') & (final_losses['label'] == 'A')]['train_loss'].mean()
    best_pre_2m_data = final_losses[(final_losses['scale'] == '2M') & (final_losses['label'] != 'A')].groupby('label')['train_loss'].mean()
    best_pre_2m = best_pre_2m_data.min() if not best_pre_2m_data.empty else np.nan
    best_pre_2m_label = best_pre_2m_data.idxmin() if not best_pre_2m_data.empty else 'N/A'
    
    a_means = [a_100k, a_2m]
    best_pre_means = [best_pre_100k, best_pre_2m]
    
    rects1 = ax.bar(x - bar_width/2, a_means, bar_width, label='Baseline A', color=get_style({'label': 'A', 'seed': '42'})[0])
    
    c100 = get_style({'label': best_pre_100k_label, 'seed': '42'})[0] if best_pre_100k_label != 'N/A' else 'gray'
    c2m = get_style({'label': best_pre_2m_label, 'seed': '42'})[0] if best_pre_2m_label != 'N/A' else 'gray'
    
    rects2 = ax.bar(x + bar_width/2, best_pre_means, bar_width, label='Best Pretrained', color=[c100, c2m])
    
    ax.set_ylabel('Mean Final Train Loss')
    ax.set_title('Scale Interaction: Regularization Flip')
    ax.set_xticks(x)
    ax.set_xticklabels(['100k', '2M'])
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=get_style({'label': 'A', 'seed': '42'})[0], label='Baseline A'),
                       Patch(facecolor='gray', label='Best Pretrained')]
    ax.legend(handles=legend_elements)
    
    for i, rect in enumerate(rects1):
        if not np.isnan(a_means[i]):
            ax.text(rect.get_x() + rect.get_width()/2., rect.get_height(), f"{a_means[i]:.4f}", ha='center', va='bottom')
    for i, rect in enumerate(rects2):
        if not np.isnan(best_pre_means[i]):
            lbl = best_pre_100k_label if i == 0 else best_pre_2m_label
            ax.text(rect.get_x() + rect.get_width()/2., rect.get_height(), f"{best_pre_means[i]:.4f}\n({lbl})", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig('docs/figures/scale_interaction.png', dpi=200)
    plt.close()

    # 7. seed_replication.png
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels_2m = ['A', 'B 0.5', 'B 0.9', 'C 0.5', 'C 0.9']
    s42_vals = []
    s1337_vals = []
    
    for lbl in labels_2m:
        val42 = final_losses[(final_losses['scale'] == '2M') & (final_losses['label'] == lbl) & (final_losses['seed'] == '42')]['train_loss']
        val1337 = final_losses[(final_losses['scale'] == '2M') & (final_losses['label'] == lbl) & (final_losses['seed'] == '1337')]['train_loss']
        s42_vals.append(val42.iloc[0] if len(val42) > 0 else 0)
        s1337_vals.append(val1337.iloc[0] if len(val1337) > 0 else 0)
        
    x = np.arange(len(labels_2m))
    bar_width = 0.35
    
    colors = [get_style({'label': l, 'seed': '42'})[0] for l in labels_2m]
    
    rects1 = ax.bar(x - bar_width/2, s42_vals, bar_width, label='Seed 42', color=colors, edgecolor='black', hatch='')
    rects2 = ax.bar(x + bar_width/2, s1337_vals, bar_width, label='Seed 1337', color=colors, edgecolor='black', hatch='//')
    
    ax.set_ylabel('Final Train Loss')
    ax.set_title('Seed Replication (2M Scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_2m)
    ax.legend()
    
    all_vals = [v for v in s42_vals + s1337_vals if v > 0]
    if all_vals:
        ax.set_ylim(min(all_vals) * 0.95, max(all_vals) * 1.05)
        
    plt.tight_layout()
    plt.savefig('docs/figures/seed_replication.png', dpi=200)
    plt.close()

if __name__ == '__main__':
    main()
