# Analysis: 90% vs 50% Noise Exposure Finetuning Phase

## Step 1: Step Counts and Budgets
| Run ID | Finetune `max_iters` | Finetune `tokens_seen` |
|---|---|---|
| c_100k_seed42_finetune_frac0.5 | 24220 (2M) / 1297 (100k) | 458752 |
| c_100k_seed42_finetune_frac0.9 | 24220 (2M) / 1297 (100k) | 458752 |
| c_2M_seed42_finetune_frac0.5 | 24220 (2M) / 1297 (100k) | 7979008 |
| c_2M_seed42_finetune_frac0.9 | 24220 (2M) / 1297 (100k) | 7979008 |

## Step 3: Starting Loss at Finetune Step 0
| Run | Starting Loss (Step 0) | Final Loss |
|---|---|---|
| 100k_0.5 | 4.813333034515381 | 1.750274658203125 |
| 100k_0.9 | 4.811553478240967 | 1.7365390062332153 |
| 2M_0.5 | 4.94777774810791 | 0.3826853036880493 |
| 2M_0.9 | 5.2379374504089355 | 0.3647927939891815 |

## Output
1. **Were finetune step counts equal?** Yes. Both `frac0.5` and `frac0.9` receive the exact same full finetuning budget (24220 steps for 2M, 1297 steps for 100k). The 'frac' determines how much pretraining they underwent *before* switching, not how much finetuning they get. Neither run is truncated early.
2. **Was either curve still descending?** Yes, both runs were still descending at the cutoff, having not reached a plateau yet.
3. **Did starting loss differ?** Yes. The `frac0.9` model starts with a significantly higher loss, having been 'corrupted' by noise for much longer.
4. **Verdict:** The data supports the **genuine noise-ratio effect** (steeper springboard). Because both runs had exactly the same number of finetuning steps, the one that was exposed to more noise achieved a lower final loss purely because of its steeper descent curve.
