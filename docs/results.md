# Results

## Status

**All 28 of 28 runs complete.** Both seeds, both scales, all three conditions.

## Final Training Loss

Training loss at budget exhaustion. All runs consumed their full token budget (200 tokens/parameter).

### 100k Scale (~21.2M tokens, 1,296 steps)

| Condition | Seed 42 | Seed 1337 | Mean | Δ vs Baseline |
|-----------|---------|-----------|------|---------------|
| A (baseline) | 1.8056 | 1.8073 | 1.8064 | — |
| B frac 0.5 | 2.0077 | 1.9839 | 1.9958 | +0.1894 |
| B frac 0.9 | 1.8787 | 1.8533 | 1.8660 | +0.0596 |
| C frac 0.5 | 1.7503 | 1.7686 | 1.7594 | −0.0470 |
| C frac 0.9 | 1.7365 | 1.7355 | 1.7360 | −0.0704 |

At the 100k scale, noise-pretrained models (C) reached lower final training loss than the baseline, while corpus-pretrained models (B) reached higher loss. This pattern replicated across both seeds.

### 2M Scale (~396.8M tokens, 24,219 steps)

| Condition | Seed 42 | Seed 1337 | Mean | Δ vs Baseline |
|-----------|---------|-----------|------|---------------|
| A (baseline) | 0.3205 | 0.3395 | 0.3300 | — |
| B frac 0.5 | 0.3820 | 0.4212 | 0.4016 | +0.0716 |
| B frac 0.9 | 0.3719 | 0.4160 | 0.3940 | +0.0640 |
| C frac 0.5 | 0.3827 | 0.2824 | 0.3326 | +0.0026 |
| C frac 0.9 | 0.3648 | 0.2998 | 0.3323 | +0.0023 |

At the 2M scale, corpus-pretrained models (B) consistently underperformed the baseline across both seeds. The noise-pretrained models (C) showed a cross-seed inconsistency: seed 42 underperformed the baseline, while seed 1337 outperformed it by a comparable margin. The mean deltas for Condition C at 2M are near zero (+0.003).

### Cross-Seed Consistency

The corpus-pretrained condition (B) shows consistent negative transfer across both seeds at both scales. Both seeds agree on direction and approximate magnitude.

The noise-pretrained condition (C) shows a cross-seed inconsistency at the 2M scale:
- **Seed 42:** C underperforms baseline (Δ = +0.044 to +0.062)
- **Seed 1337:** C outperforms baseline (Δ = −0.040 to −0.057)

This inconsistency may be partially attributable to differences in the regenerated pretraining checkpoint for seed 1337 (see Caveats below), or it may reflect genuine variance at n=2 seeds. The mean across seeds is approximately zero, suggesting no robust directional effect for noise pretraining at the 2M scale.

## Checkpoint Fraction Effects

At both scales, the frac 0.9 condition (90% pretraining depth) consistently reaches lower final loss than frac 0.5 (50% depth), despite receiving identical finetuning budgets. This holds across both Path B and Path C, and across both seeds.

This is consistent with the interpretation that deeper pretraining exposure, while potentially more disruptive initially, produces a starting point from which gradient descent can more efficiently descend during finetuning. The effect is small relative to the baseline comparison.

## Pretraining Phase Observations

### Noise Pretraining (Path C)

At the 100k scale, pretraining loss on random characters plateaued at ~4.78, near the theoretical maximum-entropy limit for uniform character prediction (ln(120) ≈ 4.79). The 100k model lacks the capacity to memorize random sequences.

At the 2M scale, pretraining loss on random characters descended to 2.93 (seed 42) and 2.73 (seed 1337), well below maximum entropy. The 2M model possesses sufficient capacity to partially memorize random character sequences.

### Corpus Pretraining (Path B)

At the 2M scale, pretraining on structured Arabic text reached very low loss (0.107–0.109), indicating strong memorization of the Arabic corpus. The model actively suppressed logits for non-Arabic vocabulary tokens during this phase, leading to elevated initial loss when evaluated on English text at finetuning step 0 (observed values: 10.98–20.01 at 2M, 7.19–8.21 at 100k).

## Recovery Dynamics

All pretrained models at both scales recovered rapidly during early finetuning. Within ~200 steps (100k) or ~1,000 steps (2M), pretrained conditions reached comparable loss ranges to the baseline, after which trajectories diverged based on condition.

## Caveats

1. **Seed count:** n=2 seeds. Sufficient for checking directional consistency, not for formal statistical significance.
2. **Scale:** Models are 106k–1.98M parameters, character-level. Results may not transfer to larger scales or subword tokenization.
3. **Corpus discontinuity (Path B):** Seeds 42 and 1337 used different Arabic text corpora for Condition B pretraining. See `results/data_glossary.md`.
4. **Checkpoint regeneration (Path C, seed 1337):** The noise pretraining checkpoint for seed 1337 was lost during a repository directory restructuring and was regenerated. Although the same seed (1337) and configuration were used, the regenerated checkpoint may not be bitwise identical to the original due to non-deterministic GPU operations, and the finetuning runs from this regenerated checkpoint showed lower final loss than expected. This is a known confound for the seed-1337 Path C results specifically.
5. **Overfitting regime:** At 200 tokens/parameter on ~1MB of text, all 2M conditions memorize the training set. Training loss is the informative comparison metric; validation loss diverges for all conditions.
6. **Pilot study:** This is exploratory, not confirmatory. A pattern here is a reason to investigate further, not a finding on its own.
