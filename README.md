# Fixed-Budget Pretraining Transfer in Small Transformers

A pilot study on whether pretraining a small transformer to extreme, fixed-token-budget exposure on one corpus — before training on an unrelated one — changes downstream performance versus training from random initialization.

> **Status:** 23 of 28 runs complete. Seed-1337 replication at 2M scale in progress.
> Live results: **[mohamedhossammohamed.github.io/overfit-init-transfer](https://mohamedhossammohamed.github.io/overfit-init-transfer)**

## Experimental Design

Three conditions are compared at two model sizes (~106k and ~1.98M parameters) across two random seeds (42, 1337):

| Condition | Pretraining | Finetuning |
|-----------|-------------|------------|
| **A** (baseline) | None — random init | Target corpus (Tiny Shakespeare) |
| **B** (corpus-pretrained) | Structured Arabic-script corpus | Target corpus, from pretrained checkpoint |
| **C** (noise-pretrained) | Uniformly sampled random characters | Target corpus, from pretrained checkpoint |

- Training budget: 200 tokens per parameter (~10× Chinchilla compute-optimal), deliberately inducing memorization.
- Conditions B and C produce two finetuning runs each, from checkpoints at 50% and 90% of the pretraining budget.
- All finetuning runs receive the same total step count, regardless of checkpoint depth.

The Arabic-script corpus was chosen for near-zero character overlap with the English target, so any measured effect cannot be attributed to shared content or vocabulary statistics. Condition C controls for whether any prior gradient exposure — regardless of data structure — produces the same effect.

## Current Results

At the 2M scale (seed 42, the only seed with all conditions complete):

| Condition | Final Training Loss |
|-----------|-------------------|
| A (baseline) | 0.3205 |
| B (corpus, frac 0.9) | 0.3719 |
| C (noise, frac 0.9) | 0.3648 |

The corpus-pretrained (B) and noise-pretrained (C) conditions converge to similar final loss, both underperforming the random-init baseline. This indicates that, at this scale, prior training exposure on a structured non-English corpus does not confer a measurable benefit over noise pretraining — the structured nature of the pretraining data does not appear to be the relevant variable.

At the 100k scale (both seeds complete), the noise-pretrained condition slightly outperforms the baseline, while the corpus-pretrained condition underperforms it.

**This is a tentative reading from a single seed at the 2M scale. Seed-1337 replication is in progress.**

## Repository Structure

```
├── docs/                    # GitHub Pages status site
│   ├── index.html           # Live results page
│   └── figures/             # Analysis figures (generated)
├── results/
│   ├── runs.jsonl           # Step-by-step metrics for every run
│   ├── run_summaries.jsonl  # Final metrics per completed run
│   ├── manifest.json        # Planned run matrix (28 runs)
│   ├── health_log.jsonl     # Automated health check log
│   ├── data_glossary.md     # Corpus definitions and known discontinuities
│   └── */                   # Per-run checkpoint directories
├── train.py                 # Core training loop
├── experiment_runner.py     # Orchestrates the full run matrix
├── generate_figures.py      # Produces analysis figures from runs.jsonl
├── health_check.py          # Automated background monitoring
├── model.py                 # GPT model definition
└── data_preparation.py      # Dataset preparation
```

## Data Tracking

All metrics are logged to `results/` as JSON Lines:

- **`runs.jsonl`** — Per-step telemetry: `run_id`, `step`, `tokens_seen`, `train_loss`, `val_loss`, `weight_update_norm`, `grad_norm`, `lr`, and architecture parameters.
- **`run_summaries.jsonl`** — Final metrics per completed run: `final_loss`, `loss_at_50pct_tokens`, `loss_at_90pct_tokens`, `still_descending_at_budget_end`.
- **`manifest.json`** — The 28-run matrix with expected step counts and token budgets.
- **`health_log.jsonl`** — Automated health checks: memory, disk, active run, loss trajectory.

## Known Limitations

- **Scale:** Models are 106k–1.98M parameters, character-level. Results may not generalize to larger scales or subword tokenization.
- **Seeds:** N=2 seeds. Sufficient for checking directional consistency, not for statistical significance claims.
- **Corpus discontinuity:** Seeds 42 and 1337 use different underlying text for Path B pretraining (original corpus vs. byte-and-token-matched Arabic Wikipedia replacement). Paths A and C are true replicates across seeds. See `results/data_glossary.md`.
- **Overfitting regime:** At 200 tokens/parameter on ~1MB of text, all 2M conditions memorize the training set. Training loss is the informative comparison metric at this scale.

## Reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install torch numpy
python data_preparation.py
python experiment_runner.py
```

### Hardware Note

Runs executed on Apple Silicon (MPS backend). Batch size / gradient accumulation calibrated per model size to maximize throughput while keeping the effective step size identical (64 × 256 = 16,384 tokens/step):
- **100k:** `batch_size=64, grad_accum=1`
- **2M:** `batch_size=16, grad_accum=4` (~25,600 tokens/sec)

## License

Apache License 2.0
