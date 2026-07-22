# Fixed-Budget Pretraining Transfer in Small Transformers

> **Status:** All 28 of 28 runs complete. Full matrix: 3 conditions × 2 scales × 2 seeds × 2 checkpoint fractions.
> Live results: **[mohamedhossammohamed.github.io/overfit-init-transfer](https://mohamedhossammohamed.github.io/overfit-init-transfer)**

---

## TLDR

**Question:** Does pretraining a small transformer on an unrelated corpus help, hurt, or do nothing when you then train it on a different corpus?

**Setup:** Three conditions — random init baseline (A), pretrained on structured Arabic text (B), pretrained on random noise (C) — at two scales (100k, 2M params), two seeds, fixed token budget of 200×param_count.

**Results (final training loss, lower = better):**

| | 100k (mean of 2 seeds) | 2M (mean of 2 seeds) |
|---|---|---|
| **A — Baseline** | 1.806 | 0.330 |
| **B — Arabic pretrain** | 1.866 to 1.996 | 0.394 to 0.402 |
| **C — Noise pretrain** | 1.736 to 1.759 | 0.332 to 0.333 |

**Key observations:**
- **Corpus pretraining (B) consistently hurt** — higher final loss than baseline at both scales, both seeds. The structured Arabic pretraining did not transfer useful structure to English character prediction.
- **Noise pretraining (C) showed a scale-dependent pattern** — at 100k, it slightly outperformed baseline (possibly acting as a regularizer). At 2M, the mean effect was approximately zero, with a cross-seed inconsistency (seed 42: slight negative transfer; seed 1337: slight positive transfer).
- **B ≈ C at 2M** in seed 42 — the corpus-pretrained and noise-pretrained conditions converged to similar loss, suggesting the structured nature of the pretraining data was not the relevant variable. However, seed 1337 did not replicate this equivalence.

**Caveats:** n=2 seeds, 100k–2M parameter scale, character-level tokenization, pilot/exploratory study. The seed-1337 noise pretrain checkpoint was regenerated after a data loss event, which may confound those specific results. See [docs/results.md](docs/results.md) for full discussion.

---

## Experimental Design

Three conditions compared at two model sizes (~106k and ~1.98M parameters) across two random seeds (42, 1337):

| Condition | Pretraining | Finetuning |
|-----------|-------------|------------|
| **A** (baseline) | None — random init | Target corpus (Tiny Shakespeare) |
| **B** (corpus-pretrained) | Structured Arabic-script corpus | Target corpus, from pretrained checkpoint |
| **C** (noise-pretrained) | Uniformly sampled random characters | Target corpus, from pretrained checkpoint |

- **Token budget:** 200 tokens per parameter (~10× Chinchilla compute-optimal), applied identically to pretraining and finetuning.
- **Checkpoint fractions:** B and C each produce finetuning runs from checkpoints at 50% and 90% of the pretraining budget.
- **All finetuning runs receive the same total step count**, regardless of checkpoint depth.
- The Arabic-script corpus has near-zero character overlap with the English target.

## Repository Structure

```
├── docs/
│   ├── index.html           # GitHub Pages status site
│   ├── methodology.md       # Full experimental methodology
│   ├── results.md           # Full results with tables and caveats
│   └── figures/             # Analysis figures (generated)
├── results/
│   ├── runs.jsonl           # Per-step telemetry (all 28 runs)
│   ├── run_summaries.jsonl  # Final metrics per completed run
│   ├── manifest.json        # The 28-run matrix spec
│   ├── data_glossary.md     # Corpus definitions, known discontinuities
│   └── health_log.jsonl     # Automated health check log
├── train.py                 # Training loop
├── model.py                 # GPT model definition
├── experiment_runner.py     # Run matrix orchestrator
├── generate_figures.py      # Analysis figure generation
├── data_preparation.py      # Dataset preparation
└── health_check.py          # Automated monitoring
```

## Documentation

- **[docs/methodology.md](docs/methodology.md)** — Full experimental design, architecture, hyperparameters, hardware, metric selection
- **[docs/results.md](docs/results.md)** — Complete results tables, cross-seed analysis, checkpoint fraction effects, caveats
- **[results/data_glossary.md](results/data_glossary.md)** — Corpus definitions and known discontinuities across seeds

## Reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install torch numpy
python data_preparation.py
python experiment_runner.py
```

### Hardware Note

Runs executed on Apple Silicon (MPS backend). Batch size / gradient accumulation calibrated per model size:
- **100k:** `batch_size=64, grad_accum=1`
- **2M:** `batch_size=16, grad_accum=4` (~25,600 tokens/sec)

## License

Apache License 2.0
