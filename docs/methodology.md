# Methodology

## Overview

This document describes the experimental design, implementation, and analysis methodology for a pilot study on fixed-budget pretraining transfer in small transformers.

## Research Question

Does pretraining a small transformer to extreme, fixed-token-budget exposure on one corpus — before training on an unrelated one — change downstream performance versus training from random initialization?

## Experimental Design

### Conditions

Three training conditions are compared:

| Condition | Label | Pretraining Phase | Finetuning Phase |
|-----------|-------|-------------------|------------------|
| **A** | Baseline | None (random init) | Target corpus |
| **B** | Corpus-pretrained | Structured Arabic-script text | Target corpus |
| **C** | Noise-pretrained | Uniformly sampled random characters | Target corpus |

The target corpus is a subset of Tiny Shakespeare (English, character-level). The Arabic-script pretraining corpus was chosen for near-zero character overlap with the English target, so any measured transfer effect cannot be attributed to shared content or vocabulary statistics.

Condition C serves as a control: it tests whether *any* prior gradient exposure — regardless of data structure — produces the same effect as structured pretraining.

### Model Architecture

GPT-2-style character-level transformer (decoder-only), implemented in PyTorch.

| Parameter | 100k Scale | 2M Scale |
|-----------|-----------|----------|
| Layers | 2 | 10 |
| Attention Heads | 4 | 4 |
| Embedding Dim | 64 | 128 |
| Total Parameters | ~106k | ~1.98M |
| Vocabulary | 120 shared characters (Arabic + English + punctuation) |

### Token Budget

Training budget is set at **200 tokens per parameter** — approximately 10× the compute-optimal ratio identified by Hoffmann et al. (2022). This ratio was chosen deliberately to induce memorization rather than avoid it. The same budget applies to both pretraining and finetuning phases.

- **100k scale:** ~21.2M tokens total, 1,296 optimizer steps
- **2M scale:** ~396.8M tokens total, 24,219 optimizer steps

### Checkpoint Fractions

For Conditions B and C, intermediate checkpoints are saved at two points during pretraining:
- **frac 0.5:** Checkpoint at 50% of the pretraining token budget
- **frac 0.9:** Checkpoint at 90% of the pretraining token budget

Each checkpoint produces a separate finetuning run. All finetuning runs receive the **same total step count** as the baseline, regardless of checkpoint depth. The `frac` parameter controls pretraining depth, not finetuning budget.

### Seeds

Two random seeds (42, 1337) are used per configuration. This provides directional consistency checks but is insufficient for formal statistical significance claims.

**Known corpus discontinuity:** Seeds 42 and 1337 use different underlying text for Condition B pretraining. Seed 42 uses the original Arabic religious text corpus; seed 1337 uses a byte-and-token-count-matched Arabic Wikipedia replacement. Conditions A and C use identical data across seeds and are true replicates. See `results/data_glossary.md` for details.

### Run Matrix

The full matrix contains **28 runs**:
- 8 pretraining runs (2 conditions × 2 scales × 2 seeds)
- 4 baseline runs (Condition A × 2 scales × 2 seeds)
- 16 finetuning runs (2 conditions × 2 fractions × 2 scales × 2 seeds)

### Optimizer and Hyperparameters

- **Optimizer:** AdamW
- **Effective batch size:** 16,384 tokens per optimizer step (64 sequences × 256 tokens)
- **100k:** batch_size=64, grad_accum=1
- **2M:** batch_size=16, grad_accum=4
- **Learning rate schedule:** Cosine decay with warmup

### Hardware

All runs executed sequentially on Apple Silicon (MPS backend). Real training throughput at 2M scale: ~25,600 tokens/second with the 16×4 batch/accumulation configuration.

## Metric Selection

At the 2M scale with 200 tokens per parameter on ~1MB of target text, all conditions substantially overfit the training set. Validation loss diverges above 4.0 while training loss falls below 0.4. Under these conditions, **training loss** is the informative comparison metric: it reflects how efficiently each initialization condition memorizes the target data under identical compute budgets.

At the 100k scale, overfitting is moderate and the pattern is consistent across both training and validation loss.

## Analysis Approach

- Final training loss is compared at matched step budgets across conditions
- Delta values (condition loss minus baseline loss) quantify relative performance
- Cross-seed consistency is assessed by comparing deltas across seeds 42 and 1337
- Learning curve trajectories are plotted to assess convergence dynamics
- Weight update norms (‖ΔW‖) are tracked per step to characterize optimization dynamics
- Early finetuning recovery is examined to understand how quickly pretrained models adapt

## Data Artifacts

All metrics are logged to `results/` as JSON Lines:
- `runs.jsonl` — Per-step telemetry for all 28 runs
- `run_summaries.jsonl` — Final metrics per completed run
- `manifest.json` — The 28-run matrix specification
- `health_log.jsonl` — Automated health check log
- `data_glossary.md` — Corpus definitions, known discontinuities, and seed notes
