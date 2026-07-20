# Overfit Init Transfer Pilot

This repository houses the scripts and tracking files for an exploratory pilot trial testing whether extreme overfitting on one dataset (a structured Arabic-script corpus) leaves a reusable "trace" of weights that accelerates learning on a completely different dataset (Tiny Shakespeare) when compared to learning from scratch or random pre-training.

> **Status:** Pilot trial is currently **in progress**.
> Follow the live progress updates at: [GitHub Pages Site](https://mohamedhossammohamed.github.io/overfit-init-transfer)

*Note: The core training scripts (`experiment_runner.py`, `train.py`, `health_check.py`) remain at the repo root during the active pilot run to ensure process stability and avoid breaking active cron jobs or internal file paths.*

## Experimental Design

The trial spans 36 runs testing 2 model sizes (100k, 2M) across 2 random seeds to assess consistency. The design follows three parallel training paths:

- **Path A (Control):** Train a randomly initialized model on Tiny Shakespeare for 2000 steps.
- **Path B (Experiment):** Pre-train a model on a structured Arabic-script corpus for a fixed token budget (checkpointing exactly at 50% and 90% of the token budget), then fine-tune those checkpoints on Tiny Shakespeare for the same duration.
- **Path C (Random Control):** Pre-train a model on a random character distribution for a fixed token budget, then fine-tune on Tiny Shakespeare (controls for the mere act of having seen *any* data).

## Honest Limitations

This is a **small-scale, exploratory pilot trial**, not a confirmatory final result. 
- **Scale:** The models are extremely small (100k and 2M parameters).
- **Domain:** It uses character-level tokenization.
- **Power:** N=2 seeds is sufficient to check if a result is "same/different" across seeds, but is too weakly powered to establish definitive statistical significance. Any observed effect will require larger seed replication to be fully trusted.

## Reproducing This Work

To run the pipeline from scratch on your own machine:
1. Initialize the Python environment: `python -m venv venv && source venv/bin/activate`
2. Prepare the datasets and dummy models: `python data_preparation.py`
3. Generate the execution manifest and launch the trial: `python experiment_runner.py`

## Data and Results Tracking

All data is logged actively into the `results/` directory as JSON Lines.
- `runs.jsonl`: Contains the step-by-step metrics for every active run. Schema includes `run_id`, `step`, `train_loss`, `val_loss_shakespeare`, and `weight_update_norm` (the Euclidean norm of parameter changes to track plasticity).
- `manifest.json`: Defines the expected runs for the current experiment.
- `health_log.jsonl`: Output of the automated 4-hour background health checks tracking memory, pace, and loss trajectory.

## Token Budget Paradigm

Training budget is set to 200 tokens per parameter — 10x the compute-optimal ratio identified by Hoffmann et al. (2022, "Training Compute-Optimal Large Language Models"). We deliberately overshoot this ratio to induce memorization rather than avoid it. The 10x multiplier itself is a chosen heuristic and is documented here for reproducibility. Checkpoints are taken deterministically at 50% and 90% of the token budget.

### Training Hardware Optimization
For training runs on Apple Silicon hardware (MPS backend), we calibrated execution throughput by adjusting the batch size vs gradient accumulation split. To maximize tokens/second while keeping the effective optimizer step size (64 sequences of 256 tokens = 16,384 tokens) mathematically identical:
*   **100k Model:** Trained using `batch_size = 64` and `grad_accum = 1`.
*   **2M Model:** Evaluated `16x4` (baseline), `32x2`, and `64x1`. While synthetic micro-benchmarks favored `64x1` due to larger parallel tensor operations on the GPU, real training runs revealed severe latency bottlenecks under `64x1` (1.16 sec/step vs 0.64 sec/step for baseline) due to Metal Memory Allocator overhead on larger active activation tensors and validation-evaluation scale shifts. Consequently, we finalized the 2M configuration at `batch_size = 16, grad_accum = 4` to achieve maximum real-run throughput (~25,600 tokens/second).

## License

Apache License 2.0
