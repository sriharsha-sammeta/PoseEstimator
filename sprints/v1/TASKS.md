# Sprint v1 — Tasks

> **Convention**: P0 = must have (pipeline broken without it), P1 = should have (needed for real training), P2 = nice to have (polish)
> **Primary checkpoint metric**: `val_mpjpe` (mean per-joint position error in mm)

---

## Phase 1: Project Scaffolding

- [ ] **Task 1**: Create project structure and `requirements.txt` (P0)
  - Acceptance: Directory layout exists; `pip install -r requirements.txt` in a fresh venv completes without error on Python 3.11; `python -c "import torch, h5py, wandb, tqdm"` passes
  - Files:
    - `requirements.txt`
    - `dataset/` (empty dir with `__init__.py`)
    - `model/` (empty dir with `__init__.py`)
    - `utils/` (empty dir with `__init__.py`)

---

## Phase 2: Dataset

- [ ] **Task 2**: Implement `dataset/egodex_dataset.py` — sequence indexer and pick-and-place filter (P0)
  - Acceptance:
    - `EgoDexDataset` scans a directory for `*.hdf5` files and reads `llm_description` from each without loading video
    - `filter_pick_place(sequences, keywords)` filters by configurable keyword list (default: `["pick", "place", "grasp", "put", "pick up", "set down"]`) and prints: `"Pick-and-place filter: train=N, val=M, test=K sequences"`
    - Dataset correctly reports `__len__()` after filtering
    - Works even if video `.mp4` files are absent (metadata-only mode)
  - Files: `dataset/egodex_dataset.py`

- [ ] **Task 3**: Implement `EgoDexDataset.__getitem__` — load frame + hand pose targets (P0)
  - Acceptance:
    - Returns a dict: `{"image": (3, H, W) float tensor, "instruction": str, "hand_joints": (T, 25, 3) float tensor, "confidence": (T, 25) float tensor}`
    - Extracts right-hand joint positions from `transforms[:, 18:43, :3, 3]` (translation column of SE(3))
    - Samples a random context frame and T=16 future frames from the sequence
    - If `mp4` is absent and metadata-only mode is active, returns a zero image tensor without crashing
    - Handles `torchcodec` import failure by falling back to `decord` then `cv2`
  - Files: `dataset/egodex_dataset.py`

- [ ] **Task 4**: Implement `dataset/synthetic_dataset.py` — dry-run synthetic data (P0)
  - Acceptance:
    - `SyntheticEgoDexDataset(n_samples=32)` returns batches with identical shape to real `EgoDexDataset.__getitem__` output
    - All tensors are random; `__len__` returns `n_samples`
    - Used automatically when `--dry_run` is passed
  - Files: `dataset/synthetic_dataset.py`

---

## Phase 3: Model

- [ ] **Task 5**: Implement `model/being_h0_wrapper.py` — load Being-H0 and define forward pass (P0)
  - Acceptance:
    - `load_being_h0(model_name, hf_token)` clones/imports Being-H GitHub repo if not present, loads `BeingHPolicy` checkpoint
    - `--dry_run` loads the **1B checkpoint** (`BeingBeyond/Being-H0-1B`) instead of the default 8B, so real model loading is always tested
    - Default `--model_name` is `BeingBeyond/Being-H0-8B-2508`; dry run overrides to `BeingBeyond/Being-H0-1B` automatically (can be overridden explicitly)
    - If download fails due to auth, prints: `"ERROR: Model download failed. Set HF_TOKEN env var or pass --hf_token_env. Model may be gated."`
    - `BeingH0Wrapper.forward(image, instruction, camera_intrinsics)` returns `{"pred_joints": (B, T, 50, 3)}` — both hands (50 joints: right 0–24, left 25–49) in meters
  - Files: `model/being_h0_wrapper.py`

---

## Phase 4: Training & Evaluation Loop

- [ ] **Task 6**: Implement `utils/metrics.py` — pose prediction metrics (P0)
  - Acceptance:
    - `compute_mpjpe(pred, gt)` — mean per-joint position error in mm; input shapes `(B, T, 50, 3)` in meters (both hands)
    - `compute_mean_l2(pred, gt)` — mean L2 across all joints and timesteps, mm
    - `compute_final_step_error(pred, gt)` — MPJPE at `pred[:, -1, :, :]`, mm
    - `compute_pct_within_threshold(pred, gt, threshold_mm=10.0)` — fraction of joints within threshold
    - All functions handle batch dimension; return scalar floats
    - Unit test: `test_metrics_smoke()` — random pred/gt of shape `(2, 16, 50, 3)` should return finite floats
  - Files: `utils/metrics.py`

- [ ] **Task 7**: Implement `train.py` — argument parser, W&B setup, and main entry point (P0)
  - Acceptance:
    - All CLI flags from PRD are present: `--model_name`, `--dataset_root`, `--output_dir`, `--batch_size`, `--num_workers`, `--max_steps`, `--epochs`, `--learning_rate`, `--weight_decay`, `--eval_every`, `--save_every`, `--resume_from`, `--wandb_project`, `--wandb_run_name`, `--dry_run`, `--hf_token_env`, `--eval_only`, `--task_filter`
    - `python train.py --help` prints all flags without error
    - W&B is initialized if `--wandb_project` is set; skipped (with warning) if `wandb` is not installed or `WANDB_API_KEY` is unset
    - `--dry_run` sets `batch_size=2`, `max_steps=5`, `eval_every=5`, uses `SyntheticEgoDexDataset`
  - Files: `train.py`

- [ ] **Task 8**: Implement `train_loop()` and `eval_loop()` in `train.py` (P0)
  - Acceptance:
    - `train_loop` iterates DataLoader, calls `model.forward()`, computes MSE loss, calls `loss.backward()`, steps optimizer, logs `train/loss` and `train/lr` to W&B every step
    - `eval_loop` runs model in `torch.no_grad()`, accumulates predictions, computes all 4 metrics, logs `val/mpjpe`, `val/mean_l2`, `val/final_step_error`, `val/loss` to W&B
    - Best `val_mpjpe` is tracked; checkpoint is saved to `--output_dir/best_model.pt` when improved
    - LR schedule: cosine decay with warmup (100 steps or 10% of total steps, whichever is smaller)
    - `--resume_from` loads `optimizer.state_dict()` and `global_step` from checkpoint; training resumes from correct step
  - Files: `train.py`

---

## Phase 5: README and Validation

- [ ] **Task 9**: Write `README.md` with full setup, all run modes, and dataset notes (P1)
  - Acceptance: README includes all sections listed below; no broken commands; tested against actual CLI flags in `train.py`
  - Required sections:
    1. **Quickstart** — 5-command venv setup
    2. **Hugging Face auth** — how to set `HF_TOKEN` env var; what to do if model is gated
    3. **W&B setup** — `wandb login` or `WANDB_API_KEY`
    4. **MacBook dry run** — exact command: `python train.py --dry_run --output_dir ./runs/dry_run`
    5. **Eval-only baseline** — exact command with all flags explained
    6. **Single-GPU Runpod training** — exact command with recommended flags
    7. **Multi-GPU Runpod (optional)** — `torchrun` command, note it requires DDP which is v2
    8. **EgoDex download** — test split curl command; note full training set size; dry-run requires no download
    9. **Pick-and-place filtering** — how keyword filter works; how to customize keywords; limitation note
    10. **Metrics explained** — table of all 4 metrics; `val_mpjpe` is checkpoint metric
  - Files: `README.md`

- [ ] **Task 10**: Dry-run smoke test — verify end-to-end pipeline on MacBook (P0)
  - Acceptance:
    - `python train.py --dry_run --output_dir /tmp/pose_test` completes without error
    - Prints sample counts (synthetic: 32 train, 8 val)
    - Runs exactly 5 training steps; prints step loss for each
    - Runs 1 validation pass; prints `val_mpjpe`, `val_mean_l2`, `val_final_step_error`
    - Saves a checkpoint to `/tmp/pose_test/best_model.pt`
    - Total wall time under 60 seconds on a MacBook (CPU mode)
    - `python train.py --eval_only --dry_run --output_dir /tmp/pose_eval` also completes without error
  - Files: `train.py` (fix any issues found during smoke test)

---

## Summary Table

| Task | Description | Priority | Phase |
|---|---|---|---|
| 1 | Project structure + requirements.txt | P0 | Scaffolding |
| 2 | EgoDexDataset indexer + pick-and-place filter | P0 | Dataset |
| 3 | EgoDexDataset.__getitem__ (frames + poses) | P0 | Dataset |
| 4 | SyntheticEgoDexDataset for dry run | P0 | Dataset |
| 5 | Being-H0 wrapper + MockBeingH0 | P0 | Model |
| 6 | Metrics: mpjpe, mean_l2, final_step, pct_threshold | P0 | Eval |
| 7 | train.py argparser + W&B init + main() | P0 | Training |
| 8 | train_loop() + eval_loop() + checkpointing | P0 | Training |
| 9 | README.md | P1 | Docs |
| 10 | MacBook dry-run smoke test | P0 | Validation |

**Total P0 tasks**: 9 | **Total P1 tasks**: 1 | **Total**: 10
