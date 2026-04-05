# Sprint v3 — PRD: Real Finetuning on EgoDex + Evaluation

## Overview

Run the full finetuning experiment on Runpod with real EgoDex data: establish baseline metrics for the raw pretrained Being-H0-8B model, finetune with frozen backbone (head-only) and unfrozen backbone, evaluate both against the baseline, log all curves to W&B, and document findings in `report.md` with architectural suggestions in `suggestions.md`.

---

## Goals

- **Done** = Baseline eval of raw pretrained Being-H0-8B on EgoDex pick-and-place test split is recorded
- **Done** = Head-only finetuning run completes with W&B logging and best checkpoint saved
- **Done** = Unfrozen-backbone finetuning run completes with W&B logging and best checkpoint saved
- **Done** = Both finetuned models are evaluated on test split; metrics compared to baseline
- **Done** = `report.md` contains all metrics, loss curves summary, and analysis
- **Done** = `suggestions.md` contains actionable architectural improvements for v4

---

## User Stories

- As a researcher, I want baseline numbers so I can measure whether finetuning actually helps.
- As a researcher, I want W&B dashboards for both runs so I can compare loss curves and pick the best approach.
- As a researcher, I want a report summarizing what worked, what didn't, and what to try next.
- As a researcher, I want a persistent Runpod volume so I don't re-download EgoDex and model weights for every experiment.

---

## Technical Architecture

### Infrastructure

```
┌────────────────────────────────────────────────────────────────┐
│  Runpod Network Volume (persistent, ~200 GB)                    │
│  ├── egodex/test/          ← 16 GB, downloaded once             │
│  ├── egodex/train/         ← 1.5 TB (if budget allows)          │
│  ├── huggingface_cache/    ← model weights cached here           │
│  └── results/              ← checkpoints + logs persist          │
└────────────────────────────────────────────────────────────────┘
                              │
                              │ mounted at /workspace
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Runpod Pod (ephemeral — attached to volume)                    │
│  GPU: RTX A6000 (48 GB) @ $0.33/hr                             │
│  Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel           │
│                                                                  │
│  /workspace/                                                     │
│  ├── PoseEstimator/        ← cloned from GitHub                  │
│  ├── egodex/ → volume      ← symlink or direct mount            │
│  └── results/ → volume     ← checkpoints persist across pods    │
│                                                                  │
│  Experiment plan:                                                │
│    Run 1: --eval_only (baseline, ~10 min)                        │
│    Run 2: --freeze_backbone (head-only, ~2-4 hrs)                │
│    Run 3: --unfreeze_backbone (full finetune, ~4-8 hrs)          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  W&B: project "being_h0_egodex"                                 │
│  ├── run: baseline_8b_eval_only                                  │
│  ├── run: finetune_head_only                                     │
│  └── run: finetune_full_backbone                                 │
│  Logged: train/loss, train/lr, val_mpjpe, val_mean_l2,          │
│          val_final_step_error, val_pct_within_10mm, best_mpjpe   │
└────────────────────────────────────────────────────────────────┘
```

### Budget estimate ($20 cap)

| Run | GPU | Est. time | Est. cost |
|---|---|---|---|
| Data download (test 16GB) | RTX A6000 | ~15 min | $0.08 |
| Baseline eval | RTX A6000 | ~15 min | $0.08 |
| Finetune head-only (5 epochs) | RTX A6000 | ~3 hrs | $1.00 |
| Finetune full backbone (3 epochs) | RTX A6000 | ~6 hrs | $2.00 |
| Final eval (both checkpoints) | RTX A6000 | ~30 min | $0.17 |
| **Total estimate** | | **~10 hrs** | **~$3.30** |

If the test split is too small for meaningful training, we download train splits incrementally (1 zip at a time, ~300GB each). Budget allows up to ~60 GPU hours.

### Code fixes needed (from v2 findings)

1. **Fix `--dry_run` model override**: `load_being_h0()` shouldn't override `--model_name` when the user explicitly passes it
2. **Fix `_KNOWN_HIDDEN_DIMS`**: Update 8B fallback from 4096 to 3584
3. **Create `scripts/setup_pod.sh`**: Reliable pod setup with proper beingvla patch

### Environment variables

| Variable | Source |
|---|---|
| `RUNPOD_API_KEY` | `/Users/sriharsha/Developer/keys_tokens/export_keys.sh` |
| `HF_TOKEN` | `/Users/sriharsha/Developer/keys_tokens/export_keys.sh` |
| `WANDB_API_KEY` | `/Users/sriharsha/Developer/keys_tokens/export_keys.sh` |

---

## Out of Scope (v3)

- Multi-GPU / DDP training
- LoRA / PEFT backbone finetuning (full unfreeze instead for v3)
- Downloading full 1.5TB training set (start with test split; expand if needed)
- Native motion token finetuning (MANO / GRVQ-8K)
- Hyperparameter sweeps
- Joint rotation supervision (SE(3) rotation matrices)

---

## Dependencies

- **v1 code** on GitHub: `sriharsha-sammeta/PoseEstimator`
- **v2 validated**: pipeline runs on Runpod GPU with 8B model
- **Runpod MCP tools**: `create-pod`, `stop-pod`, `delete-pod`, `create-network-volume`
- **SSH access**: `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`
- **Tokens**: `keys_tokens/export_keys.sh`

---

## Deliverables

| File | Content |
|---|---|
| `sprints/v3/report.md` | Baseline metrics, finetuning results, loss curve analysis, head-only vs full comparison |
| `sprints/v3/suggestions.md` | Architectural improvements, next experiments, what to change |
| `sprints/v3/PRD.md` | This document |
| `sprints/v3/TASKS.md` | Task list with completion status |
| `scripts/setup_pod.sh` | One-command pod setup script |
