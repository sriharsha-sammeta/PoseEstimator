# Sprint v3 — Tasks: Real Finetuning on EgoDex

> **Convention**: P0 = must have, P1 = should have, P2 = nice to have
> **Budget cap**: $20 on Runpod GPU time
> **W&B project**: `being_h0_egodex`

---

## Phase 1: Code Fixes + Setup

- [ ] **Task 1**: Fix v2 bugs in train.py and being_h0_wrapper.py (P0)
  - Fix `load_being_h0()` — don't override `--model_name` when user explicitly passes it
  - Fix `_KNOWN_HIDDEN_DIMS["8b"]` from 4096 to 3584
  - Create `scripts/setup_pod.sh` — reliable one-command pod setup with beingvla patch
  - Push fixes to GitHub
  - Acceptance: `python train.py --dry_run --model_name BeingBeyond/Being-H0-8B-2508` uses 8B (not 1B) locally
  - Files: `model/being_h0_wrapper.py`, `train.py`, `scripts/setup_pod.sh`

- [ ] **Task 2**: Create persistent Runpod network volume (P0)
  - Create ~200 GB network volume via Runpod MCP
  - Volume should persist across pod terminations
  - Acceptance: Volume exists, can be listed via `mcp__runpod__list-network-volumes`
  - Files: none (infrastructure)

- [ ] **Task 3**: Create Runpod pod attached to the network volume (P0)
  - RTX A6000, 50 GB container disk, attached to the persistent volume
  - Inject HF_TOKEN and WANDB_API_KEY as env vars
  - Acceptance: Pod is running, SSH accessible, volume mounted at /workspace
  - Files: none (infrastructure)

- [ ] **Task 4**: Run setup_pod.sh on the pod (P0)
  - Clone repo, install deps, patch beingvla, verify imports
  - Acceptance: `python3 -c "import torch, beingvla; print('OK')"` passes, flash-attn installed
  - Files: none (remote commands)

---

## Phase 2: Data + Baseline

- [ ] **Task 5**: Download EgoDex test split to persistent volume (P0)
  - `curl` the 16 GB test.zip to the network volume, unzip
  - Acceptance: `.hdf5` files exist under `/workspace/egodex/test/`, count printed
  - Files: none (remote download)

- [ ] **Task 6**: Run baseline eval on raw pretrained Being-H0-8B (P0)
  - `python train.py --eval_only --model_name BeingBeyond/Being-H0-8B-2508 --dataset_root /workspace/egodex/test --output_dir /workspace/results/baseline --wandb_project being_h0_egodex --wandb_run_name baseline_8b_eval_only`
  - Record baseline metrics in `report.md`
  - Acceptance: val_mpjpe, val_mean_l2, val_final_step_error, val_pct_within_10mm printed and logged to W&B
  - Files: `sprints/v3/report.md` (started)

---

## Phase 3: Finetuning

- [ ] **Task 7**: Run head-only finetuning (frozen backbone) (P0)
  - `python train.py --model_name BeingBeyond/Being-H0-8B-2508 --dataset_root /workspace/egodex/test --output_dir /workspace/results/finetune_head_only --freeze_backbone --epochs 5 --batch_size 4 --eval_every 100 --save_every 500 --wandb_project being_h0_egodex --wandb_run_name finetune_head_only`
  - Acceptance: Training completes, best_model.pt saved, W&B run shows loss curve
  - Files: `sprints/v3/report.md` (updated with head-only results)

- [ ] **Task 8**: Run full-backbone finetuning (unfrozen) (P0)
  - `python train.py --model_name BeingBeyond/Being-H0-8B-2508 --dataset_root /workspace/egodex/test --output_dir /workspace/results/finetune_full --unfreeze_backbone --epochs 3 --batch_size 2 --learning_rate 5e-5 --eval_every 100 --save_every 500 --wandb_project being_h0_egodex --wandb_run_name finetune_full_backbone`
  - Acceptance: Training completes (or OOM is documented), best_model.pt saved
  - Files: `sprints/v3/report.md` (updated with full-backbone results)

---

## Phase 4: Evaluation + Reporting

- [ ] **Task 9**: Evaluate both finetuned models on test split and compile report (P0)
  - Run eval_only with each checkpoint: `--resume_from /workspace/results/finetune_head_only/best_model.pt` and `--resume_from /workspace/results/finetune_full/best_model.pt`
  - Add comparison table to `report.md` (baseline vs head-only vs full-backbone)
  - Add W&B dashboard links if available
  - Write `suggestions.md` with next steps
  - Acceptance: `report.md` has a complete comparison table; `suggestions.md` has 5+ actionable suggestions
  - Files: `sprints/v3/report.md` (finalized), `sprints/v3/suggestions.md`

- [ ] **Task 10**: Stop pod (keep volume for future use) (P0)
  - Stop pod via Runpod MCP — do NOT delete the network volume
  - Acceptance: Pod terminated, volume still listed in Runpod account
  - Files: none

---

## Summary Table

| Task | Description | Priority | Phase |
|---|---|---|---|
| 1 | Fix v2 bugs + create setup script | P0 | Setup |
| 2 | Create persistent network volume | P0 | Setup |
| 3 | Create pod attached to volume | P0 | Setup |
| 4 | Run setup_pod.sh on pod | P0 | Setup |
| 5 | Download EgoDex test split | P0 | Data |
| 6 | Baseline eval (raw pretrained 8B) | P0 | Baseline |
| 7 | Head-only finetuning (frozen backbone) | P0 | Finetuning |
| 8 | Full-backbone finetuning (unfrozen) | P0 | Finetuning |
| 9 | Evaluate + compile report + suggestions | P0 | Reporting |
| 10 | Stop pod, keep volume | P0 | Cleanup |

**Total tasks**: 10 (all P0)
