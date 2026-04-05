# Sprint v2 — Tasks: Runpod GPU Validation

> **Convention**: P0 = must have, P1 = should have, P2 = nice to have
> **Sprint goal**: Verify the v1 pipeline runs on Runpod GPU with the real 8B model

---

- [ ] **Task 1**: Create a Runpod GPU pod via MCP or API (P0)
  - Acceptance: Pod is running with an A100 (or equivalent) GPU, SSH or exec access is available
  - Use Runpod MCP tools to create the pod
  - Pod config: PyTorch template, 50GB+ disk, 1x A100 40GB (or cheapest available GPU with 40GB+ VRAM)
  - Files: none (infrastructure only)

- [ ] **Task 2**: Clone repo and install all dependencies on the pod (P0)
  - Acceptance: `python -c "import torch, h5py, wandb, tqdm, transformers; print('OK')"` passes on the pod
  - Commands to run on pod:
    1. `git clone https://github.com/sriharsha-sammeta/PoseEstimator /workspace/PoseEstimator`
    2. `cd /workspace/PoseEstimator && python3 -m venv .venv && source .venv/bin/activate`
    3. `pip install -r requirements.txt`
    4. `git clone --depth=1 https://github.com/BeingBeyond/Being-H0.git third_party/Being-H0`
    5. `pip install flash-attn --no-build-isolation`
    6. `pip install einx timm opencv-python-headless`
    7. Apply the beingvla internvl_adapter.py patch
  - Files: none (all commands run on pod)

- [ ] **Task 3**: Set environment variables on the pod (P0)
  - Acceptance: `echo $HF_TOKEN` prints a valid token on the pod
  - Export `HF_TOKEN` from local `keys_tokens/export_keys.sh` onto the pod
  - Files: none

- [ ] **Task 4**: Run dry_run with real 8B model on GPU (P0)
  - Acceptance: The following are all true in the output:
    - `device=cuda`
    - `dtype=torch.bfloat16`
    - `flash_attn=True`
    - `hidden_dim=4096`
    - 5 training steps complete without error
    - Validation metrics printed (val_mpjpe, val_loss)
    - Checkpoint saved to `runs/gpu_dry_run/best_model.pt`
  - Command: `python train.py --dry_run --model_name BeingBeyond/Being-H0-8B-2508 --output_dir ./runs/gpu_dry_run`
  - Files: none

- [ ] **Task 5**: Run eval_only with dry_run on GPU (P0)
  - Acceptance: Eval-only mode completes, prints metric table (val_mpjpe, val_mean_l2, val_final_step_error, val_pct_within_10mm, val_loss)
  - Command: `python train.py --eval_only --dry_run --model_name BeingBeyond/Being-H0-8B-2508 --output_dir ./runs/gpu_eval`
  - Files: none

- [ ] **Task 6**: Terminate the Runpod pod (P0)
  - Acceptance: Pod is stopped/terminated, no ongoing charges
  - Use Runpod MCP tools to stop the pod
  - Files: none

---

## Summary Table

| Task | Description | Priority |
|---|---|---|
| 1 | Create Runpod GPU pod | P0 |
| 2 | Clone repo + install deps | P0 |
| 3 | Set env vars (HF_TOKEN) | P0 |
| 4 | Dry run with 8B model on GPU | P0 |
| 5 | Eval-only dry run on GPU | P0 |
| 6 | Terminate pod | P0 |

**Total tasks**: 6 (all P0)
