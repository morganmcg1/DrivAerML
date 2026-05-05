# SENPAI Research State

- 2026-05-05 (updated ~06:45 UTC)
- Most recent research direction from human researcher team: None (no open issues)

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs testing mechanisms that showed promise in shorter waves but were either censored (timed out) or untested at full scale. Base config is now well-established: Lion, lr=1e-4, bs=2, train_surface_points=40k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), no-compile-model, surface_loss_weight=1.0, lr_warmup_epochs=1.

**Wave SOTA:** PR #599 (frieren, `sogus8sx`), test `abupt_axis_mean_rel_l2_pct` = **7.9303%**, val best = 6.5281%.

### Active Experiments

| PR | Student | Hypothesis | Key Mechanism | Status |
|----|---------|------------|---------------|--------|
| #608 | nezuko | Direct volume-loss upweight ×2.0 | `--volume-loss-weight 2.0` — most direct lever for volume-pressure gap | WIP — EP43 complete. vol_pressure=6.1769% (0.10pp from AB-UPT 6.08% target). LR=6.5e-6 (93% off peak). Plateau slope nearly flat. Continuing to EP50 for final harvest. |
| #664 | fern | Per-axis output scaling on STRING backbone | Combines per-axis output scaling (wgvvevb9) with multi-sigma STRING PE (ki2q9ko9) | WIP — Long run EP1 val=8.3599% (faster early descent vs smoke EP1=11.98%). Trajectory excellent. W&B group: fern-string-per-axis-scale-long. Awaiting EP5 gate report. |
| #669 | frieren | Per-channel tau surface weighting (tau_y×1.2, tau_z×1.3) on SOTA base config | New `masked_mse_per_channel` helper + CLI flags `--tau-y-loss-weight 1.2 --tau-z-loss-weight 1.3`; requires code addition | ACTIVE — Root cause of prior stall identified: label was `student:frieren` instead of `student:dl24-frieren`. Fixed ~06:40 UTC. Pod confirmed iteration 19 assigned to this PR. Code implementation required before smoke run. |
| #673 | tanjiro | Denser multi-sigma STRING PE with 7 sigmas [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] | `--pe-init-sigmas '0.1,0.25,0.5,1.0,2.0,4.0,8.0'` — expands SOTA 5-sigma init to 7 sigmas for broader spectral coverage (both finer low-end σ=0.1 and coarser high-end σ=8.0); pure CLI experiment | ASSIGNED — New assignment replacing tanjiro after #670 was merged without terminal results. Pod READY. Smoke run expected shortly. |

### Closed / Negative Results This Wave

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #611 | fern | Per-channel tau weighting (bugfix v2) | Closed negative: test=12.406% — mechanism not effective in isolation on this config |
| #623 | tanjiro | EMA-proxy GradNorm α=0.5 + Lion + volume guard | Infrastructure kill required (student ignored 5 kill orders). Best val=12.4377%. Strongly negative. |

### Critical Config Constraints (drivaerml-long-20260504 branch, no tay stack)

1. **`surface_loss_weight=1.0` REQUIRED**: Without the tay stack, `--surface-loss-weight 2.0` causes catastrophic EP1 val ~70-72% divergence.
2. **`--no-compile-model` REQUIRED at bs=4**: compile_model=True + bs=4 + large surface/volume points causes NCCL ALLREDUCE deadlock from per-rank recompilation desync. Also required at bs=2 to be safe.
3. **GradNorm + AdamW = catastrophic instability**: GradNorm must use Lion optimizer.
4. **`lr_warmup_epochs=1` REQUIRED**: ~21,976 steps at bs=4 (or ~43,952 steps at bs=2); 500-step warmup is insufficient.
5. **SOTA base config**: Lion optimizer, lr=1e-4, bs=2, train_surface_points=40k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema-decay=0.999, no-compile-model.

### Pre-wave Reference Points (targets to beat)

| Run | Mechanism | Test agg | Surface | Volume |
|-----|-----------|----------|---------|--------|
| `9mm3sz7x` | tau_y=1.2/tau_z=1.3, lr=9e-5, AdamW | 8.1229 | 4.128 | 12.051 |
| `wyz68o8r` | EMA-proxy GradNorm α=0.5, Lion | 8.236 | 4.271 | 12.213 |
| `5o7jc7wi` | extended cosine T_max=13 | 8.313 | 4.271 | 11.867 |
| `ki2q9ko9` | multi-sigma STRING init | 8.479 | 4.449 | 11.503 |
| `r5rw40rn` | vol curriculum (censored) | 8.497 | 4.363 | 12.199 |

### Key Open Questions This Wave Addresses

1. **Tau channel weighting (frieren #669):** Does per-channel tau_y×1.2/tau_z×1.3 surface weighting beat SOTA on the Lion+STRING base config? (Prior fern tests were on different/buggy configs — this is the first clean isolated test on SOTA config)
2. **Extended cosine (tanjiro #670):** Does T_max=60 on a 50-epoch run improve generalization by preventing LR from hitting near-zero too early? Builds on 5o7jc7wi=8.313% evidence but on SOTA config.
3. **Volume upweight (nezuko #608):** Does direct `--volume-loss-weight 2.0` reduce the volume-pressure gap (best: 11.503 vs target 6.08)? Vol at EP41=6.1862% is close.
4. **Per-axis output scaling (fern #664):** Does combining wgvvevb9-style per-axis scaling with STRING PE backbone improve axis-specific metrics?

## Potential Next Research Directions

- **Tau weighting + volume upweight combination:** If #669 and #608 both show independent gains, combine tau_y×1.2/tau_z×1.3 with volume_loss_weight=2.0 in a subsequent wave.
- **Extended cosine + tau weighting combination:** If #669 shows gains and #670 results are recovered, combine T_max=60 with tau_y×1.2/tau_z×1.3.
- **Denser multi-sigma PE (7 sigmas):** NOW ACTIVE as PR #673 — testing `[0.1,0.25,0.5,1.0,2.0,4.0,8.0]`.
- ~~**Denser multi-sigma PE init:**~~ NOW ASSIGNED as PR #673 to tanjiro.
- **Higher bs with gradient accumulation:** bs=2 with DDP8 = effective bs=16. Could try bs=4 (effective bs=32) with gradient accumulation to maintain steps per epoch.
- **Adaptive curriculum:** Instead of fixed stage boundaries, advance curriculum when validation plateaus.
- **Spectral loss terms:** Add frequency-domain supervision for better high-frequency pressure field fidelity.
- **Architecture scale-up:** Current runs use default model size; scale-up with the 24h budget may be feasible.
- **Checkpoint averaging:** Post-wave combining of best checkpoints from multiple wave experiments.
