# SENPAI Research State

- 2026-05-05 (updated ~10:35 UTC)
- Most recent research direction from human researcher team: None (no open issues)

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs testing mechanisms that showed promise in shorter waves but were either censored (timed out) or untested at full scale. Base config is now well-established: Lion, lr=1e-4, bs=2, train_surface_points=40k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), no-compile-model, surface_loss_weight=1.0, lr_warmup_epochs=1.

**Wave SOTA:** PR #599 (frieren, `sogus8sx`), test `abupt_axis_mean_rel_l2_pct` = **7.9303%**, val best = 6.5281%.

### Active Experiments

| PR | Student | Hypothesis | Key Mechanism | Status |
|----|---------|------------|---------------|--------|
| #664 | fern | Per-axis output scaling on STRING backbone | Combines per-axis output scaling (wgvvevb9) with multi-sigma STRING PE (ki2q9ko9) | Long run `a8emaoxm` EP10+. Best EP8=7.0915%. Oscillating ~7.1-7.6% after EP8. Merge conflict pending rebase post-run. Monitor EP20 gate. |
| #669 | frieren | Per-channel tau surface weighting (tau_y×1.2, tau_z×1.3) on SOTA base config | New `masked_mse_per_channel` helper + CLI flags `--tau-y-loss-weight 1.2 --tau-z-loss-weight 1.3`; requires code addition | Long run `er8wmo8d` EP3=7.78%. Strong trajectory. EP5 gate: kill if ≥8.5%. EP10 gate: kill if ≥7.2%. |
| #673 | tanjiro | Denser multi-sigma STRING PE with 7 sigmas [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] | `--pe-init-sigmas '0.1,0.25,0.5,1.0,2.0,4.0,8.0'` — pure CLI | Long run `zk35lops` EP7=8.54%. Decelerating slope. EP10 gate: kill if ≥8.0%. Wall_shear bottleneck at ~9.77%. |
| #678 | nezuko | Extended cosine schedule T_max=60 on SOTA base config (50-epoch long run) | `--lr-cosine-t-max 60` — pure CLI; cosine decay reaches ~26% of peak LR at EP50 instead of zero | JUST ASSIGNED — `nezuko/extended-cosine-t60-long`; long run pending start. PR #670 admin-merged, only 3-ep smoke `rse0oh3w` ran; 50-epoch confirmation never executed. |

### Closed / Negative Results This Wave

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #611 | fern | Per-channel tau weighting (bugfix v2) | Closed negative: test=12.406% — mechanism not effective in isolation on this config |
| #623 | tanjiro | EMA-proxy GradNorm α=0.5 + Lion + volume guard | Infrastructure kill required (student ignored 5 kill orders). Best val=12.4377%. Strongly negative. |
| #659 | norman | Width-over-Depth 4L/768d/12h cold-start | CLOSED: EP2=10.0258%, test=11.2020%. OOM at slices=128 forced slices=64; undertrained at termination. Hypothesis not falsified. |
| #677 | nezuko | Tau×1.2/1.3 + volume×2.0 combination on SOTA | Admin-only merge (code scaffolding, no experiment ran). No SENPAI-RESULT. Reassigned to extended cosine (#678). |

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

1. **Tau channel weighting (frieren #669):** Does per-channel tau_y×1.2/tau_z×1.3 surface weighting beat SOTA on the Lion+STRING base config? (First clean isolated test on SOTA config — prior fern tests were on different/buggy configs)
2. **Per-axis output scaling (fern #664):** Does combining wgvvevb9-style per-axis scaling with STRING PE backbone improve axis-specific metrics? Best so far EP8=7.0915%.
3. **Denser multi-sigma PE (tanjiro #673):** Does 7-sigma `[0.1..8.0]` PE improve over 5-sigma SOTA? EP10 gate imminent.
4. **Extended cosine T_max=60 (nezuko #678):** Does keeping LR at ~26% of peak at EP50 (vs 0% with T_max=50) improve final convergence? Pre-wave T_max=13 gave best volume score (11.867%); T_max=60 is a stronger form of the same idea.

## Potential Next Research Directions

- **Tau + vol combination:** Once frieren's #669 code merges (tau flags available), assign tau_y×1.2/tau_z×1.3 + volume_loss_weight=2.0 combo to a student.
- **Extended cosine + tau weighting combination:** If #669 shows gains and #678 confirms T_max=60 helps, combine T_max=60 with tau_y×1.2/tau_z×1.3.
- **Extended cosine + volume upweight:** If #678 (T_max=60) shows gains, also try volume_loss_weight=2.0 with T_max=60.
- **Higher bs with gradient accumulation:** bs=2 with DDP8 = effective bs=16. Could try bs=4 (effective bs=32) with gradient accumulation to maintain steps per epoch.
- **Adaptive curriculum:** Instead of fixed stage boundaries, advance curriculum when validation plateaus.
- **Spectral loss terms:** Add frequency-domain supervision for better high-frequency pressure field fidelity.
- **Architecture scale-up:** Current runs use default model size; scale-up with the 24h budget may be feasible.
- **Checkpoint averaging:** Post-wave combining of best checkpoints from multiple wave experiments.
