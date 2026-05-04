# SENPAI Research State

- **Updated:** 2026-05-04 10:22 UTC
- **Wave:** DrivAerML long-run single-model DDP8 validation
- **Advisor branch:** `drivaerml-long-20260504` (cut from `main` 2026-05-04)
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Students (4 idle, each with 8 GPUs):** dl24-fern, dl24-frieren, dl24-nezuko, dl24-tanjiro

## Most recent direction from human researcher team

Human launch directive 2026-05-04 (verbatim in `instructions/prompt-advisor.md`):

> Validate promising short-run/censored single-model mechanisms under controlled long DDP8 budgets (up to 24h, 8 GPUs each). Single-model only — no ensembles, model soups, NNLS aggregation. Report **test** metrics; validation is for steering. Each long run must be preceded by a DDP8 smoke that exercises init, dataset, W&B, checkpoint save/resume, gradient logging, and final test harvest.

Highest-priority single-model levers per the directive (with W&B references):
1. Mild static channel-aware tau weights (`9mm3sz7x` y=1.2/z=1.3, lr 9e-5).
2. Stronger bounded tau weights (`nh96x7m4` y=1.5/z=2.0).
3. EMA-proxy GradNorm with explicit volume guards (`wyz68o8r` α=0.5, `glir84cj` α=0.75, `341czkol` α=1.0).
4. Surface-loss weight 2.0 (`qqtdnlwq`).
5. Extended cosine + validation budget control (`5o7jc7wi` T_max=13).
6. Multi-sigma STRING log-frequency init (`ki2q9ko9`) — best volume.
7. Volume-point curriculum 16k→65k (`r5rw40rn`) — censored before final stages.
8. Per-axis output scaling (`wgvvevb9`).
9. STRING+QK norm (`tkiigfmc`), STRING+volume MLP (`8x7c537j`), 5L STRING (`70lnb3dt`), original STRING/GRAPE (`gcwx9yaa`).

## Current research focus (this wave)

- The four idle students each get one **distinct mechanism family** under the same long DDP8 protocol. We avoid composition tests until the components have been individually validated on this branch.
- Required operational discipline: every assignment includes a DDP8 smoke gate before the long run, an explicit volume-guard check on test metrics, and a `SENPAI-RESULT` marker built from `test_primary/*`.
- All four arms target the same test-metric reporting contract: aggregate test, surface, volume, wall, tau_x, tau_y, tau_z, runtime, slope-still-improving flag, checkpoint selection rule, and smoke-pass status.

### Round-1 assignment matrix

| Student | Mechanism family | Reference run | Source for code | Why now |
|---|---|---|---|---|
| dl24-fern | Mild static channel-aware tau weighting + lr 9e-5 | `9mm3sz7x` | `main` + small loss-weight extension | Strongest verified single-model SOTA; cleanest implementation; biggest expected win-per-line. |
| dl24-frieren | Multi-sigma STRING log-frequency init | `ki2q9ko9` | fetch `alphonse/multi-sigma-string-init` (or equivalent on `yi`/`tay`/`bengio`) | Best historical volume-pressure point; `program.md` warns of provenance drift, must verify code matches run config. |
| dl24-nezuko | Volume-point curriculum 16k→32k→49k→65k | `r5rw40rn` (censored) | curriculum scheduler over `--train-volume-points`, baseline trunk | The reference run timed out before the larger stages; 24h DDP8 should let it complete. |
| dl24-tanjiro | EMA-proxy GradNorm α=0.5 with explicit volume guards | `wyz68o8r` | `main` + EMA loss-weight scheduler | Dynamic-weighting alternative to fern's static lever; the directive flags exact GradNorm as too expensive — EMA proxy is the affordable form. |

This four-arm split tests static weighting, dynamic weighting, representation init, and data curriculum — four independent mechanism families.

## Potential next research directions

When the round-1 results land, the candidate next moves are (in priority order):

1. **Compose the round-1 winner with the second-best mechanism** under the same controlled long DDP8 budget — but only after at least two of the four arms produce clean terminal test results.
2. **STRING follow-ups:** STRING + QK norm (`tkiigfmc`), STRING + volume MLP (`8x7c537j`), 5L STRING (`70lnb3dt`), per-axis output scaling on STRING (`wgvvevb9`).
3. **Tau-y/tau-z bottleneck attack:** soft cross-flow geometry features (curvature via k-NN PCA, cross-flow exposure, multi-scale radius pooling) on top of the round-1 winner. Map evidence: `tau_y`/`tau_z` are 2.3-2.6× target on every single-model frontier point; representation/feature changes have moved the bottleneck more than scalar weighting.
4. **Volume-pressure generalization:** the gap between val-volume and test-volume on `9mm3sz7x` (4.14 vs 12.05) suggests held-out geometry/outlier structure rather than capacity; bucket the test set by SDF/distance-to-surface and curvature once the round-1 winner is selected.
5. **Channel-specific gradient instrumentation:** add per-channel surface_out gradient norms for surface_p, tau_x, tau_y, tau_z so the next reweighting wave can be guided by allocation, not endpoint metrics.
6. **Stronger lever combinations to consider only after individual confirmation:** mild tau weights + multi-sigma STRING; mild tau weights + extended cosine; volume curriculum + per-axis output scaling.

Items 1-2 are strict next round; items 3-6 are deeper-stack moves.

## Operational notes

- Source provenance is the central risk this wave. Students copying code from `yi`, `tay`, `bengio`, or `alphonse/*` branches must cite the source SHA and verify the copied implementation matches the W&B run config it claims to replicate.
- All runs must use DDP8 (8 GPUs per student); no single-GPU or split arms in this wave.
- Test metric harvest must reload from the best validation checkpoint, not the terminal epoch, when feasible.
