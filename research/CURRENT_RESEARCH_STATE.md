# SENPAI Research State
- 2026-05-12 ~05:40 UTC

## Human Research Directive (Issue #882)
**TOP PRIORITY — Volume Pressure Focus:**
- The **TEST volume pressure L2 error** is the only metric that matters for new experiment design
- Do NOT degrade surface error or wall shear stress metrics
- Published SOTA models show significantly better volume pressure test metrics are achievable — large headroom to close
- All new student assignments must be designed with volume pressure improvement as the singular focus

## Wave SOTA (Test — Merged on Branch)

**PR #740** (dl24-fern, run `5x8wofzm`): `abupt_axis_mean_rel_l2_pct` = **7.5195%**
| Metric | Value |
|--------|-------|
| test_abupt | 7.5195% |
| surf_p | 3.8810% |
| vol_p | 10.7580% |
| wall_shear | 7.0610% |

**Central unsolved problem:** val vol_p ≈ 3.8–4.0%, test vol_p ≈ 10.7–12.2% — systematic +7–8pp val→test gap confirmed across ALL completed long runs. All active experiments are designed to close this gap.

**15+ model-side interventions FALSIFIED on this axis:** WD variants (0.005/0.01), GradNorm α-variants (0.25/0.75), fixed loss weights (2.0/3.0 with/without GradNorm), extended cosine T_max=60, EMA (decay=0.999/0.9999), BBox normalization, TTA Y-symmetry, DropPath, vol coord noise, stochastic vol subsampling, SDF-stratified sampling (far-field + near-surface), InstanceNorm across vol tokens, Lookahead. Gap is structurally embedded in the train/test distribution split.

## Active Experiments (2026-05-12 ~12:00 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #999 | dl24-nezuko | **SWA (Stochastic Weight Averaging)** — uniform epoch-snapshot averaging EP20–EP30; `--use-swa --swa-start-epoch 20 --swa-freq 1`; bs=1 DDP8 | `f8rc8ahi` | **Running** — step 212,392 (EP19.35). EP20 gate ≤6.70% imminent (~7k steps). SWA will activate at EP20. | val_abupt=**6.223%**, val_vol_p=**3.878%** | EP15 PASS ✓ (6.277%), EP20 gate approaching — current abupt 6.223% gives +0.477pp margin over gate. SWA not yet activated; watch for `train/swa/n_snapshots≥1`. |
| #1014 | dl24-fern | **Poisson pressure physics regularization** — auxiliary loss λ=0.01 × k-NN Laplacian smoothness penalty on predicted pressure; `--use-poisson-reg --poisson-lambda 0.01 --poisson-k 8 --poisson-m 2048`; 4L architecture | `l5urrdmk` (rank0) | **Running** — step 223,719 (EP20.38). EP20 gate ≤6.70% PASSED ✓ (+0.284pp). Next: EP25 gate ≤6.65% at step ~274,375. | val_abupt=**6.416%**, val_vol_p=**4.071%** | `train/poisson_loss` stable. Physics regularization active. **NOTE: 4L not 6L** — absolute metrics lag 6L SOTA but mechanism signal is what matters. EP20 PASS confirmed, next gate EP25. |
| #1025 | dl24-frieren | **Vol-token LayerNorm WITHOUT GradNorm** — second `nn.LayerNorm(hidden_dim=512)` on volume_hidden + fixed task weights (no GradNorm feedback); hypothesis reassigned from CLOSED #1023 | `ttnva184` (rank0) | **Running** — step 69,002 (EP6.29). EP5 gate ≤7.5% PASSED ✓ (6.567%). Next: EP10 gate ≤7.20% at step ~109,750. | val_abupt=**6.567%**, val_vol_p=**3.620%** | Excellent convergence — already at 6.567% in EP6, below EP10 gate threshold. vol_p 3.620% strong. Healthy trajectory. |
| #1033 | dl24-tanjiro | **Online focal-vol reweight with Adam bias correction (bug fix of #1026)** — per-case EMA ratio `(case_ema/global_ema)`, each EMA bias-corrected by `1 - decay^n`; clipped `[0.5, 3.0]` | `e316e9iz` (rank0) | **Running** — step 24,194 (EP2.20). 4-epoch screen ongoing (EP4 ~step 43,900). | val_abupt=**7.418%**, val_vol_p=**5.112%** | **CRITICAL ISSUE**: `train/focal_vol/scale=3` (ceiling) at EP2 — same symptom as PR #1026 bug that this PR was meant to fix. Diagnostic comment posted requesting `scale_mean`, `scale_std`, `per_case_ema_mean`, `bias_correction_mean`. Awaiting tanjiro response. EP2 abupt 7.418% is below 8.5% kill threshold so run continues. |

## Recently Closed (since 2026-05-09)

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #1026 | Online focal vol reweight (initial, dl24-tanjiro) | **CLOSED** — `train/focal_vol/scale=3` ceiling degeneration: global EMA converges in O(steps), per-case EMA converges in O(case_touches), ratio ≈10.2x → always clips to max. Bug: no bias correction for asymmetric convergence rates. Replaced by #1033 with bias correction fix. | Bias correction REQUIRED for multi-rate EMA ratios. |
| #972 | SDF-stratified far-field sampling (α=2.0) | **CLOSED** — test_vol_p=11.827% (+1.069pp WORSE than baseline 10.758%), val→test gap +8.012pp UNCHANGED | SDF sampling FULLY FALSIFIED. AXIS CLOSED. |
| #968 | Stochastic vol subsampling (fresh random draw every batch) | **CLOSED** — test_vol_p=12.114%, gap +8.115pp **WIDEST EVER** | Sampling strategy does NOT address structural val→test shift. AXIS CLOSED. |
| #1015 | InstanceNorm across volume tokens (tanjiro) | **CLOSED** — predecessor of #1023/#1025 vol-token LN line | Path led to vol-token LayerNorm direction. |
| #1023 | Vol-token LN (tanjiro) | **CLOSED — unresponsive student** | Hypothesis reassigned to frieren as #1025. |
| #998 | Lookahead Lion (tanjiro) | **CLOSED** — EP5 FAIL | Lookahead direction exhausted. |
| #1003 | PCGrad gradient surgery (fern) | **CLOSED** — bad config runs, hypothesis unconverted | PCGrad implementation needs re-think before retry. |

## Key Insights

1. **The val→test vol_p gap is structural and unsolved.** Persistent +7–8pp gap across 15+ falsified interventions. Almost certainly covariate shift between training and test aerodynamic configurations.

2. **Current 4 active gap-closing candidates** all attack the gap via different mechanisms:
   - **#999 SWA**: flat-minima weight averaging hypothesis — averaging EP20–EP30 snapshots may converge to wider basin with better OOD generalization
   - **#1014 Poisson**: direct physics constraint — Laplacian smoothness penalty enforces physically consistent pressure fields
   - **#1025 vol-token LN no GradNorm**: representation-level regularization without adaptive loss balancing feedback
   - **#1026 online focal vol reweight**: per-case dynamic loss reweighting targets chronically high-error volumes

3. **Weight decay is load-bearing.** WD axis fully exhausted — neither WD=0.005 nor WD=0.01 closes gap.

4. **GradNorm α=0.5 is optimal.** α=0.25 causes test regression; α=0.75 causes catastrophic instability at EP16.

5. **String multisigma PE (5-octave) is confirmed best.** σ=[0.25, 0.5, 1.0, 2.0, 4.0].

6. **Sampling AXIS CLOSED.** Both far-field SDF upweighting (#972) and stochastic subsampling (#968) failed. Gap +8.115pp on #968 (WIDEST EVER).

## Gate Schedule

| Gate | Standard Threshold | Steps (std, bs=2, DDP8) | Steps (bs=1, DDP8) |
|------|--------------------|---------------------|----------------------|
| EP5  | ≤7.5% | ~27,469 | ~54,930 |
| EP10 | ≤7.2% | ~54,938 | ~109,860 |
| EP15 | ≤6.80% | ~82,407 | ~164,790 |
| EP20 | ≤6.70% | ~109,876 | ~219,720 |
| EP25 | ≤6.65% | ~137,345 | — |
| EP30 | ≤6.60% | ~164,814 | — |

## Critical Config Constraints

1. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
2. **`--train-volume-points 65000` (or higher)** REQUIRED: default 16384 inverts volume:surface gradient ratio.
3. **`--lr-warmup-epochs 1` NOT `--lr-warmup-steps 500`**: epoch-based warmup is correct at 6L 65k.
4. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion optimizer.
5. **`--volume-loss-weight` BUG**: When `--use-gradnorm` is active, `--volume-loss-weight` is a no-op.
6. **`--model-pe string_multisigma` REQUIRED when using STRING PE**: omitting causes `--pe-init-sigmas` to be silently ignored.
7. **`--pe-init-sigmas` must be COMMA-separated**: `0.25,0.5,1.0,2.0,4.0` NOT space-separated.
8. **lion-pytorch pod environment drift**: `ModuleNotFoundError: No module named 'lion_pytorch'` can occur. Fix: `uv pip install --system lion-pytorch` (resolves to 0.2.4).
9. **Kill threshold operator is `<`**: NOT `>`. PR #945 had operator bug causing inverted logic.
10. **EMA decay=0.999 (NOT 0.9999)**: 0.9999 gives ~10,000-step lookback (too slow); 0.999 gives ~1,000-step lookback.
11. **Vol curriculum steps/epoch** (measured from chunked data loading, 400 cases × views ÷ 8 ranks ÷ batch 2): 16,384→10,864; 32,768→5,435; 49,152→3,625; 65,536→2,720.
12. **Steps/epoch at bs=1 DDP8**: ~10,975–10,986.
13. **Steps/epoch at bs=2 DDP8 (standard)**: ~10,975.

## Confirmed Dead Ends (Do Not Retry)

- No weight decay: overfits at EP9 (PR #898)
- 7L depth without full regularization: catastrophic bounce (PR #873)
- 8L depth: EP11 bounce +0.222pp (PR #965)
- Dropout=0.1: consistent degradation (PR #899)
- vol-loss-weight=2.0 WITH GradNorm: self-cancelling (PR #911)
- vol-loss-weight=2.0 WITHOUT GradNorm: actively harmful, EP5=9.010% (PR #936)
- vol-loss-weight=3.0 WITHOUT GradNorm: WORST EVER test_abupt=8.0190%, gap +8.12pp (PR #964) — AXIS FULLY CLOSED
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- 96k+60k balanced sampling on 5L at default lr=1e-4: catastrophic divergence EP5=15.22% (PR #938)
- Proportional sampling 96k+60k at 6L: slower convergence, vol_p oscillation (PR #951)
- WD=0.01: does not close val→test vol_p gap (PR #900)
- WD=0.005: val matches WD=0.01; test gap persists (PR #914)
- QK-Norm: consistent failures (multiple PRs)
- Y-sym p=1.0: over-augmentation (PR #866)
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (multiple PRs): terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16
- Extended cosine T_max=60 (PR #946): destabilizing in training tail
- EMA decay=0.999 (PR #954): does not close val→test vol_p gap (test_vol_p=11.28%)
- TTA Y-symmetry (PR #979): gap UNCHANGED. AXIS CLOSED.
- SDF-stratified importance sampling (PR #972): test_vol_p=11.827% WORSE than baseline. AXIS FULLY CLOSED.
- DropPath regularization (PR #987): EP5 gate FAIL; gap +7.91pp UNCHANGED. FALSIFIED.
- Lookahead Lion (PR #998): EP5 FAIL. FALSIFIED.
- Vol coordinate noise (PR #990): EP5=8.54% gate FAIL. FALSIFIED.
- Bbox normalization (PR #978): regression. CLOSED.
- Stochastic vol subsampling (PR #968): test_vol_p=12.114%, gap +8.115pp WIDEST EVER. AXIS FULLY CLOSED.
- InstanceNorm across vol tokens (PR #1015): closed before terminal — direction continued as #1025 vol-token LN.

## Potential Next Directions (Not Yet Assigned)

If the 4 active gap-closing experiments fail to close the gap, escalate per Plateau Protocol — bold architecture-level moves:

1. **Independent vol_p transformer tower** — fully separate encoder for volume head; train surface and volume jointly but with no shared backbone parameters. Tests whether the val→test gap is shared-feature interference.
2. **DETR-style learned query positions for volume decoder** — replace coordinate-based vol queries with N learned query embeddings, allowing model to learn its own canonical volume sampling pattern.
3. **Voxel-based aggregation with spatial attention** — discretize volume into voxel grid, apply 3D attention; treats vol prediction as a structured grid problem rather than scattered point regression.
4. **Domain adversarial training** — train a discriminator on backbone features to predict train-vs-test domain; backprop adversarial loss to make features distribution-invariant.
5. **Top-K val checkpoint averaging** — instead of best single val checkpoint, average the K best by val; orthogonal to SWA epoch-snapshot averaging.
6. **Geometric conditioning on body shape for vol prediction** — explicit shape descriptor (e.g., spherical harmonics of car silhouette) prepended to volume queries.
7. **Pure Fourier neural operator (FNO) for vol_p** — completely replace volume decoder with FNO on a regular grid, interpolated to query points; tests whether spectral aggregation generalizes better than per-point MLP.

_Last updated: 2026-05-12 ~12:00 UTC. Key changes: (1) #999 nezuko SWA at EP19.35 (step=212,392), abupt=6.223%, EP20 gate (≤6.70%) imminent with +0.477pp margin; SWA activates at EP20. (2) #1014 fern Poisson at EP20.38 (step=223,719), EP20 gate PASSED ✓ (abupt=6.416%, +0.284pp margin); next EP25 gate ≤6.65%. (3) #1025 frieren vol-LN at EP6.29 (step=69,002), abupt=6.567% — excellent early trajectory, EP10 gate ≤7.20% comfortably cleared already. (4) #1026 CLOSED (focal-vol scale degeneration bug — global vs per-case EMA asymmetric convergence); replaced by #1033 with Adam bias correction fix. #1033 tanjiro at EP2.20 (step=24,194), scale=3 still ceiling-stuck — diagnostic comment posted, awaiting tanjiro response. (5) EP20 gate PASS comment posted on #1014._
