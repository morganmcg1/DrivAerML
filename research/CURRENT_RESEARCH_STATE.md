# SENPAI Research State
- 2026-05-09 ~23:00 UTC

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

**Val wave leader (awaiting new challenger):** frieren #972 CLOSED — test EP20 checkpoint: vol_p=**11.827%** (+1.069pp WORSE than baseline 10.758%), abupt=7.480%, surf_p=3.574%, wall=6.726%. Val→test gap = +8.012pp (structural gap UNCHANGED). SDF hypothesis FULLY FALSIFIED. Three experiments active (nezuko #999, fern #1014, tanjiro #1015).

**Central unsolved problem:** val vol_p ≈ 3.8–4.0%, test vol_p ≈ 10.7–11% — systematic +7pp val→test gap confirmed across ALL completed long runs. All active experiments are designed to close this gap.

**Vol-loss-weighting direction CLOSED:** PR #911 + PR #936 + PR #964 (gap +8.12pp WORST EVER). Static vol upweighting conclusively does NOT close the val→test gap. FULLY CLOSED.

**EMA AXIS CLOSED:** PR #954 — test_vol_p=11.28%, gap unchanged. EMA weight averaging does not reduce val→test gap.

**SDF/Sampling AXIS FULLY CLOSED:** PR #972 TERMINAL test eval (EP20 checkpoint): test_vol_p=11.827%, gap +8.012pp. PR #968 stochastic vol subsampling: gap +8.115pp WIDEST EVER. Sampling strategy (far-field SDF, near-surface SDF, stochastic subsampling) does NOT close gap. AXIS FULLY CLOSED.

## Active Experiments (2026-05-09 ~23:00 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Known Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #999 | dl24-nezuko | **SWA (Stochastic Weight Averaging)** — uniform epoch-snapshot averaging EP20–EP30; `--use-swa --swa-start-epoch 20 --swa-freq 1`; bs=1 DDP8 | `f8rc8ahi` | **Running** — step ~94,288 (EP8.6). EP5 gate CLEARED (abupt 6.703% ≤7.5% ✓). EP10 gate next at step ~109,750 (threshold ≤7.2%). | val_abupt=**6.478%** (EP8), val_vol_p=4.063%; slope improving | Healthy convergence. GradNorm w_vol_p=0.222 (not collapsed). SWA collection begins EP20 (~step 219,500). EP10 gate ~1,500 steps away. |
| #1014 | dl24-fern | **Poisson pressure physics regularization** — auxiliary loss λ=0.01 × k-NN Laplacian smoothness penalty on predicted pressure; `--use-poisson-reg --poisson-lambda 0.01 --poisson-k 8 --poisson-m 2048`; group `poisson-pressure-reg` | `l5urrdmk` (rank0) | **Running** — step ~19,499 (EP1.78). EP1 val: abupt=8.581%, vol_p=6.266%. EP5 gate at step ~54,875 (threshold ≤7.5%). | val_abupt=**8.581%** (EP1) | Physics regularization directly constrains vol_p spatial consistency. EP1 convergence within normal early-epoch range. |
| #1015 | dl24-tanjiro | **InstanceNorm across volume tokens** — `nn.InstanceNorm1d(hidden_dim, affine=True)` applied across ~65k vol tokens per channel | `48pi1dn4` (rank0) | **Running** — step ~5,753 (EP0.5). All 8 DDP ranks confirmed live. EP1 val at step ~10,975. | No val yet | All 8 rank IDs: `48pi1dn4, 0nj8yj7b, dnobec5q, 11qquima, yllm3wjd, a9gptjc4, nlty089a, 7hnsh7o3`. Cross-token normalization of vol activations targets val→test distribution shift. |

## Closed This Wave (Recent)

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #972 | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym + **SDF-stratified importance sampling** (far-field bias, α=2.0) | **CLOSED** — test_vol_p=11.827% (+1.069pp WORSE than baseline 10.758%), abupt=7.480%, surf_p=3.574%, wall=6.726%; val→test gap +8.012pp UNCHANGED; run `56bcqp3m` | SDF sampling FULLY FALSIFIED. Far-field upweighting marginally improved surf_p/wall but WORSENED vol_p. Val→test gap completely structural. AXIS CLOSED. |
| #968 | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym + **stochastic vol subsampling** (fresh random draw every batch) | **CLOSED** — test_abupt=7.6157%, test_vol_p=12.1140%, surf_p=3.9440%, wall=7.0470%; gap +8.115pp **WIDEST EVER**; val_abupt best EP15=6.2806%, val_vol_p=3.999% | Stochastic vol subsampling compressed val_vol_p but test EXPLODES. Gap +8.115pp worst ever. Sampling strategy (far-field OR stochastic) does NOT close the structural val→test shift. |
| #995 | Pre-xattn volume LayerNorm — single `nn.LayerNorm(hidden_dim=512)` on volume_hidden before surf_to_vol_xattn | **CLOSED** — superseded/not in active WIP | Frieren reassigned to SDF long run #972. |
| #996 | Near-surface SDF-stratified sampling — `weight = exp(-alpha × |sdf|)` concentrates gradient on near-surface vol points; Arm A: alpha=1.0, Arm B: alpha=2.0 | **CLOSED** — not in active drivaerml-long WIP | Near-surface SDF axis subsumed into broader SDF sampling FALSIFIED conclusion. |
| #994 | LR warmup decoupled from vol curriculum (`--lr-warmup-steps 8000`) | **CLOSED** — superseded by Lookahead (#998) | LR warmup decoupling low-priority lever; Lookahead higher expected impact. |
| #990 | Vol coordinate noise augmentation (σ=0.005 on vol query coords) | **CLOSED** — EP5 abupt=8.54% gate FAIL (>7.5%); hypothesis FALSIFIED | Coordinate noise does not improve OOD vol_p generalization. |
| #987 | DropPath + Lookahead combo (k=5, α=0.5 wrapping Lion) | **CLOSED** — EP5=7.8846% gate FAIL; test_abupt=9.1036%, test_vol_p=14.278%, gap +7.91pp UNCHANGED | Adversarial GradNorm interaction. Lookahead alone re-tested in #998. |
| #979 | TTA Y-symmetry ensemble | **CLOSED** — gap +7.860pp → +7.863pp UNCHANGED (noise-level) | TTA Y-sym is free-lunch only, does not close val→test gap. |
| #978 | Bbox normalization of input coords | **CLOSED** — regression, no gap improvement | Coord normalization does not address val→test shift. |
| #964 | 6L + vol_loss_weight=3.0 + NO GradNorm | **CLOSED** — test_abupt=8.0190%, test_vol_p=12.52%, gap +8.12pp WORST EVER | Static vol upweighting (any value) does NOT close val→test gap. Axis FULLY CLOSED. |
| #954 | EMA decay=0.999 + eval-raw-vs-ema 6L | **CLOSED** — test_abupt=7.5476%, test_vol_p=11.2803% | EMA does not close val→test gap. EMA axis FULLY REJECTED. |
| #951 | 6L + proportional sampling (96k vol + 60k surf) | **CLOSED** — slower convergence, EP15 gate AT RISK | Proportional 1.6:1 vol:surf ratio shows vol_p oscillation. |
| #946 | Extended cosine T_max=60 | **CLOSED** — EP7 regression +0.21pp | T_max=60 keeps LR too high in tail. Default T_max=50 confirmed optimal. |

## Key Insights

1. **The val→test vol_p gap is structural and unsolved.** Val vol_p ≈ 3.8–4.0%, test vol_p ≈ 10.7–11%. Gap persists across ALL WD values, EMA, loss-weighting, architecture-depth changes, sampling strategies (far-field SDF, near-surface SDF, stochastic), coordinate noise, TTA, and optimizer changes tried so far. Almost certainly covariate shift between training and test aerodynamic configurations.

2. **Current active gap-closing candidates:**
   - #999 SWA epoch-snapshot averaging: EP8 abupt=6.478%, EP10 gate at step ~109,750 ≤7.2%. SWA averaging begins EP20.
   - #1014 Poisson physics regularization: EP1 abupt=8.581%, EP5 gate at step ~54,875 ≤7.5%. Direct physics constraint on vol_p.
   - #1015 InstanceNorm across vol tokens: EP0.5 launched, EP1 val pending at step ~10,975. Cross-token normalization hypothesis.

3. **Weight decay is load-bearing.** WD axis fully exhausted — neither WD=0.005 nor WD=0.01 closes gap, but required to prevent overfitting.

4. **GradNorm α=0.5 is optimal.** α=0.25 causes test regression; α=0.75 causes catastrophic instability at EP16.

5. **String multisigma PE (5-octave) is confirmed best.** σ=[0.25, 0.5, 1.0, 2.0, 4.0].

6. **Sampling AXIS CLOSED.** Both far-field SDF upweighting (#972 EP7) and stochastic vol subsampling (#968) failed. Gap +8.115pp on #968 (WIDEST EVER). Sampling strategy does not address the structural distribution shift.

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
12. **Steps/epoch at bs=1 DDP8**: ~10,975–10,986 (nezuko #999 SWA).
13. **Steps/epoch at bs=2 DDP8 (standard)**: ~10,975 (all other PRs).

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
- TTA Y-symmetry (PR #979): gap +7.860pp → +7.863pp UNCHANGED. AXIS CLOSED. Add `--use-tta` as free-lunch only.
- SDF-stratified importance sampling far-field upweighting (PR #972, TERMINAL EP20 test eval): test_vol_p=11.827% (+1.069pp WORSE than baseline), val→test gap = +8.012pp. HYPOTHESIS FALSIFIED. AXIS FULLY CLOSED.
- DropPath regularization (PR #987): EP5=7.8846% gate FAIL; test_vol_p=14.278%, gap +7.91pp UNCHANGED. FALSIFIED.
- Lookahead+DropPath combo (PR #987): EP5=7.885% gate FAIL; adversarial GradNorm interaction. Re-testing Lookahead alone in #998.
- Vol coordinate noise augmentation (PR #990, σ=0.005): EP5 abupt=8.54% gate FAIL. FALSIFIED.
- Bbox normalization (PR #978): regression, no gap improvement. CLOSED.
- **Stochastic vol subsampling (PR #968)**: val_vol_p compressed to 3.999% but test_vol_p=12.1140%, gap +8.115pp **WIDEST EVER**. Sampling strategy does NOT address structural val→test distribution shift. AXIS FULLY CLOSED.

## Potential Next Directions (Not Yet Assigned)

**Targeting val→test vol_p gap (primary unsolved problem):**

1. **Physics-informed regularization** — Poisson residual on pressure field as auxiliary loss; direct physics constraint for vol_p generalization. Dirichlet/Neumann boundary conditions encoded as soft constraints.
2. **Data distribution analysis** — Profile train vs test aerodynamic configurations. What makes test OOD? Build augmentations that explicitly mimic test distribution shift.
3. **Domain adaptation** — If we can identify what makes the test split OOD, train a domain discriminator and use adversarial training to make backbone features distribution-agnostic.
4. **Checkpoint averaging (top-3 val)** — Average top-3 val checkpoints instead of best single; known to reduce overfit to val noise.
5. **Feature disentanglement** — Explicit bottleneck between surface and volume prediction paths; train surface and volume heads with independent gradient flows (separate encoder).
6. **Test-time adaptation (TTA) augmentation ensemble** — Beyond Y-symmetry: rotate/reflect configurations at test time and average predictions. Geometric ensemble for OOD robustness.
7. **If all active gap-closing hypotheses fail** — Escalate to plateau protocol: (a) separate surface/volume encoder architecture, (b) physics-based Poisson regularization, (c) domain adversarial training, (d) explicit geometric conditioning on body shape for volume prediction.

_Last updated: 2026-05-09 ~23:00 UTC. Key changes: (1) PR #972 CLOSED — SDF-stratified sampling TERMINAL test eval: test_vol_p=11.827% (+1.069pp WORSE than baseline), val→test gap +8.012pp UNCHANGED. SDF axis FULLY CLOSED. (2) Frieren removed from active experiments table. (3) PR #999 (nezuko SWA) updated to EP8 abupt=6.478%, step ~94,288; EP10 gate imminent. (4) PR #1014 (fern Poisson) updated to EP1 abupt=8.581%, step ~19,499; EP5 gate at step ~54,875. (5) PR #1015 (tanjiro InstanceNorm) confirmed launched — all 8 DDP rank IDs live, step ~5,753, EP1 val pending._
