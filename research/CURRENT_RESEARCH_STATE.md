# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-02 17:00 UTC (Round 21 launched; 3 new experiments assigned after Round 20 cleanup)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #358 (thorfinn STRING-sep + QK-norm), val_abupt 7.4648%

**NEW SOTA: thorfinn PR #358 merged. QK-norm (RMSNorm on Q and K projections) stacked on STRING-sep beats PR #311 by −0.081pp (7.4648% vs 7.546%). W&B run `tkiigfmc`, EP10.**

All future PRs must beat val_abupt < **7.4648%**.

| Metric | PR #358 SOTA (val) | AB-UPT | Val Gap |
|---|---:|---:|---:|
| `abupt` mean | **7.4648%** | — | — |
| `surface_pressure` | 4.9919% | 3.82 | ×1.31 |
| `volume_pressure` | 4.5871% | 6.08 | (val<AB-UPT, test gap ×2.05) |
| `wall_shear` | 8.4538% | 7.29 | ×1.16 |

Convergence was smooth and still descending at EP10 — model underfit, more epochs will help.

Merge bar = val_abupt < **7.4648%** (PR #358 `tkiigfmc`).

---

## Currently in-flight

### Tay branch — Round 21 (5 active WIP PRs)

| PR | Student | Hypothesis | W&B group | State |
|---|---|---|---|---|
| #387 | alphonse | STRING-sep num_features sweep (16/32/64) on QK-norm base | `alphonse-string-sep-feat-qknorm` | RECOVERING — prior arms failed ~22s after launch; rebasing to QK-norm base |
| #393 | edward | STRING-sep extended epochs (string_sep only, no QK-norm) | `edward-string-sep-extended` | EP6 val=8.47%, healthy descent; borderline new bar 7.4648% |
| #351 | fern | Soft tangential penalty on tau, λ=1.0 run `o78w9geu` | `fern-tangent-frame-soft-r18` | EP2 val=49%, destabilized; rebase to QK-norm base on completion |
| #359 | frieren | STRING-sep + separate volume decoder head | `frieren-string-sep-vol-decoder` | EP7 val=8.20%, vol_p=5.08% healthy; borderline new bar |
| #365 | nezuko | layers=5 on QK-norm base (rebase required) | `tay-nezuko-layers5-string-qknorm` | Waiting on student rebase + rerun |
| #422 | tanjiro | Multi-scale STRING-sep (8/32/128 feats) on QK-norm | `tay-tanjiro-multiscale-rff` | NEW — just assigned |
| #423 | thorfinn | Local tangent-frame input features on QK-norm | `tay-thorfinn-tangent-frame` | NEW — just assigned |
| #424 | askeladd | Per-axis L2 loss weights for tau_y/tau_z (correct weighting, no Huber) | `tay-askeladd-per-axis-weight` | NEW — just assigned |

### Round 20 CLOSED

PRs #394 (chihiro), #395 (emma), #396 (gilbert), #400 (haku), #401 (kohaku), #403 (norman), #404 (senku), #405 (violet) — **CLOSED as misassignments**. Those students have no `senpai-drivaerml-ddp8` GPU pods. The underlying hypothesis ideas (tangent frame, multi-sigma RFF, curvature features, log1p norm, attention bias, EdgeConv, radius pooling) are valuable and have been re-assigned to DrivAerML-capable students in Round 21.

PR #353 (askeladd Huber loss) — **CLOSED NEGATIVE**. Channel reweighting bug + Huber clips hard-channel gradients. Replaced by #424 (per-axis L2 weights).

### Notes on in-flight experiments

**PR #387 alphonse:** All 8 ranks failed within 22s (state=failed, no steps logged). Likely import error with STRING-sep + QK-norm stack on rebased branch. Advisor pinged with rebase instructions + updated merge bar 7.4648%.

**PR #393 edward:** Running on STRING-sep only (no QK-norm). At EP6 val=8.47%, slope -0.48pp/epoch. Will likely finish 7.5-7.7% — borderline new bar 7.4648%. Valuable reference for STRING-sep headroom without QK-norm.

**PR #351 fern:** λ=1.0 soft tangential penalty: severe destabilization at EP1-2 (72% vs SOTA 52% at EP1). Grad-clip accidentally 1.0 instead of 0.5. Physics correctness proven (tau_nf dropping to 0.06 vs 0.09 plateau). Need to rebase onto QK-norm base + fix grad-clip + use λ=0.3 for next run.

**PR #359 frieren:** Separate volume decoder head on STRING-sep (no QK-norm). EP7 vol_p=5.08% — if this holds to test, significant improvement vs PR #311 test vol_p=12.44%. Need EP10-11 to see if val_abupt crosses 7.4648%.

**PR #365 nezuko:** layers=5 on STRING-sep (no QK-norm) got val=7.5250% which beat old bar 7.546% but not new bar 7.4648%. Rebasing to QK-norm base + retesting with layers=5 + QK-norm.

**PR #422 tanjiro:** Multi-scale STRING-sep concatenating 3 heads (8/32/128 features = 48+192+768 = 1008-dim encoding) on QK-norm stack. Targets tau_y/tau_z via multi-scale spectral content. New assignment.

**PR #423 thorfinn:** Local tangent frame (t1, t2 orthogonal to normal) added as 6 extra input features. 7→13 input dim, no new parameters. Targets tau_y/tau_z via explicit geometric context. New assignment.

**PR #424 askeladd:** Per-axis L2 loss weights: sp=1.0, tau_x=1.0, tau_y=2.0, tau_z=2.5 (run A), then tau_y=3.0, tau_z=4.0 (run B). Single masked_mean call to avoid PR #353 reweighting bug. Correct approach to tau_y/tau_z outlier pushing without Huber's gradient-clipping side effect.

---

---

## GRAPE-M Arm C — RESOLVED NEGATIVE (historical)

Edward's GRAPE-M Arm C (group `tay-round18-grape-ablation`, run `hrgo2pk1`) reached val_abupt=9.892% at ep7.9 — substantially worse than STRING-sep SOTA 7.546%. **Conclusion:** GRAPE-M (multiplicative, learned non-axis-aligned rank-2 generators) does NOT beat STRING-sep (separable per-axis learnable freq/phase) on this 3D point-cloud CFD setting. Issue #285 closed-positive on STRING-sep, closed-negative on GRAPE-M.

---

## Active Human Research Directives

**Issue #252 (Modded-NanoGPT):**
- Muon: CLOSED NEGATIVE (askeladd #299, +4.09pp)
- Post-attention RMSNorm: CLOSED NEGATIVE (tanjiro #300 diverged)
- QK-norm: CLOSED NEGATIVE on old base (alphonse #287, +0.10pp); CLOSED on STRING-sep base (thorfinn #358, Round 19)
- U-net skips: CLOSED NEGATIVE (fern #320, +0.555pp)
- Surface-tangent loss: CLOSED (fern #351, Round 19)

**Issue #285 (GRAPE/Representational Position Encoding):** RESOLVED — STRING-sep (PR #311) merged as new SOTA. GRAPE-M (Arm C) confirmed negative at ep7.9 (val=9.892%).

---

## Key Learnings (cumulative, final state)

| Lever | Status | Best result |
|---|---|---|
| LR | CLOSED | 1e-4 SOTA |
| EMA decay | CLOSED | 0.999 SOTA |
| Lion beta2 / beta1 | CLOSED | 0.99 / 0.9 SOTA |
| Weight decay | CLOSED | 5e-4 SOTA |
| Vol/surf loss weights | CLOSED | 1.0/1.0 SOTA |
| Vol/surf points | CLOSED | 65536/65536 SOTA |
| mlp_ratio | CLOSED NEGATIVE | 4 SOTA |
| Dropout | CLOSED | 0.0 SOTA |
| Tau axis weights | CLOSED | 1.0 SOTA |
| model_heads | CLOSED SOTA | 4H beats 8H (#232 merged) |
| model_layers | CLOSED | 4 SOTA; 5L on OLD base=8.9938%; 5L on STRING-sep base (nezuko #365, Round 19) |
| model_slices / hidden_dim | CLOSED NEGATIVE | 128/512 SOTA |
| lr_cosine T_max / warmup | CLOSED | T_max=50, warmup=0 SOTA |
| grad-clip-norm | MERGED | 0.5 SOTA (#309) |
| **STRING-separable PE** | **MERGED NEW SOTA** | **val 7.546% / test 8.771% (#311)** |
| STRING-sep num_features | In-flight | alphonse #387 (16/32/64 sweep on QK-norm base) |
| RFF (isotropic Gaussian) | CLOSED NEGATIVE | WORSE than no-encoding |
| **QK-norm** | **MERGED** | **thorfinn #358 −0.081pp → val 7.4648% NEW SOTA** |
| U-net skips | CLOSED NEGATIVE | +0.555pp vs SOTA |
| MLP activation (SwiGLU/ReLU²) | CLOSED NEGATIVE | Near-zero improvement; ReLU² OOM |
| Per-channel output-head scaling | CLOSED MISMATCH | chihiro #394 closed (no DrivAerML pod) |
| Sandwich-norm (RMSNorm) | CLOSED NEGATIVE | diverged |
| Muon (canonical) | CLOSED NEGATIVE | +4.09pp |
| Channel-selective Huber on tau | CLOSED NEGATIVE | askeladd #353 — bug + wrong loss type |
| Per-axis L2 loss weights on tau | In-flight | askeladd #424 (Round 21) |
| MLP volume decoder | In-flight | frieren #359 (STRING-sep base, EP7 vol_p=5.08%) |
| Extended training | In-flight | edward #393 (STRING-sep only, no QK-norm) |
| log1p tau normalization | CLOSED | haku #400 — no DrivAerML pod (re-assignable) |
| Normal-dot geometric attention bias | CLOSED | kohaku #401 — no DrivAerML pod (re-assignable) |
| k-NN EdgeConv pre-encoder | CLOSED | norman #403 — no DrivAerML pod (re-assignable) |
| Multi-scale radius pooling | CLOSED | senku #404 — no DrivAerML pod (re-assignable) |
| Surface curvature features (κ_H, κ_G) | CLOSED | violet #405 — no DrivAerML pod (re-assignable) |
| Multi-scale STRING-sep (8/32/128 feats) | In-flight | tanjiro #422 (Round 21) |
| Local tangent-frame input features | In-flight | thorfinn #423 (Round 21) |
| Soft tangential penalty (λ=1.0) | In-flight | fern #351 — λ=0.1 proved physics; λ=1.0 destabilized |
| layers=5 on QK-norm base | In-flight | nezuko #365 — rebase pending |

---

## Largest Remaining Gaps to AB-UPT (from PR #311 test metrics)

1. **volume_pressure** ×2.05 (12.438% vs 6.08%) — senku #404 (multi-scale radius pooling at 5cm/15cm/45cm)
2. **tau_y** ×2.53, **tau_z** ×2.88 — chihiro #394 (per-axis output scaling), gilbert #396 (tangent-frame features), haku #400 (log1p tau norm), violet #405 (surface curvature features)
3. **wall_shear** ×1.13 (8.227% vs 7.29%) — norman #403 (EdgeConv local geometry)
4. **surface_pressure** ×1.17 (4.485% vs 3.82%) — kohaku #401 (normal-dot attention bias), norman #403

---

## Next Priorities

1. **Monitor frieren #359** — vol_p=5.08% at EP7 is very promising; if EP10-12 val_abupt < 7.4648%, merge immediately. This would also update the volume_pressure test gap.
2. **Monitor alphonse #387** — rerunning with QK-norm base after import errors; if 64-feature arm wins, assign 128-feature arm
3. **Monitor edward #393** — STRING-sep only (no QK-norm); trajectory at EP6=8.47%. Useful reference even if below new bar
4. **Round 21 launches** (#422 tanjiro, #423 thorfinn, #424 askeladd) — watch EP1-2 for healthy convergence
5. **Nezuko #365 rebase** — layers=5 + QK-norm compound; if it holds, merge new SOTA
6. **Fern #351** — λ=1.0 destabilized; when complete, rebase to QK-norm base with λ=0.3 + grad-clip 0.5 for clean single-delta test
7. **Re-assignable ideas** (need DrivAerML student slots): log1p tau norm, k-NN EdgeConv, multi-scale radius pooling, curvature features, normal-dot attention bias — these are all good hypotheses, assigned to wrong students only
8. **Future compound round**: once Round 21 closes, stack best winner(s) with extended training on QK-norm base
