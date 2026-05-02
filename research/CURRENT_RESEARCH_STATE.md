# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-02 13:10 UTC (Round 19 in flight; alphonse reassigned to STRING-sep feature-count sweep after QK-norm null result)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #311 (edward STRING-separable PE), val_abupt 7.546%

**This is the new SOTA — merged 2026-05-01 ~10:35 UTC. All future PRs must beat 7.546% val.**

STRING-sep replaces fixed isotropic Gaussian RFF with learnable per-axis frequency/phase (`log_freq` + `phase` as `nn.Parameter`). The axis-separable factorization learns independent spectral emphasis per spatial axis, matching the anisotropic structure of automotive aerodynamics.

| Metric | PR #311 SOTA (val/test) | AB-UPT | Test Gap |
|---|---:|---:|---:|
| `abupt` mean | **7.546% / 8.771%** | — | — |
| `surface_pressure` | 4.867% / **4.485%** | 3.82 | ×1.17 |
| `wall_shear` | 8.527% / **8.227%** | 7.29 | ×1.13 |
| `volume_pressure` | 4.525% / **12.438%** | 6.08 | **×2.05 ← biggest gap** |
| `tau_x` | — / **7.253%** | 5.35 | ×1.36 |
| `tau_y` | — / **9.233%** | 3.65 | **×2.53** |
| `tau_z` | — / **10.449%** | 3.63 | **×2.88** |

**Key finding from 3-arm ablation:** RFF-32 (Arm A) val=9.710% is *worse* than no-encoding SOTA (9.039%). Fixed isotropic Gaussian is the wrong spectral prior for anisotropic automotive aerodynamics. The STRING-sep win comes from learning the per-axis spectral structure.

**Convergence diagnostics:** All val slopes still negative at PR #311 termination — model underfit, more epochs will help.

Merge bar = val_abupt < 7.546%.

---

## Currently in-flight

### Tay branch (8 active WIP PRs)

| PR | Student | Hypothesis | Status (13:10Z) | Latest val_abupt |
|---|---|---|---|---:|
| #387 | alphonse | STRING-sep num_features sweep (16/32/64) — optimal spectral resolution | Just assigned — warmup | — |
| #323 | tanjiro | 2-layer MLP vol decoder + STRING-sep (combined re-launch) | `8x7c537j` ep1.1 — warmup | 37.95% |
| #351 | fern | Soft tangent-frame loss on tau (asymmetric, λ=0.1) | `la5hrm16` ep4.4 — tracking SOTA closely | 13.75% |
| #353 | askeladd | Channel-selective Huber loss on tau (δ=0.5) | `nhr4uj3q` ep8.2 — descending but behind SOTA | 10.91% |
| #357 | edward | STRING-sep extended epochs | **OFF-TASK** — running unauthorized GRAPE-M arm-C; redirect order sent 12:38Z | — |
| #358 | thorfinn | STRING-sep + QK-norm stack | `tkiigfmc` ep1.1 — warmup | 52.05% |
| #359 | frieren | STRING-sep + separate volume decoder head | **OFF-TASK** — running unauthorized SwiGLU arm-D; redirect order sent 12:38Z | — |
| #365 | nezuko | model-layers=5 + STRING-sep PE | `70lnb3dt` group `tay-nezuko-layers5-string-sep` ep~2.5 | descending |

### Notes on each (Round 19, 13:10Z snapshot)

**PR #387 alphonse STRING-sep feature count sweep:** 3-arm W&B sweep under group `alphonse-string-sep-features`. Arm A=16 features (96-dim encoding), Arm B=32 (192-dim, SOTA control), Arm C=64 (384-dim, double expressivity). Win condition: any arm beats val_abupt < 7.546%. Watch ep5-8 for Arm C vs Arm B divergence — if 64 wins by ≥0.3pp it points toward even larger feature counts (128) as next step.

**PR #323 tanjiro vol-decoder + STRING-sep (combined):** Just relaunched 12:07Z after rebase onto STRING-sep. Single-delta on top of SOTA = `--model-vol-decoder-depth 2`. Watch ep5-6 vol_pressure — if it drops below 4.0% val (vs SOTA 4.525%) it's the merge signal.

**PR #351 fern soft-tangent-frame:** Pivot from broken hard projection (which zeroed tau_z gradients and caused tau_z divergence to 113%) to soft normal-component penalty `λ·mean((pred·n̂)²)` with λ=0.1, primary MSE intact. ep1-2: tau_z 69→33% (descending properly), abupt 22.94% vs SOTA 23.24% (slightly ahead). Strong early signal — let it run to completion (~ep11-12).

**PR #353 askeladd Huber-tau:** ep8.2 best 10.91%, descending but ~3.4pp behind SOTA. May close as null at ep10-12 unless tau_z specifically improves vs STRING-sep baseline (10.449% test).

**PR #357 edward STRING-sep extended-epochs (BLOCKED state, just redirected):** Edward never relaunched the assigned run after STRING-sep code was pushed. Has been running unauthorized GRAPE-M arm-C since 09:16Z, reached 9.892% val_abupt at ep7 — confirming GRAPE-M does NOT beat plain STRING-sep. Redirect order sent 12:38Z; close PR if no launch by 12:55Z.

**PR #358 thorfinn STRING-sep + QK-norm:** Just launched 12:07Z post-unblock (`tkiigfmc`). Watch ep5-7 — should track SOTA closely if QK-norm is benign on Lion+STRING-sep, ahead if it adds value.

**PR #359 frieren STRING-sep + separate vol-decoder (BLOCKED state, just redirected):** Frieren did not relaunch after unblock. Running unauthorized MLP-activation arm-D SwiGLU-uniform at val 9.190% — also not productive. Redirect order sent 12:38Z.

**PR #365 nezuko model-layers=5 + STRING-sep:** Launched 11:20Z, ep~2.5. nezuko did the bugfix recovery work pre-launch (caught flag-name mismatches, missing rff_num_features, lr-warmup-epochs, grad-clip-norm). On-task and progressing.

---

## GRAPE-M Arm C — RESOLVED NEGATIVE

Edward's GRAPE-M Arm C (group `tay-round18-grape-ablation`, run `hrgo2pk1`) was kept running after the PR #311 merge (against advisor instructions — see Edward off-task escalation 11:24Z). At ep7.9 it reached val_abupt=9.892% — substantially worse than STRING-sep SOTA's 7.546% at the same compute. **Conclusion:** GRAPE-M (multiplicative, learned non-axis-aligned rank-2 generators) does NOT beat STRING-sep (separable per-axis learnable freq/phase) on this 3D point-cloud CFD setting. Issue #285 closed-positive on STRING-sep, closed-negative on GRAPE-M.

---

## Active Human Research Directives

**Issue #252 (Modded-NanoGPT):**
- Muon: CLOSED NEGATIVE (askeladd #299, +4.09pp)
- Post-attention RMSNorm: CLOSED NEGATIVE (tanjiro #300 diverged)
- **QK-norm: IN-FLIGHT** alphonse #287 (old base) + thorfinn #358 (STRING-sep base)
- U-net skips: CLOSED NEGATIVE (fern #320, +0.555pp)
- Surface-tangent loss: IN-FLIGHT (fern #351)

**Issue #285 (GRAPE/Representational Position Encoding):** RESOLVED — STRING-sep (PR #311) merged as new SOTA. GRAPE-M (Arm C) still running for completeness.

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
| model_layers | In-flight | 4 SOTA; 5L on OLD base=8.9938%; 5L on STRING-sep base IN-FLIGHT (#365) |
| model_slices / hidden_dim | CLOSED NEGATIVE | 128/512 SOTA |
| lr_cosine T_max / warmup | CLOSED | T_max=50, warmup=0 SOTA |
| grad-clip-norm | MERGED | 0.5 SOTA (#309) |
| **STRING-separable PE** | **MERGED NEW SOTA** | **val 7.546% / test 8.771% (#311)** |
| STRING-sep num_features | In-flight | alphonse #387 (16/32/64 sweep) |
| RFF (isotropic Gaussian) | CLOSED NEGATIVE | WORSE than no-encoding |
| QK-norm | CLOSED NEGATIVE (old base) / In-flight (new base) | alphonse #287 null (+0.10pp worse); thorfinn #358 on STRING-sep base in-flight |
| U-net skips | CLOSED NEGATIVE | +0.555pp vs SOTA |
| MLP activation (SwiGLU/ReLU²) | CLOSED NEGATIVE | Near-zero improvement; ReLU² OOM |
| Per-channel output-head scaling | CLOSED MISMATCH | frieren #352 closed (PR/run mismatch) |
| Sandwich-norm (RMSNorm) | CLOSED NEGATIVE | diverged |
| Muon (canonical) | CLOSED NEGATIVE | +4.09pp |
| Channel-selective Huber on tau | In-flight | askeladd #353 |
| MLP volume decoder | In-flight | tanjiro #323 (old base), frieren #359 (STRING-sep base) |
| Extended training | In-flight | edward #357 |

---

## Largest Remaining Gaps to AB-UPT (from PR #311 test metrics)

1. **volume_pressure** ×2.05 (12.438% vs 6.08%) — frieren #359 + tanjiro #323 targeting this
2. **tau_y** ×2.53, **tau_z** ×2.88 — fern #351 (tangent-frame loss), askeladd #353 (Huber tau)
3. **wall_shear** ×1.13 (8.227% vs 7.29%) — fern #351 also targets this via tangent projection
4. **surface_pressure** ×1.17 (4.485% vs 3.82%) — closest to reference, may need dedicated sweep

---

## Next Priorities (as students complete)

1. **Monitor alphonse #387 STRING-sep feature count sweep** — if 64-feature arm beats 32-feature control, merge and consider 128 as follow-up
2. **Monitor PR #357 edward extended epochs** — may push STRING-sep val below 7.0%
3. **Monitor nezuko #365** — model-layers=5 + STRING-sep compound; depth=5 may compound with richer STRING-sep geometry
4. **Await tanjiro #323 completion** — if vol-decoder works on old base, supersede with frieren #359
5. **Compound round** (next free students): depth=5 + STRING-sep; or STRING-sep + QK-norm + depth (three-way stack if all work independently)
6. **STRING-sep hyperparameter sweep**: init scale, per-axis vs isotropic hybrid (feature count now covered by alphonse #387)
