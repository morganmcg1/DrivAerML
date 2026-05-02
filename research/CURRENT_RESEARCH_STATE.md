# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 ~11:15 UTC (post-PR #311 merge cycle; nezuko assigned PR #365)
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

| PR | Student | Hypothesis | Status | Latest val |
|---|---|---|---|---:|
| #357 | edward | STRING-sep extended epochs (≥16 epochs) | NEW — assigned | — |
| #358 | thorfinn | STRING-sep + QK-norm stack | NEW — assigned | — |
| #359 | frieren | STRING-sep + separate volume decoder head | NEW — assigned | — |
| #287 | alphonse | QK-norm on OLD SOTA stack | In-flight (Arm B running) | TBD |
| **#365** | **nezuko** | **model-layers=5 + STRING-sep PE (NEW ASSIGNMENT)** | **NEWLY ASSIGNED** | — |
| #323 | tanjiro | 2-layer MLP vol decoder head (v3 re-launch) | ep6~10.96%, converging | 10.96% |
| #351 | fern | Surface-tangent-frame projection loss for tau_y/tau_z | ep2=47.74%, early | 47.74% |
| #353 | askeladd | Channel-selective Huber loss on tau (delta=0.5) | Early — no val yet | — |

### Notes on each

**PR #287 alphonse QK-norm (old base):** Testing QK-norm on the old SOTA stack (PR #309 baseline, not STRING-sep). The new SOTA is now 7.546%, so alphonse's result must beat this. If QK-norm is strong, it will still beat 7.546% (since the old SOTA is only 9.039%, a QK-norm improvement would need to be very large). Thorfinn PR #358 tests QK-norm stacked with STRING-sep — whichever approach works better will be the next merge.

**PR #365 nezuko model-layers=5 + STRING-sep:** Compound hypothesis. PR #283 showed depth=5 on the OLD SOTA base reached val~8.9938% (doesn't beat STRING-sep SOTA of 7.546%, but the depth signal is real). Now we stack depth=5 on the STRING-sep base — the richer learnable per-axis positional encoding should give the extra layer more structure to compose over. Run command is identical to SOTA except `--model-layers 5`. ~10-12 epochs expected in 270-min budget.

**PR #323 tanjiro vol-decoder:** Testing separate volume head on the OLD SOTA base. If it works, will be superseded by frieren PR #359 (same idea but on STRING-sep base). Still tracking the convergence — ep6=10.96% (normal trajectory at ep6).

**PR #351 fern surface-tangent:** Projecting tau onto surface tangent plane to remove unphysical normal-direction loss signal. Targets tau_y/tau_z gaps. Still very early (ep2).

**PR #353 askeladd Huber tau:** Channel-selective Huber loss (delta=0.5) on tau channels (surface_preds[:, 1:4]) targeting heavy-tailed wall-shear error distribution. No val epochs yet.

---

## GRAPE-M Arm C

Edward's PR #311 Arm C (GRAPE-M — minimal learned spectral projection, B as nn.Parameter) was still running at PR merge time. All 8 DDP ranks in state=running, group `tay-round18-grape-ablation`. Will evaluate when it finishes. If GRAPE-M also beats the STRING-sep baseline (unlikely but possible given val slopes show room to grow), we'll stack GRAPE-M on top.

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
| RFF (isotropic Gaussian) | CLOSED NEGATIVE | WORSE than no-encoding |
| QK-norm | In-flight | alphonse #287 (old base), thorfinn #358 (new base) |
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

1. **Monitor alphonse #287 QK-norm Arm B** — if it beats STRING-sep SOTA, merge and compound
2. **Monitor PR #357 edward extended epochs** — may push STRING-sep val below 7.0%
3. **Monitor nezuko #365** — model-layers=5 + STRING-sep compound; depth=5 may compound with richer STRING-sep geometry
4. **Await tanjiro #323 completion** — if vol-decoder works on old base, supersede with frieren #359
5. **Compound round** (next free students): depth=5 + STRING-sep; or STRING-sep + QK-norm + depth (three-way stack if all work independently)
6. **STRING-sep hyperparameter sweep**: feature count (16/32/64), init scale, per-axis vs isotropic hybrid
