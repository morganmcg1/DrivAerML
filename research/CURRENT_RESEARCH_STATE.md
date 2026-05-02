# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-02 ~06:50 UTC (cycle re-entry)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #232 (askeladd model-heads=4), val_abupt 9.0650% (ep11)

**Verified SOTA config (W&B run `r8s2dtnq`):**
- lr=1e-4, weight_decay=5e-4, optimizer=lion, lion_beta1=0.9, lion_beta2=0.99
- use_ema=True, ema_decay=0.999, lr_cosine_t_max=50 (≈flat for 9–11ep), **lr_warmup_epochs=0**
- **model_layers=4, model_hidden_dim=512, model_heads=4, model_slices=128, model_mlp_ratio=4**
- train/eval surface_points=65536, train/eval volume_points=65536
- volume_loss_weight=1, surface_loss_weight=1, batch_size=4, compile_model=False
- **rff_num_features=0** (no RFF — was never part of true SOTA)

| Metric | PR #232 SOTA | AB-UPT | Gap |
|---|---:|---:|---:|
| `abupt` mean | **9.0650 (val) / 10.190 (test)** | — | — |
| `surface_pressure` | **5.461** | 3.82 | ×1.43 |
| `wall_shear` | **9.910** | 7.29 | ×1.36 |
| `volume_pressure` | **12.656** | 6.08 | **×2.08** |
| `tau_y` | **11.952** | 3.65 | **×3.27** |
| `tau_z` | **12.447** | 3.63 | **×3.43** |

val→test ratio: 1.124. Merge bar = val_abupt < 9.0650%.

## In-flight — Round 16-18 fleet (8 active WIP PRs, 0 review-ready, 0 idle, 06:50 UTC)

**Active DrivAerML students (ddp8 pods):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn.

| PR | Student | Hypothesis | Best W&B | Current val_abupt | vs SOTA | Status |
|---|---|---|---|---:|---:|---|
| #311 | edward | STRING-separable / GRAPE-M / RFF (3-arm position encoding) | `gcwx9yaa` ArmB | **9.235% (ep5)** | +0.17pp | **CLOSING — closest to SOTA** |
| #309 | thorfinn | grad-clip-norm=0.5 (Lion+EMA, warmup=0) | `ztdhodw1` ArmB | 9.494% (ep9) | +0.43pp | Closing fast, monotone-decreasing |
| #283 | nezuko | model-layers=5 | `z6xc97gg` rank0 | 9.993% (ep7) | +0.93pp | Trending down, ~3 epochs remain |
| #287 | alphonse | QK-norm (per-head L2 norm) | `nesrmoi9` ArmA | 10.413% (ep6) | +1.35pp | Trending down |
| #320 | fern | U-net skip connections | `1d2c2a6q` ArmB | 10.604% (ep4) | +1.54pp | Early, slow |
| #280 | frieren | MLP activation (SwiGLU/ReLU²/GELU) | `k76fngw1` ArmC SwiGLU | 10.976% (ep5) | +1.91pp | ArmA GELU done @9.196% (config drift); ArmB ReLU² OOM; ArmC running. Branch needs rebase after ArmC. |
| #299 | askeladd | Muon optimizer (canonical lr=0.02) | `t3o9jib0` | 16.056% (ep6) | +6.99pp | Slow convergence, wait until ep10 to assess |
| #323 | tanjiro | 2-layer MLP volume decoder head | `ux5s5id6` (KILLED) | DIVERGED (84.83% ep9) | — | **Restarting** with new init/normalization |

## Active Human Research Directives

**Issue #252 (Modded-NanoGPT):** Mine Modded-NanoGPT techniques. Status:
1. Muon optimizer — **in-flight** askeladd #299 (canonical lr=0.02)
2. Post-attention RMSNorm — **NEGATIVE** (tanjiro #300 closed)
3. QK-norm — **in-flight** alphonse #287
4. U-net skip connections — **in-flight** fern #320
5. Sequence packing / FlexAttention — throughput lever (deferred)

**Issue #285 (GRAPE/Representational Position Encoding):** 3-arm sweep — **IN-FLIGHT** edward #311 (Arm B already at 9.235%, very promising).

## Current Research Focus

**Top priority — edward #311 STRING-separable (val=9.235% ep5):**
At rate of -1.2pp/epoch, ep6-7 should land at ≤9.0% which would be a **new SOTA**. This is the first new-architecture (vs HP-tuning) win in many rounds. Watch ep6-8 closely.

**Watching for SOTA crosses:**
- thorfinn #309 clip=0.5 (val=9.494%, only 0.43pp behind, monotone-decreasing). Also showing it avoids the SOTA's ep8 regression (12.12% → 9.73% on this run).
- nezuko #283 layers=5 (val=9.993%, ~3 epochs remain).

**Architecture under test:**
- alphonse #287 QK-norm — clean A/B test, trending but unlikely to beat SOTA at current pace.
- fern #320 U-net skips — early.
- frieren #280 SwiGLU/GELU/ReLU² — ArmA underperformed control (config drift on old tay base, branch needs rebase). ArmC SwiGLU running.

**Optimizer experiments:**
- askeladd #299 Muon canonical lr=0.02 — slow ep1-6, decision at ep10.

**Decoder head experiments:**
- tanjiro #323 — diverged at ep9, restarting with corrected init / norm placement.

## Key Learnings (cumulative)

| Lever | Status | Outcome |
|---|---|---|
| LR | CLOSED | 1e-4 SOTA; ceiling confirmed |
| EMA decay | CLOSED | 0.999 SOTA |
| Lion beta2 | CLOSED | 0.99 SOTA; 0.999 negative |
| Lion beta1 | CLOSED | 0.9 SOTA; 0.8 + 0.95 negative |
| Weight decay | CLOSED | 5e-4 SOTA |
| Vol loss weight | CLOSED | 1.0 SOTA |
| Vol points | CLOSED | 65536 SOTA |
| Surface points | CLOSED | 65536 SOTA |
| mlp_ratio | CLOSED NEGATIVE | 4 SOTA; 6 + 8 negative |
| Dropout | CLOSED | 0.0 SOTA |
| Tau axis weights | CLOSED | 1.0 SOTA; Lion sign neutralizes per-channel weighting |
| model_layers | In-flight | 4 SOTA; 5L (nezuko #283) trending +0.93pp |
| model_heads | CLOSED — NEW SOTA | 4H beats 8H (PR #232 merged) |
| model_slices | CLOSED NEGATIVE | 128 SOTA; 64 negative |
| model_hidden_dim | CLOSED NEGATIVE | 512 SOTA |
| lr_cosine T_max sweep | CLOSED NEGATIVE | T_max=50 SOTA |
| lr_warmup_epochs | CLOSED | warmup=0 in actual SOTA r8s2dtnq |
| grad-clip-norm | In-flight | thorfinn #309: clip=0.5 closing fast (val=9.494% ep9, -2.39pp vs SOTA's ep8 regression) |
| U-net skips | In-flight | fern #320: val=10.60% ep4 |
| Dedicated MLP vol decoder | Restarting | tanjiro #323: ArmA diverged at ep9 → 84.8% (training instability) |
| GRAPE-M / STRING / RFF | In-flight | edward #311: ArmB STRING-separable val=9.235% ep5 |
| MLP activation | In-flight | frieren #280: ArmA GELU 9.196% (config drift), ArmB ReLU² OOM, ArmC SwiGLU val=10.98% ep5 |
| QK-norm | In-flight | alphonse #287: val=10.41% ep6 |
| Sandwich-norm (RMSNorm) | CLOSED NEGATIVE | tanjiro #300: val=79.98% diverged |
| Muon optimizer (canonical) | In-flight | askeladd #299: val=16.06% ep6, slow convergence |
| batch_size | CLOSED NEGATIVE | 4 SOTA |
| compile_model | CLOSED ANOMALOUS | False SOTA |

## Largest Remaining Gaps to AB-UPT

1. **volume_pressure** ×2.08 (12.656% vs 6.08%) — tanjiro MLP decoder restarting; nezuko layers=5 also targeting this axis.
2. **tau_y** ×3.27, **tau_z** ×3.43 — Shear stress direction prediction. Best lever: surface-tangent-frame projection.

## Next Priorities (when students free up)

1. **Edward #311 SOTA fast-track** — if ArmB STRING-separable hits <9.065%, immediate merge.
2. **Surface-tangent-frame projection** for tau_y/tau_z (×3.27/×3.43 gaps) — geometry-aware shear head.
3. **Compound winners:** if STRING-separable wins + clip=0.5 wins, compound them on top of SOTA.
4. **Sequence packing / FlexAttention** — throughput lever to enable more epochs within budget.
5. **Muon-with-warmup** — if vanilla Muon stalls, retry with lr_warmup_epochs=1 (Modded-NanoGPT recipe).
6. **Volume-decoder iteration** — if tanjiro restart works, sweep MLP depth/init/bottleneck variants.
