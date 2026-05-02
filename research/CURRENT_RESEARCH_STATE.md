# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-02 ~09:05 UTC (cycle re-entry)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #309 (thorfinn grad-clip-norm=0.5), val_abupt 9.0389% (ep11)

**Verified SOTA config (W&B run `ztdhodw1`):**
- lr=1e-4, weight_decay=5e-4, optimizer=lion, lion_beta1=0.9, lion_beta2=0.99
- use_ema=True, ema_decay=0.999, lr_cosine_t_max=50, lr_warmup_epochs=0
- model_layers=4, model_hidden_dim=512, model_heads=4, model_slices=128, model_mlp_ratio=4
- train/eval surface_points=65536, train/eval volume_points=65536
- volume_loss_weight=1, surface_loss_weight=1, batch_size=4, compile_model=False
- rff_num_features=0
- **grad-clip-norm=0.5** (new in #309)

| Metric | PR #309 SOTA | AB-UPT | Gap |
|---|---:|---:|---:|
| `abupt` mean | **9.0389 (val) / 10.126 (test)** | — | — |
| `surface_pressure` | **5.395** | 3.82 | ×1.41 |
| `wall_shear` | **9.883** | 7.29 | ×1.36 |
| `volume_pressure` | **12.484** | 6.08 | **×2.05** |
| `tau_y` | **11.941** | 3.65 | **×3.27** |
| `tau_z` | **12.407** | 3.63 | **×3.42** |

Merge bar = val_abupt < 9.0389%.

## BREAKING — THREE SOTA-CROSSING SIGNALS (~09:05 UTC)

### edward #311 STRING-separable — **val=7.5458% (-1.49pp vs SOTA)** — DECISIVE WIN, still improving

W&B run `gcwx9yaa`. Monotone-decreasing, still running (runtime ~4.5h).

| Epoch | val_abupt | vs SOTA |
|---:|---:|---:|
| 5 | 9.235% | +0.20pp |
| 6 | 8.478% | -0.56pp (crossed) |
| 7 | 8.159% | -0.88pp |
| 8 | 7.909% | -1.13pp |
| 9 | 7.742% | -1.30pp |
| latest summary | **7.546%** | **-1.49pp** |

**First new-architecture (vs HP-tuning) win in many rounds.** Fast-track: let it finish, run test-set eval from best-val checkpoint, mark ready for review for immediate merge.

### nezuko #283 model-layers=5 — **val=8.9938% ep12 (-0.045pp vs SOTA)** — CROSSED

W&B run `z6xc97gg`. Lion ep10 spike → ep11/12 recovery played out cleanly.

| Epoch | val_abupt | vs SOTA |
|---:|---:|---:|
| 9 | 9.254% | +0.21pp |
| 10 | 10.896% | (Lion spike) |
| 11 | 9.523% | +0.48pp |
| 12 | **8.9938%** | **-0.045pp (crossed)** |

Still training (~6 epochs left of budget). Plan: let run to completion, test-set eval from best-val checkpoint, then merge.

### alphonse #287 QK-norm — LABELING CORRECTION (do not merge yet)

Earlier celebration of `nesrmoi9` was incorrect — that is **Arm A control (no `--qk-norm`)**, not the treatment. Control itself reached ep10=8.9532% (-0.086pp), informative for seed/init variance characterization but NOT a QK-norm signal.

- Arm B treatment `pz7zp49v` launched 08:43Z, currently <ep1.
- Merge criterion: Arm B must beat Arm A's best-val by ≥0.4pp to credibly attribute to QK-norm. Within ±0.2pp = null result.

## In-flight — Round 16-18 fleet (8 active WIP PRs; 2 SOTA-crossing, 1 wrap-up, 1 idle, 08:35 UTC)

**Active DrivAerML students (ddp8 pods):** alphonse, askeladd, edward, frieren, fern, nezuko, tanjiro, thorfinn.

| PR | Student | Hypothesis | W&B | Latest val | vs SOTA | Status |
|---|---|---|---|---:|---:|---|
| #311 | edward | STRING-separable / GRAPE-M / RFF (3-arm) | `gcwx9yaa` ArmB | **7.546% latest** | **-1.49pp** | **WINNER — fast-track merge after test eval** |
| #287 | alphonse | QK-norm (per-head L2) | Arm A `nesrmoi9` (CONTROL, ep10=8.953%); Arm B `pz7zp49v` <ep1 | TBD | TBD | Awaiting Arm B treatment results |
| #345 | thorfinn | grad-clip-norm=2.0 retest | `7see7ryk` | 13.42% (epoch unconfirmed) | — | Pinged for status — possible divergence |
| #283 | nezuko | model-layers=5 | `z6xc97gg` | **8.9938% ep12** | **-0.045pp** | **CROSSED — let finish, test eval, merge** |
| ~~#280~~ | ~~frieren~~ | ~~MLP activation (SwiGLU/ReLU²/GELU)~~ | `k76fngw1` ArmC | 9.153% | +0.114pp | **CLOSED — informative-negative** |
| **#352** | **frieren** | **Per-channel output-head scaling** — but active W&B run `9yme70s3` is `arm-D-swiglu-uniform` (PR #280 family). PRpr/branch mismatch flagged to student. | `9yme70s3` (mismatched?) | 58.24% (ep1) | — | **NEEDS CLARIFICATION** |
| ~~#299~~ | ~~askeladd~~ | ~~Muon optimizer (canonical lr=0.02)~~ | `t3o9jib0` | 13.13% ep11 | +4.09pp | **CLOSED — confirmed negative** |
| **#353** | **askeladd** | **Channel-selective Huber loss on tau (delta=0.5)** | — | — | — | **NEW — assigned (H04 Tier-1)** |
| #323 | tanjiro | 2-layer MLP volume decoder head | restarted | (in restart) | — | Restarting after divergence |
| #351 | fern | surface-tangent-frame projection loss for tau_y/tau_z | — | — | — | WIP |

## Active Human Research Directives

**Issue #252 (Modded-NanoGPT):**
1. Muon optimizer — in-flight askeladd #299 (canonical lr=0.02)
2. Post-attention RMSNorm — NEGATIVE (tanjiro #300 closed)
3. **QK-norm — WINNING in-flight** alphonse #287 (-0.086pp)
4. ~~U-net skip connections~~ — NEGATIVE (fern #320 closed at +0.555pp)
5. Sequence packing / FlexAttention — throughput lever (deferred)

**Issue #285 (GRAPE/Representational Position Encoding):** 3-arm sweep — **WINNING** edward #311 STRING-separable (-1.30pp, decisive).

## Current Research Focus

**Top priority — close out SOTA winners (best-first merge order):**
1. edward #311 STRING-separable (val≈7.55%, -1.49pp): wait for run to finish, run test-set eval from best-val checkpoint, merge first.
2. nezuko #283 model-layers=5 (val=8.9938%, -0.045pp): wait for run to finish, test eval, merge after #311 (deltas orthogonal: PE vs depth).
3. alphonse #287 QK-norm: await Arm B treatment `pz7zp49v` results before any merge decision (Arm A was control).

**Compound experiment planning:**
- If both #311 and #287 merge, compound experiment = STRING-separable + QK-norm + grad-clip=0.5 SOTA stack. Assign to next free student. The deltas are architecturally orthogonal (position encoding vs attention normalization vs gradient stabilization).

**Architecture watch:**
- nezuko #283 layers=5: 9.523% at ~ep11, monotone but unlikely to cross — wait for finish.
- frieren #280: CLOSED informative-negative (SwiGLU 9.153% > GELU 9.196%, +0.114pp above SOTA; ReLU² OOM on 4L/512d/65k+65k DDP8). frieren reassigned to #352 (per-channel output-head scaling).
- thorfinn #345: RFF retest on Lion+EMA+heads=4 stack starting fresh.

**Optimizer:**
- askeladd #299 Muon canonical lr=0.02: **CLOSED** — 13.13% at ep11 (+4.09pp), confirmed negative. Backup option for later: Muon-with-warmup retry.

**Loss reformulation:**
- askeladd #353 Huber-on-tau (delta=0.5): channel-selective Huber loss for tau_x/tau_y/tau_z (surface_preds[:, 1:4]); MSE retained on surface_pressure (channel 0) and all volume channels. Targets the heavy-tailed wall-shear distribution driving tau_y x3.27 and tau_z x3.42 gaps. Near-orthogonal to architecture deltas in-flight.

**Decoder head:**
- tanjiro #323 MLP volume decoder restart targeting volume_pressure ×2.05 gap.

## Key Learnings (cumulative)

| Lever | Status | Outcome |
|---|---|---|
| LR | CLOSED | 1e-4 SOTA |
| EMA decay | CLOSED | 0.999 SOTA |
| Lion beta2 / beta1 | CLOSED | 0.99 / 0.9 SOTA |
| Weight decay | CLOSED | 5e-4 SOTA |
| Vol/surf loss weights | CLOSED | 1.0 / 1.0 SOTA |
| Vol/surf points | CLOSED | 65536 / 65536 SOTA |
| mlp_ratio | CLOSED NEGATIVE | 4 SOTA |
| Dropout | CLOSED | 0.0 SOTA |
| Tau axis weights | CLOSED | 1.0 SOTA |
| model_layers | In-flight | 4 SOTA; 5L (#283) trending +0.48pp |
| model_heads | CLOSED — SOTA | 4H beats 8H (#232 merged) |
| model_slices / hidden_dim | CLOSED NEGATIVE | 128 / 512 SOTA |
| lr_cosine T_max / lr_warmup | CLOSED | T_max=50 / warmup=0 SOTA |
| **grad-clip-norm** | **CLOSED — NEW SOTA** | **0.5 beats no-clip (#309 merged, 9.0389%)** |
| **STRING-separable PE (GRAPE-M)** | **WINNING in-flight** | **edward #311 ArmB val=7.742% ep9 (-1.30pp)** |
| **QK-norm (per-head L2)** | **WINNING in-flight** | **alphonse #287 val=8.953% ep10 (-0.086pp)** |
| U-net skips | CLOSED NEGATIVE | fern #320 final 9.594% (+0.555pp) |
| MLP activation (SwiGLU/ReLU²) | CLOSED NEGATIVE | frieren #280: SwiGLU 9.153 > GELU 9.196 (0.043pp, below noise), ReLU² OOM at 4L/512d/65k+65k DDP8 |
| Per-channel output-head scaling | In-flight | frieren #352: s∈R^4 init=1 on surface_out + s∈R^1 on volume_out; targets tau_y/z magnitude calibration |
| Sandwich-norm (RMSNorm) | CLOSED NEGATIVE | tanjiro #300 diverged |
| Muon (canonical) | CLOSED NEGATIVE | askeladd #299 13.13% ep11 (+4.09pp) — confirmed negative |
| Channel-selective Huber on tau | In-flight | askeladd #353 (delta=0.5 on surface_preds[:, 1:4]) |
| MLP volume decoder | Restarting | tanjiro #323 |
| RFF on Lion stack | New | thorfinn #345 retest |
| batch_size / compile_model | CLOSED | 4 / False SOTA |

## Largest Remaining Gaps to AB-UPT

1. **volume_pressure** ×2.05 (12.484% vs 6.08%) — tanjiro MLP decoder restart; nezuko layers=5 also hits this axis.
2. **tau_y** ×3.27, **tau_z** ×3.42 — three parallel approaches in-flight: fern #351 (tangent-frame projection loss), frieren #352 (per-channel output-head scaling), and askeladd #353 (channel-selective Huber loss on tau).

## Next Priorities (when students free up)

1. **Compound stack:** STRING-separable (#311) + QK-norm (#287) + grad-clip=0.5 SOTA. Assign as soon as both winners merge.
2. **Surface-tangent-frame projection** for tau_y/tau_z — fern #351 in-flight.
   **Per-channel output-head scaling** — frieren #352 in-flight (complementary tau_y/z lever).
3. **Sequence packing / FlexAttention** — throughput lever to enable more epochs within budget.
4. **Muon-with-warmup** — if vanilla Muon stalls, retry with `lr_warmup_epochs=1`.
5. **Volume-decoder iteration** — if tanjiro restart works, sweep MLP depth/init/bottleneck.
6. **STRING-separable hyperparam sweep** post-merge — feature count, frequency band, learnable-vs-fixed.
