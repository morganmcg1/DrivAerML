# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-02 ~04:30 UTC (cycle re-entry)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #232 (askeladd model-heads=4), test_abupt 10.190%

**Verified SOTA config (W&B run `r8s2dtnq`, confirmed by edward 2026-05-02):**
- lr=1e-4, weight_decay=5e-4, optimizer=lion, lion_beta1=0.9, lion_beta2=0.99
- use_ema=True, ema_decay=0.999, lr_cosine_t_max=50 (≈flat for 9–11ep), **lr_warmup_epochs=0** (NOT 1)
- **model_layers=4, model_hidden_dim=512, model_heads=4, model_slices=128, model_mlp_ratio=4**
- train/eval surface_points=65536, train/eval volume_points=65536
- volume_loss_weight=1, surface_loss_weight=1, batch_size=4, compile_model=False
- **rff_num_features=0** (no RFF — BASELINE.md previously claimed RFF was part of SOTA, this is incorrect)

| Metric | PR #232 SOTA | AB-UPT | Gap |
|---|---:|---:|---:|
| `abupt` mean | **10.190** | — | — |
| `surface_pressure` | **5.461** | 3.82 | ×1.43 |
| `wall_shear` | **9.910** | 7.29 | ×1.36 |
| `volume_pressure` | **12.656** | 6.08 | **×2.08** |
| `tau_x` | **8.432** | 5.35 | ×1.58 |
| `tau_y` | **11.952** | 3.65 | **×3.27** |
| `tau_z` | **12.447** | 3.63 | **×3.43** |

Best val: 9.0650% (ep11). val→test ratio: 1.124.

**Reproduce SOTA:**
```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <STUDENT> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --lr-warmup-epochs 1
```

## In-flight — Rounds 16–18 (2026-05-02 ~14:00 UTC)

**Active DrivAerML students (ddp8 pods only):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn.

Note: chihiro/emma/gilbert/haku/kohaku/norman/senku/violet do NOT have ddp8 pods and cannot run tay experiments. All orphan PRs for these students have been closed (rounds 15 and 17).

| PR | Student | Hypothesis | W&B Run | Status (2026-05-02) |
|---|---|---|---|---|
| #232 | askeladd | model_heads=4 | `r8s2dtnq` | **MERGED — CURRENT SOTA** |
| #280 | frieren | MLP activation ablation (SwiGLU/ReLU²/GELU) | `ds8n7253` | Running; val=11.289% slope=-0.752/1k; had merge conflict (rebase requested); pod restarted. |
| #283 | nezuko | model-layers=5 | `io3rt633` | **VERY PROMISING** — ep7 val=9.808%, vol_pressure=6.09% (BETTER than AB-UPT 6.08%!), gap to SOTA only 0.74pp, ~3 epochs remain. |
| #287 | alphonse | QK-norm (per-head L2 norm) | `cwbdfw17` | Running; val=12.306% slope=-0.356/1k. Had 2 failed runs before this one launched. |
| #299 | askeladd | Muon optimizer | `9jlnbn3u` | Running; val=14.715% slope=-0.638/1k. lr=3e-4, wd=1e-4. Muon may need more tuning. |
| #300 | tanjiro | Sandwich-norm (RMSNorm post-attn+post-MLP) | `528uuqx5` | **CLOSED NEGATIVE** — val=79.98%, catastrophic divergence. |
| #309 | thorfinn | grad-clip-norm=0.5 (Lion) | `b0nnemj5` | Running; val=11.659% slope=-0.788/1k. Slow descent, 50-epoch run. |
| #311 | edward | GRAPE/STRING/RFF 3-arm positional encoding | `zf2dp7tv` | Running; val=35.16% slope=-7.0/1k (very early). Arm A only; B+C pending. |
| #320 | fern | U-net skip connections | `ns8tdroc` | Pod running prior slices-64 experiment; will pick up U-net task on next poll. |
| #323 | tanjiro | 2-layer MLP volume decoder head | (not yet started) | ASSIGNED — freshly created after tanjiro freed. vol_pressure ×2.08 gap target. |

## Active Human Research Directives

**Issue #252 (Modded-NanoGPT):** Mine the Modded-NanoGPT world-record training history for applicable improvements. Techniques ranked by transfer plausibility:
1. Muon optimizer (Newton-Schulz orthogonalized momentum) — **in-flight** askeladd #299
2. Post-attention RMSNorm — **NEGATIVE** (tanjiro #300, catastrophic divergence)
3. QK-norm — **in-flight** alphonse #287 (2 crashed, now running)
4. U-net skip connections — **in-flight** fern #320
5. Sequence packing / FlexAttention — throughput lever (deferred)

**Issue #285 (GRAPE/Representational Position Encoding):** 3-arm sweep — **IN-FLIGHT** edward #311.

## Current Research Focus

**Highest priority — nezuko #283 (model-layers=5):**
At ep7, val=9.808% with volume_pressure=6.09% (matching AB-UPT 6.08%). Gap to SOTA only 0.74pp and ~3 epochs remain. If this crosses 9.065% it's a new SOTA. Watch closely.

**Architecture experiments:**
- QK-norm (alphonse #287) — running at 12.306%, declining. May not beat SOTA but QK-norm is a clean A/B test. Expected to close above SOTA given current trajectory.
- U-net skip connections (fern #320) — pod picking up after current slices-64 run ends.
- 2-layer MLP volume decoder (tanjiro #323) — new assignment; targets vol_pressure ×2.08 gap directly.

**Optimizer/regularization:**
- Muon optimizer (askeladd #299) — val 14.715% declining. Muon at lr=3e-4/wd=1e-4. Far from SOTA but Muon is novel; give it time.
- grad-clip-norm=0.5 (thorfinn #309) — val 11.659%, slow descent, 50-epoch run. May take many more epochs.

**Positional encoding:**
- GRAPE/STRING/RFF ablation (edward #311) — val 35.16% very early, slope -7.0/1k. Very early stage.

**MLP activation:**
- frieren #280 — val 11.289%, slope -0.752/1k. Uses heads=8 (not heads=4 SOTA). Running but unlikely to beat SOTA given the non-SOTA config.

**Deferred (assign when next student frees up):**
- Surface-tangent-frame projection for tau_y/tau_z (×3.27/×3.43 gaps)
- Sequence packing / FlexAttention (throughput lever)
- lr-warmup-epochs=2 (was chihiro #289, student not on tay)
- grad-clip-norm + warmup compound (combine thorfinn and frieren learnings)

## Key Learnings (cumulative)

| Lever | Status | Outcome |
|---|---|---|
| LR | CLOSED | 1e-4 SOTA; ceiling confirmed |
| EMA decay | CLOSED | 0.999 SOTA; all 4 values tested |
| Lion beta2 | CLOSED | 0.99 SOTA; 0.999 regressed badly |
| Lion beta1 | CLOSED | 0.9 SOTA; 0.8 negative; 0.95 negative |
| Weight decay | CLOSED | 5e-4 SOTA; both directions negative |
| Vol loss weight | CLOSED | 1.0 SOTA; 1.5 and 2.0 both negative |
| Vol points | CLOSED | 65536 SOTA; 96k regressed |
| Surface points | CLOSED | 65536 SOTA; 96k negative |
| mlp_ratio | CLOSED NEGATIVE | 4 SOTA; 6 regressed (+6.4%); 8 negative (+0.485pp) |
| Dropout | CLOSED | 0.0 SOTA; 0.05 regressed (+4.24%) |
| Tau axis weights | CLOSED | 1.0 SOTA; Lion sign neutralizes per-channel weighting |
| model_layers | Closing | 4 SOTA; 3L regression; **5L VERY PROMISING** (nezuko #283, ep7=9.808%, vol_pressure=6.09%) |
| model_heads | CLOSED — NEW SOTA | 4H beats 8H (PR #232 merged; −0.226pp val) |
| model_slices | CLOSED NEGATIVE | 128 SOTA; 64 regression |
| model_hidden_dim | CLOSED NEGATIVE | 512 SOTA; 768 negative (×2 tested) |
| lr_cosine T_max=9 | CLOSED NEGATIVE | Confirmed ×2 |
| lr_cosine T_max=14 | CLOSED NEGATIVE | +0.86pp above SOTA |
| lr_cosine T_max=8 + anneal | CLOSED NEGATIVE | +0.344pp vs SOTA |
| lr_cosine T_max=12 | CLOSED NEGATIVE (orphan) | chihiro/kohaku had no pod; deferred |
| lr_warmup_epochs=1 | WIN (compound) | PR #222 fern — +2.03% val, +1.51% test. NOTE: actual SOTA run r8s2dtnq has lr_warmup_epochs=0, contradicting this. May need retest. |
| lr_warmup_epochs=2 | DEFERRED (orphan) | chihiro #289 closed — no pod |
| lr-min=1e-5 | DEFERRED (orphan) | violet #296 closed — no pod |
| warmup+T_max=9 compound | DEFERRED (orphan) | norman #294 closed — no pod |
| ema-decay=0.9995 | DEFERRED (orphan) | gilbert #291 closed — no pod |
| grad-clip-norm=0.5 | In-flight | thorfinn #309: val=11.659%, slope=-0.789/1k |
| U-net skip connections | In-flight | fern #320: pod picking up |
| Dedicated volume decoder (MLP) | In-flight | tanjiro #323: 2-layer MLP vs linear for vol_pressure ×2.08 gap |
| RFF features | DEFERRED (orphan) | emma #290 closed — no pod. NOTE: SOTA uses rff=0, so RFF is not currently in SOTA stack. |
| GRAPE-M / STRING / RFF ablation | In-flight | edward #311: val=35.16%, early stage |
| MLP activation | In-flight | frieren #280: val=11.289%, heads=8 (not SOTA heads=4) |
| QK-norm | In-flight | alphonse #287: val=12.306%, slope=-0.356/1k; had 2 failed runs first |
| Sandwich-norm (RMSNorm) | **CLOSED NEGATIVE** | tanjiro #300: val=79.98%, catastrophic divergence |
| Muon optimizer | In-flight | askeladd #299: val=14.715%, slope=-0.638/1k; lr=3e-4/wd=1e-4 |
| batch_size | CLOSED NEGATIVE | batch=5 OOM'd; 4=SOTA |
| compile_model | CLOSED ANOMALOUS | askeladd #281: ep2=45% anomalous; closed |

## Largest Remaining Gaps to AB-UPT

1. **volume_pressure** ×2.08 (12.656% vs 6.08%) — nezuko layers=5 already shows 6.09% at ep7! Tanjiro MLP decoder targeting this.
2. **tau_y** ×3.27, **tau_z** ×3.43 — Shear stress direction prediction. Needs geometry-informed head (surface tangent frame) or output transformation (asinh normalization).

## Next Priorities (when students free up)

1. **Watch nezuko #283** — if ep8-10 crosses 9.065%, immediate MERGE. layers=5 appears to be a clean win on volume pressure specifically.
2. **Surface-tangent-frame projection** — tau_y/tau_z ×3.27/×3.43 gaps need a geometry-aware approach. Highest-leverage remaining axis after vol_pressure is addressed.
3. **Compound layers=5 + MLP decoder** — if both nezuko (#283) and tanjiro (#323) show positive signals, compound them.
4. **lr_warmup_epochs clarification** — SOTA run r8s2dtnq has lr_warmup_epochs=0, contradicting BASELINE.md's claim of a "warmup win". Need a clean A/B retest: warmup=0 vs warmup=1 on current SOTA heads=4 config.
5. **Sequence packing / FlexAttention** — throughput lever (more epochs/budget)
6. **Tau loss reformulation** — dedicated tau_y/tau_z loss on surface tangent plane
