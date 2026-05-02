# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-02 ~08:00 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #232 (askeladd model-heads=4), test_abupt 10.190%

**Verified SOTA config (W&B run `r8s2dtnq`):**
- lr=1e-4, weight_decay=5e-4, optimizer=lion, lion_beta1=0.9, lion_beta2=0.99
- use_ema=True, ema_decay=0.999, lr_cosine_t_max=50 (≈flat for 9–11ep), **lr_warmup_epochs=1**
- **model_layers=4, model_hidden_dim=512, model_heads=4, model_slices=128, model_mlp_ratio=4**
- train/eval surface_points=65536, train/eval volume_points=65536
- volume_loss_weight=1, surface_loss_weight=1, batch_size=4, compile_model=False

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

## In-flight — Rounds 13–18 (2026-05-02 ~08:00 UTC)

**Active DrivAerML students (ddp8):** alphonse, askeladd, chihiro, edward, emma, fern, frieren, gilbert, haku, kohaku, nezuko, norman, senku, tanjiro, thorfinn, violet.

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #232 | askeladd | model_heads=4 | **MERGED — CURRENT SOTA** |
| #240 | frieren | mlp_ratio=8 | **CLOSED NEGATIVE** — best val=9.5498% at ep7.52 (+0.485pp). |
| #251 | fern | T_max=8+warmup+lr-min=5e-6 | **CLOSED NEGATIVE** — best val=9.4088% at ep9, +0.344pp vs SOTA. T_max=8 cosine hypothesis falsified. Branch deleted. |
| #280 | frieren | MLP activation ablation (SwiGLU/ReLU²/GELU) | 4-arm sequential DDP8 sweep; started ~22:59 UTC 2026-05-01; ~18h total; results due ~17:00 UTC 2026-05-02. Branch has merge conflict — rebase reminder posted. |
| #282 | edward | lion-beta1=0.95 | Running; ep3=26.10% (gap narrowing, down from ep2=53.3%). Monitor for convergence. |
| #283 | nezuko | model-layers=5 depth sweep | Running; ~2.7h elapsed, current val=11.41%, steep descent slope (-0.615%/1k steps), best_epoch=None. Monitor for completion. |
| #287 | alphonse | QK-norm on SOTA stack (per-head L2 norm) | Round 16 — pod starting; 0 comments. |
| #289 | chihiro | lr-warmup-epochs=2 (1ep→2ep) | Round 17 — pod starting; 0 comments. |
| #290 | emma | RFF retest on current SOTA stack (heads=4+warmup=1ep) | Round 17 — pod starting; 0 comments. |
| #291 | gilbert | ema-decay=0.9995 with lr-warmup-epochs=1 | Round 17 — pod starting; 0 comments. |
| #292 | haku | grad-clip-norm=0.5 (Lion stabilization) | **CLOSED** — superseded; haku reassigned to #321 |
| #293 | kohaku | lr-cosine-t-max=12 (fill T_max=9 neg / T_max=14 neg gap) | Round 17 — pod starting; 0 comments. |
| #294 | norman | warmup=1ep + lr-cosine-t-max=9 (warmup+anneal compound) | Round 17 — pod starting; 0 comments. |
| #295 | senku | model-hidden-dim=768 + muP lr=8.2e-5 (rescue tanjiro w/ correct LR scaling) | Round 17 — pod starting; 0 comments. |
| #296 | violet | lr-min=1e-5 on SOTA stack (cosine LR floor) | Round 17 — pod starting; 0 comments. |
| #299 | askeladd | Muon optimizer (Newton-Schulz orthogonalized momentum) | Round 17 — pod starting; 0 comments. |
| #300 | tanjiro | Post-attention + post-MLP RMSNorm (sandwich-norm) | Round 17 — pod starting; 0 comments. |
| #309 | thorfinn | grad-clip-norm=0.5 (Lion+EMA+warmup+heads=4) | Round 17 — pod starting; 0 comments. |
| #311 | edward | GRAPE/Positional Encoding 3-arm ablation (RFF ctrl / STRING separable / GRAPE-M learned) | Round 18 — assigned ~02:00 UTC 2026-05-02; ~13.5h total (~15:30 UTC results). Focus: tau_y/tau_z ×3.27/×3.43 gap. |
| #320 | fern | U-net skip connections in Transformer backbone (geometric feature preservation) | Round 18 — assigned ~08:00 UTC 2026-05-02; code changes to model.py + train.py required; single DDP8 arm. |
| #321 | haku | Dedicated 2-layer MLP volume decoder head (vol_pressure ×2.08 gap) | Round 18 — assigned ~08:00 UTC 2026-05-02; code changes to model.py + train.py required; 2-arm sweep (depth=2 vs depth=1 ctrl). |

## Active Human Research Directives

**Issue #252 (Modded-NanoGPT):** Mine the Modded-NanoGPT world-record training history for applicable improvements. Techniques ranked by transfer plausibility:
1. Muon optimizer (Newton-Schulz orthogonalized momentum) — **in-flight** askeladd #299
2. Post-attention RMSNorm / QK-norm — **in-flight** tanjiro #300 (RMSNorm) + alphonse #287 (QK-norm)
3. Linear warmdown LR (approximated by norman #294 + fern #251)
4. U-net skip connections — cheap residual across layers (deferred)
5. Sequence packing / FlexAttention — throughput lever (deferred)

**Issue #285 (GRAPE/Representational Position Encoding):** 3-arm sweep (RFF control / 3D STRING separable / full GRAPE-M learned non-axis-aligned planes) — **unassigned; deferred to next free slot.**

## Current Research Focus

**Primary focus — Round 17 (architecture + optimizer):**
- Muon optimizer (askeladd #299) — highest-variance bet; Newton-Schulz orthogonalization replaces Adam/Lion
- Sandwich-norm RMSNorm (tanjiro #300) — post-attn + post-MLP normalization; stabilizes Transolver attention
- QK-norm (alphonse #287) — per-head L2 normalization; never tested

**Secondary focus — Round 17 (hyperparameter closure):**
- MLP activation ablation (frieren #280) — SwiGLU/ReLU²/GELU 4-arm sweep; results due ~17:00 UTC 2026-05-02
- lion-beta1=0.95 (edward #282) — closing lion_beta1 axis
- model-layers=5 (nezuko #283) — closing depth axis

**Deferred (assign when next student frees up):**
- ~~GRAPE/Group Representational Position Encoding (Issue #285)~~ — **IN-FLIGHT** edward #311 (3-arm: RFF ctrl / STRING / GRAPE-M)
- ~~U-net skip connections~~ — **IN-FLIGHT** fern #320
- ~~Dedicated volume decoder head~~ — **IN-FLIGHT** haku #321
- Sequence packing / FlexAttention (throughput lever)
- Surface-tangent-frame projection for tau_y/tau_z (×3.27/×3.43 gaps)

## Key Learnings (cumulative)

| Lever | Status | Outcome |
|---|---|---|
| LR | CLOSED | 1e-4 SOTA; ceiling confirmed |
| EMA decay | CLOSED | 0.999 SOTA; all 4 values tested |
| Lion beta2 | CLOSED | 0.99 SOTA; 0.999 regressed badly |
| Lion beta1 | Closing | 0.9 SOTA; 0.8 negative; 0.95 in-flight (#282) |
| Weight decay | CLOSED | 5e-4 SOTA; both directions negative |
| Vol loss weight | CLOSED | 1.0 SOTA; 1.5 and 2.0 both negative |
| Vol points | CLOSED | 65536 SOTA; 96k regressed |
| Surface points | CLOSED | 65536 SOTA; 96k (PR #206) negative |
| mlp_ratio | CLOSED NEGATIVE | 4 SOTA; 6 regressed (+6.4%); 8 negative (PR#240, +0.485pp) |
| Dropout | CLOSED | 0.0 SOTA; 0.05 regressed (+4.24%) |
| Tau axis weights | CLOSED | 1.0 SOTA; Lion sign neutralizes per-channel weighting |
| model_layers | Closing | 4 SOTA; 3L regression (PR #233); 5L in-flight (#283) |
| model_heads | CLOSED — NEW SOTA | 4H beats 8H (PR #232 merged; −0.226pp val) |
| model_slices | CLOSED NEGATIVE | 128 SOTA; 64 regression (PR #231) |
| model_hidden_dim | CLOSED NEGATIVE | 512 SOTA; 768 tanjiro #241 negative; 768+muP in-flight (#295) |
| lr_cosine T_max=9 | CLOSED NEGATIVE | Confirmed ×2 (edward #195, tanjiro #202) |
| lr_cosine T_max=14 | CLOSED NEGATIVE | thorfinn #247: ep7=9.925%, +0.86pp above SOTA |
| lr_cosine T_max=8 + anneal | **CLOSED NEGATIVE** | fern #251: best val=9.4088% at ep9, +0.344pp vs SOTA |
| lr_cosine T_max=12 | In-flight | kohaku #293: fills T_max gap; pod starting |
| lr_warmup_epochs=1 | WIN (compound) | PR #222 fern — +2.03% val, +1.51% test |
| lr_warmup_epochs=2 | In-flight | chihiro #289: pod starting |
| lr-min=1e-5 | In-flight | violet #296: pod starting |
| warmup+T_max=9 compound | In-flight | norman #294: pod starting |
| grad-clip-norm=0.5 | Closing | thorfinn #309: pod starting; haku #292 closed |
| U-net skip connections | In-flight | fern #320: cross-layer residuals in 4L backbone; learnable scale gates |
| Dedicated volume decoder (MLP) | In-flight | haku #321: 2-layer MLP vs linear head for vol_pressure ×2.08 gap |
| RFF features | In-flight | emma #290: SOTA-stack retest (heads=4+warmup); pod starting |
| GRAPE-M / STRING / RFF ablation | In-flight | edward #311: 3-arm positional encoding ablation; started ~02:00 UTC 2026-05-02 |
| ema-decay=0.9995 | In-flight | gilbert #291: with warmup=1ep; pod starting |
| batch_size | CLOSED NEGATIVE | batch=5 OOM'd; batch-size lever closed; 4=SOTA |
| compile_model | CLOSED ANOMALOUS | askeladd #281: ep2=45% anomalous; closed |
| MLP activation | In-flight (long run) | SwiGLU/ReLU²/GELU ablation (frieren #280); ~18h sweep |
| QK-norm | In-flight | Per-head L2 norm (alphonse #287); pod starting |
| Sandwich-norm (RMSNorm) | In-flight | Post-attn+post-MLP (tanjiro #300); pod starting |
| Muon optimizer | In-flight | Newton-Schulz orthogonalization (askeladd #299); pod starting |

## Largest Remaining Gaps to AB-UPT

1. **volume_pressure** ×2.08 — Not a loss-weighting problem (closed). Likely architectural: dedicated volume decoder, richer SDF features, hierarchical multi-scale volume heads.
2. **tau_y** ×3.27, **tau_z** ×3.43 — Shear stress direction prediction. Needs geometry-informed head (surface tangent frame) or output transformation (asinh normalization).

## Next Priorities (after Rounds 16/17/18 close)

1. ~~**GRAPE/Representational Position Encoding**~~ — **IN-FLIGHT** edward #311
2. ~~**U-net skip connections**~~ — **IN-FLIGHT** fern #320
3. ~~**Volume architecture (dedicated decoder)**~~ — **IN-FLIGHT** haku #321
4. **Tau head reform** — surface-tangent-frame projection to close tau_y/tau_z gap (next free slot)
5. **Compound winners** — stack Round 16/17/18 winners once identified
6. **Sequence packing / FlexAttention** — throughput lever (more epochs/budget)
