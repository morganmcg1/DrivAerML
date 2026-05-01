# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-02 ~01:00 UTC
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

## In-flight — Rounds 13–16 (2026-05-02 ~01:00 UTC)

**Active DrivAerML students (ddp8):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn.

**NOTE:** Round 15 PRs (#263-272) assigned to non-existent students were closed 2026-05-02. Deferred hypotheses listed under Key Learnings.

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #232 | askeladd | model_heads=4 | **MERGED — CURRENT SOTA** |
| #240 | frieren | mlp_ratio=8 | **CLOSED NEGATIVE** — best val=9.5498% at ep7.52 (+0.485pp). |
| #241 | tanjiro | hidden_dim=768 (lr=1e-4) | Running; **tracking NEGATIVE** (~10.5% likely final). |
| #247 | thorfinn | lr-cosine-t-max=14 | Running; ep7=9.9253%, **tracking NEGATIVE** (+0.860pp). |
| #251 | fern | T_max=8+warmup+lr-min=5e-6 | Running; ep5=11.835%, **tracking NEGATIVE** (behind frieren at every epoch). |
| #280 | frieren | MLP activation ablation (SwiGLU/ReLU²) | WIP — pod waiting for #240 cleanup. |
| #281 | askeladd | --compile-model on SOTA stack | Running; ep2=45.04% (**anomalous — 2x worse than SOTA ep2=22%**). Monitor. |
| #282 | edward | --lion-beta1 0.95 | Running; ep1=63.7% (normal). Monitor ep2. |
| #283 | nezuko | --model-layers 5 depth sweep | Running; ep1=64.78% (normal). Monitor ep2. |
| **#287** | **alphonse** | **QK-norm on SOTA stack** | **NEW Round 16 — pod pickup pending.** |

## Active Human Research Directives

**Issue #252 (Modded-NanoGPT):** Mine the Modded-NanoGPT world-record training history for applicable improvements. Techniques ranked by transfer plausibility:
1. Muon optimizer (Newton-Schulz orthogonalized momentum) — would require code change; high priority next free slot
2. Post-attention RMSNorm / QK-norm — stabilizes Transolver slot attention
3. Linear warmdown LR (being approximated by norman #269 + fern #251)
4. U-net skip connections — cheap residual across layers
5. Sequence packing / FlexAttention — throughput lever (more epochs/budget)

## Current Research Focus

**Primary focus:** Architecture and attention normalization (Round 16).
- QK-norm on SOTA stack (alphonse #287) — per-head L2 normalization, never tested
- MLP activation ablation SwiGLU/ReLU² vs GELU (frieren #280) — activation function sweep

**Secondary focus (closing out active runs):**
- compile_model efficiency (askeladd #281) — anomalous ep2 val; monitor for recovery
- model-layers=5 depth sweep (nezuko #283) — closing the depth axis
- lion-beta1=0.95 (edward #282) — closing the lion_beta1 sweep axis
- LR anneal T_max=8 (fern #251) — tracking NEGATIVE; close when done
- LR cosine T_max=14 (thorfinn #247) — tracking NEGATIVE; close when done
- Width 768d (tanjiro #241) — tracking NEGATIVE; close when done

**Deferred (hypotheses from closed Round 15 orphans — assign to ddp8 students):**
- lr-warmup-epochs=2 (was chihiro #263)
- RFF retest on current SOTA stack (was emma #264)
- ema-decay=0.9995 WITH warmup=1ep (was gilbert #265)
- grad-clip-norm=0.5 (was haku #267)
- lr-cosine-t-max=12 (was kohaku #268)
- warmup=1ep + cosine T_max=9 (was norman #269)
- model-hidden-dim=768 + muP lr=8.2e-5 (was senku #271)
- lr-min=1e-5 (was violet #272)

**Next priority (after Round 16):** Modded-NanoGPT code-change ideas
- Muon optimizer (Newton-Schulz orthogonalized gradient) — highest expected gain
- Post-attention RMSNorm (stabilize Transolver attention, ~4-line change)
- U-net skip connections (cross-layer residuals)

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
| mlp_ratio | **CLOSED NEGATIVE** | 4 SOTA; 6 regressed (+6.4%); 8 NEGATIVE (PR#240, +0.485pp, timeout-bounded) |
| Dropout | CLOSED | 0.0 SOTA; 0.05 regressed (+4.24%); model underfits |
| Tau axis weights | CLOSED | 1.0 SOTA; Lion sign neutralizes per-channel weighting |
| model_layers | Closing | 4 SOTA; 3L regression (PR #233); 5L in-flight (#283) |
| model_heads | **CLOSED — NEW SOTA** | 4H beats 8H (PR #232 merged; −0.226pp val) |
| model_slices | Closing | 128 SOTA; 64 tracking regression (PR #231) |
| lr_cosine T_max=9 | CLOSED NEGATIVE | Confirmed ×2 (edward #195, tanjiro #202) |
| lr_cosine T_max=14 | NEGATIVE (in-flight) | thorfinn #247: ep7=9.925%, tracking +0.86pp above SOTA |
| lr_cosine T_max=8 + anneal | NEGATIVE (in-flight) | fern #251: ep5=11.83%, behind frieren at every epoch |
| lr_warmup_epochs=1 | WIN (compound) | PR #222 fern — +2.03% val, +1.51% test |
| lr-min | DEFERRED | 1e-5 was violet #272 (no pod — orphan closed) |
| grad-clip-norm | DEFERRED | 0.5 was haku #267 (no pod — orphan closed) |
| RFF features | DEFERRED | Emma #264 retest on SOTA (no pod — orphan closed) |
| model_hidden_dim | NEGATIVE (in-flight) | 512 SOTA; 768 tanjiro #241 tracking plateau |
| Warmup 2ep | DEFERRED | chihiro #263 (no pod — orphan closed) |
| batch_size | CLOSED NEGATIVE | batch=5 OOM'd; batch-size lever closed; 4=SOTA |
| compile_model | Anomalous (in-flight) | askeladd #281: ep2=45% vs SOTA 22% — monitoring for recovery |
| lion_beta1 | Closing | 0.9 SOTA; 0.8 negative; 0.95 in-flight (edward #282) |
| MLP activation | In-flight | SwiGLU/ReLU²/GELU ablation (frieren #280) |
| QK-norm | In-flight | Per-head L2 norm (alphonse #287) — new Round 16 |

## Largest Remaining Gaps to AB-UPT

1. **volume_pressure** ×2.08 — Not a loss-weighting problem (closed). Likely architectural: dedicated volume decoder, richer SDF features, hierarchical multi-scale volume heads.
2. **tau_y** ×3.27, **tau_z** ×3.43 — Shear stress direction prediction. Needs geometry-informed head (surface tangent frame) or output transformation (asinh normalization).

## Next Priorities (after Rounds 15/16 close)

1. **Muon optimizer** — Newton-Schulz orthogonalized gradient, highest-variance Modded-NanoGPT bet
2. **Post-attention RMSNorm** — stabilize Transolver slot attention, cheap code change
3. **Volume architecture** — dedicated volume decoder head; potentially copy from AB-UPT paper's design
4. **Tau head reform** — surface-tangent-frame projection to close tau_y/tau_z gap
5. **Compound winners** — stack Round 15/16 winners once identified
