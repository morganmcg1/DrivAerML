# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-02 ~00:00 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #222 (fern lr_warmup_epochs=1), test_abupt 10.420%

**Verified SOTA config (W&B run `ut1qmc3i`):**
- lr=1e-4, weight_decay=5e-4, optimizer=lion, lion_beta1=0.9, lion_beta2=0.99
- use_ema=True, ema_decay=0.999, lr_cosine_t_max=50 (≈flat for 9ep), **lr_warmup_epochs=1**
- model_layers=4, model_hidden_dim=512, model_heads=8, model_slices=128, model_mlp_ratio=4
- train/eval surface_points=65536, train/eval volume_points=65536
- volume_loss_weight=1, surface_loss_weight=1, batch_size=4, compile_model=False

| Metric | PR #222 SOTA | AB-UPT | Gap |
|---|---:|---:|---:|
| `abupt` mean | **10.420** | — | — |
| `surface_pressure` | **5.550** | 3.82 | ×1.45 |
| `wall_shear` | **10.185** | 7.29 | ×1.40 |
| `volume_pressure` | **12.737** | 6.08 | **×2.09** |
| `tau_x` | **8.629** | 5.35 | ×1.61 |
| `tau_y` | **12.329** | 3.65 | **×3.38** |
| `tau_z` | **12.854** | 3.63 | **×3.54** |

Best val: 9.2910% (ep9). val→test ratio: 1.121.

**Reproduce SOTA:**
```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <STUDENT> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.999 --lr-warmup-epochs 1
```

## In-flight — Rounds 12–15 (2026-05-02 ~00:00 UTC)

All 16 student GPUs occupied. 8 slots filling now as students pick up Round 15 assignments.

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #231 | nezuko | model_slices=64 | Running; ep4=15.63%, **tracking regression** (+6.15pp). Continue to ep9 to confirm. |
| #232 | askeladd | model_heads=4 | Running; ep5=11.18%, **tracking regression** (+1.69pp). Has merge conflict — close after ep9. |
| #233 | edward | model_layers=3 | Running; ep5=13.52%, **tracking regression** (+4.04pp). Has merge conflict — close after ep9. |
| #240 | frieren | mlp_ratio=8 | Running; no student report yet — second ping sent. |
| #241 | tanjiro | hidden_dim=768 (lr=1e-4) | Running; no student report yet — second ping sent. |
| #247 | thorfinn | lr-cosine-t-max=14 | Running; rank0=`rldium3l`, ep2=22.6%, promising early trajectory. Has merge conflict. |
| #250 | alphonse | batch-size=5 (OOM fallback from 8, then 6) | Running; rank0=`okwbc8i6`, eff_batch=40. Approved batch=5 fallback. |
| #251 | fern | T_max=8+warmup+lr-min=5e-6 | Running; rank0=`uederk7o`, early ep1. |
| #263 | chihiro | lr-warmup-epochs=2 | **NEW** — just assigned Round 15. |
| #264 | emma | RFF coord features retest on SOTA stack | **NEW** — just assigned Round 15. |
| #265 | gilbert | ema_decay=0.9995 re-test WITH warmup | **NEW** — just assigned Round 15. |
| #267 | haku | grad-clip-norm=0.5 (never varied) | **NEW** — just assigned Round 15. |
| #268 | kohaku | lr-cosine-t-max=12 (cosine sweep midpoint) | **NEW** — just assigned Round 15. |
| #269 | norman | warmup=1ep + T_max=9 (warmdown-with-warmup) | **NEW** — just assigned Round 15. |
| #271 | senku | model-hidden-dim=768 + muP lr=8.2e-5 | **NEW** — just assigned Round 15. |
| #272 | violet | lr-min=1e-5 (never varied) | **NEW** — just assigned Round 15. |

## Active Human Research Directives

**Issue #252 (Modded-NanoGPT):** Mine the Modded-NanoGPT world-record training history for applicable improvements. Techniques ranked by transfer plausibility:
1. Muon optimizer (Newton-Schulz orthogonalized momentum) — would require code change; high priority next free slot
2. Post-attention RMSNorm / QK-norm — stabilizes Transolver slot attention
3. Linear warmdown LR (being approximated by norman #269 + fern #251)
4. U-net skip connections — cheap residual across layers
5. Sequence packing / FlexAttention — throughput lever (more epochs/budget)

## Round 15 Hypotheses (just assigned)

| PR | Student | Key delta | Rationale |
|---|---|---|---|
| #263 | chihiro | `--lr-warmup-epochs 2` | Extend confirmed-winning warmup lever. 1ep won; does 2ep help more? |
| #264 | emma | `--rff-num-features 32 --rff-sigma 1.0` | RFF was +2.04pp in PR #33 but never re-tested on current compound SOTA stack |
| #265 | gilbert | `--ema-decay 0.9995` | Re-test with warmup — PR #194 tested without warmup, regressed; warmup may change interaction |
| #267 | haku | `--grad-clip-norm 0.5` | Grad clip norm = 1.0 has NEVER been varied in ~60 tay experiments |
| #268 | kohaku | `--lr-cosine-t-max 12` | Fills the 9-fail / 12-? / 14-inflight / 50-SOTA cosine sweep |
| #269 | norman | `--lr-cosine-t-max 9 --lr-min 1e-6` + warmup | Warmdown-with-warmup (Modded-NanoGPT directive); distinct from PR #202 (T_max=9 without warmup, regressed) |
| #271 | senku | `--model-hidden-dim 768 --lr 8.2e-5` | Width scaling with muP-scaled LR (vs tanjiro #241 same width, no LR scaling) |
| #272 | violet | `--lr-min 1e-5` | LR floor never varied; probes end-of-schedule sensitivity |

## Current Research Focus

**Primary focus:** LR schedule shape post-warmup. The PR #222 win (warmup=1ep) proved the gradient entry trajectory matters. The current round tests:
- Warmup duration (1ep vs 2ep)
- LR decay shape (flat=SOTA, cosine T_max=8/12/14/9+warmup)
- LR floor (lr_min 1e-5 vs 1e-6)

This completes the LR schedule space. If none of the cosine-anneal variants beat SOTA, the "flat post-warmup LR" conclusion will be firm.

**Secondary focus:** Architecture / representation expansion
- Width scaling (768d, senku with muP-LR vs tanjiro without)
- RFF coordinate features on current SOTA stack (emma)

**Tertiary focus (next round):** Modded-NanoGPT code-change ideas
- Muon optimizer (new PR needed)
- Post-attention RMSNorm/QK-norm (new PR needed)
- These require model.py or train.py code changes — highest-variance bets

## Key Learnings (cumulative)

| Lever | Status | Outcome |
|---|---|---|
| LR | CLOSED | 1e-4 SOTA; ceiling confirmed |
| EMA decay | CLOSED | 0.999 SOTA; all 4 values tested |
| Lion beta2 | CLOSED | 0.99 SOTA; 0.999 regressed badly |
| Lion beta1 | Mostly closed | 0.9 SOTA; 0.8 negative; 0.95+ untested |
| Weight decay | CLOSED | 5e-4 SOTA; both directions negative |
| Vol loss weight | CLOSED | 1.0 SOTA; 1.5 and 2.0 both negative |
| Vol points | CLOSED | 65536 SOTA; 96k regressed |
| Surface points | CLOSED | 65536 SOTA; 96k (PR #206) negative |
| MLp_ratio | CLOSED | 4 SOTA; 6 regressed (+6.4%) |
| Dropout | CLOSED | 0.0 SOTA; 0.05 regressed (+4.24%); model underfits |
| Tau axis weights | CLOSED | 1.0 SOTA; Lion sign neutralizes per-channel weighting |
| model_layers | Closing | 4 SOTA; 3L tracking regression (PR #233) |
| model_heads | Closing | 8 SOTA; 4H tracking regression (PR #232) |
| model_slices | Closing | 128 SOTA; 64 tracking regression (PR #231) |
| lr_cosine T_max=9 | CLOSED NEGATIVE | Confirmed ×2 (edward #195, tanjiro #202) |
| lr_warmup_epochs=1 | NEW SOTA | PR #222 fern — +2.03% val, +1.51% test; warmup confirmed lever |
| batch_size | In-flight | 4 SOTA; 5 (effective) in-flight PR #250 |
| lr-min | Never varied | 1e-5 being tested (violet #272) |
| grad-clip-norm | Never varied | 0.5 being tested (haku #267) |
| RFF features | Re-test needed | +2.04pp in PR #33 but on old SOTA stack; re-test with current SOTA (emma #264) |
| model_hidden_dim | In-flight | 512 SOTA; 768 in-flight (tanjiro #241, senku #271) |
| Cosine T_max 12/14 | In-flight | T_max=14 (thorfinn #247), T_max=12 (kohaku #268), T_max=8 (fern #251) |
| Warmup 2ep | In-flight | Being tested chihiro #263 |

## Largest Remaining Gaps to AB-UPT

1. **volume_pressure** ×2.09 — Not a loss-weighting problem (both 1.5 and 2.0 closed). Likely architectural: dedicated volume decoder, richer SDF features, hierarchical multi-scale volume heads.
2. **tau_y** ×3.38, **tau_z** ×3.54 — Shear stress direction prediction. Lion sign update neutralizes per-axis loss weighting. Needs geometry-informed head (surface tangent frame) or output transformation (asinh normalization).

## Next Priorities (after Round 15 closes)

1. **Muon optimizer** — Newton-Schulz orthogonalized gradient, highest-variance Modded-NanoGPT bet
2. **Post-attention RMSNorm** — stabilize Transolver slot attention, cheap code change
3. **Volume architecture** — dedicated volume decoder head; potentially copy from AB-UPT paper's design
4. **Tau head reform** — surface-tangent-frame projection to close tau_y/tau_z gap
5. **Compound winners** — stack Round 15 winners once identified
