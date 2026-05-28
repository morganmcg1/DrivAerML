# SENPAI Research State

_Last updated: 2026-05-28 20:57Z (H150 EP13=6.6567% steady descent; H151 EP14=6.836%; H154 EP6=7.0035% kill PASSED + val_VP under H147; **H155 EP1=7.8037% — −5pp vs H147 EP1=12.82%**)_

---

## Current research focus: Wave 41 — closing β-grid + tier-shift to tau_z and lr-axis

**Primary target:** Beat test_WSS < 5.85% (Transolver-3 SOTA) without degrading test_VP ≤ 3.643% or test_SP ≤ 3.577%.  
**Current single-model SOTA:** PR #1344 (H147) — test_WSS = **6.5409%** — gap to target = −0.69pp.  
**Hard constraint (Issue #1056):** WSS is the main focus. NO ENSEMBLES. ~1 day until shutdown (2026-05-29).

---

## Most recent human research directive (Issue #1056)

- "test_wss is the main focus now!!" (comment #92)
- "NO MORE ENSEMBLES! Its the lazy route to better results" (comment #17)
- "there are only 2 days left before we shut down this experiment" (~2026-05-29)
- Test VP and SP floors from PR #972 must NOT be regressed: test_VP ≤ 3.643%, test_SP ≤ 3.577%

---

## Wave 41: β-grid disentanglement — COMPLETE (2026-05-28 18:30Z)

### Final β-grid verdict

```
        β1=0.93    β1=0.95    β1=0.97
β2=0.97  H149⛔    H152⛔     —
β2=0.98            H147⭐     H153⛔
β2=0.985           —          H150⏳
```

| Direction | Outcome | Evidence |
|-----------|---------|----------|
| β1↓ + β2↓ | ⛔ falsifying | H149 aborted EP3 val_WSS=7.083% |
| β2↓ pure | ⛔ falsifying | H152 aborted EP5.4 val_WSS=7.113% (β1=0.95, β2=0.97) |
| β1↑ pure | ⛔ destabilizes EP1 | H153 aborted EP1 val_WSS=13.86% (β1=0.97, β2=0.98) |
| β1↑ + β2↑ joint | ⏳ delayed converger | H150 (β1=0.97, β2=0.985) — see below |
| **β-grid optimum** | **⭐ H147 (0.95, 0.98)** | SOTA at 6.5409% test_WSS |

**β-space is fully exhausted.** Only H150 remains as an active β-grid arm and it is a delayed converger.

### Fleet status (2026-05-28 20:57Z)

| Run ID | Student | H# | Epoch | val_WSS | val_VP | val_SP | val_ABUPT | State | Δ vs H147 EP10=6.64% |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 5bgp2ryq | tanjiro | H150 (β1=0.97/β2=0.985) | 13.97 | **6.6567%** | 3.562% | 3.906% | 5.917% | running | **+0.017pp** behind WSS; **VP/SP/ABUPT all below H147 EP10** |
| d20sf8th | nezuko | H151 (extended 45EP canonical) | 14.24 | 6.8362% | 3.703% | 4.030% | 6.098% | running | +0.20pp behind (stable RNG drift) |
| 8w7qtm5e | frieren | H154 (tau_z=1.3) | 5.95 | **7.0035%** | **3.382%** | 3.996% | 6.134% | running | **EP6 kill PASSED; val_VP UNDER H147 EP10** |
| 9xo566ws | fern | H155 (lr=9e-5 primary) | 1.0 | **7.8037%** | — | — | — | running | **EP1 = -5.02pp vs H147 EP1 = 12.82%** |
| u3vbwwhd | (frieren old) | H149 (β1=0.93/β2=0.97) | 3.1 | 7.083% | — | — | — | **crashed** (abort) | β1↓+β2↓ falsifying |
| 2h8cddnz | (fern old) | H152 (β1=0.95/β2=0.97) | 5.4 | 7.113% | — | — | — | **crashed** (abort) | β2↓ falsifying |

### H147 SOTA trajectory reference

| EP | 1 | 2 | 3 | 5 | 10 | final test |
|---:|---:|---:|---:|---:|---:|---:|
| H147 val_WSS | 12.82% | 7.26% | 6.98% | 6.75% | 6.64% | **6.5409%** |

### H150 per-epoch val trajectory (5bgp2ryq)

| EP | 1 | 2 | 5 | 8 | 10 | 11 | 12 | **13** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_WSS | 11.19% | 7.27% | 6.83% | 6.73% | 6.694% | 6.6715% | 6.6656% | **6.6567%** |
| val_VP | 11.72% | 4.89% | 3.80% | 3.64% | 3.608% | 3.589% | 3.570% | **3.562%** |
| val_SP | — | — | — | — | — | — | — | **3.906%** |
| val_ABUPT | 11.28% | 6.67% | 6.10% | 5.99% | 5.955% | 5.936% | 5.925% | **5.917%** |
| Δ WSS vs H147 EP10=6.64% | +4.55pp | +0.62pp | +0.19pp | +0.09pp | +0.05pp | +0.031pp | +0.025pp | **+0.017pp** |

H150 plateau-recovered then resumed steady -0.01pp/EP descent. **val_VP, val_SP, val_ABUPT all already below H147 EP10**. WSS expected to cross H147 EP10 by EP15-16. If linear -0.01pp/EP holds: EP20 ~6.62%, EP30 ~6.55% (below H147 final 6.5409%). This is the most promising β-grid arm.

### H151 per-epoch val trajectory (d20sf8th)

| EP | 7 | 9 | 10 | 11 | 12 | 13 | **14** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| val_WSS | 6.906% | 6.891% | 6.870% | 6.868% | 6.848% | 6.844% | **6.8362%** |
| val_VP | — | — | — | — | — | — | **3.703%** |
| val_ABUPT | — | — | — | — | — | — | **6.098%** |

Linear descent −0.01pp/EP. Drift to H147 stable at +0.20pp. Run has ~31 EP remaining of its 45-EP budget. Extrapolation: EP30 ~6.68%, EP45 ~6.53%. Decisive whether 45-EP-canonical can beat 30-EP-canonical.

### H154 per-epoch val trajectory (8w7qtm5e)

| EP | 1 | 2 | 3 | 4 | **5** |
|---:|---:|---:|---:|---:|---:|
| val_WSS | 15.54% | 7.77% | 7.2375% | 7.0698% | **7.0035%** |
| val_VP  | 8.175% | 3.790% | 3.488% | 3.396% | **3.382%** |
| val_ABUPT | 13.90% | 6.836% | 6.345% | 6.184% | **6.134%** |

**EP6 kill PASSED** (gate >7.20%; EP5.95=7.0035%). Descent rate slowing: -0.54pp/EP (EP2→3), -0.17pp/EP (EP3→4), -0.07pp/EP (EP4→5). Risk of plateau above 6.80% by EP10. **val_VP=3.382% is BELOW H147 EP10=3.608%** — tau_z reweighting lifts volume_p too, surprising positive cross-head effect.

### H155 primary EP1 — DRAMATIC SIGNAL (9xo566ws)

Created 19:37:55Z. Config verified: `lr=9e-5`, `lion_beta1=0.95`, `lion_beta2=0.98`, `tau_z_loss_weight=1`, `epochs=30`, `lr_cosine_t_max=30`, `ema_decay=0.999`, `ema_start_step=500`. DDP8 active.

| Stack | EP1 val_WSS |
|---|---:|
| H147 (lr=1e-4) | 12.82% |
| H150 (β↑↑, lr=1e-4) | 11.19% |
| H154 (tau_z=1.3, lr=1e-4) | 15.54% |
| **H155 (lr=9e-5)** | **7.8037%** ← **−5.02pp vs H147** |

**Cleanest EP1 ever observed on this stack.** Lower lr during warmup avoids the EP1 overshoot. EP1 kill (>17%) passed by 9.2pp. Even with conservative descent rate (half H147's -6.18pp/9EP), H155 EP10 trajectory plausibly lands at 6.0-6.5% — could beat H147 final 6.5409%.

---

## Decisive epochs ahead (active runs)

- **H155 EP3** (~22:15Z, step 32927) — gate val_WSS > 7.80% (trivial pass expected given EP1=7.80%)
- **H154 EP8** (~22:30Z, step 87807) — descent-rate diagnostic; >6.95% suggests plateau forming
- **H150 EP15-16** (~22:50-23:30Z, step 165k-176k) — projected H147 EP10 crossover
- **H154 EP10 hard kill** (~23:30Z, step 109759) — gate val_WSS > 6.80%
- **H155 EP5** (~23:30Z) — gate val_WSS > 7.20%
- **H151 EP20-25** — RNG-drift trajectory check; will compound budget pay off?

---

## Potential next research directions (post β-sweep)

### If H150 plateau holds (likely)
1. **Capacity tier**: hidden_dim 512→768 or layers 6→8 on H147 stack (orthogonal axis)
2. **Surface point density**: 65k→130k surface points on H147 stack
3. **Per-axis WSS Charbonnier**: τ_x/τ_y/τ_z loss reformulation
4. **Architectural tier**: cross-attention surface-volume coupling

### If H154 (tau_z=1.3) wins
1. **tau_z grid**: 1.2, 1.4, 1.5 on H147 stack
2. **Compound**: H154 tau_z + H155 lr (if both win)
3. **tau_y_loss_weight** code add: parallel τ_y axis lever (would need train.py extension)

### If H155 (lr=9e-5) wins
1. **lr grid**: 7e-5, 8e-5, 1.1e-4 on H147 stack
2. **lr × cosine T_max**: longer T_max with lower lr
3. **Compound**: lr=9e-5 + tau_z=1.3 if both win independently

### Bold tier (if all current arms plateau)
1. **Flash attention 2 + RoPE PE** replacing STRING PE
2. **Multi-resolution FPS hierarchical processing**
3. **Physics-guided pretraining** on synthetic RANS solutions

---

## Key technical constraints (canonical H147 config)

```bash
torchrun --nproc_per_node=8 target/train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --batch-size 1 --weight-decay 0.005 \
  --train-surface-points 65000 --train-volume-points 65000 \
  --eval-surface-points 65536 --eval-volume-points 65536 \
  --epochs 30 --lr 1e-4 --lr-cosine-t-max 30 --lr-warmup-epochs 1 \
  --optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
  --ema-decay 0.999 --ema-start-step 500 \
  --model-hidden-dim 512 --model-layers 6 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" --pe-num-features 16 \
  --surface-out-width-factor 2.0
```

**Dataset:** `rawcanon_20260511` (corrected split — old split had +7-8pp artifact)  
**EMA:** eval EMA with decay=0.999, start_step=500 (already wired in train.py)  
**Optimizer:** Lion (NOT Adam) — β changes have outsized impact on early-epoch trajectory  
**DDP8:** 8-GPU distributed training, effective batch=8  

---

## Historical context: How we arrived at H147 SOTA

| Wave | Key finding | Best test_WSS |
|------|-------------|--------------|
| Pre-wave | H39 wider surface_out MLP (factor=2.0) | 6.6506% (PR #1284) |
| Wave 36 (H138-H148) | Disentanglement: β-drift is single driver; wd-drift null; curvature null | 6.5409% (H147) |
| Wave 41 (H149-H155) | β-grid exhausted; tier-shift to tau_z (H154) + lr-axis (H155) | TBD |

The gap from current SOTA (6.5409%) to Transolver-3 SOTA (5.85%) is **0.69pp**. β-sweep closed 0pp (H147 stays optimum). Remaining headroom must come from tier-shift: tau_z weighting (H154), lr-axis (H155), capacity/density, or architecture/loss reformulation.

---

## References

- Issue #1056 — active research directive and hard constraints
- BASELINE.md — locked scoreboard with corrected-split results
- PR #1344 — H147 merge commit with full β-attribution analysis
- PR #1359 — H150 β1=0.97/β2=0.985 (tanjiro, running)
- PR #1360 — H151 45-EP extended (nezuko, running)
- PR #1367 — H154 tau_z=1.3 (frieren, running)
- PR #1368 — H155 lr=9e-5 (fern, running)
- `RESEARCH_IDEAS_*.md` — researcher-agent design briefs from prior waves
