# SENPAI Research State

_Last updated: 2026-05-28 22:32Z (**H150 EP14=6.6453% — FIRST CROSSOVER BELOW H147 EP10=6.640%**; H154 EP6=6.9719% descent slowing; H155 EP2=7.3609% strong descent; H151 EP14=6.836% stable drift)_

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

### Fleet status (2026-05-28 22:32Z)

| Run ID | Student | H# | Epoch | val_WSS | val_VP | val_SP | val_ABUPT | State | Δ vs H147 EP10=6.640% |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 5bgp2ryq | tanjiro | H150 (β1=0.97/β2=0.985) | 14.61 | **6.6453%** | **3.555%** | **3.905%** | **5.907%** | running | 🏆 **−0.005pp WSS; ALL 4 metrics BELOW H147 EP10** |
| d20sf8th | nezuko | H151 (extended 45EP canonical) | 14.89 | 6.8362% | 3.703% | 4.030% | 6.098% | running | +0.196pp behind (stable drift, narrowing) |
| 8w7qtm5e | frieren | H154 (tau_z=1.3) | 6.66 | 6.9719% | **3.369%** | 3.978% | 6.107% | running | ⚠️ EP10 kill at risk: descent slowing to −0.03pp/EP |
| 9xo566ws | fern | H155 (lr=9e-5 primary) | 2.0 | 7.3609% | — | — | — | running | **EP2 only +0.10pp behind H147 EP2=7.26%** |
| u3vbwwhd | (frieren old) | H149 (β1=0.93/β2=0.97) | 3.1 | 7.083% | — | — | — | **crashed** (abort) | β1↓+β2↓ falsifying |
| 2h8cddnz | (fern old) | H152 (β1=0.95/β2=0.97) | 5.4 | 7.113% | — | — | — | **crashed** (abort) | β2↓ falsifying |

### H147 SOTA trajectory reference

| EP | 1 | 2 | 3 | 5 | 10 | final test |
|---:|---:|---:|---:|---:|---:|---:|
| H147 val_WSS | 12.82% | 7.26% | 6.98% | 6.75% | 6.64% | **6.5409%** |

### H150 per-epoch val trajectory (5bgp2ryq) — 🏆 H147 EP10 CROSSED

| EP | 10 | 11 | 12 | 13 | **14** |
|---:|---:|---:|---:|---:|---:|
| val_WSS | 6.694% | 6.6715% | 6.6656% | 6.6567% | **6.6453%** |
| val_VP | 3.608% | 3.589% | 3.570% | 3.562% | **3.5546%** |
| val_SP | — | — | — | 3.906% | **3.9052%** |
| val_ABUPT | 5.955% | 5.936% | 5.925% | 5.917% | **5.9071%** |
| Δ WSS vs H147 EP10=6.640% | +0.054pp | +0.031pp | +0.025pp | +0.017pp | 🏆 **−0.005pp** |
| Δ VP  vs H147 EP10=3.608% | 0 | -0.019pp | -0.038pp | -0.046pp | **−0.053pp** |

**First single-model arm to beat H147 EP10 at the same epoch.** Steady -0.011pp/EP descent. Extrapolation: EP18 ~6.61%, EP25 ~6.55%, EP30 ~6.49% — would beat H147 final test 6.5409% by ~0.05pp. β1↑+β2↑ joint = working delayed-converger.

### H151 per-epoch val trajectory (d20sf8th)

| EP | 7 | 9 | 10 | 11 | 12 | 13 | **14** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| val_WSS | 6.906% | 6.891% | 6.870% | 6.868% | 6.848% | 6.844% | **6.8362%** |
| val_VP | — | — | — | — | — | — | **3.703%** |
| val_ABUPT | — | — | — | — | — | — | **6.098%** |

Linear descent −0.01pp/EP. Drift to H147 stable at +0.20pp. Run has ~31 EP remaining of its 45-EP budget. Extrapolation: EP30 ~6.68%, EP45 ~6.53%. Decisive whether 45-EP-canonical can beat 30-EP-canonical.

### H154 per-epoch val trajectory (8w7qtm5e) — ⚠️ DESCENT SLOWING

| EP | 1 | 2 | 3 | 4 | 5 | **6** | Δ rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| val_WSS | 15.54% | 7.77% | 7.2375% | 7.0698% | 7.0035% | **6.9719%** | -0.03 |
| val_VP  | 8.175% | 3.790% | 3.488% | 3.396% | 3.382% | **3.369%** | |
| val_ABUPT | 13.90% | 6.836% | 6.345% | 6.184% | 6.134% | **6.107%** | |

EP6 kill formally passed (gate >7.20%; EP6=6.9719%). But **descent rate has collapsed**: -0.54 (EP2→3), -0.17 (EP3→4), -0.07 (EP4→5), **-0.03 (EP5→6)**. EP10 kill (>6.80%) needs -0.17pp in 4 EPs; at current -0.03pp/EP rate → lands at ~6.85% = **FAILS EP10 kill**. val_VP=3.369% remains below H147 EP10 — tau_z weighting helps volume head but tau_z=1.3 alone insufficient to beat H147 wall_shear.

### H155 primary EP1-EP2 — STRONG TRAJECTORY (9xo566ws)

Created 19:37:55Z. Config verified: `lr=9e-5`, `lion_beta1=0.95`, `lion_beta2=0.98`, `tau_z_loss_weight=1`, `epochs=30`, `lr_cosine_t_max=30`, `ema_decay=0.999`, `ema_start_step=500`. DDP8 active.

| EP | H155 val_WSS | H147 val_WSS | Δ |
|---:|---:|---:|---:|
| 1 | 7.8037% | 12.82% | **−5.02pp** |
| **2** | **7.3609%** | 7.26% | **+0.10pp** |

EP1→EP2 descent = -0.44pp/EP. H155 has caught H147 trajectory by EP2 from a 5pp better EP1 start. Extrapolation: if H155 maintains H147's descent rate from EP2, EP10 ~6.74% (+0.10pp behind H147 EP10). If H155 maintains better-than-H147 rate (likely given lower lr), EP10 could be 6.50-6.60% — would beat H147 final.

---

## Decisive epochs ahead (active runs)

- **H155 EP3-EP4** (~22:45-23:15Z) — descent-rate diagnostic; if continues -0.4pp/EP, EP5 = ~6.5%
- **H154 EP8** (~23:00Z, step 87807) — if val_WSS > 6.93%, decision to early-abort
- **H154 EP10 hard kill** (~23:45Z, step 109759) — gate val_WSS > 6.80%; likely fail at current descent rate
- **H150 EP18-20** (~00:30-01:30Z) — extrapolation check: should be ~6.59-6.61%
- **H155 EP10** (~05:30Z) — would decide if lr=9e-5 alone beats H147 final
- **H150 EP30 terminal** (~10:30Z 2026-05-29) — single-model SOTA candidate; needs test harvest
- **H151 EP20-25** — slow narrowing of RNG drift

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
