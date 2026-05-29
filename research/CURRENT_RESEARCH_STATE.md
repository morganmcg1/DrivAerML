# SENPAI Research State

_Last updated: 2026-05-29 00:40Z (**H156 PRIMARY LAUNCHED** rank0=ugpyo62a EP0.71; **H150 EP18=6.6363%** descent rate fallen 10x to -0.0012pp/EP — plateau zone, riding to EP21; **H155 EP7.6=7.131%** passed EP5 gate but EP10 projection 7.07% > 6.80% hard kill; H151 EP18.7 stable RNG-baseline)_

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

### Fleet status (2026-05-29 00:40Z)

| Run ID | Student | H# | Epoch | val_WSS | val_VP | val_ABUPT | State | Δ vs H147 EP10=6.640% |
|---|---|---|---:|---:|---:|---:|---|---|
| 5bgp2ryq | tanjiro | H150 (β1=0.97/β2=0.985) | **18.38** | **6.6363%** | 3.555% | **5.897%** | running | 🏆 **-0.0037pp WSS vs H147 EP10; plateau zone (-0.0012/EP)** |
| d20sf8th | nezuko | H151 (extended 45EP canonical) | 18.66 | 6.826% | 3.703% | 6.089% | running | +0.190pp stable drift (RNG baseline confirmed) |
| 9xo566ws | fern | H155 (lr=9e-5 primary) | **7.58** | **7.131%** | 3.488% | 6.203% | running | EP5 gate ✅; EP10 gate at risk (proj 7.07% vs 6.80% kill) |
| ugpyo62a (+7 ranks) | frieren | **H156 (β1=0.97/β2=0.985 + lr=9e-5)** | **0.71** | warm-up | — | — | **running primary (smoke `grdap1rg` passed 00:09Z)** | 🎯 EP1 expected ~00:52Z |
| 8w7qtm5e | (frieren prev) | H154 (tau_z=1.3) | 7.64 | 6.9906% | 3.379% | 6.123% | **CLOSED (PR #1367 22:57Z)** | ⛔ EP7 falsified; volume_p lever preserved |
| u3vbwwhd | (frieren old) | H149 (β1=0.93/β2=0.97) | 3.1 | 7.083% | — | — | — | crashed (abort) | β1↓+β2↓ falsifying |
| 2h8cddnz | (fern old) | H152 (β1=0.95/β2=0.97) | 5.4 | 7.113% | — | — | — | crashed (abort) | β2↓ falsifying |

### H147 SOTA trajectory reference

| EP | 1 | 2 | 3 | 5 | 10 | final test |
|---:|---:|---:|---:|---:|---:|---:|
| H147 val_WSS | 12.82% | 7.26% | 6.98% | 6.75% | 6.64% | **6.5409%** |

### H150 per-epoch val trajectory (5bgp2ryq) — 🏆 PASSED H147 EP10, IN PLATEAU ZONE

| EP | 10 | 13 | 14 | 15 | **18** |
|---:|---:|---:|---:|---:|---:|
| val_WSS | 6.694% | 6.6567% | 6.6453% | 6.6400% | **6.6363%** |
| val_VP | 3.608% | 3.562% | 3.5546% | 3.555% | **~3.555%** |
| val_ABUPT | 5.955% | 5.917% | 5.9071% | 5.9026% | **5.897%** |
| Δ WSS vs H147 EP10=6.640% | +0.054pp | +0.017pp | -0.005pp | 0.000pp | **-0.0037pp** ✅ |
| Δ WSS per-EP | — | -0.009 | -0.011 | -0.005 | **-0.0012** (10x slower) |

**Plateau decision EP18 reading (6.6363%) is in AMBIGUOUS zone:** above harvest threshold (>6.62%) and above ride threshold (≤6.60%). **Decision: ride to EP21** (cosine LR at EP21 ≈ 25% of base; stair-step descent re-acceleration common at deep cosine).

| EP30 extrapolation | val | test (with H147 -0.10pp val→test gap) | Beats H147 6.5409%? |
|---|---:|---:|---:|
| Current rate (-0.0012/EP) | 6.622% | ~6.52% | wins by 0.02pp (below 0.05pp threshold) |
| Re-accelerates to H147 avg (-0.005/EP) | 6.576% | ~6.48% | ✅ wins by 0.06pp |
| Geometric halving | 6.633% | ~6.53% | matches H147 |

**EMA best-val checkpoint at EP18-20 is the safety net** — even at current plateau-rate, the EMA-best harvest should give a defensible single-model dl24 SOTA candidate. Other primary metrics (val_VP 3.555% / val_ABUPT 5.897%) already pass H147 EP10 floors.

### H151 per-epoch val trajectory (d20sf8th)

| EP | 7 | 9 | 10 | 11 | 12 | 13 | 14 | **15** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_WSS | 6.906% | 6.891% | 6.870% | 6.868% | 6.848% | 6.844% | 6.8362% | **6.8306%** |
| val_VP | — | — | — | — | — | — | 3.703% | **3.703%** |
| val_ABUPT | — | — | — | — | — | — | 6.098% | **6.092%** |

Linear descent −0.005pp/EP at EP14→EP15 (matching H150's slowdown). Drift to H147 stable at +0.20pp. Confirms **H150's improvement is genuine β-config gain, not RNG**: same EP15 same descent, only +0.20pp gap. Run has ~30 EP remaining. Extrapolation: EP45 ~6.71% (no longer projected to beat H147 final). Useful as RNG-baseline confirmation arm only.

### H154 per-epoch val trajectory (8w7qtm5e) — ⛔ ABORTED EP7 DESCENT REVERSAL

| EP | 1 | 2 | 3 | 4 | 5 | 6 | **7** | Δ EP6→EP7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_WSS | 15.54% | 7.77% | 7.2375% | 7.0698% | 7.0035% | 6.9719% | **6.9906%** | **+0.019pp** |
| val_WSS_x | 13.41% | 6.66% | 6.236% | 6.122% | 6.067% | 6.045% | **6.072%** | +0.028pp |
| val_WSS_y | 17.76% | 8.92% | 8.127% | 7.857% | 7.755% | 7.695% | **7.686%** | -0.009pp ✅ |
| val_WSS_z | 20.69% | 10.37% | 9.741% | 9.526% | 9.470% | 9.446% | **9.479%** | **+0.033pp** ⚠️ |
| val_VP  | 8.175% | 3.790% | 3.488% | 3.396% | 3.382% | 3.369% | **3.379%** | +0.009pp |
| val_ABUPT | 13.90% | 6.836% | 6.345% | 6.184% | 6.134% | 6.107% | **6.123%** | +0.016pp |

**Falsification at EP7:** WSS_z (the channel tau_z=1.3 was specifically weighting) reversed by +0.033pp — the weighted head saturated. WSS_y (unweighted) still descending = direct evidence the tau_z weighting caused the reversal. Aborted PR #1367 by advisor decision at 22:28Z.

**Mechanistic learning preserved:** tau_z reweighting works as **early-EP volume_p lever** (val_VP reached H147 EP30-territory by EP6=3.369%) but does NOT produce terminal WSS gain on canonical lr=1e-4 stack. Would need lr-floor adjustment (i.e., compound with H155 lr=9e-5) to test full hypothesis.

### H155 primary EP1-EP7.6 — EP5 GATE PASSED; EP10 GATE AT RISK (9xo566ws)

Created 19:37:55Z. Config verified: `lr=9e-5`, `lion_beta1=0.95`, `lion_beta2=0.98`, `tau_z_loss_weight=1`, `epochs=30`, `lr_cosine_t_max=30`, `ema_decay=0.999`, `ema_start_step=500`. DDP8 active.

| EP | H155 val_WSS | H147 val_WSS | Δ | per-EP descent |
|---:|---:|---:|---:|---:|
| 1 | 16.34% | 12.82% | +3.52pp | (warmup-only) |
| 2 | 7.804% | 7.26% | +0.54pp | −8.54pp |
| 3 | 7.361% | 6.98% | +0.38pp | −0.44pp |
| 4 | 7.223% | (~6.87%) | +0.36pp | −0.14pp |
| **7.6** | **7.131%** | (~6.72%) | **+0.41pp** | **−0.026/EP** |

**EP5 kill gate >7.20% PASSED** ✅ (well below). **EP10 hard kill gate >6.80% at risk:** at current −0.026pp/EP rate, EP10 projects to 7.07%, failing by 0.27pp. Would need descent rate to **double** between EP7.6→EP10 for the gate to clear. Possible if cosine deepens significantly but not guaranteed. **Hold position: continue to EP10**; if killed, H155 still provides valuable lr=9e-5 isolation data for H156 compound analysis.

---

## Decisive epochs ahead (active runs)

- **H156 EP1 kill check** (~00:52Z 2026-05-29) — gate >13.5%; expected 8.5-10.5% (rank 0 = `ugpyo62a`)
- **H150 EP21 plateau re-decision** (~02:30Z) — current rate -0.0012/EP; if EP21 maintains, harvest EMA best-val at EP20; if re-accelerates ≥-0.003/EP, ride to EP30
- **H156 EP3 kill check** (~03:30Z) — gate >7.50%
- **H155 EP10 hard kill** (~04:30Z) — gate >6.80%; at current rate proj 7.07% would FAIL; needs descent doubling
- **H156 EP5 kill check** (~06:30Z) — gate >7.05%
- **H156 EP10 soft kill** (~10:30Z) — gate >6.65% (must beat H147 EP10 floor)
- **H150 EP30 terminal harvest** (~10:30Z 2026-05-29) — single-model SOTA candidate; test_WSS vs H147 6.5409%; safety net = EMA best-val checkpoint at EP18-20
- **H151 EP25-30** — RNG-baseline confirmation; useful only as control
- **H156 EP30 terminal harvest** (~21:30Z 2026-05-29) — last in-wave run; must beat H147 by ≥0.05pp to declare win

---

## Potential next research directions (post β-sweep, post H154 falsification)

### H156 (frieren next) — HIGH PRIORITY: β-config + lr-axis compound
**Rationale:** H150-β (β1=0.97/β2=0.985) is the only single-arm that has matched H147 EP10 (val_WSS=6.6400% at EP15). H155-lr (lr=9e-5) delivered the cleanest EP1 ever observed (-5.02pp vs H147 EP1). Both axes orthogonal — compound should produce:
- Better EP1 (lower lr eliminates warmup overshoot)
- Steeper cosine tail (β2=0.985 stable late-LR descent)
- Expected EP30: 6.4-6.5% test_WSS range

Config: `--lion-beta1 0.97 --lion-beta2 0.985 --lr 9e-5` on canonical H147 stack. Single ablation axis vs H155 = β-config; vs H150 = lr.

### If H150 plateau holds at EP18
1. **Harvest test from H150 EP15 EMA best-val checkpoint** — likely single-model dl24 SOTA candidate; val_VP/SP/ABUPT already pass H147 EP10 floors
2. **Capacity tier**: hidden_dim 512→768 or layers 6→8 on canonical stack (orthogonal axis)
3. **Surface point density**: 65k→130k surface points
4. **Architectural tier**: cross-attention surface-volume coupling

### If H155 EP10 wins (val_WSS < 6.80%)
1. **lr grid**: 7e-5, 8e-5, 1.1e-4 around 9e-5 optimum
2. **lr × cosine T_max**: longer T_max with lower lr
3. **lr-9e-5 × β-config compound** (= H156 candidate above)

### Bold tier (if all current arms plateau before deadline)
1. **Flash attention 2 + RoPE PE** replacing STRING PE
2. **Multi-resolution FPS hierarchical processing**
3. **Per-axis WSS Charbonnier loss** reformulation

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
| Wave 41 (H149-H156) | β-grid exhausted; tau_z falsified on canonical lr; lr-axis viable; compound H156 = highest-EV next | TBD |

The gap from current SOTA (6.5409%) to Transolver-3 SOTA (5.85%) is **0.69pp**. β-sweep closed 0pp (H147 stays optimum). Remaining headroom must come from compound mechanism (H156 = β + lr), early-harvest from H150 plateau, capacity/density, or architecture/loss reformulation.

---

## References

- Issue #1056 — active research directive and hard constraints
- BASELINE.md — locked scoreboard with corrected-split results
- PR #1344 — H147 merge commit with full β-attribution analysis
- PR #1359 — H150 β1=0.97/β2=0.985 (tanjiro, running)
- PR #1360 — H151 45-EP extended (nezuko, running)
- PR #1367 — H154 tau_z=1.3 (frieren, CLOSED 22:57Z aborted_descent_reversal_ep7)
- PR #1368 — H155 lr=9e-5 (fern, running)
- PR #1369 — H156 β1=0.97/β2=0.985 + lr=9e-5 compound (frieren, draft 22:58Z) — LAST in-wave run
- `RESEARCH_IDEAS_*.md` — researcher-agent design briefs from prior waves
