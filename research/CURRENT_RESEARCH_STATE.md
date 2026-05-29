# SENPAI Research State

_Last updated: 2026-05-29 02:19Z (**H150 EP20 RE-ACCELERATION** val_WSS=6.6223% at -0.007/EP — 5x faster than EP18 rate, SOTA-beating projection restored; **H155 EP9=7.109% — EP10 KILL CONFIRMED** (descent stuck -0.003/EP since EP6, will fail 6.80% gate by 0.3pp); **H156 EP2=7.572%** tracks H147 with +0.30pp offset (ABUPT=7.15% concerning); H151 EP20.8 stable +0.190pp drift)_

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

### Fleet status (2026-05-29 02:19Z)

| Run ID | Student | H# | Epoch | val_WSS | val_VP | val_ABUPT | State | Δ vs H147 EP10=6.640% |
|---|---|---|---:|---:|---:|---:|---|---|
| 5bgp2ryq | tanjiro | H150 (β1=0.97/β2=0.985) | **20.56** | **6.6223%** | 3.555% | **5.883%** | running | 🏆 **-0.0177pp WSS vs H147 EP10; RE-ACCEL -0.007/EP** |
| d20sf8th | nezuko | H151 (extended 45EP canonical) | 20.84 | 6.8215% | 3.703% | 6.081% | running | +0.190pp stable drift (RNG baseline confirmed) |
| 9xo566ws | fern | H155 (lr=9e-5 primary) | **9.98** | **7.109%** | 3.488% | 6.176% | running | ⛔ **EP10 KILL TRIGGERED** — descent stuck -0.003/EP since EP6 |
| ugpyo62a (+7 ranks) | frieren | **H156 (β1=0.97/β2=0.985 + lr=9e-5)** | **2.85** | **7.572%** | — | — | **running primary** | EP1=13.15%, EP2=7.57% — tracks H147 +0.30pp offset |
| 8w7qtm5e | (frieren prev) | H154 (tau_z=1.3) | 7.64 | 6.9906% | 3.379% | 6.123% | **CLOSED (PR #1367 22:57Z)** | ⛔ EP7 falsified; volume_p lever preserved |
| u3vbwwhd | (frieren old) | H149 (β1=0.93/β2=0.97) | 3.1 | 7.083% | — | — | — | crashed (abort) | β1↓+β2↓ falsifying |
| 2h8cddnz | (fern old) | H152 (β1=0.95/β2=0.97) | 5.4 | 7.113% | — | — | — | crashed (abort) | β2↓ falsifying |

### H147 SOTA trajectory reference

| EP | 1 | 2 | 3 | 5 | 10 | final test |
|---:|---:|---:|---:|---:|---:|---:|
| H147 val_WSS | 12.82% | 7.26% | 6.98% | 6.75% | 6.64% | **6.5409%** |

### H150 per-epoch val trajectory (5bgp2ryq) — 🏆 RE-ACCELERATION CONFIRMED AT DEEP COSINE

| EP | 10 | 13 | 14 | 15 | 18 | **20** |
|---:|---:|---:|---:|---:|---:|---:|
| val_WSS | 6.694% | 6.6567% | 6.6453% | 6.6400% | 6.6363% | **6.6223%** |
| val_VP | 3.608% | 3.562% | 3.5546% | 3.555% | ~3.555% | **~3.555%** |
| val_ABUPT | 5.955% | 5.917% | 5.9071% | 5.9026% | 5.897% | **5.883%** |
| Δ WSS vs H147 EP10=6.640% | +0.054pp | +0.017pp | -0.005pp | 0.000pp | -0.0037pp | **-0.0177pp** ✅✅ |
| Δ WSS per-EP | — | -0.009 | -0.011 | -0.005 | -0.0012 | **-0.0070** (RE-ACCEL 5x) |

**Stair-step descent CONFIRMED.** Plateau at EP15-18 was transient; cosine deepening to ~30% of base (EP20) re-engaged optimization in a tighter basin. Decision to ride to EP21 paid off — now ride to EP30.

| EP30 extrapolation | val | test (with H147 -0.10pp val→test gap) | Beats H147 6.5409%? |
|---|---:|---:|---:|
| Sustained -0.007/EP | **6.552%** | **~6.45%** | ✅ wins by **0.09pp** |
| Half-rate decay (-0.0035/EP) | 6.587% | ~6.49% | ✅ wins by 0.05pp |
| Re-plateau immediately | 6.622% | ~6.52% | wins by 0.02pp (marginal) |

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

### H155 primary EP1-EP9 — EP10 KILL CONFIRMED ⛔ (9xo566ws)

Created 19:37:55Z. Config verified: `lr=9e-5`, `lion_beta1=0.95`, `lion_beta2=0.98`, `tau_z_loss_weight=1`, `epochs=30`, `lr_cosine_t_max=30`, `ema_decay=0.999`, `ema_start_step=500`. DDP8 active.

| EP | H155 val_WSS | H155 val_ABUPT | per-EP Δ |
|---:|---:|---:|---:|
| 1 | 16.34% | 14.87% | (warmup) |
| 2 | 7.804% | 6.837% | −8.54 |
| 3 | 7.361% | 6.416% | −0.44 |
| 4 | 7.223% | 6.293% | −0.14 |
| 5 | 7.169% | 6.240% | −0.054 (EP5 gate ✅) |
| 6 | 7.134% | 6.208% | −0.035 |
| 7 | 7.131% | 6.203% | −0.003 |
| 8 | 7.112% | 6.179% | −0.019 |
| **9** | **7.109%** | **6.176%** | **−0.003** |

**EP10 hard kill gate (>6.80%): WILL FAIL by ~0.30pp.** Descent stuck at -0.003pp/EP since EP6 — cosine deepening has NOT triggered re-acceleration as it did for H150. The lr=9e-5 effective LR at deep cosine (~5.7e-5) is too low to make further progress.

**Lr-axis falsified standalone:** lr=9e-5 on canonical H147 stack does NOT beat H147 at any epoch. The early-EP catch-up from EP1→EP5 reverses into plateau drift by EP6+.

**Action triggered: kill request to fern (PR #1368 comment).** Awaiting terminal SENPAI-RESULT with EMA best-val test metrics, then close PR + assign H157.

### H156 primary EP1-EP2 — β + lr compound TRACKING H147 +0.30pp (ugpyo62a)

Launched 00:40Z after `grdap1rg` smoke passed clean. Config: `lion_beta1=0.97`, `lion_beta2=0.985`, `lr=9e-5`, canonical H147 stack otherwise. DDP8 active, 8 W&B run-IDs in group (rank 0 = `ugpyo62a` canonical).

| EP | H156 val_WSS | H147 reference | H155 reference | val_ABUPT |
|---:|---:|---:|---:|---:|
| 1 | 13.154% | 12.82% | 16.34% | 13.168% |
| 2 | 7.572% | 7.26% | 7.804% | 7.146% |

**Compound β + lr partially recovers H155's lr penalty:** EP1 only +0.33pp vs H147 (vs H155 +3.52pp), EP2 only +0.31pp vs H147 (vs H155 +0.54pp). The higher-β momentum config compensates for ~85% of the lower-lr early-EP overshoot reduction.

**Trajectory parallel to H147 + 0.30pp offset (if persists):**
- H156 EP3 ≈ 7.29% (expected to PASS EP3 gate >7.50%)
- H156 EP10 ≈ 6.94% (would FAIL EP10 soft kill >6.65% by 0.29pp)
- H156 EP30 ≈ 6.85% val → ~6.75% test (would NOT beat H147 SOTA)

**Risk flag:** ABUPT=7.146% at EP2 is HIGH (H155 EP2 ABUPT=6.837%). The compound may have a worse ABUPT trajectory than its WSS suggests. Will track ABUPT specifically through EP5-10.

**Hold position to EP3 final reading (~03:00Z).** If +0.30pp offset persists, plan to terminate at EP6-10. If H156 closes the gap to H147 by EP3-5, ride further.

---

## Decisive epochs ahead (active runs)

- **H155 EP10 kill triggered** (02:19Z) — PR #1368 awaiting fern's terminal SENPAI-RESULT
- **H156 EP3 kill check** (~03:00Z) — gate >7.50%; expected ~7.29% from +0.30pp H147 offset projection
- **H150 EP21+ ride** (~04:30Z) — sustained -0.007/EP rate → EP30 val=6.55%, test=~6.45% (beats H147 by 0.09pp)
- **H156 EP5 kill check** (~06:00Z) — gate >7.05%
- **H156 EP6 ABUPT trajectory check** (~07:00Z) — kill if ABUPT plateau > H147 EP6 ABUPT + 0.50pp
- **H156 EP10 soft kill** (~10:00Z) — gate >6.65% (must beat H147 EP10 floor)
- **H150 EP30 terminal harvest** (~10:30Z 2026-05-29) — single-model SOTA candidate; test_WSS vs H147 6.5409%; safety net = EMA best-val checkpoint at EP18-22
- **H151 EP25-30** — RNG-baseline confirmation; useful only as control
- **H157 (fern, queued) EP1 kill check** — once fern's H155 terminal lands; gate >13.5%

---

## Potential next research directions (post H155 kill, post H150 re-accel)

### H157 (fern next, QUEUED) — HIGH PRIORITY: WSS Charbonnier auxiliary loss on H150-β stack
**Rationale:** `wss_charbonnier_weight` (train.py:137) is a single-flag lever **never enabled** in any wave. Adds a Charbonnier loss term on WSS axes (default `axes=all`, covers tau_x/tau_y/tau_z surface-channel indices 1:4). Charbonnier is L1-like in the tails but smooth at zero — robust against outlier surface points where WSS is poorly localized. Directly targets the primary metric.

Config: H150 winning β (β1=0.97, β2=0.985) + `--wss-charbonnier-weight 0.5 --wss-charbonnier-axes all`, canonical H147 lr=1e-4, 30 EP cosine.

Test compounds: H150-β-win + direct WSS upweighting. If H150 EP30 hits SOTA, H157 stacks further. If H150 EP30 misses by hair, charbonnier may push H157 across the threshold.

**Kill ladder:**
- EP1 > 13.5% kill
- EP3 > 7.40% kill (must be tighter than H156 since charbonnier should help, not hurt early)
- EP10 > 6.65% soft kill (must beat H147 EP10 floor)
- EP18 plateau gate as in H150

### If H150 EP30 wins (test_WSS < 6.5409%)
1. **Merge winner** + cleanup PR removing H149-H156 abort branches
2. **Compound on H150**: H157 (charbonnier, queued) extends this
3. **Next compound**: try H150-β + EMA decay 0.9995 (slower averaging on the winning config)
4. **Final push**: H150-β + Charbonnier + slower EMA — stacked single-arm

### If H150 EP30 misses by < 0.05pp
1. **Harvest EMA best-val checkpoint** from H150 — likely captured a tighter local optimum
2. **45-EP extension** of H150-β (H151 analog on winning β) — does duration extension help?
3. **H157 Charbonnier** still high-EV — same single-axis test

### If H150 EP30 plateaus above H147
1. **β-grid optimum confirmed at H147 (0.95, 0.98)** — abandon β-axis tuning
2. **Architecture tier**: more sigmas in STRING PE, hidden_dim 512→768
3. **Per-axis WSS reformulation**: Charbonnier on `axes=z` only (tau_z most informative)
4. **Schedule reformulation**: cosine restarts, polynomial decay

### Bold tier (if shutdown ≈ 24h away & nothing landing)
1. **Per-axis WSS Charbonnier** reformulation (z-only)
2. **RoPE PE** replacing STRING PE
3. **Multi-resolution FPS** hierarchical processing

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
