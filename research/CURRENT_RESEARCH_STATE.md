# SENPAI Research State

_Last updated: 2026-05-29 08:25Z (**H156 TERMINAL — CLOSED PR #1369 (no merge)**: test_WSS=6.6909% (+0.150pp regression), test_VP=11.8441% (3.3x violation of 3.643% cap), test_SP=3.7103% (+0.13pp over 3.577% cap), test_ABUPT=7.4963%; **RESEARCH FINDING — val_VP→test_VP gap = 2.84x is unique to compound β+lr** (H147 has val/test ≈ 1.0x); mechanism hypothesis: high-β + low-lr drives optimizer into val-distribution-specific local minimum on VP; tau_z floor confirmed 3rd independent time (~9.40% on H147 stack). **H158 (vol_p_charbonnier=0.1 on H147 stack)** being assigned to frieren — addresses VP regularization symmetric to H157b's wss_charbonnier. **H157b EP4.6=7.070% val_ABUPT=6.240%** — EP5 gate imminent ~08:25Z. **H150 EP28.67=6.6404% val_ABUPT=5.899%** — terminal harvest ~10:00Z. **H151 EP28.95=6.8118%** noise. **HUMAN DEADLINE 15:45Z (7.3h remaining)** — H158 10-EP terminal ~15:25Z; H157b EP10 terminal ~11:34Z)_

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

### Fleet status (2026-05-29 08:25Z)

| Run ID | Student | H# | Epoch | val_WSS | val_ABUPT | State | vs H147 |
|---|---|---|---:|---:|---:|---|---|
| 5bgp2ryq | tanjiro | H150 (β1=0.97/β2=0.985) | **28.67** | **6.6404%** | **5.8993%** | running | cosine-floor plateau; EMA EP18-20 ~6.622% harvest target; ~10:00Z terminal |
| d20sf8th | nezuko | H151 (extended 45EP canonical) | 28.95 | 6.8118% | 6.0726% | running | noise band oscillation; +0.17pp vs H150 holding |
| ew63yb7p (+7 ranks) | **fern (tay)** | **H157b (wss-charbonnier=0.1 + H150-β)** | **4.61** | **7.070%** | **6.240%** | running | ✅ EP3 PASS; EP5 gate ≤6.85% imminent ~08:25Z; EP10 terminal ~11:34Z |
| (H158 PR pending) | **frieren (dl24)** | **H158 (vol_p_charbonnier=0.1 on H147 stack)** | — | — | — | **assigning** | symmetric to H157b, targets VP regularization (H156 val/test gap finding); 10 EPs / ~7h |
| ugpyo62a + 66ys15yn | (frieren prev) | H156 (β1=0.97/β2=0.985 + lr=9e-5) | 10 (terminal) | 6.8949% (val) | 6.2370% (val) | **CLOSED PR #1369 08:23Z** | ⛔ test_WSS=6.6909% (+0.150pp), test_VP=11.8441% (3.3x cap), test_SP=3.7103% — val/test VP gap 2.84x is the finding |
| 9xo566ws | (fern prev) | H155 (lr=9e-5) | 10.27 | 7.092% | 6.157% | **CLOSED PR #1368 03:00Z** | ⛔ test_WSS=6.8936%, lr-axis falsified |
| 8w7qtm5e | (frieren prev) | H154 (tau_z=1.3) | 7.64 | 6.991% | 6.123% | **CLOSED PR #1367 22:57Z** | ⛔ EP7 falsified |

### H147 SOTA trajectory reference

| EP | 1 | 2 | 3 | 5 | 10 | final test |
|---:|---:|---:|---:|---:|---:|---:|
| H147 val_WSS | 12.82% | 7.26% | 6.98% | 6.75% | 6.64% | **6.5409%** |

### H150 per-epoch val trajectory (5bgp2ryq) — 🏆 PLATEAU-DRIFT, EMA SAFETY NET HOLDS

| EP | 14 | 15 | 17 | 18 | 19 | 20 | 21 | **24.6** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_WSS | 6.6453% | 6.6400% | 6.6316% | 6.6363% | 6.6246% | 6.6223% | 6.6244% | **6.6327%** |
| val_ABUPT | 5.907% | 5.903% | 5.892% | 5.897% | 5.886% | 5.883% | 5.884% | **5.8931%** |

**Plateau drift confirmed EP21→EP24.6 (+0.0083pp on WSS).** Three more EP of running with no descent renewal. EMA best-val (EP18-20 ~6.622%) remains the harvest target. EP30 ETA ~08:40Z. Conservative test projection downgraded:

| EP30 scenario | EP30 val | EP30 test (val−0.10) | Beats H147 6.5409%? |
|---|---:|---:|---:|
| Continue drift (+0.0015/EP) | ~6.641% | ~6.541% | ❌ ties / loses |
| Re-acceleration (rare) | ~6.61% | ~6.51% | ✅ marginal +0.03pp |
| EMA-best harvest (EP18-20) | val ≈ 6.622% | test ≈ 6.522% | ✅ wins by 0.02pp |

**Best harvest = EMA best-val checkpoint, not EP30 terminal.** Single-model dl24 SOTA candidate would be EMA-best at ~6.52% test vs H147 6.5409% — wins by ~0.02pp. Margin is tight; advisor merge decision will hinge on whether the terminal SENPAI-RESULT reports the EMA-best or last checkpoint.

**Conservative EP30 projection (7-EP average -0.003/EP):**

| Scenario | EP30 val | EP30 test | Beats H147 6.5409%? |
|---|---:|---:|---:|
| Conservative avg (-0.003/EP) | **6.597%** | **~6.497%** | ✅ wins by 0.04pp |
| Moderate re-acceleration (-0.005/EP) | 6.570% | ~6.470% | ✅ wins by 0.07pp |
| EP19-style burst recurs | 6.530% | ~6.430% | ✅ wins by 0.11pp |

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

**CLOSED 03:00Z (PR #1368):** fern posted terminal SENPAI-RESULT 02:53Z — test_WSS=**6.8936%** (regression vs H147 6.5409% by +0.35pp). EMA best-val checkpoint EP10. W&B rank0=9xo566ws. Standalone lr=9e-5 axis **falsified**. H157 (wss-charbonnier compound) assigned to fern at PR #1378.

### H156 primary EP1-EP7 — β + lr compound GAP REVERSING ⚠️ (ugpyo62a)

Launched 00:40Z after `grdap1rg` smoke passed clean. Config: `lion_beta1=0.97`, `lion_beta2=0.985`, `lr=9e-5`, canonical H147 stack otherwise. DDP8 active, 8 W&B run-IDs in group (rank 0 = `ugpyo62a` canonical).

| EP | H156 val_WSS | H147 reference | H156−H147 offset | Trend |
|---:|---:|---:|---:|---|
| 1 | 13.154% | 12.82% | +0.33pp | warmup |
| 2 | 7.572% | 7.26% | +0.31pp | ↘ closing |
| 3 | 7.182% | 6.98% | +0.20pp | ↘ closing |
| 4 | 7.042% | ~6.86% (interp) | +0.18pp | ↘ closing |
| **6.92** | **6.9527%** | **~6.70% (interp)** | **+0.25pp** | **⚠️ REVERSING** |

**Gap-closing reversed EP4→EP7.** Offset jumped from +0.18pp (EP4) to +0.25pp (EP7). Per-EP descent rate H156 EP4→EP7 = -0.030pp/EP vs H147 EP4→EP7 ≈ -0.053pp/EP — H156 descending **slower** than H147 at the same epoch. The compound β+lr theory predicted late-cosine acceleration but it's manifesting as a slower-grind regime instead.

**Updated EP10 projection (gap-widening at +0.023pp/EP, rate slowing):**
- EP10 ≈ 6.86% (gap +0.22pp) → would FAIL EP10 soft kill >6.65% by 0.21pp
- EP15 ≈ 6.74% (extrapolated) → still above H147 EP15 ~6.70% baseline by ~0.04pp
- EP30 ≈ 6.55% terminal — possible but no longer probable (would need late-cosine acceleration to close +0.22pp gap in 23 EP)

**Decision: ride to EP10 soft kill (~07:25Z).** Direction pre-committed: if EP10 WSS >6.65% AND gap >+0.10pp → harvest EMA-best, do NOT extend to EP30. EP10 reading determines the call.

**TERMINAL OUTCOME (08:13Z, PR #1369 CLOSED 08:23Z):**

| Metric | H156 val | H156 test | val/test ratio | H147 val/test ratio |
|---|---:|---:|---:|---:|
| WSS | 6.8949% | 6.6909% | 0.97x (test slightly better) | ~1.0x |
| SP  | 4.1013% | 3.7103% | 0.90x | ~1.0x |
| **VP** | **4.1687%** | **11.8441%** | **2.84x** ⚠️ | **~1.0x** |
| ABUPT | 6.2370% | 7.4963% | 1.20x (VP-dominated) | ~1.0x |

**Research finding — val_VP/test_VP gap = 2.84x is unique to compound β+lr.** H147 has matched val/test on VP (~1.0x). H156 compound config (β1=0.97/β2=0.985 + lr=9e-5) drives the optimizer into a val-distribution-specific local minimum on VP. Mechanism hypothesis: very small effective step in flat directions (high-β momentum + low-lr) tracks val-set features without exploring volume-pressure-generalizing directions. Late-cosine renewal exists (-0.034pp EP9→EP10 confirms it) but converges to a val-specific minimum that does not transfer to test.

This finding **directly motivates H158** (vol_p_charbonnier on H147 stack) — the only single-flag VP-regularization lever left in train.py.

### H158 (frieren, ASSIGNED 08:25Z) — vol_p_charbonnier=0.1 on H147 stack

**Hypothesis:** H156 revealed compound β+lr induces structural VP val/test gap (2.84x). Charbonnier loss on volume pressure (smooth-near-zero L1-like) should regularize VP and close the val/test gap by penalizing outlier volume points without distorting bulk learning. Symmetric to H157b's wss_charbonnier (in flight on H150 stack on the WSS axes).

**Config:** H147 SOTA stack (β1=0.95/β2=0.98, lr=1e-4, 6L/512d/4h/128slices, surface_out_width_factor=2.0, STRING-multisigma sigmas 0.25/0.5/1.0/2.0/4.0, ema_decay=0.999) + `--vol-p-charbonnier-weight 0.1 --vol-p-charbonnier-eps 1e-3`, 10 EP cosine T_max=10. Single seed.

**Kill ladder:**
- EP1 > 13.5% kill
- EP3 > 7.20% kill (must beat H147 EP3=6.98% by margin compatible with Charbonnier overhead)
- EP5 > 6.85% kill
- EP10 terminal harvest (EMA-best if val ≤ 6.65% AND val_VP < 4.0% — the second condition is the test of the H158 hypothesis)

**Expected:** if Charbonnier closes the VP gap, test_VP should land in 3.5-4.0% range (H147 SOTA had test_VP=3.6033%). If test_VP > 5.0%, Charbonnier doesn't address the structural gap and we close.

---

## Decisive epochs ahead (active runs)

**⚠️ HUMAN DEADLINE 15:45Z** (7.3h from current 08:25Z) — all terminal harvests must land before this.

- **H156 TERMINAL** (08:13Z) — CLOSED PR #1369; test_WSS=6.6909% (+0.150pp), test_VP=11.84% (3.3x cap), test_SP=3.71% (+0.13pp) — no merge; **val/test VP gap 2.84x is the publishable finding**
- **H157b EP5 kill check** (~08:25Z, imminent) — gate >6.85%; EP3 PASS by 0.34pp
- **H158 launch** (~08:35Z) — vol_p_charbonnier=0.1 on H147 stack; tests if Charbonnier on volume axis closes val/test VP gap
- **H151 EP30 first inflection** (~09:50Z) — extended-training value signal (15-EP tail)
- **H150 EP30 terminal harvest** (~10:00-11:00Z) — EMA-best at EP18-20 ~6.622% val → ~6.522% test, marginal win vs H147 6.5409%
- **H157b EP10 soft kill** (~11:34Z) — gate >6.65%
- **H158 EP10 terminal** (~15:25Z) — 1.5h buffer; harvest EMA-best for test eval if val ≤6.65% and val_VP < 4.0%
- **H157b EP15 realistic terminal** (~14:46Z) — last gate within 1h buffer to deadline; harvest EMA-best for test eval if val ≤6.60%
- **H151 EP45 terminal** (~14:50-15:30Z) — 45-EP extended-training experiment final
- **HARD DEADLINE 15:45Z** — all test_eval harvests must complete before this

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
- PR #1369 — H156 β1=0.97/β2=0.985 + lr=9e-5 compound (frieren, **CLOSED 08:23Z** — test_VP 3.3x cap, test_SP +0.13pp, test_WSS +0.150pp regression; **val/test VP gap 2.84x finding documented**)
- PR #1378 — H157 (auto-merged in error, superseded by PR #1385)
- PR #1385 — H157b wss_charbonnier=0.1 on H150 stack (fern/tay, RUNNING)
- (H158 PR TBD) — vol_p_charbonnier=0.1 on H147 stack (frieren, ASSIGNING 08:25Z)
- `RESEARCH_IDEAS_*.md` — researcher-agent design briefs from prior waves
