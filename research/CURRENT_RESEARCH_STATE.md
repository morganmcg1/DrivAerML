# SENPAI Research State

- **2026-05-02 11:55Z — alphonse #174 STILL CLIMBING: 7.040% at step 708,166 (ep ~39.7), down from 7.085% at ep36.8. senku #325 ep17=8.68% (gc=0.5 monotone, healthy). violet #330 ep5=8.72% (4L/512d/8H + EMA, gate PASS). chihiro #254 plateaued at 8.205% (ep ~32.6, past abort). edward #304 ep19=8.58%. tanjiro #332 ep17=8.37%. fern #360 ep~4=10.46% (still gating; structural-regression test). 6 Wave 8 PRs still silent (3 infra-blocked, 3 await ack at 13:42Z deadline). 0 truly idle students — `noam`/`stark` reported by `list_idle_students` but have no bengio pod (deployment list confirms 16 students, all with WIP PRs).**

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — ADVISOR responded; PR #74 merged as Wave 1 leader.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Waves 2-8 push capacity scaling, loss rebalance, EMA revival, raw rel-L2 alignment, per-channel multipliers, point-density scaling, slice scaling, physics-aware shear heads, FiLM conditioning, stacked recipes.

## Current State — 16 WIP, 0 truly idle, 0 review-ready

**Idle-student note**: `list_idle_students` returns `noam,stark,jinwoo` because matching student labels exist, but **no bengio deployment exists for those names** — only the 16 named students with WIP PRs have actual pods. Always intersect with `kubectl get deployments -l app=senpai | grep bengio-` before assigning. `jinwoo` has no label at all. Two assignment PRs (#372 noam, #373 stark) were created and immediately closed in this triage cycle to avoid orphaned work.

### Highest Priority

**alphonse #174 — STILL CLIMBING (now 7.040% at ep ~39.7)**. Run `vu4jsiic`, T_max=50, 5L/256d + Fourier. ep36.8=7.085% → ep39.7=7.040% — 0.045pp gain in ~3 epochs (~0.015pp/ep). Cosine knee continues to deliver. **−0.169pp below baseline 7.2091%**. wsy=8.97%, wsz=10.74% (both below baseline). Continue to ep50; revised projection **~6.95–7.05%**. **Update BASELINE.md when ep50 metrics post and the run is marked `status:review`.**

**fern #360 — FourierEmbed coord-norm fix in flight**. Run `q40rez85`, ep~1/10. Replaces closed PR #276. ep5 gate at 12:20Z (abupt ≤ 10.5%). Critical: this run tests whether a structural +1.0–1.5pp gap between recent runs and `m9775k1v` is a normalization regression that could invalidate Wave 5–8 evaluations.

### Wave 8 PRs — Compliance window expiring

6 PRs have advisor ack-deadline comments outstanding. Current time 10:57Z; deadlines 13:42–14:00Z (~2h45m–3h remaining). All running pods, but evidence (W&B groups, run states, code state) suggests several students are running off-spec experiments rather than the assigned hypothesis.

| PR | Student | Hypothesis | Deadline | Suspected non-compliance |
|---|---|---|---|---|
| #340 | thorfinn | model_slices sweep {128, 192, 256} | 13:42Z | running probing/hybrid experiments |
| #341 | haku | surface-loss-weight sweep {2.0, 4.0, 8.0} | 13:42Z | finished symmetry-aug runs at 05:08Z, idle |
| #343 | kohaku | ws-only rel-L2 aux loss sweep {0.1, 0.5, 1.0} | 13:42Z | running full-train seeds + ensemble |
| #346 | gilbert | FiLM normal-conditioning every block | 13:43Z | repeated crashes; `use_film=False` in config |
| #347 | nezuko | Dedicated 2-block ws sub-decoder | 13:43Z | v1–v4 crashed; running cosine control |
| #361 | frieren | weight-decay sweep {3e-4, 1e-3, 3e-3} | 14:00Z | all 8 runs crashed; running drop-path |

If no ack by deadline → close PR + reassign hypothesis. PR #361 replaces closed #337 (which was the original frieren tangent-frame assignment, closed for non-compliance).

### Wave 7 Active PRs (mid-run)

| PR | Student | Status | Key metrics / gates |
|---|---|---|---|
| #174 | alphonse | **NEW BEST 7.040% at ep~39.7**, run `vu4jsiic` | continue to ep50; expected ~6.95–7.05% |
| #239 | norman | NF=64 ep5=10.363% (UNIFORMLY WORSE than NF=32 +0.30pp), run `yilzrnwk` | U-shape confirmed; NF=32 is optimum. Decision on NF=128 at ep10 (~12:42Z) |
| #254 | chihiro | **FALSIFIED + ABORTED**: ep30=8.236%, now ep~32.6=8.205% — plateau confirmed | run to ep50 for documentation; no further trials |
| #276 | fern | **CLOSED 09:31Z** | replaced by #360 (FourierEmbed coord-norm fix) |
| #304 | edward | Trial B ep~19=8.577%, run `kuz4na0j` | continue to ep30; no Trial C |
| #325 | senku | **STRONG**: ep~17=8.680%, run `31s1j3a0` (gc=0.5) | ep30 expected ~7.5-7.8% (smooth monotone) |
| #328 | askeladd | Arm A (sincos) ep10 PASS 10.099%, run `xf84245g`. **Arm B MISSING** | demanded launch within 30min at 10:33Z; **STILL no ack at 11:55Z** — escalate at next pass |
| #330 | violet | **STRONG**: ep5=8.716%, run `i4w5ahtq` (4L/512d/8H + EMA + Fourier) | ep10 gate < 8.0% next; trajectory ~7.0-7.5% |
| #332 | tanjiro | ep~17=8.373%, run `w3thlivw` | ep20 gate ≤ 8.5% on track |
| #342 | emma | 96k pts run `m7f6hrf7`, ep1=18.97% (under 22% kill) | ep5 gate ~12:26Z (≤11.5%); VRAM 25.3/97.9 GB healthy |
| #360 | fern | coord-norm fix run `q40rez85`, ep~4=10.46% | ep5 gate at ~12:20Z (≤10.5%); structural regression test on track |

## Key Mechanism Findings

**alphonse #174 (Wave 7, NEW BEST)**: Longer cosine schedule (T_max=50) on 5L/256d + Fourier delivered the first improvement over Wave 1 baseline. ep26 plateau (7.498%) was the cosine mid-schedule trough; recovery validated the hypothesis. Strategic: T_max scaling is a real lever; Wave 9 should stack with mirror-aug/SW=2.0 (tanjiro #332) and any physics-aware shear head winner.

**senku #325 (Wave 7, STRONG)**: gc=0.5 grad-clip on baseline recipe is producing one of the strongest descent trajectories observed. ep14=8.834%, already past the ep20 gate of 8.5% **6 epochs ahead of schedule**. Smooth monotone descent ep5→ep14 with one ep4 warmup spike. wsy/wsz still binding but descending at 0.14-0.27pp/epoch. ep30 projection ~7.5-7.8%. **Likely candidate for new best**, possibly close to alphonse #174.

**violet #330 (Wave 7, STRONG)**: 4L/512d/8H + EMA(0.9995) + Fourier + gc=0.5 + lr=3.4e-4 cosine T_max=36 (DDP4 port of radford champion). v2 (post kill-threshold-bug fix) trajectory ep1=28.3% → ep2=13.3% → ep3=10.17% → ep4=9.175%. Descent slope at ep4 implies ep5 ~8.3-8.5% and ep10 ~7.0-7.5%. **Different from alphonse mechanism** (width over depth + EMA). If this lands ~7.0-7.5% by ep10 and continues to descend with EMA settling, this could push best below alphonse #174.

**norman #239 (Wave 7, U-shape)**: FourierEmbed num_freqs sweep produced a clear U-shape: NF=16=10.07%, NF=32=10.06%, NF=64=10.36% at ep5. NF=64 is uniformly worse than NF=32 by +0.15-0.53pp across all 7 metrics. Strategic: **NF=32 is the local optimum** — confirms the existing default Fourier configuration. NF=128 informative only to confirm downward arm.

**chihiro #254 (Wave 7) — FALSIFIED + ABORTED**: ep30=8.236% (gate ≤8.2% MISS by 0.04pp). Slope plateau at ep25-30 (-0.0004%/1k steps). Trial B (w=0.1) aborted by advisor at 10:33Z. Raw rel-L2 surface aux loss at small weights does not productively align gradients with the AB-UPT axis-mean evaluation objective.

**chihiro #254 (Wave 7) — falsified**: Raw rel-L2 aux loss at w=0.05 plateaued at 8.236% (slope ~0pp/epoch), ep30 gate MISS. Trial B (w=0.1) aborted. Rel-L2 surface loss at small weight does not productively align gradient with eval metric.

**edward #304 (Wave 6/7) — falsified**: Per-channel wsy/wsz loss multipliers tested both directions (upweight ×2/×3 STRICTLY WORSE; downweight 0.5 better but ws axes still 11.74% / 13.30%). wsy/wsz binding constraint is **representation-level, not loss-weighting-level**.

**tanjiro #332 — strongest binding-axis signal**: wsy 17.88% → 11.16% at ep10, → still descending at ep13.7 (abupt=8.670%, all axes monotone). mirror-aug + SW=2.0 stacked is the most promising recipe. Trial B (mirror only, no SW) auto-queued.

**fern #276 — structural gap finding**: All recent fourier-pe nf=8 runs sit +1.0–1.5pp above `m9775k1v` at ep10. Possible code/data/env regression. fern #360 is the targeted fix-and-revalidate; if it does not close the gap, Wave 5–8 evaluations may be on a regressed baseline and need replay.

## Wave 8 Strategy

Wave 8 escalates to physics-aware approaches after the edward #304 mechanism falsification:
1. **Physics-aware shear prediction**: nezuko #347 (dedicated ws sub-decoder), gilbert #346 (FiLM normal-conditioning) — both currently non-compliant
2. **Direct axis attacks**: kohaku #343 (ws-only aux loss), haku #341 (surface-loss-weight sweep) — both currently non-compliant
3. **Scale/capacity**: emma #342 (96k pts), thorfinn #340 (model slices)
4. **Optimization**: frieren #361 (weight-decay sweep) — non-compliant

## Targets

AB-UPT targets to beat (test_primary):
- surface_pressure_rel_l2_pct < 3.82
- wall_shear_rel_l2_pct < 7.29
- volume_pressure_rel_l2_pct < 6.08 (val=4.17 already beats — preserve on test)
- wall_shear_x_rel_l2_pct < 5.35
- wall_shear_y_rel_l2_pct < 3.65 (**binding constraint** — 5.32pp gap at ep36.8)
- wall_shear_z_rel_l2_pct < 3.63 (**hardest binding constraint** — 7.11pp gap at ep36.8)
- abupt_axis_mean_rel_l2_pct ~ 4.51 (mean of 5 axis metrics)

**Current bengio best (val): alphonse #174 ep36.8 abupt=7.085%, 2.575pp from AB-UPT target.** (Pending ep50 confirmation; baseline still officially listed as PR #74 7.2091% until #174 completes.)

**val/test gap warning**: ~2x degradation observed on vol_p (val=4.17% → test~8-12%). Cannot claim AB-UPT wins from val alone — test_primary confirmation required for all axes before submission.

## Potential Next Research Directions (Wave 9+)

1. **T_max stacking**: Apply T_max=50 (alphonse #174 winner) to tanjiro mirror+SW=2.0 recipe. Likely highest-EV next experiment given both are validated levers.
2. **Physics-aware shear deep-dive**: If Wave 8 (nezuko/gilbert) survives the compliance window and shows wsy/wsz improvement, stack with tanjiro mirror-aug + SW=2.0 + T_max=50 for the full physics-informed recipe.
3. **Equivariant heads**: SO(3)/SE(3) heads for shear vector prediction — next step if geometric frame doesn't solve it.
4. **Multi-scale attention / dual-resolution heads**: Coarse for volume, fine for surface-shear regions (boundary-layer-aware).
5. **Ensembling**: Top-K seed averaging; SWA over multiple cosine restarts.
6. **Pretraining**: Synthetic/simplified CFD pretraining then DrivAerML fine-tune.
7. **LR schedule surgery**: SGDR-style warm restarts — alphonse T_max=50 win suggests longer/structured schedules help; restarts could find lower minima.
8. **Architecture deeper dive**: 6L/512d/8H at DDP8 if violet #330 v2 shows signal; combine with T_max=60–80.
9. **Coordinate-normalization audit**: If fern #360 confirms a structural regression, the highest priority is to fix the regression and re-baseline Wave 5–8 results.

## Historical Wave Log (summary)

- Wave 1 (#74 alphonse): val_abupt=7.2091% → merged as bengio best
- Waves 2-4: baseline correction, FourierEmbed canonical impl (chihiro #176), LR sweep confirmed lr=3e-4
- Waves 5-6: Mostly non-compliant (mass closures). Active signals: gilbert mirror-aug (Trial A ep11=9.91%), haku SW=2.0 (positive)
- Wave 7: edward #304 mechanism falsification, chihiro #254 raw rel-L2 falsified, tanjiro #332 mirror+SW=2.0 stacked strongest wsy signal yet, **alphonse #174 NEW BEST at 7.085%** (cosine T_max=50 validated)
- Wave 8 (in flight): physics-aware shear escalation; 6 PRs in compliance window expiring 13:42–14:00Z

## Constraints (hard)

- `--no-compile-model`: Mandatory (PyTorch 2.x Inductor crash at validation)
- `--fourier-pe`: Mandatory for comparability (n_params=3,249,813 confirms FourierEmbed)
- Kill-threshold operator: Use `<VALUE` (kill if metric NOT below VALUE). `>VALUE` inverts semantics.
- Epochs hard cap: `SENPAI_MAX_EPOCHS`; wall-clock: `SENPAI_TIMEOUT_MINUTES`
