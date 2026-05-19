# SENPAI Research State

- **Date:** 2026-05-19 (latest invocation: 2026-05-19 ~11:55 UTC)
- **Branch:** tay
- **W&B project:** wandb-applied-ai-team/senpai-v1-drivaerml-ddp8
- **Thread share note:** Issue #1056 is shared with another advisor ("dl24") running a parallel fleet on `drivaerml-long-20260504`. The dl24-prefixed students are real but **NOT under tay advisorship** — visible context for cross-pollination only.

## Latest invocation actions (2026-05-19 ~11:55Z) — PR #1204 H55 TAU-Z-LOSS-CURRICULUM killed at EP1 by **6th invocation of the lr-warmup-1 EP1 gate recipe-bug** (50 minutes after H54 same-pattern kill); RELAUNCH directive posted (edward to rerun on same branch with corrected gates); MEMORY entry updated with BATCH-RISK FINDING (H54 and H55 both written in same invocation, both inherited the bug, both fired)

### Headline updates (11:55Z)

1. **PR #1204 edward H55 TAU-Z-LOSS-CURRICULUM v1 killed at EP1** (run `cq1dkiau`, killed at step 10864 with val_abupt 26.35% > 10% gate). Edward's diagnosis sharp: compared to H48 baseline EP1=25.94% (same recipe minus curriculum) — 0.41pp gap is within noise. The τz curriculum is NOT damaging training at EP1; the kill was 100% due to my miscalibrated gate. Curriculum firing correctly (`train/tau_z_loss_weight_curriculum` 5.000→4.500 linear decay verified). 6th invocation of this exact pattern — H54 (PR #1203) was killed 50 min earlier by the same bug at 11:05Z.

2. **Relaunch directive posted to PR #1204** ([comment 4488115172](https://github.com/morganmcg1/DrivAerML/pull/1204#issuecomment-4488115172)) — edward to relaunch on the same branch `edward/h55-tau-z-loss-curriculum` with corrected `--kill-thresholds "32592:val_primary/abupt_axis_mean_rel_l2_pct<7.5,32592:val_primary/surface_pressure_rel_l2_pct<5.5,65184:val_primary/abupt_axis_mean_rel_l2_pct<6.5"`. NO EP1 gate (lr-warmup-1 makes EP1 too noisy). EP3 binding + EP6 intermediate. No code changes needed.

3. **Memory entry `feedback_ep_thresholds_recipe_dependent.md` updated** with BATCH-RISK FINDING: when writing multiple PR bodies in a single invocation, ALL of them inherit the same recipe-bug until one fires. H54 (08:25Z) and H55 (08:55Z) both written before H54's kill alerted me to the bug. Mitigation: pre-flight each PR body separately against memory before pushing assignment commit.

4. **Wave 31 fleet status** — 8/8 WIP, 0 idle, 0 review-ready (H54 v2 awaiting relaunch, H55 v2 awaiting relaunch, both same advisor recipe-bug pattern):
   - **H50 askeladd (PR #1198)** — STRONGEST POSITION; val_abupt **6.227% at EP6.25** (+0.100pp above merge gate), val_VP **3.679%** (+0.036pp above floor — IMMINENT crossing). Terminal ~14:45-15:00Z. **First likely merge of Wave 31**.
   - **H53 tanjiro (PR #1202)** — STRONGEST SLOPE; val_abupt **6.268% at EP4.8**, slope **−0.020 pp/1k** (3.4× steeper than H47, 3.9× steeper than H50). val_VP 3.671% +0.028pp above floor (TIGHTEST). Terminal projection 5.83-6.10% range = strong merge case.
   - **H47 nezuko (PR #1194)** — val_abupt 6.283% at step 62,492, slope decelerating to −0.0041 pp/1k. Terminal ~17:00Z, borderline merge case.
   - **H52 frieren (PR #1200)** — mid-EP4 healthy, mechanism alive std 0.154.
   - **H33 askeladd (PR #1187)** — wait, askeladd is H50 above. Let me restate: H50 askeladd is the COORDSLICE run (PR #1198). The H33 PR was the older SLICEPE one (#1187) — askeladd was reassigned to H50.
   - **H54 alphonse (PR #1203)** — code correct, v1 killed by my recipe-bug 11:05Z, v2 awaiting relaunch.
   - **H55 edward (PR #1204, THIS ENTRY)** — code correct, v1 killed by my recipe-bug 11:12Z, v2 awaiting relaunch.
   - **H56 fern (PR #1205)** — H51-RELAUNCH NPCA+SSFL+slices=192+ema=0.9999, pre-EP1.
   - **H57 thorfinn (PR #1206)** — MULTI-SCALE-RFF-EXPANDED, pre-EP1.

5. **Strategic notes**:
   - **Recipe-bug pattern catalog now 9 items** with #4 (lr-warmup-aware EP1 thresholds) firing TWICE in this invocation cycle (H54 + H55). Pattern #4 has now fired 6 times since 2026-05-15. Memory entry exists; failure mode is batch-writing assignments without pre-flighting each one.
   - **Wave 31 has 3 active merge candidates** progressing to terminal (H47, H50, H53) with orthogonal mechanism classes. If 2+ merge, Wave 32 has a 2-3-way stacking opportunity.
   - **H55 v2 expected behavior**: τz curriculum should produce reduced test_WSS_z by ≥0.3pp from baseline ~9.5%. EP3 reading should show val_abupt 6.5-7.0% (similar to H48 baseline at EP3) and val_WSS_z 9.5-10.0%; if val_WSS_z is lower than baseline at EP3, mechanism is alive.

## Previous invocation actions (2026-05-19 ~11:30Z) — PR #1197 H49 SDORTH-FULL CLOSED as **mechanism-positive null with structural finding** (test mean deflection failed binding gate, all 5 paper-facing metrics DEGRADE vs baseline); THORFINN REASSIGNED H57 MULTI-SCALE-RFF-EXPANDED (PR #1206, encoder freq-band expansion 5σ→8σ / 4 octaves → 7 octaves, mechanism-class FDCE new, recipe-only change, attacks τz axis from encoder side)

### Headline updates (11:30Z)

1. **PR #1197 thorfinn H49 SDORTH-FULL CLOSED** ([close comment 4487236446](https://github.com/morganmcg1/DrivAerML/pull/1197#issuecomment-4487236446)). 14h full 13-ep run, no timeout cut. Terminal verdict: val_abupt **6.221%** FAIL merge gate 6.126% by +0.095pp; test_abupt **6.080%** FAIL baseline 5.844% by +0.236pp; test_SP 3.861% FAIL floor by +0.284pp; test_VP 3.662% FAIL floor by +0.019pp (close); test_WSS 6.981% FAIL baseline by +0.254pp; **binding test τz/τx mean 1.480 FAIL binding gate <1.44** (back in [1.44, 1.55] band attractor). The H46 EP3-test 1.431 reading was a TRANSIENT mid-training value; H49's gradual 13-ep cosine gives the attractor enough anneal time to re-magnetize the mean.

2. **Structural finding — variance-class subclassification** (most important takeaway): the variance-class mechanism splits into TWO sub-classes:
   - **Variance-class-encoder-input — WINS** (H26 NPCA, H31 WALLDIST, H35 NPCA+SSFL — all MERGED). Encoder feature variance translates into lower test error.
   - **Variance-class-decoder-weight — NULL at translation** (H46/H49 SDORTH). Decoder weight init variance produces persistent variance signature (test std 1.75× baseline, 48% cars outside band) but ALL 5 paper-facing axes DEGRADE.
   - Hypothesis: encoder-input variance creates per-token signal heterogeneity that the model CAN aggregate; decoder-weight variance creates output heterogeneity that bypasses the aggregation capacity and only produces per-car ratio dispersion.

3. **Wave 31 mechanism-class taxonomy after H49 closure (7 classes)**:
   | # | Class | Status |
   |---|---|---|
   | 1 | variance-class-encoder-input | ✅ WINS (H26/H31/H35 MERGED) |
   | 2 | variance-class-decoder-sublayer | TBD (H47 V-DEPTH borderline merge) |
   | 3 | variance-class-cp-loss-weight | TBD (H53 second merge candidate) |
   | 4 | shared-capacity-surface | TBD (H54 relaunching) |
   | 5 | mean-shift-class | ❌ null (H48 closed) |
   | 6 | cross-channel-weight-space | ❌ null (H45 closed) |
   | 7 | variance-class-decoder-weight | ❌ null (**H49 closed THIS PR**) |

4. **PR #1206 thorfinn H57 MULTI-SCALE-RFF-EXPANDED assigned** ([PR #1206](https://github.com/morganmcg1/DrivAerML/pull/1206), branch `thorfinn/h57-multiscale-rff-expanded`). **Mechanism class: frequency-domain capacity expansion (FDCE) — NEW**, attacking the τz axis from the encoder side. Recipe change: `--rff-init-sigmas` from `"0.25,0.5,1.0,2.0,4.0"` (5 sigmas, 4 octaves) to `"0.125,0.25,0.5,1.0,2.0,4.0,8.0,16.0"` (8 sigmas, 7 octaves). Pure recipe change, no model.py edits expected (pre-flight smoke check required before full launch in case shape mismatch). Pre-flight aware: NO EP1 gate (lr-warmup-1); EP3 binding `32592:val_abupt<7.5 + val_SP<5.5`; EP6 hard kill `65184:val_abupt<6.5`. Binding mechanism gate: test_WSS_z reduction by ≥0.3pp from baseline ~9.5%. Mechanism orthogonality: all 7 in-flight Wave 31 PRs attack different axes (slice PE / NPCA aug / loss weight / decoder depth / τz curriculum / NPCA stack / nothing on encoder freq band).

5. **Wave 31 fleet status** — 8/8 WIP, 0 idle, 0 review-ready (after H57 thorfinn assignment):
   - **H47 nezuko (PR #1194)** — STRONGEST MERGE CANDIDATE; latest val_abupt 6.309% at step 56,147 (+0.183pp above merge gate). Slope decelerating −0.0118 → −0.0058 pp/1k. Borderline merge case at terminal.
   - **H53 tanjiro (PR #1202)** — SECOND MERGE CANDIDATE; EP4 val_abupt 6.433%, val_SP 4.149% fleet-leading, val_VP 3.787% tightest +0.144pp above floor. Mechanism orthogonal to H47.
   - **H52 frieren (PR #1200)** — mid-EP4 healthy, mechanism ALIVE std 0.154.
   - **H33 askeladd (PR #1187)** — mid-EP6 SLICEPE v2, primary mech falsified at this point but val_VP descent continues.
   - **H54 alphonse (PR #1203)** — code correct, v1 killed by my recipe-bug at EP1; v2 launching with corrected gates.
   - **H55 edward (PR #1204)** — TAU-Z-LOSS-CURRICULUM, pre-EP1.
   - **H56 fern (PR #1205)** — H51-RELAUNCH NPCA+SSFL+slices=192+ema=0.9999, pre-EP1.
   - **H57 thorfinn (PR #1206, this entry)** — MULTI-SCALE-RFF-EXPANDED, EP1 launch when student picks up.

6. **Strategic notes (post-H49 structural finding)**:
   - **Wave 32 attack-map narrowing**: drop test-mean-deflection axis (H46/H49 transient signal, not a Wave 32 binding target). Drop decoder-weight variance perturbation (mechanism-positive null at translation). Focus on (a) encoder-input variance [proven], (b) encoder freq-band [H57 new], (c) decoder-sublayer depth [H47 proven borderline], (d) per-axis loss curricula [H55 in flight], (e) loss-weight reallocation [H53 in flight].
   - **τz remains the dominant bottleneck** across all Wave 31 PRs (test_WSS_z 9.0-9.9% vs τx 6.1-6.5%, 50% gap unchanged). H55 attacks τz from loss side; H57 attacks τz from encoder freq-band side — orthogonal mechanisms, may stack in Wave 32.
   - **Three-mechanism Wave 32 stack candidate**: H35 (encoder feature variance) + H47 (decoder sublayer depth) + H57 (encoder freq-band) — three orthogonal mechanism classes; all could land additively if H47 and H57 each cross merge gate.

## Previous invocation actions (2026-05-19 ~11:05Z) — PR #1203 H54 SURFACE-DEEP killed at EP1 by **8th recipe-bug** (5th invocation of EP1-gate-too-tight-under-lr-warmup-1 pattern); RELAUNCH directive posted, alphonse to rerun on same branch with corrected gates (no new PR; code is correct)

### Headline updates (11:05Z)

1. **PR #1203 H54 SURFACE-DEEP killed at EP1** (run `lmhzak2l`, group `wave31_h54_surface_deep`, 2.0h runtime, finished at step 10,864 with val_abupt=28.49% > 9.5% gate). **Advisor's recipe-bug**: I wrote `10864:val_primary/abupt_axis_mean_rel_l2_pct<9.5` in the PR body. Under `--lr-warmup-epochs 1`, EP1 cold-start val_abupt is structurally guaranteed to land 25-35% — any gate value below ~35 will false-kill a healthy run. **Alphonse's implementation (commit `8c6bd64`) was correct** — surf_deep_blocks with zero-init residuals, identity-at-init bit-exact baseline, 8-rank DDP synchronous, nonfinite=0. Only the gate value was wrong.

2. **Relaunch directive posted to PR #1203** ([comment 4487132006](https://github.com/morganmcg1/DrivAerML/pull/1203#issuecomment-4487132006)) — alphonse to relaunch on the same branch with corrected `--kill-thresholds "32592:val_primary/abupt_axis_mean_rel_l2_pct<7.5,32592:val_primary/surface_pressure_rel_l2_pct<5.5,65184:val_primary/abupt_axis_mean_rel_l2_pct<6.5"`. NO EP1 gate (lr-warmup-1 makes EP1 too noisy). EP3 binding + EP6 intermediate. No code changes needed — same `alphonse/h54-surface-decoder-depth` branch.

3. **Memory entry `feedback_ep_thresholds_recipe_dependent.md` updated** with RED-FLAG PATTERN section: if a `--kill-thresholds` spec contains `10864:val_abupt<N` for N < 30 under `--lr-warmup-epochs 1`, that's a recipe-bug. Added pre-flight checklist for future PR body kill threshold specs. This is the 5th invocation of this exact pattern (askeladd H33, edward H34, nezuko #1113, alphonse H54, plus one prior). The memory has existed since 2026-05-15; failure mode is not consulting it when writing new PR bodies.

4. **Wave 31 fleet status** — 8/8 WIP, 0 idle, 0 review-ready:
   - **H47 nezuko (PR #1194)** — STRONGEST MERGE CANDIDATE; latest val_abupt 6.309% at step 56,147 (+0.183pp above merge gate). Slope decelerating from −0.0118 → −0.0058 pp/1k. Borderline merge case at terminal (conservative 6.25% / most-likely 6.05-6.18% / optimistic 5.82%).
   - **H53 tanjiro (PR #1202)** — SECOND MERGE CANDIDATE; EP4 val_abupt 6.433% (+0.307pp above merge gate, tied with H47), val_SP 4.149% fleet-leading, val_VP 3.787% tightest in fleet (+0.144pp above floor). Mechanism orthogonal to H47.
   - **H52 frieren (PR #1200)** — mid-EP4 step 43,747 last check (09:35Z); val_abupt 6.901% descending; mechanism ALIVE (std 0.154 just past threshold at EP3).
   - **H50 askeladd, H49 thorfinn** — in-flight at various stages.
   - **H54 alphonse (PR #1203, this entry)** — code correct, awaiting relaunch with fixed gates; v2 EP1 launch ~12:00-12:30Z.
   - **H55 edward (PR #1204)** — TAU-Z-LOSS-CURRICULUM, assigned 08:55Z. EP1 read ~10:30Z (still pre-EP1 from alphonse's pace pattern).
   - **H56 fern (PR #1205)** — H51-RELAUNCH, assigned 10:05Z. EP1 read ~11:30Z.

5. **Strategic notes**:
   - **Recipe-bug pattern catalog now 8 items**: (1) flag existence + format, (2) step-indexed thresholds, (3) EMA δ^N composition, (4) lr-warmup-aware EP1 thresholds (RECURRING), (5) SENPAI-RESULT placeholder format, (6) step = exact epoch×steps_per_epoch, (7) mid-epoch mini-validation cadence, (8) (pending if pattern #4 re-fires).
   - **H47 vs H53 stacking thesis**: both tied on val_abupt at EP4; if both close to terminal val_abupt < 6.126%, they stack additively in Wave 32 (architectural depth + loss-weight reallocation, orthogonal mechanisms).
   - **H54 v2 expected behavior**: surf_deep block norms should grow EP2→EP3 mirroring H47's vol_deep growth (+26-57% across attn_proj and ffn_fc2); val_abupt should land 6.5-7.0% at EP3 if mechanism is alive on surface side.

## Previous invocation actions (2026-05-19 ~10:05Z) — PR #1199 H51 NPCA+SSFL+slices192+ema9999 CLOSED as **RECIPE-BUG CLOSURE** (advisor's fault: EMA-aware kill gate calibration error + kill-threshold step alignment off-by-2; mechanism was activating — τz/τx std doubled EP2→EP4 0.073→0.117; killed mid-EP4 by stale gate <10% under ema=0.9999 EMA-shadow lag) + FERN REASSIGNED H56 H51-RELAUNCH (PR #1205, exact same recipe with corrected EMA-aware kill gates 32592:abupt<25 / 43456:abupt<15 / 65184:abupt<7+SP<5; isolates dual hypothesis arms slices=192 + ema=0.9999 under properly calibrated gates) + 2 NEW RECIPE-BUG PATTERNS catalogued (Pattern #6: kill-threshold step = exact epoch×steps_per_epoch; Pattern #7: mid-epoch mini-validation cadence interaction) — recipe-audit checklist now 7 patterns

### Headline updates (10:05Z)

1. **PR #1199 fern H51 CLOSED as recipe-bug closure** ([close comment 4486642972](https://github.com/morganmcg1/DrivAerML/pull/1199#issuecomment-4486642972)). 6.5h run died mid-EP4 step 38,027 by my too-tight EP3 gate `<10%` under ema=0.9999. **Trained model was healthy** (train/epoch_loss matched H35 step-for-step). **Variance-class mechanism was activating** (std EP2→EP4 doubled 0.073 → 0.117). The kill was a **recipe bug, not a hypothesis failure**. W&B run `2vlx68f9` preserved as variance-class activation reference.

2. **PR #1205 fern H56 H51-RELAUNCH assigned** — exact same recipe (NPCA + SSFL + slices=192 + ema=0.9999) with EMA-aware kill gates per fern's HIGH priority follow-up suggestion. Step values corrected to exact `epoch × 10,864` boundaries: EP3 32,592 / EP4 43,456 / EP6 65,184 (not the off-by-+2 / +44 values I had been quoting). New gate schedule: EP3 catastrophic-only `<25%`, EP4 intermediate `<15%`, EP6 binding `<7.0% + SP<5.0%` (original spec intent). First fully informative read at EP6 with δ^step=0.0015.

3. **2 NEW recipe-bug patterns catalogued** (now 7 total): (#6) kill-threshold step must equal exact `epoch × steps_per_epoch`, NOT memorized constants from prior recipes that had different config. (#7) `--validation-every 1` fires mid-epoch mini-validations — kill-threshold check at step ≥ N triggers at the FIRST validation past N, which can be a mid-epoch mini-validation. Memory entry `feedback_kill_thresholds_step_indexed.md` corrected.

4. **Wave 31 status** — 8/8 WIP, 0 idle, 0 review-ready after H56 assignment:
   - **H47 nezuko (PR #1194)** — STRONGEST MERGE CANDIDATE; EP6 val_abupt 6.357% (only +0.230pp above merge gate). Terminal expected ~17:00Z, projected 6.10-6.15%.
   - **H52 frieren (PR #1200)** — mid-EP4 step 43,747 last check (09:35Z); val_abupt 6.901% descending; mechanism ALIVE (std 0.154 crossed threshold at EP3); +0.27pp gap to H44 shape tightening. EP6 gate ~14:00Z under revised pace.
   - **H50 askeladd, H53 tanjiro, H49 thorfinn** in-flight at various stages.
   - **H54 alphonse SURFACE-DEEP** (PR #1203, assigned 08:25Z) — mirror of H47 V-DEPTH on surface decoder side; EP1 read ~10:00Z.
   - **H55 edward TAU-Z-LOSS-CURRICULUM** (PR #1204, assigned 08:55Z) — time-varying loss weight curriculum, mechanism-class-novel; EP1 read ~10:30Z.
   - **H56 fern H51-RELAUNCH** (PR #1205, assigned 10:05Z) — variance-class capacity expansion under corrected gates; EP1 read ~11:30Z.

5. **Strategic notes**:
   - **Recipe-audit checklist now 7 patterns** — pre-flight every new assignment with: (1) flag existence + format, (2) step-indexed thresholds, (3) EMA δ^N composition, (4) lr-warmup-aware EP1, (5) SENPAI-RESULT placeholder format, (6) step = exact epoch×steps_per_epoch, (7) mid-epoch mini-validation.
   - **Per-car / aggregate decoupling** (H48 finding) — instrument all future PRs to report both per-car τz/τx AND aggregate WSS_z/WSS_x ratio; the aggregate is what drives val_abupt.
   - **Mechanism-class taxonomy** — avoid stacking mean-shift mechanisms (H48-style) in Wave 32; stack variance-class (H35, H56) + shared-capacity-class (H47, H54 if alive) instead.
   - **H56 mechanism prediction**: by EP6 (first fully informative read), τz/τx std target 0.15-0.20 and val_abupt target 6.0-7.0%; if both achieved, this is the second variance-class merge candidate in Wave 31 (alongside H47 V-DEPTH).

## Previous invocation actions (2026-05-19 ~08:55Z) — PR #1196 H48 TAU-Y-EQUALIZE TERMINAL CLOSED (mechanism-positive null, val_abupt 6.485% / test_abupt 6.167% FAIL by 35.9/32.3 bp; MOST EXTREME mean-shift attractor break in W30/31 history — per-car τz/τx mean 0.401 = 25× more extreme than H46 SDORTH; **MEAN-SHIFT CLASS** now 5th observed mechanism class) + EDWARD REASSIGNED H55 TAU-Z-LOSS-CURRICULUM (PR #1204, mechanism-class-novel time-varying loss weight, front-load τ_z 5.0→2.0 by EP6, hold through EP13)

### Headline updates (08:55Z)

1. **PR #1196 edward H48 TAU-Y-EQUALIZE CLOSED** ([close comment 4486160052](https://github.com/morganmcg1/DrivAerML/pull/1196#issuecomment-4486160052)). Terminal EP10 EMA best-ckpt: val_abupt **6.485%** fails merge gate 6.126% by 0.359pp; test_abupt **6.167%** fails baseline 5.844% by 0.323pp. Test_VP 3.671% near-tie at floor (+0.028pp). All other binding gates fail (val_SP +0.700pp, test_SP +0.321pp). **MEAN-SHIFT MECHANISM EXTREME**: per-car τz/τx mean **0.401** val / **0.420** test (−1.04 / −1.02 below [1.44, 1.55] band lower edge), n_outside 34/34 val + 50/50 test — 25× more extreme than H46 SDORTH's prior record (−0.04). Per-axis WSS hardness **INVERTED vs hypothesis**: τ_x became EASIEST (6.437% val_WSS_x) and τ_z became HARDEST (9.899% val_WSS_z); model redirected gradient capacity to τ_x not τ_z when τ_y was loosened. STRUCTURAL FINDING: aggregate WSS_z/WSS_x ratio stayed in band (1.538 val) even when per-car ratios extreme — **per-car / aggregate decoupling** is the underlying signal pattern; val_abupt depends on aggregate channel not per-car. **Mean-shift class formally cemented as 5th mechanism-class observation** alongside variance-class encoder, variance-class decoder sublayer, cross-channel weight-space, shared-capacity surface.

2. **PR #1204 edward H55 TAU-Z-LOSS-CURRICULUM assigned** ([PR #1204](https://github.com/morganmcg1/DrivAerML/pull/1204), branch `edward/h55-tau-z-loss-curriculum`). **Mechanism class: time-varying loss weight curriculum (NEW)**, distinct from H48 static τ_y reduction, H53 static cp upweight, and existing GradNorm dynamic balancing. Attacks the test_WSS_z gap from the opposite direction of H48: instead of *relaxing* τ_y statically, **front-load τ_z early** in training (high attention to τ_z patterns during the high-LR backbone-shaping phase EP1-3), then linearly decay to standard 2.0 by EP6 end (step 65228), hold through EP13. Conceptually a learning-rate warmup on loss-weight space. Schedule values: start=5.0, end=2.0, decay_steps=65228 (EP6 end on the standard 13-ep 18h recipe). Edward needs to add 3 new flags: `--tau-z-loss-weight-start`, `--tau-z-loss-weight-end`, `--tau-z-loss-weight-decay-steps`, and per-step modify `surface_channel_weights[3]` (τ_z slot, train.py:754) with linear interpolation. EP3 binding gate val_abupt<7.5%; EP1 elevated kill threshold val_abupt<10% (high tau_z front-load may temporarily elevate abupt).

3. **Memory entry filed** at `feedback_senpai_result_template_placeholders.md`. BOTH alphonse (#1192 H45) and edward (#1196 H48) independently flagged the same issue this invocation cycle: my advisor check-in SENPAI-RESULT template lines containing `<best_val_abupt>` angle-bracket placeholders broke the student's `mark_ready_for_review` JSON parse guard. Same recipe-bug pattern, second confirmation in one invocation. Memory now lists 5 recipe-audit patterns: (1) flag existence+format, (2) step-indexed thresholds, (3) EMA-step δ^N composition, (4) lr-warmup-aware EP1 thresholds, (5) SENPAI-RESULT template placeholders. Going forward: replace angle-bracket placeholders with concrete numeric placeholders (`"value":0`) or use prose to describe the template instead of pasting a code-block scaffold.

4. **Wave 31 status** — 8/8 WIP, 0 idle, 0 review-ready:
   - **H47 nezuko EP6** read at 08:20Z showed val_abupt 6.357% (+0.230pp above merge gate). Slope EP5→EP6 −0.0118 pp/1k projects terminal at 6.10-6.15%. **Strongest merge candidate of Wave 31**. Next check-in EP7-EP8.
   - **H54 alphonse** (assigned 08:25Z) — earliest mechanism diagnostic ~1.6h after pickup (mirror of H47 V-DEPTH on surface decoder side).
   - **H55 edward** (assigned 08:55Z) — earliest informative read EP1 end ~1.5h after pickup; mechanism diagnostic per-axis WSS at EP3 firing (step 32594).
   - **H50 askeladd**, **H51 fern**, **H52 frieren**, **H53 tanjiro**, **H49 thorfinn** in-flight (various stages).

5. **Strategic notes for next invocations**:
   - **H47 V-DEPTH terminal** expected ~17:00Z — if it crosses merge gate, it becomes Wave 31's first merge.
   - **Mean-shift class avoidance**: don't stack mean-shift mechanisms in Wave 32 — they produce stable equilibria far from band but val null. Stack variance-class (H35-style) and shared-capacity-class (H47 V-DEPTH, H54 SURFACE-DEEP if it lives) mechanisms.
   - **Per-car / aggregate decoupling** is the new signal pattern to instrument: future PRs should report both per-car τz/τx and aggregate WSS_z/WSS_x ratio, since the aggregate is what drives val_abupt.

## Previous invocation actions (2026-05-19 ~08:25Z) — PR #1192 H45 CROSSCHAN-DEC CLOSED (mechanism-positive null, val_abupt 6.3523% / test_abupt 6.0751% FAIL by 22.6/23.1 bp) + ALPHONSE REASSIGNED H54 SURFACE-DEEP (#1203, mirror of H47 V-DEPTH on surface decoder) + H47 EP6 STRONG (val_abupt 6.357% only +0.230pp above merge gate)

### Headline updates (08:25Z)

1. **PR #1192 alphonse H45 CROSSCHAN-DEC CLOSED** ([close comment 4486017957](https://github.com/morganmcg1/DrivAerML/pull/1192#issuecomment-4486017957)). Terminal val_abupt 6.3523% fails merge gate 6.126% by 22.6 bp; test_abupt 6.0751% fails baseline 5.844% by 23.1 bp. **MECHANISM-POSITIVE NULL**: weight-space rank-decoupling (τz/τx out_weight_norm) peaked at 24.16 (18.6× the 1.3 PASS threshold) but val space didn't translate. Structural finding logged in EXPERIMENTS_LOG.md: band attractor is NOT in surface decoder pre-projection cross-channel structure.

2. **PR #1203 alphonse H54 SURFACE-DEEP assigned** — mirror of H47 V-DEPTH (in-flight merge candidate) on the surface decoder side. Add 2 dedicated transformer blocks before `surface_out` projection. Mechanism-class-novel vs H21 (per-channel split) and H45 (cross-channel attention). The H45 mechanism-positive null narrows the search: capacity expansion BEFORE the projection (shared-capacity-class) is the natural next axis on the surface side. Mechanism orthogonality with H47 — if both merge, they stack additively in Wave 32. Recipe identical to H47 V-DEPTH with `--use-surf-deep --surf-deep-num-blocks 2`. EP3 gate val_abupt<7.5% AND val_SP<5.5% AND surf_deep block norms growing.

3. **H47 V-DEPTH EP6 strong read** (concurrent with this invocation, queried 08:20Z): val_abupt **6.357%** (down from EP5 6.421%, descent of 0.064pp); val_SP **4.148%**; val_vol_p **3.932%**. **Only +0.230pp above merge gate 6.126%**. Step 50,941 in EP7 leg. Slope EP5→EP6 = −0.0118 pp/1k. Trajectory continues to support merge-candidate-at-terminal-EP13 hypothesis (terminal projection 6.1-6.3% range; if EP7-EP13 keeps even half the EP5→EP6 slope, terminal lands ~6.10-6.15%).

4. **H48 TAU-Y-EQUALIZE near terminal** (~10 min from invocation start) — plateau confirmed at 6.485%, fails merge gate by 0.359pp, will close as mean-shift mechanism class structural finding.

5. **Fleet 8/8 WIP, 0 idle (alphonse reassigned), 0 review-ready.** Updated priority by next check-in:
   - **H48 edward terminal** ~08:17-08:30Z (close-incoming, mean-shift class confirmed)
   - **H47 nezuko EP7** ~step 52,522 ~09:00Z — merge candidacy trajectory continuation
   - **H52 frieren EP3** firing now-ish 07:50-08:00Z (variance-class ALIVE threshold check)
   - **H53 tanjiro EP3** ~08:20Z (SP-gate confirmation)
   - **H51 fern EP3** ~08:30Z (first informative EMA read, mechanism class confirmation)
   - **H50 askeladd EP6** ~10:15Z — make-or-break gate
   - **H54 alphonse EP1** ~est launch + 1.6h after alphonse picks up — earliest mechanism diagnostic
   - **H47 nezuko terminal** ~17:00Z (likely merge candidate)
   - **H49 thorfinn terminal** ~17:30Z
   - **H53 tanjiro terminal** ~22:30Z

## Previous invocation actions (2026-05-19 ~08:10Z) — EDWARD H48 TAU-Y-EQUALIZE EP12 CHECK-IN: **PLATEAU CONFIRMED — TERMINAL ~7 MIN AWAY, NO MERGE** (best val_abupt 6.485% at EP10 fails merge gate 6.126% by 0.359pp; first Wave 31 example of MEAN-SHIFT MECHANISM CLASS finding)

### Headline updates (08:10Z)

1. **PR #1196 edward H48 EP12 check-in posted** ([comment 4485746788](https://github.com/morganmcg1/DrivAerML/pull/1196#issuecomment-4485746788)). Run `8cn5abxm` healthy at step 69,855 (mid-EP13 leg, 13.74h runtime, heartbeat fresh 08:10Z, nonfinite=0). **PLATEAU FIRMLY ESTABLISHED**: val_abupt is in 6.485-6.508% band across EP7→EP12 (a 0.023pp range over 6 epochs). Best val_abupt = 6.485% at EP10 (step 62,501) — **fails merge gate 6.126% by 0.359pp**. EP10→EP12 slight uptick (+0.003pp val_abupt, +0.007pp val_SP) is the standard mild-overfitting signal. EP13 terminal at step 70,636 expected ~08:17Z (~7 min away).

2. **Best metrics at EP9-EP10 (likely best EMA ckpt for test eval)**:
   - val_abupt: 6.485% (EP10) — merge gate fail by 0.359pp
   - val_SP: 4.277% (EP9-EP10) — +0.70pp above test_SP floor 3.577% (binding gate fail)
   - val_vol_p: 3.802% (EP9) — +0.16pp above test_vol_p floor 3.643% (tight but not crossing)
   - val_WSS: 7.351% (EP10-EP11) — +0.62pp above baseline 6.727%

3. **Structural finding: MEAN-SHIFT mechanism class established (cross-experiment value)**:
   - H48 confirmed as first Wave 31 example of mean-shift mechanism class: τz/τx mean stuck at 0.401 (below band [1.44, 1.55]) with std 0.033 compressed and n_outside 34/34 across all reported epochs
   - **Key Wave 31 mechanism-class taxonomy point**: mean-shift mechanisms produce stable equilibria that don't translate to val_abupt headroom, while variance-class mechanisms (H35 NPCA+SSFL ref, H47 V-DEPTH sublayer growth, H52 NPCA×YAW activating, H35 ALIVE-class) do
   - Cross-experiment finding worth flagging in Wave 32 stacking proposals — avoid stacking with other mean-shift mechanisms

4. **PR outcome decision**: Will close after terminal SENPAI-RESULT with test metrics. The structural mechanism finding is the valuable artifact, not a metric win. Closing reason: "Plateau confirmed at EP7+ — best val_abupt 6.485% fails merge gate 6.126% by 0.359pp; first Wave 31 confirmation of mean-shift mechanism class (τz/τx mean 0.401 stable below band, std compressed); taxonomy finding to inform Wave 32 stacking proposals."

5. **Fleet 8/8 WIP, 0 idle, 0 review-ready.** Active runs in priority by next check-in:
   - **H48 edward terminal** ~08:17Z (step 70,636) — terminal SENPAI-RESULT incoming, will close PR
   - **H47 nezuko EP6** ~08:15Z (step 48,897) — MERGE-GATE-CROSSOVER decision (concurrent firing)
   - **H52 frieren EP3** ~07:50-08:00Z (firing now) — std ALIVE threshold check
   - **H53 tanjiro EP3** ~08:20Z — SP-gate confirmation
   - **H51 fern EP3** ~08:30Z — first informative EMA read
   - **H50 askeladd EP6** ~10:15Z — make-or-break
   - **H47 nezuko terminal** ~17:00Z — likely merge candidate
   - **H53 tanjiro terminal** ~22:30Z

## Previous invocation actions (2026-05-19 ~07:55Z) — NEZUKO H47 V-DEPTH EP5 CHECK-IN: **MERGE CANDIDATE STRENGTHENING** (val_abupt 6.421% — only +0.295pp above merge gate 6.126%, monotonic descent EP3→EP4→EP5, EP6 fires ~08:15Z make-or-break early-merge signal)

### Headline updates (07:55Z)

1. **PR #1194 nezuko H47 V-DEPTH EP5 check-in posted** ([comment 4485662935](https://github.com/morganmcg1/DrivAerML/pull/1194#issuecomment-4485662935)). Run `dp7gbsjb` healthy at step 46,738 (33.1% through budget, 8.20h runtime, heartbeat fresh 07:51Z, nonfinite=0, mid-EP6 leg 2,159 steps before EP6 boundary 48,897). **MONOTONIC DESCENT EP3→EP4→EP5: val_abupt 6.784% → 6.511% → 6.421%**. EP5 sits inside EP6 gate (val_abupt<6.5%) by 0.079pp and **only +0.295pp above merge gate 6.126%**. Slope EP4→EP5 = −0.017 pp/1k steps; if continues through EP6 (5,435 steps), projected val_abupt 6.33% — already crossing EP6 gate cleanly. Terminal projection 5.9-6.1% range straddles merge gate.

2. **MERGE CANDIDATE STRENGTHENING**: H47 is the **first Wave 31 PR with realistic early-merge crossover trajectory**. EP6 reading at ~08:15Z is the critical signal:
   - If val_abupt ≤ 6.126% at EP6: H47 crosses merge gate before terminal — would justify early-confirmation discussion at EP9 (~12:00Z) or EP13 (~17:00Z)
   - If 6.13-6.30%: standard 13-epoch trajectory, terminal determines merge eligibility
   - Worst case: descent stalls and terminal lands ~6.2-6.4% — close but no merge

3. **Secondary metrics also descending**:
   - val_SP: 4.507% → 4.270% → 4.196% (only +0.62pp above test_SP floor 3.577%)
   - val_vol_p: 4.035% → 3.976% → 3.955% (only +0.31pp above test_vol_p floor 3.643%) — the architectural target of volume-decoder-depth is tightening as predicted
   - val_WSS aggregate: 7.658% → 7.346% → 7.234% (+0.51pp above test_WSS baseline 6.727%; τz still draining residual at 9.65%)

4. **Wave 31 fleet status updated leaderboard at EP3-EP5 read window**:
   - val_abupt (lowest = best): **H47 EP5 6.421% (LEADING by ~1.3pp)**; cohort H53 EP2 7.674%, H50 EP2 7.807%, H52 EP2 8.480%
   - val_SP (lowest = best): H53 EP2 4.898%, **H47 EP5 4.196% (LEADING)**, H52 EP2 5.79%
   - val_vol_p: **H47 EP5 3.955% (LEADING, closest to floor)**, H53 EP2 4.648%, H52 EP2 4.89%
   - H47 is now sweep-leading on all 3 primary metrics at the EP5 read

5. **Fleet 8/8 WIP, 0 idle, 0 review-ready.** Active runs in priority by next check-in:
   - **H52 frieren EP3** ~07:50Z (step 32,594) — mechanism ALIVE threshold (NOW firing — next check-in window)
   - **H53 tanjiro EP3** ~08:20Z (step 32,594) — SP-gate strong PASS expected
   - **H51 fern EP3** ~08:30Z (step 32,594) — FIRST informative read, mechanism class confirmation
   - **H47 nezuko EP6** ~08:15Z (step 48,897) — **MERGE-GATE-CROSSOVER decision point**
   - **H50 askeladd EP6** ~10:15Z — make-or-break
   - **H48 edward terminal** ~12:45Z
   - **H47 nezuko terminal** ~17:00Z
   - **H53 tanjiro terminal** ~22:30Z

## Previous invocation actions (2026-05-19 ~07:35Z) — FERN H51 NPCA+SSFL-SLICES192 EP2 MID-EP3-LEG CHECK-IN: **UNEXPECTED MECHANISM REGIME SHIFT** (variance-class hypothesis → mean-shift observation, EMA-shadow caveat at EP2)

### Headline updates (07:35Z)

1. **PR #1199 fern H51 EP2 check-in posted** ([comment 4485498426](https://github.com/morganmcg1/DrivAerML/pull/1199#issuecomment-4485498426)). Run `2vlx68f9` healthy at step 25,962 (18.4% through 141,232-step budget, 4.22h runtime, heartbeat fresh 07:30Z, nonfinite=0). **EP2 val_abupt 53.21% is EMA-shadow-dominated** under ema=0.9999 — δ^21728=0.115 means 11.5% of EMA weights are still initial-random. **NOT yet informative** for trajectory comparison. EP3 read at step 32,594 (~08:30Z, δ=0.038) is the first informative checkpoint; EP6 (δ=0.0015) is the first fully informative read. Stale_wip was the known long-jobs 2h-no-commits false positive.

2. **UNEXPECTED MECHANISM REGIME SHIFT** — H51 hypothesis predicted variance-class expansion (slices=128→192 → +50% variance room → std grows toward 0.15 ALIVE threshold). Actual EP2 observation:
   - **EP1 → EP2**: τz/τx mean 1.3225 → **1.2635** (mean dropped −0.059, BELOW band edge 1.44)
   - **EP1 → EP2**: τz/τx std 0.0931 → **0.0726** (std **COMPRESSED** −0.020, not expanded)
   - **EP1 → EP2**: n_outside 27/34 → **34/34** (saturated to 100% cars outside band)
   This is **MEAN-SHIFT behavior (H48-like)**, NOT variance-class behavior (H35-like). EMA-shadow contamination at EP2 means the class may still flip toward variance once EMA catches up — EP3 will tell. If continued mean-shift, H51 is producing a mechanism class orthogonal to its original hypothesis.

3. **Wave 31 mechanism-class taxonomy (updated)**:
   - **variance-class** (H35 NPCA+SSFL ref: std 0.251 EP6 peak): H52 NPCA×YAW activating (std 0.1498 EP2 +75%), H47 V-DEPTH sublayer mechanism (different metric class — direct activation magnitude)
   - **mean-shift class** (H48 TAU-Y-EQUALIZE: mean 0.40 band-break flat): **H51 NPCA+SSFL+slices192 emerging** (mean 1.2635 below band, std compressed, n_outside saturated)
   - **shift+compression** (new sub-class candidate H51): may need EP6 first-fully-informative read to confirm

4. **Hard kill gates trending PASS** for H51 despite EMA-shadow contamination: EP3 gates (val_abupt<10% AND val_SP<6%) generous-by-design for δ=0.038 contamination. All trending strong PASS by step 32,594 firing time.

5. **Fleet 8/8 WIP, 0 idle, 0 review-ready.** All 4 human issues (#1056, #285, #252, #618) checked — no new responses required, all previous ADVISOR replies are the most-recent activity. Active runs in priority by next check-in:
   - **H52 frieren EP3** ~07:50Z (step 32,594) — mechanism ALIVE threshold (std crossing 0.15)
   - **H53 tanjiro EP3** ~08:20Z (step 32,594) — SP-gate strong PASS expected
   - **H51 fern EP3** ~08:30Z (step 32,594) — FIRST informative read, mechanism class confirmation
   - **H50 askeladd EP6** ~10:15Z — make-or-break
   - **H47 nezuko EP6** ~10:30Z — MERGE CANDIDATE strengthening
   - **H48 edward terminal** ~12:45Z
   - **H47 nezuko terminal** ~17:00Z
   - **H53 tanjiro terminal** ~22:30Z

## Previous invocation actions (2026-05-19 ~07:00Z) — ISSUE #1056 ADVISOR RESPONSE POSTED (test_SP definition + test_WSS status + promising Wave 31 runs)

### Headline updates (07:00Z)

1. **Issue #1056 ADVISOR response posted** ([comment 4485312834](https://github.com/morganmcg1/DrivAerML/issues/1056#issuecomment-4485312834)). Morgan's three questions addressed cleanly:
   - **test_SP definition**: surface pressure relative L2 error (%), identical to test_surf_p; current single-model floor 3.577% from PR #972
   - **test_WSS status**: single-model SOTA 6.727%, ensemble SOTA 6.3263% (banned per CLAUDE.md "NO MORE ENSEMBLES"), Transolver-3 target 5.85%, gap 0.877pp; no new Wave 31 terminal test scores yet (test eval at EP13 terminal only); earliest terminals 18-24h out
   - **Promising Wave 31 runs**: H47 nezuko (leading merge candidate trajectory val_abupt 6.784% EP3), H53 tanjiro (fleet-leading val_SP 4.898% EP2 attacks binding gate), H52 frieren (fastest variance-class activation +75% std EP1→EP2)
   - All facts verified against recent advisor check-ins — no parallel-fork misinformation this time.

2. **Other open issues (#618, #285, #252) checked**: last ADVISOR comment was most-recent activity on each. No new human messages requiring response.

3. **Fleet 8/8 WIP unchanged from 06:50Z snapshot.** Next student check-ins: H52 frieren EP3 ~07:50Z, H53 tanjiro EP3 ~08:20Z, H51 fern EP3 ~08:30Z.

## Previous invocation actions (2026-05-19 ~06:50Z) — TANJIRO H53 CP-LOSS-WEIGHT EP2 CHECK-IN: val_SP 4.898% FLEET-LEADING by 0.210pp vs H47 — TARGETED SP-BINDING-GATE ATTACK WORKING

### Headline updates (06:50Z)

1. **PR #1202 tanjiro H53 CP-LOSS-WEIGHT EP2 check-in posted.** Run `u187bw3a` healthy at step 22,381 (3.32h runtime, EP2 cleared into EP3 leg, heartbeat fresh 06:48Z, nonfinite=0, throughput accelerated 1.63 → 1.87 it/s). **EP2 val_SP 4.898% is FLEET-LEADING by 0.210pp vs H47 (5.108%) and by 0.892pp vs H52 (5.79%)** — the targeted attack on the test_SP binding gate (0/7 vol_p crossings in Wave 31 fleet have ever crossed test_SP floor 3.577%) is producing the expected accelerated surface-pressure descent. EP2 val_abupt 7.674% is competitive mid-pack (H47 7.615%, H50 7.807%, H52 8.480%). Both EP3 kill gates already inside at EP2: val_abupt 7.674<9.5 AND val_SP 4.898<5.5. `loss/cp_loss_weight=2.0` and `train/cp_loss_weight=2.0` verified in W&B summary. EP3 gate fires ~08:20Z May 19.

2. **H53 hypothesis structurally validated**: doubling cp-channel gradient budget (cp loss weight 1.0→2.0, channels [cp=2, τx=1, τy=1.5, τz=2]) is producing precisely the accelerated surface-pressure descent the hypothesis predicted. val_SP slope EP1→EP2: −1.769 pp/1k steps. If margin holds or widens through EP3-EP6, H53 has a real shot at the test_SP binding gate (3.577% floor).

3. **Wave 31 mechanism orthogonality**: H53 (cp-channel loss-weight redirection) and H47 (volume-decoder depth) are mechanistically orthogonal — H53 attacks the SP binding gate while H47 attacks the val_abupt merge gate. Both are fleet-leading on their respective primary metrics at EP2-EP3. If both merge cleanly, **the two could stack additively in a Wave 32 follow-up** (H53's cp-channel mechanism + H47's vol-deep architecture).

4. **Updated EP2 fleet leaderboard** (Wave 31 active runs that reached EP2):
   - **val_abupt**: H47 7.615% > H53 7.674% > H50 7.807% > H52 8.480%
   - **val_SP**: H53 4.898% > H47 5.108% > H52 5.79% (THE binding gate)
   - **val_vol_p**: H47 4.035% (EP3 read) > H53 4.648% (EP2) > H52 4.89% (EP2)

5. **Fleet 8/8 WIP, 0 idle, 0 review-ready.** Active runs in priority by next check-in:
   - **H52 frieren EP3** ~07:50Z (step 32,594) — mechanism std ALIVE threshold check
   - **H53 tanjiro EP3** ~08:20Z (step 32,594) — SP gate watch
   - **H51 fern EP3** ~08:30Z (step 32,594) — gate decision under EMA-aware thresholds
   - **H50 askeladd EP6** ~10:15Z (step 65,228) — MAKE-OR-BREAK at 6.5%
   - **H47 nezuko EP6** ~10:30Z (step 65,228) — MERGE CANDIDATE trajectory
   - **H53 tanjiro EP6** ~12:30Z — SP descent confirmation
   - **H48 edward terminal** ~12:45Z — structural finding writeup
   - **H47 nezuko terminal** ~17:00Z
   - **H53 tanjiro terminal** ~22:30Z

## Previous invocation actions (2026-05-19 ~06:15Z) — FRIEREN H52 NPCA × YAW-AUG EP2 CHECK-IN (val_abupt 8.48% lagging cohort BUT mechanism std +75% EP1→EP2 fleet-leading rate) + 4TH RECIPE-BUG ACKNOWLEDGED (underscore vs hyphen)

### Headline updates (06:15Z)

1. **PR #1200 frieren H52 NPCA × YAW-AUG STACK EP2 check-in posted.** Run `3u4i7oy6` healthy at step 27,249 (4.04h runtime, mid-EP3 leg, heartbeat fresh 06:11Z, nonfinite=0). **EP2 val_abupt 8.48% LAGGING fleet cohort by 0.67-0.87pp** (vs H47 EP2 7.615% + H50 EP2 7.807%) — consistent with H44 standalone YAW-AUG's cold-start tax pattern (~+0.5-0.8pp behind at EP2/EP3 then closes gap by terminal). val_SP 5.79% borderline at gate <5.5%. **MECHANISM SIGNAL FLEET-LEADING ACTIVATION RATE**: τz/τx std 0.0855 → **0.1498** (+75% in one epoch), approaching 0.15 ALIVE threshold; mean 1.4247 → 1.4447 (slight band-break growth); n_outside 12→13/34. If std crosses 0.15 by EP3, H52 enters variance-class ALIVE regime — earliest in Wave 31 fleet history (H35 needed EP6 for fleet-peak std 0.251, H46 SDORTH PathB never reached 0.15 at EP3).

2. **4th recipe-bug acknowledgment.** Frieren silently caught and corrected another flag-format bug at H52 launch: I wrote `--use_local_frame_proj` (underscores) in the PR body but the actual argparse declares `--use-local-frame-proj` (hyphens). Argparse converts hyphens to underscores for the attribute name but requires the original hyphen form at command-line. Frieren also noted in the PR body that NPCA was on PR #1177 (closed-not-merged), not on tay as my body claimed — required cherry-pick from `thorfinn/h26-normal-projected-coord-aug`. Updated `feedback_audit_flags_in_recipe.md` memory with both patterns: underscore-vs-hyphen and stale-branch flag references.

3. **Wave 31 recipe-bug pattern tally (this invocation cycle): 4 bugs caught by 3 students.**
   - Nezuko H47: EP1 kill threshold `<9.5` was wrong (preempt-killed run, saved 18h slot)
   - Fern H51 #1: EMA-aware threshold (δ^step composition not accounted for under ema=0.9999)
   - Fern H51 #2: phantom `--use-spectral-loss` flag (doesn't exist in argparse)
   - Frieren H52: `--use_local_frame_proj` (underscores) vs `--use-local-frame-proj` (hyphens) + stale "on tay" claim
   - All 4 caught BEFORE wasting 18h GPU. Recipe-audit checklist now 4 patterns deep + 2 sub-patterns: flag existence, format conventions, branch presence, step-indexed thresholds, EMA-step δ^N composition, lr-warmup-aware EP1 thresholds.

4. **Wave 31 mechanism stacking continues**: H52 NPCA × YAW combined stack is producing FASTEST variance-class activation in fleet history at EP2 (+75% std growth in one epoch). This is exactly the prediction from H35 stacking thesis: combine NPCA's variance-class encoder enrichment with YAW's training-time symmetry-break and the signals SHOULD compound. EP3 std read (~07:50Z) will confirm if mechanism has crossed ALIVE threshold.

5. **Fleet 8/8 WIP, 0 idle, 0 review-ready.** Active runs in priority by next check-in:
   - **H51 fern EP3** ~08:30Z (step 32,594) — gate decision point under EMA-aware thresholds
   - **H52 frieren EP3** ~07:50Z (step 32,594) — mechanism ALIVE threshold check
   - **H50 askeladd EP6** ~10:15Z (step 65,228) — make-or-break gate at 6.5%
   - **H47 nezuko EP6** ~10:30Z (step 65,228) — MERGE CANDIDATE trajectory
   - **H48 edward terminal** ~12:45Z — likely structural finding (τz/τx mean 0.40 band-break)
   - **H47 nezuko terminal** ~17:00Z
   - **H49/H45/H52/H51/H53 terminals** 14:00-20:00Z range

## Previous invocation actions (2026-05-19 ~05:45Z) — NEZUKO H47 V-DEPTH EP3 GATE STRONG PASS CHECK-IN (val_abupt 6.784% LEADING WAVE 31 EP3 COHORT + MECHANISM GROWING 4-10× ABOVE FLOOR — POTENTIAL MERGE CANDIDATE)

### Headline updates (05:45Z)

1. **PR #1194 nezuko H47 V-DEPTH EP3 gate check-in posted.** Run `dp7gbsjb` healthy at step 36,771 (6h02m runtime, EP4 leg, heartbeat fresh 05:39Z, nonfinite=0). **EP3 validation completed at step 32,592 with both gates PASS with margin**: val_abupt **6.784%** (gate <7.5%, PASS by 0.72pp) and val_SP **4.507%** (gate <5.5%, PASS by 0.99pp). All 4 mechanism sublayers PASS kill gate strongly and GREW from EP2 to EP3: vol_deep_block0 attn_proj 0.171→0.221 (+29%), ffn_fc2 0.305→0.479 (+57%), vol_deep_block1 attn_proj 0.191→0.252 (+32%), ffn_fc2 0.404→0.509 (+26%). Volume decoder depth bump is doing meaningful nonlinear computation, not just routing. FFN dominance over attn persists (0.48/0.51 vs 0.22/0.25). Stale_wip is the 2h-no-commits long-jobs false positive.

2. **H47 LEADING Wave 31 EP3 cohort** by 0.15-0.18pp:
   - H47 V-DEPTH EP3 6.784% (leader, this run)
   - H33 SLICEPE EP3 6.94% (merged baseline reference)
   - H45 ANCHOR-CROSSCHAN-DEC EP3 6.96% (running, alphonse)
   - H44 YAW-AUG EP3 ~7.0% (closed at 02:15Z, terminal val_abupt 6.355% FAIL +0.229pp above gate)
   - H50 COORDSLICE EP3 7.807% step 30,008 (running, askeladd, lagging 1.02pp)

3. **MERGE CANDIDATE trajectory**: slope EP2→EP3 −0.094 pp/1k steps. Projecting:
   - EP6 (step 65,228) ~10:30Z May 19: val_abupt 6.3-6.5% projected → likely PASS gate <6.5%
   - EP13 terminal (step 141,232) ~17:00Z May 19: val_abupt **5.6-6.0% projected** → potentially CROSSES merge gate 6.126%
   - With cosine compression in EP7-EP13 + typical 1.5-2pp terminal-vs-EP3 compression, H47 has a real shot at the merge gate.

4. **Test_SP / test_vol_p / test_WSS targets at terminal**: H47's volume-decoder depth bump primarily targets vol_p (currently 4.035% descending toward floor 3.643%). val_SP at 4.507% is descending but still 0.93pp above test_SP floor 3.577% — test_SP gap may remain. val_WSS aggregate 7.658% above baseline 6.727% by 0.93pp — Wall_shear axes lagging in absolute terms (especially τz 10.156%, still draining cold-start residual).

5. **Fleet 8/8 WIP, 0 idle, 0 review-ready.** Active runs in priority order by next check-in:
   - **H47 nezuko** (THIS) — EP3 PASS, EP6 ~10:30Z, terminal ~17:00Z
   - **H48 edward** — mid-EP9 plateau at 6.508%, terminal ~12:45Z
   - **H50 askeladd** — EP3 7.807% lagging, EP6 gate ~10:15Z (make-or-break)
   - **H51 fern** — EP1 71.17% as predicted, EP3 ~08:30Z next
   - **H52 frieren, H53 tanjiro, H49 thorfinn, H45 alphonse** — terminals 14:00-18:00Z

## Previous invocation actions (2026-05-19 ~05:25Z) — FERN H51 v3 EP1 CHECK-IN (EMA-AWARE THRESHOLD DROP VALIDATED + 2ND RECIPE-BUG ACKNOWLEDGED — PHANTOM `--use-spectral-loss` FLAG)

### Headline updates (05:25Z)

1. **PR #1199 fern H51 NPCA+SSFL-SLICES192 v3 EP1 check-in posted.** Run `2vlx68f9` (launched 03:17:39Z May 19 with corrected EMA-aware thresholds + spectral-loss flag stripped) is HEALTHY at step 12,828 (9.1% through budget, 2h4m runtime, heartbeat fresh 05:20Z, nonfinite=0). **EP1 validation completed at step 10,864 with val_abupt 71.17%** — EXACTLY what the δ^10864=0.337 math predicted (≥70% guaranteed under EMA shadow contamination). **The gate-drop decision is structurally validated by actual run.** Had the original H35-copied `10864:val_abupt<9.5` threshold remained, the run would have false-killed at EP1 despite a healthy model.

2. **STRONG early mechanism signal**: `tau_zx_ratio_n_outside_band 27/34` already at EP1 — for context, H35 NPCA+SSFL (the H51 mechanism reference) hit `n_outside_band 17/34` at EP6 (cohort baseline). H51 at EP1 is already ahead of H35's EP6 reference by 10 cars (~80% of cars producing per-car ratios outside [1.44, 1.55] band attractor). EMA-shadow noise caveat applies at EP1, but EP3 persistence would confirm fleet-leading variance-class mechanism.

3. **2nd Wave 31 recipe-bug acknowledged on same PR.** Fern caught a SECOND recipe-bug: `--use-spectral-loss` flag does not exist in `train.py` argparse — spectral loss is enabled implicitly when `--lambda-spectral` is nonzero (train.py:486-494). The v2 relaunch crashed with `train.py: error: unrecognized arguments: --use-spectral-loss`. Fern stripped the flag for v3 — confirmed the original H35 reference run was also missing it. Two recipe bugs in one PR = root cause: I copied H35's recipe without auditing each flag against the current `target/train.py` argparse. Filed as **3rd Wave 31 recipe-bug pattern**: flag existence audit (companion to EMA-aware thresholds and EP-warmup-aware thresholds). New memory entry written: `feedback_audit_flags_in_recipe.md`.

4. **Recipe-audit checklist now 4 patterns deep:**
   - Flag existence (this PR — phantom `--use-spectral-loss`)
   - Step-indexed kill thresholds correctly map to global_step boundaries (no `step>=` prefix confusion)
   - EMA-step δ^N composition matches kill-threshold sensitivity (when `--ema-decay != 0.999`)
   - `--lr-warmup-epochs` aware EP1 threshold (warmup=1 recipes land val_abupt ~33% at EP1)

   Going forward I will walk down all 4 checks for every PR I assign. Three of these patterns were caught by students this invocation (fern EMA + flag, nezuko EP-warmup).

5. **Fleet 8/8 WIP, 0 idle, 0 review-ready.** Survey confirms all students assigned, no human messages requiring response. Next student check-ins expected:
   - **H47 nezuko EP3** ~05:00-06:00Z (step 32,594) — overdue, no comment yet but no stale flag
   - **H51 fern EP3** ~08:30Z (step 32,594) — gate decision point
   - **H50 askeladd EP6** ~10:15Z (step 65,228) — make-or-break gate at 6.5% val_abupt
   - **H48 edward terminal** ~12:45Z — likely structural finding, possible borderline merge
   - **H51 fern EP6** ~14:50Z (step 65,228)
   - **H52/H53/H49/H45 terminals** 14:00-18:00Z range

## Previous invocation actions (2026-05-19 ~03:50Z) — EDWARD H48 TAU-Y-EQUALIZE MID-EP9 CHECK-IN (val_abupt PLATEAUED 6.508% × 3 — mechanism HELD 9 EPOCHS — likely structural finding > merge candidate)

### Headline updates (03:50Z)

1. **PR #1196 edward H48 TAU-Y-EQUALIZE mid-EP9 check-in posted.** Run `8cn5abxm` healthy at step 58,599 (10.4h runtime, mid-EP9, heartbeat fresh). All hard kill gates pass. **val_abupt PLATEAUED at 6.508% across 3 consecutive validations** (EP7 → EP8 → mid-EP9, slope −0.007 pp/1k → 0.000 pp/1k). EP13 terminal projection 6.40-6.50% (+30-40bps above merge gate 6.126%) — likely closing as **structural finding > merge candidate**. **MECHANISM HELD 9 EPOCHS**: τz/τx mean 0.4009 unchanged from EP1 (drift +0.018 over 47,735 steps = +0.0006 pp/1k = statistically flat), 34/34 cars persistently outside band [1.44, 1.55], std 0.0328. Most stable extreme attractor reading in Wave 30/31 history. Decision: continue to natural budget cutoff (~12:45Z May 19) — structural finding deserves full terminal trace regardless of merge probability; cosine compression EP10-EP13 may bend val_abupt by 0.05-0.15pp giving 5-15% merge probability; test eval at best EMA ckpt is the decision point (val→test gap could compress or expand).

2. **H48 will be the FIRST training-time-only band-break demonstration** in Wave 30/31. Single `--tau-y-loss-weight` flag change, zero code modification. Independent of merge outcome, this opens "mechanism × baseline-architecture" stacking experiments (H48 + H26 NPCA + H31 WALLDIST simultaneously).

3. **Fleet 8/8 WIP, 0 idle, 0 review-ready.** Survey confirms all students assigned, no human messages requiring response. Next student check-ins expected:
   - **H47 nezuko EP3** ~05:00Z (step 32,594) — currently fleet-leading val_abupt 7.615% at EP2
   - **H50 askeladd EP6** ~10:15Z (step 65,228) — currently lagging cohort 0.85pp val_abupt 7.807%
   - **H48 edward EP12/13** terminal ~12:45Z — likely structural finding writeup, possible merge
   - **H51 fern, H52 frieren, H53 tanjiro, H49 thorfinn, H45 alphonse**: terminal landings 14:00-18:00Z range based on launch times

4. **Wave 31 binding-gate intelligence:** test_SP 0/7 vol_p crossings — remains unyielding. H48's mean-shift mechanism (vs H35/H36's variance-class mechanisms which show val→test divergence) is the most cleanly-stacked novel mechanism in flight, but val_abupt plateau suggests it alone won't close merge gate. Composition with NPCA / WALLDIST may be required.

5. **Test_WSS gap to Transolver-3 SOTA: −0.877pp to go.** Baseline test_WSS 6.727%, SOTA target 5.85%. H48 alone unlikely to close gap (val_WSS plateauing 7.375%, +0.65pp above baseline). H52 NPCA×YAW-AUG + H53 CP-LOSS-WEIGHT + H47 V-DEPTH remain the strongest WSS-targeting bets.

## Previous invocation actions (2026-05-19 ~03:35Z) — NEZUKO H47 V-DEPTH EP2 CHECK-IN (HEALTHY, FLEET-LEADING EP2 + STRONG MECHANISM SIGNAL) + 2ND WAVE 31 RECIPE-BUG ACKNOWLEDGMENT

### Headline updates (03:35Z)

1. **PR #1194 nezuko H47 V-DEPTH EP2 check-in posted.** Run `dp7gbsjb` (relaunched 23:35Z after preemptive kill of `8d1dzmqk`) is healthy at step 24,820 (17.6% through 141,232-step budget, 3h55m runtime, heartbeat fresh, nonfinite=0). **val_abupt 7.615% at EP2 boundary LEADING fleet** — beats H50 COORDSLICE which sits at 7.807% at step 30,008 (7,700 steps later). val_SP 5.108% already inside EP3 gate <5.5%; val_vol_p 4.354% descending toward floor 3.643%. **MECHANISM STRONGLY ACTIVE**: all 4 vol_deep_block{0,1} sublayer max_abs values 6-8× above EP3 kill floor 0.05 (`diag/vol_deep_block0/attn_proj` 0.171, `ffn_fc2` 0.305, `vol_deep_block1/attn_proj` 0.191, `ffn_fc2` 0.404). FFN dominating attn in both blocks suggests volume-decoder depth doing meaningful nonlinear computation, not just routing. Stale_wip is known long-jobs false positive — student-committed next check-in at EP3 gate ~05:00Z May 19.

2. **2nd Wave 31 recipe-bug acknowledgment (companion to H51 EMA fix).** Nezuko caught and preemptively fixed a broken EP1 kill threshold I'd embedded in the recipe: `10864:val_abupt<9.5` would have guaranteed a false kill on a healthy mechanism (every 13-ep cosine-warmup run lands 25-30% val_abupt at EP1). The student killed `8d1dzmqk` at 71% through EP1 (~45min compute discarded) to save the 18h slot, citing PR Step 8 spec `<30`, H30 V2S 28.30%, H47 6h 25.43% as precedents. Now run `dp7gbsjb` passed EP1 cleanly with val_abupt 27.28%. **Pattern**: copying step-indexed kill thresholds across PRs without precedent-checking against actual EP1/3/6 reads from the matching recipe = recipe bug. Same pattern as fern's H51 EMA-aware fix earlier this invocation. Both caught by students before doomed run.

3. **Fleet 8/8 WIP**, all students confirmed active. Active runs trajectory snapshot:
   - **askeladd H50 COORDSLICE** (EP3 step 30,008): val_abupt 7.807% lagging cohort, Block 2 dominance emergent — EP6 gate at step 65,228 make-or-break (~10:15Z May 19)
   - **nezuko H47 V-DEPTH** (EP2 step 24,820): val_abupt 7.615% LEADING fleet, mechanism strong — EP3 gate at step 32,594 (~05:00Z May 19)
   - **edward H48 TAU-Y-EQUALIZE** (EP4): val_abupt 6.597% descending −0.106 pp/1k, τz/τx mean 0.40 persisting all 34/34 cars — potential merge candidate at terminal ~12:45Z May 19
   - **fern H51 NPCA+SSFL-SLICES192** (relaunching): EMA-aware kill thresholds applied, terminal ~17:00Z May 19+
   - **frieren H52 NPCA×YAW-AUG** (assigned at 02:15Z): combines `--use_local_frame_proj` + `--yaw-aug-theta-max 5.0`, expected EP6 ~10:00Z May 19+
   - **tanjiro H53 CP-LOSS-WEIGHT** (assigned at 02:50Z): single-arm `--cp-loss-weight 2.0` SP-binding-gate attack, expected EP6 ~10:00Z May 19+
   - **alphonse, edward, thorfinn**: in-flight per prior snapshots

4. **Wave 31 binding-gate intelligence remains the same:** test_SP 0/7 vol_p crossings — surface pressure remains the unyielding gate. H53 is the dedicated attack. H47 v-depth's volume-decoder capacity bump targets vol_p (not SP), so even a clean EP6 gate pass leaves SP merge-gate question open at terminal.

5. **Test_WSS gap to Transolver-3 SOTA: −0.877pp to go.** Baseline test_WSS 6.727%, SOTA target 5.85%. All in-flight runs (H48, H45, H50, H47, H52, H53, H51) need to compound improvements to close the gap.

## Previous invocation actions (2026-05-19 ~03:15Z) — FERN H51 SENT BACK (EMA-AWARE THRESHOLD FIX) + H50 ASKELADD CHECK-IN (EP3 LAGGING) + ISSUE #1056 CORRECTION POSTED

### Headline updates (03:15Z)

1. **PR #1199 fern H51 NPCA+SSFL-SLICES192 — SENT BACK to status:wip with adjusted EMA-aware kill thresholds.** Fern correctly diagnosed that the H51 recipe I assigned at 00:15Z had a critical bug: kill threshold `10864:val_primary/abupt_axis_mean_rel_l2_pct<9.5` is incompatible with `--ema-decay 0.9999`. Math: δ^10864 = 0.337 → at EP1 the EMA shadow is 34% initial random weights + 66% trained → val_abupt 72.18% guaranteed → falsely kills run before mechanism evaluates. The student killed 2 runs and posted a clean diagnosis with three options. **Approved option 1** with relaxed thresholds: drop EP1 gate (no informative signal possible under ema=0.9999); EP3 abupt<10% + SP<6% (3.8% initial contamination remaining); EP6 abupt<7% + SP<5% (EMA caught up); EP13 verdict on val_abupt vs 6.126% merge gate. Preserves dual-hypothesis isolation (slices=128→192 capacity expansion × ema=0.999→0.9999 cosine window). Filed as **Wave 31 recipe-bug pattern**: when EMA-decay deviates from 0.999 baseline, step-indexed kill thresholds need recalibration based on δ^step composition.

2. **PR #1198 askeladd H50 COORDSLICE EP3 check-in posted.** Run `biw3rtli` at step 30,008 (76% through EP3, 4.57h runtime, heartbeat fresh). Hard kill gates pass (val_abupt 7.807% < 9.5%, val_SP 5.257% < 5.5%). ⚠️ **val_abupt LAGGING EP3 cohort by ~0.85pp** (H33 6.94%, H45 6.96%, H44 similar → H50 7.807%). 🟢 **MECHANISM SHIFTED: Block 2 NOW DOMINATES SPATIAL DIFFERENTIATION** — at mid-EP2 the pattern was L0-INVERSION (L0=0.453 LOWEST, L4=0.647 HIGHEST); at EP3 it has shifted to Block 2 = NEW LOWEST inter_slice_cos 0.093 with centroid_range_x 15.88 (2-4× wider than other blocks). The model is dynamically choosing WHERE to do spatial differentiation, not having it imposed by depth — novel Wave 31 emergent behavior (5th instance of mechanism-class emergent observation). EP6 gate at step 65,228 (~10:15Z) is make-or-break: val_abupt > 6.5% would kill.

3. **Issue #1056 correction posted.** A parallel check-human-issues invocation posted an incorrect "overnight update" claiming H44 was "MERGED WINNER" (it was CLOSED, NOT-A-MERGE) with wrong val_abupt 7.166% (actual 6.355%) and claiming H36 was a "LIVE CANDIDATE" (it was CLOSED at 02:33Z). Posted correction with accurate state: 3 dead ends in Wave 31 (H35, H44, H36), 3 vol_p crossings (5th/6th/7th), 0 merges. test_SP remains 0/7 crossings — binding gate unyielding. **Reminder**: trust verbatim PR state over fork summaries.

4. **Active research-direction implications restated to human team:**
   - test_SP is the binding gate (0 crossings in 26 attempts). H53 PR #1202 (cp-channel up-weighting via `--cp-loss-weight 2.0`) is the first targeted SP attack — uses same single-flag lever as H48 TAU-Y-EQUALIZE's τz/τx breakthrough.
   - Variance-class mechanisms hit a ceiling (H26/H35/H36 show val→test divergence: val std 0.195+ → test std 0.118-0.140); H48's mean-shift mechanism is more stable across the val→test gap (34/34 cars persistence).
   - Mechanism-stacking is proven (H35 NPCA+SSFL → H52 NPCA×YAW-AUG, H53 + H44 follow-up).

5. **Test_WSS gap to Transolver-3 SOTA: −0.877pp to go.** Baseline test_WSS 6.727%, SOTA target 5.85%. All in-flight runs (H48, H45, H50, H52, H53) need to compound improvements to close the gap.

6. **Fleet 8/8 WIP**, but fern's H51 about to relaunch with corrected thresholds — temporary 7/8 active until fern relaunches.

## Previous invocation actions (2026-05-19 ~02:50Z) — TANJIRO H53 RE-ASSIGNED ON CORRECT BRANCH (PR #1201 closed because branch was `dl24-tanjiro/h53-cp-loss-weight` which is dl24-prefix and would not be picked up by our tanjiro's student-polling on `tanjiro/...` branches; recreated as PR #1202 on `tanjiro/h53-cp-loss-weight` with refined single-arm 18h DDP-8 recipe `--cp-loss-weight 2.0` — clean orthogonal-arm discipline avoids compounding with H48's parallel `--tau-y-loss-weight 1.0` test; EP3 SP gate <4.7%, EP6 SP gate <4.3%, terminal target SP ≤3.577%); H36 ANCHOR-SLICE-QUERIES EXPERIMENTS_LOG entry added (full 26th DE writeup: 3rd variance-class confirmation + DEEPEST vol_p crossing −0.165pp + anchor sparsity finding max ~66× mean + val/test variance divergence 0.195→0.140 = variance-class overfits to training distribution); FLEET 8/8 WIP

### Headline updates (02:50Z)

1. **PR #1201 H53 CP-LOSS-WEIGHT CLOSED + RE-ASSIGNED as PR #1202.** Original #1201 was created on branch `dl24-tanjiro/h53-cp-loss-weight` — wrong prefix (dl24 convention) that our tanjiro's student-polling does NOT match. Closed with explanatory comment, recreated as PR #1202 on `tanjiro/h53-cp-loss-weight` with same hypothesis (`--cp-loss-weight` cp-channel up-weighting to attack test_SP binding gate) but refined recipe: SINGLE arm at `--cp-loss-weight 2.0` (vs original two-arm 2.0/3.0) to preserve orthogonal-arm discipline with H48's parallel `--tau-y-loss-weight 1.0` test, plus new step-indexed kill thresholds at EP6 (val_SP < 4.3% gate to catch failed SP descent early).

2. **H36 closure entry added to EXPERIMENTS_LOG.md.** Documents the 26th Wave 30/31 DE + 7th vol_p floor crossing (DEEPEST at −0.165pp) + 3rd variance-class mechanism confirmation. New Wave 31 mechanism observations: (a) ANCHOR SPARSITY — `anchor_mod_abs_max 348.76` vs `anchor_mod_abs_mean 5.27` = 66× concentration on a few slots, possibly driving SP regression; (b) VAL/TEST VARIANCE DIVERGENCE — std 0.195 val → 0.140 test = variance-class mechanisms partially overfit training distribution, DISTINCT from H48's mean-shift mechanism which projects all-34/34 cars to all-50/50 cars.

3. **Wave 31 emerging structural insight (now 4 instances).** mechanism class determines per-car gradient routing distribution NOT aggregate axis ratios — H35 stacking + H44 cross-channel WSS regularization + H45 GATE 1/GATE 2 decoupling + H48 τz/τx 0.40 persistence + H36 variance/mean decoupling. Mean-shift mechanisms (H48) appear to generalize better than variance-class (H26, H35, H36) on the val→test gap.

4. **vol_p floor crossing tally remains at 7** (H31 WALLDIST, H26 NPCA, H46 SDORTH PathB, H33 SLICEPE, H35 NPCA+SSFL, H44 YAW-AUG, H36 ANCHOR-SLICE). 5 of 7 have val_abupt FAIL on merge dim. **test_SP remains 0/7 crossings** — the binding gate is unyielding. H53 PR #1202 is the dedicated SP-binding-gate attack.

5. **Fleet remains 8/8 WIP.** All other students continue per 02:15Z snapshot.

## Previous invocation actions (2026-05-19 ~02:30Z) — TANJIRO H36 ANCHOR-SLICE-QUERIES CLOSED EP10 (**26TH Wave 30/31 DE on merge dim** + **7TH test_vol_p floor crossing at 3.4780% −0.1650pp — DEEPEST vol_p crossing in Wave 31 history** — but test_SP 3.7169% FLOOR BREACH +0.1399pp, val_abupt 6.2379% MISS +0.1119pp above gate 6.126%); **TANJIRO H53 CP-LOSS-WEIGHT INITIALLY ASSIGNED** (PR #1201, later closed and re-created as #1202 due to branch-naming issue); FLEET STILL 8/8 WIP

### Headline updates (02:30Z, superseded by 02:50Z H53 re-assignment)

1. **PR #1191 tanjiro H36 ANCHOR-SLICE-QUERIES CLOSED EP10.** Terminal SENPAI-RESULT. Run `vu93lzgc` budget cutoff ~06:10Z. val_abupt **6.2379% FAIL** (+0.1119pp above merge gate 6.126%), test_SP **3.7169% FLOOR BREACH** (+0.1399pp above 3.577% floor), test_VP **3.4780% PASS** (−0.1650pp below floor — 7TH and DEEPEST vol_p floor crossing in Wave 31 history). Per-axis val_WSS ahead on ALL 3 axes vs baseline at EP6/EP7; test_WSS 7.0xx% (+0.37pp above baseline, not improved). The 7th vol_p crossing is the deepest (−0.165pp vs H31 WALLDIST's −0.155pp), but surface-pressure and merge-gate metrics remain unbeaten.

2. **vol_p floor crossing tally now at 7** (H31 WALLDIST, H26 NPCA, H46 SDORTH PathB, H33 SLICEPE, H35 NPCA+SSFL, H44 YAW-AUG, H36 ANCHOR-SLICE). 4 of 7 have val_abupt FAIL on merge dim. **test_SP remains 0/7 crossings** — the binding gate is unyielding.

3. **PR #1201 tanjiro H53 CP-LOSS-WEIGHT INITIALLY ASSIGNED.** Closed + reassigned at 02:50Z as PR #1202 on correct branch `tanjiro/h53-cp-loss-weight`. See 02:50Z entry above for refined recipe.

4. **Fleet remains 8/8 WIP.** All other students continue per 02:15Z snapshot.

## Previous invocation actions (2026-05-19 ~02:15Z) — FRIEREN H44 YAW-AUG CLOSED EP13 (**25TH Wave 30/31 DE on merge dim** + **6TH test_vol_p floor crossing at 3.608% −0.035pp + FIRST DATA-AUGMENTATION AXIS CROSSING in DrivAerML fleet history** + cleanest cross-channel WSS regularization signal in Wave 31 — τz channel improved by −0.88pp at test DESPITE yaw rotation NOT mixing τz, NOVEL MECHANISM CLASS not predicted by hypothesis; all 3 WSS axes ahead of baseline at test; per-car std(τz/τx) val 0.198 PASSES 0.15 gate, test 0.118 partial; beats AB-UPT on vol_p −2.47pp and WSS aggregate −0.23pp); **FRIEREN H52 NPCA × YAW-AUG STACK ASSIGNED** (PR #1200, combines H26 NPCA encoder-input enrichment + H44 yaw aug θ_max=5°, mechanistically orthogonal axes: encoder-input × data-distribution; H35 just proved stacking works; predicted val_abupt 5.95-6.18% BORDERLINE merge candidate, test_vol_p 3.45-3.55% well below floor, NPCA frame-invariance expected to soften H44's surface-pressure tax); FLEET STILL 8/8 WIP

### Headline updates (02:15Z)

1. **PR #1190 frieren H44 YAW-AUG CLOSED.** Terminal SENPAI-RESULT at 02:08Z. Run `6scw4nto` finished EP13 EMA best, 51,622s runtime. val_abupt 6.355% (+0.229pp above merge gate), test_abupt 6.116% (+0.272pp above baseline), test_SP 3.853% (+0.276pp surface tax breach), test_vol_p **3.608%** (−0.035pp below floor — 6TH floor crossing). **First data-augmentation axis crossing in DrivAerML fleet history.**

2. **Cross-channel WSS regularization — novel Wave 31 mechanism class.** All 3 WSS axes ahead of baseline at test: τx −0.25pp, τy −0.51pp, τz −0.88pp. **τz NOT mixed by yaw rotation, yet improved by −0.88pp.** Rotation-equivariance prior provides cross-channel regularization on axes that aren't directly rotated. Novel finding — H44 hypothesis predicted only τx/τy improvement.

3. **Per-car std(τz/τx) variance gate analysis:** val 0.198 PASSES 0.15 gate, test 0.118 (between 0.05 KILL and 0.15 ALIVE — partial). 35% of test cars below 1.44 band edge — yaw rotation shifting some cars away from band attractor entirely.

4. **PR #1200 frieren H52 NPCA × YAW-AUG STACK ASSIGNED.** Combines `--use_local_frame_proj` (H26 NPCA, merged) + `--yaw-aug-theta-max 5.0` (H44 YAW-AUG). Mechanistically orthogonal: encoder input enrichment × data distribution rotation. **NPCA is rotation-equivariant by construction** — `[p·n, p·t1, p·t2]` are scalars under any rotation that rotates both `p` and `n` consistently. Predicted to soften H44's surface-pressure tax via NPCA frame-invariance while preserving vol_p + per-axis WSS gains.

5. **Wave 30/31 floor crossing tally — now 6** (H31 WALLDIST, H26 NPCA, H46 SDORTH, H33 SLICEPE, H35 NPCA+SSFL, **H44 YAW-AUG**). First data-aug axis. 3 of 6 (or 4 of 6 incl. H46 PathB) crossings have val_abupt fail — merge dimension consistently harder than floor.

## Previous invocation actions (2026-05-19 ~01:30Z) — EDWARD H48 TAU-Y-EQUALIZE EP4 stale_wip CLEARED with **τz/τx MEAN 0.40 PERSISTING ACROSS 4 EPOCHS** (EP1 0.383 → mid-EP3 0.400 → EP4 0.401, drift +0.0006 pp/1k essentially flat, std stable 0.033, n_outside 34/34 all cars stable — most stable extreme attractor reading in Wave 30/31 history) + **POTENTIAL MERGE CANDIDATE** val_abupt 6.597% at EP4 descending at −0.106 pp/1k from mid-EP3, projected terminal 5.95-6.25% at budget cutoff EP10-EP11 (~12:45Z May 19); **4TH INSTANCE of per-car / aggregate decoupling pattern**: per-car τz/τx mean 0.40 (all 34/34 cars below band) but aggregate wall_shear_z/wall_shear_x 1.526 (IN band [1.44, 1.55]) — joins H35 + H44 + H45 as emerging Wave 31 structural finding

## Previous invocation actions (2026-05-19 ~01:05Z) — TANJIRO H36 ANCHOR-SLICE-QUERIES EP7 region stale_wip CLEARED with **DOUBLE-POSITIVE LIVE MERGE CANDIDATE** (val_VP 3.629% **−0.014pp BELOW val/test floor** — val-side floor crossing CONFIRMED in flight; val_abupt 6.245% **+0.119pp above merge gate 6.126%** — closest in-flight to merge gate, projected terminal 6.10-6.18% borderline; per-axis val_WSS AHEAD on ALL 3 axes vs baseline EP6 — τx −0.31pp, τy −0.56pp, τz −0.39pp); budget cutoff at ~06:10Z May 19 lands at ~EP9-EP10 (NOT EP13 — student's prior projection off); terminal SENPAI-RESULT expected ~07:30Z after test eval

## Previous invocation actions (2026-05-19 ~00:45Z) — ASKELADD H50 COORDSLICE mid-EP2 stale_wip CLEARED with **L0-INVERSION CONFIRMATION** (inter_slice_cos_post_pe L0=0.453 LOWEST, L4=0.647 HIGHEST — exact INVERSE of H33's L0-dominance failure pattern; coordinate-conditioned anchors deliver positional disambiguation at L0 the layer that needs it most; EP1 val_abupt 28.90% normal cold-start, student's kill-threshold fix from step 10864 → 21729 validated since H33/H45 all land EP1 ~28-30%); per-block centroid spread healthy 0.076-0.122 spread along vehicle length 0.94-1.38 (slices anchoring on real 3D positions, NOT collapsing); all 8 ranks DDP healthy

### Headline updates (01:05Z)

1. **PR #1191 tanjiro H36 EP7 region — LIVE MERGE CANDIDATE.** Run `vu93lzgc` at step 67,932 (EP6→EP7 region, 13.16h runtime, heartbeat fresh). Two converging positive signals:
   - **val_VP 3.629%** ← BELOW val-side floor 3.643% by 0.014pp (val floor crossing CONFIRMED in flight)
   - **val_abupt 6.245%** ← +0.119pp above merge gate 6.126%, descending at −0.013 to −0.030 per validation
   - **Projected terminal val_abupt: 6.10-6.18% at budget cutoff** (~EP9-EP10) — BORDERLINE merge candidate
   - val_VP slope sequence: stall mid-EP6 then REACTIVATED 3.682 → 3.661 → 3.650 → 3.629 (cosine compression catching up)

2. **Per-axis val_WSS AHEAD on ALL 3 axes vs baseline EP6.** τx −0.313pp, τy −0.564pp, τz −0.391pp. The encoder-side anchor-slice-query mechanism delivering BALANCED improvement across all wall_shear axes simultaneously. No band-break (mean stays at 1.534 in [1.44, 1.55]) but per-axis errors are smaller. Aggregate val_WSS 7.083% projects to 6.85-6.95% at terminal (better than baseline EP6 7.43%, slightly worse than baseline terminal 6.727%).

3. **Budget reality check — EP9-EP10 cap, NOT EP13.** Run started 11:50:20Z May 18. 1100min cutoff at **06:10Z May 19** (~5h09min remaining). At observed 1.85 it/s, lands at step ~102,200 = mid-EP10 (EP10 ends 108,640). Student's earlier EP13 ETA ~01:00Z was off — actual pace ~30% slower than projected.

4. **No other PR action needed this invocation.** All other fleet members continuing per 22:26Z + 00:15Z + 00:30Z + 00:45Z snapshots.

## Previous invocation actions (2026-05-19 ~00:30Z) — ALPHONSE H45 ANCHOR-CROSSCHAN-DEC EP3 stale_wip CLEARED with **GATE 1 / GATE 2 DECOUPLING FINDING** (out_weight_norm τz/τx ratio 9.75 ≫ 1.3 gate, STRONGEST weight-space asymmetry in Wave 31; but output-space wall_shear τz/τx 1.517 STAYING IN BAND — weight asymmetry and output band-break are INDEPENDENT mechanisms, val_abupt 6.96% on baseline-beat trajectory); EMERGING WAVE 31 PATTERN: mechanism class determines gradient routing, NOT loss target (joins H35 stacking + H44 cross-channel regularization)

### Headline updates (00:45Z)

1. **PR #1198 askeladd H50 mid-EP2 healthy.** Run `biw3rtli` at step 14,078 (30% through EP2, 2.15h runtime, heartbeat fresh). EP1 cold-start val_abupt 28.90% — within normal cohort range (H33 29.63%, H44 32.97%, H45 27.81%, H48 25.94%). Student caught a critical kill-threshold bug at +30min: the PR-body's EP1 gate `<9.5%` at step 10864 would have killed every cold-start run in this cohort. Moved EP1 hard kill to step 21729 (EP2 boundary) where val_abupt is normally <9.5%. Fix validated.

2. **L0-INVERSION CONFIRMATION — H50 is doing exactly what H33's L0-dominance failure analysis predicted.** Per-block `inter_slice_cos_post_pe` at step 14,078:
   - L0 = **0.453** ← LOWEST (most differentiated slices at the layer that attends to raw geometry)
   - L1 = 0.454
   - L2 = 0.482
   - L3 = 0.547
   - L4 = **0.647** ← HIGHEST (deeper layers attend to content, not slice-position)

   H33's terminal pattern was OPPOSITE: L0 had HIGHEST inter_slice_cos (LEAST differentiated at the layer that needs it most). H50 delivers L0-targeted positional disambiguation via coordinate-conditioned RFF anchors — this validates the L0-dominance hypothesis structurally, even before val convergence.

3. **Centroid spread healthy — slices anchoring on real 3D positions.** Per-block centroid range x (vehicle length, normalized): 0.94 → 1.32 → 1.13 → 1.01 → 1.38 across L0-L4 — slices spread along vehicle length, not collapsing. Centroid range y (lateral, vehicle is bilaterally symmetric) is narrow 0.02-0.04 — correct, slices shouldn't need lateral differentiation. Centroid range z (vertical): 0.20-0.40, moderate height differentiation between top vs underbody.

4. **No other PR action needed this invocation.** All other fleet members continuing per 22:26Z + 00:15Z + 00:30Z snapshots.

## Previous invocation actions (2026-05-19 ~00:15Z) — FERN H35 NPCA+SSFL STACK CLOSED EP13 (24th Wave 30/31 DE on merge dim + **5TH test_vol_p floor crossing** at 3.585% −0.058pp + **FIRST PROVEN MECHANISM-STACKING EXPERIMENT** — NPCA per-slice variance signal SURVIVED SSFL spectral regularization, τz/τx std grew monotonically 0.078 → 0.251 fleet-peak through 13 epochs, n_outside_band 17/34, AB-UPT victory on p_s/p_v/τ); FERN H51 ASSIGNED (PR #1199, NPCA+SSFL stack at `--model-slices 192` + `--ema-decay 0.9999` — combines student's #1 capacity-scaling and #6 EMA-decay-fix priority follow-ups, expected +0.10-0.35pp val_abupt gain vs H35, potentially crosses 6.126% gate); FLEET STILL FULLY BOOKED 8/8 WIP

### Headline updates (00:30Z)

1. **PR #1192 alphonse H45 EP3 reading — GATE 1 MASSIVELY POSITIVE, GATE 2 DECOUPLED.** Run `lhivsp6j` at EP4 step ~32,810 (5h25m runtime, 1.69 it/s steady). EP3 readings:
   - val_abupt **6.960%** — under early-positive trigger 7.2% by 24bp; on baseline-beat trajectory (vs H26 EP3 7.039%, H31 EP3 7.117%)
   - val_VP 4.221% (descending toward floor 3.643%)
   - val_SP 4.608%, val_WSS 7.832%
   - **`crosschan_out_weight_norm` τz/τx ratio: EP1 0.83 → EP2 3.92 → EP3 9.75** ← 7.5× the 1.3 gate, STRONGEST weight-space asymmetry in Wave 31
   - τy/τx out_weight ratio: 0.62 → 5.21 → **12.78** ← τy channel loaded EVEN MORE than τz
   - cp + τx out_weight_norm COLLAPSED 27× and 31× from EP1 → EP3 (channels deemed "fine without correction")
   - But output-space `wall_shear τz/τx` ratio: 1.370 → 1.507 → **1.517** ← STAYING IN BAND [1.44, 1.55], not shrinking

2. **NEW Wave 31 structural insight: GATE 1 / GATE 2 DECOUPLING.** H45 demonstrates that weight-space channel asymmetry and output-space band-break are INDEPENDENT mechanisms. The cross-channel module CAN load 10× more capacity on τy/τz channels without affecting the output ratio. Joins TWO other recent Wave 31 findings showing the same pattern:
   - H35 NPCA+SSFL (closed 00:15Z): τz/τx std grew 3.2× but val_abupt only improved 0.17pp vs H26 → variance USED but capacity bottlenecked
   - H44 YAW-AUG (running): τz channel improved by +0.131pp despite yaw rotation NOT touching τz → cross-channel regularization, NOT direct-axis improvement
   - **H45 CROSSCHAN-DEC (running)**: out_weight_norm τz/τx 9.75× but output ratio 1.52 in band → weight asymmetry without output break

   **Emerging Wave 31 publishable insight**: mechanism class determines gradient routing, NOT loss target. The band attractor and aggregate accuracy are decoupled.

3. **H45 continuation decision: RUN ON to natural budget cutoff ~11:17Z May 19 (EP10-EP11).** Cosine LR schedule set for 13 epochs; running 10-11 means cosine tail partially traversed, EMA captures late-cosine descent. Hard kill at EP6: val_abupt > 6.5% → CLOSE. Early-positive at EP6: val_abupt < 6.35% → CONFIRM run + propose paper-facing eval.

4. **No other PR action needed this invocation.** All other fleet members continuing per 22:26Z + 00:15Z snapshots.

### Headline updates (00:15Z)

1. **PR #1189 fern H35 CLOSED EP13.** Terminal `7zkdf9xv` at 00:09:45Z May 19: val_abupt 6.298% FAIL (+0.172pp), test_SP 3.771% FAIL (+0.194pp), test_WSS 6.926% FAIL (+0.199pp), **test_vol_p 3.585% PASS (−0.058pp below floor)**. 24th Wave 30/31 DE on merge dim, 5th vol_p floor crossing in series (H31 merged, H26 merged, H46 PathB, H33 SLICEPE, **H35 NPCA+SSFL stack**). Beats AB-UPT on p_s, p_v, and vector τ; loses only on ABUPT meta-aggregate (+0.151pp).

2. **STACKING MECHANISM PROVEN — FIRST proven independence between Wave 30 mechanism classes.** NPCA per-slice variance signal SURVIVED SSFL spectral regularization without interference. τz/τx std grew MONOTONICALLY through 13 epochs: 0.078 → 0.121 → 0.155 → 0.184 → 0.205 → 0.223 → 0.236 → 0.244 → 0.249 → 0.251 (fleet peak). n_outside_band 17/34 vs H26 NPCA standalone 9/34. Confirms the working hypothesis that band-break mechanisms live in orthogonal subspaces — validates the broader stacking program (NPCA × WALLDIST, NPCA × SDORTH, etc.).

3. **Variance room insight — why val_abupt didn't cross.** H35 has 2.3× the τz/τx variance of H26 NPCA but only 0.17pp better val_abupt. Variance is being USED (n_outside_band 17/34 vs 9/34) but model lacks CAPACITY to convert per-slice heterogeneity into per-axis accuracy gain. 128 slices × 2.3× variance budget = ~295 effective slice-modes needed, but only 128 raw slice queries available → bottleneck.

4. **EMA-decay 0.999 issue — structural config bug helping ALL runs.** At lr-cosine-t-max=13 with ~141k total steps, EMA-decay 0.999 has effective window ~1000 steps but cosine tail spans ~70k steps where best checkpoints live. **0.9999 (10× longer window) is the correct value for 13-ep recipes.** Could help every closed run by 0.05-0.15pp val_abupt. H51 is the first run with the corrected EMA — provides reference for retrofit consideration.

5. **PR #1199 fern H51 ASSIGNED — NPCA+SSFL stack at slices=192 + ema-decay=0.9999.** Combines student priority #1 (slices=192 capacity scaling) and #6 (ema-decay fix). VRAM safety: H35 peak 78.43 GiB; +50% slice capacity may push to ~88-92 GiB; fallback `--batch-size 2` documented if OOM. Expected gain stacking: +0.10-0.35pp val_abupt vs H35 → potentially closes 6.126% gate.

## Previous invocation actions (2026-05-18 ~22:26Z) — EDWARD H48 TAU-Y-EQUALIZE mid-EP3 stale_wip CLEARED with **UNPRECEDENTED EXTREME BAND-BREAK SIGNAL** (τz/τx mean 0.400 PERSISTING from EP1 → mid-EP3, ALL 34/34 cars outside band, std stable 0.033 — 25× larger mean shift than H46 SDORTH's 0.04, 3rd mechanism class producing band-break, FIRST via training-time loss weight with zero code change); FLEET STILL FULLY BOOKED 8/8 WIP

## Previous invocation actions (2026-05-18 ~22:18Z) — NEZUKO H47 V-DEPTH SENT BACK FOR FULL 18H RERUN (mechanism positive but 6h truncated; test_VP +0.151pp above floor at 20% budget — credible 5th vol_p floor crossing candidate at full budget); ALPHONSE H45 ANCHOR-CROSSCHAN-DEC mid-EP3 stale_wip CLEARED with **STRONG MECHANISM SIGNAL** (τz/τx out_weight_norm ratio 3.91 ≫ 1.3 mech gate; τy/τz channels loading 4-5× over cp/τx); FRIEREN H44 YAW-AUGMENTATION mid-EP6 stale_wip CLEARED with **CRITICAL VAL_VP SIGNAL** (val_VP 3.687% ALREADY BELOW baseline EP13 final 3.80%, descent continuing toward floor 3.643%); FLEET FULLY BOOKED (8/8 WIP, zero idle students)

### Headline updates (22:26Z)

1. **PR #1196 edward H48 TAU-Y-EQUALIZE mid-EP3 — UNPRECEDENTED EXTREME BAND-BREAK SIGNAL.** Run `8cn5abxm` mid-EP3 step 27,220 (84% through EP3, 4.0h runtime, heartbeat 0.1 min ago). **τz/τx mean = 0.400 at mid-EP3, holding stable since EP1 (0.383)** — vs band [1.44, 1.55] lower edge 1.44, that's **−1.04 below band edge**, an order of magnitude larger shift than any prior Wave 30/31 result. **ALL 34/34 val cars OUTSIDE band** (vs H46's 47%). std stays tight at 0.033 (mech is mean SHIFT, NOT std spread — opposite of H26/H31 std-spread mechanism class). val_VP 4.52% at mid-EP3 ahead of baseline EP3 5.90% (val_VP positive signal). val_abupt 7.74% mid-EP3 slightly above baseline 7.11% but EP3 not yet finished.

2. **Wave 30/31 band-break mechanism classes now 3.** H18 (per-vertex area weighting, merged outlier, val 1.46 in band + test 1.418 below band), H46 SDORTH (decoder weight init, val 1.490 rebound + test 1.431 below band — FIRST test-only deflection), and now **H48 TAU-Y-EQUALIZE (training-time loss weight, val + test 0.40 = COMPLETE all-cars deflection)**. H48 mechanism class is fundamentally new: zero code change, single CLI flag, persistent across epochs, every car outside band. **Independent of architecture — could stack with H26, H31, H46 simultaneously.**

3. **Interpretation — reverse difficulty asymmetry.** Baseline τz/τx ratio 1.50 means τz error is 1.5× τx error (τz harder). H48 ratio 0.40 means τz is now EASIER than τx — the model has reversed which channel is the bottleneck. Hypothesis: reducing tau-y-loss-weight 1.5 → 1.0 dis-favors τx via the surface decoder Linear(512, 4) row coupling that the H46 path-dependent attractor proof established. **H48 globally perturbs the training trajectory (every step), producing a much stronger trajectory deflection than H46's init-only perturbation.**

4. **No other PR action needed this invocation.** alphonse H45 EP3 gate already landing ~22:30Z (will check in next invocation); fern H35 / frieren H44 / tanjiro H36 continuing per 22:18Z snapshot; nezuko H47 18h rerun pickup pending; askeladd H50 + thorfinn H49 pickup pending.

### Headline updates (22:18Z)

1. **PR #1194 nezuko H47 V-DEPTH SENT BACK — mechanism unambiguously alive, 6h budget insufficient.** Terminal at mid-EP3 step 28,586 (≈63% through EP3) when 270-min training timeout fired. Per-block residual diagnostic: ALL 4 SUBLAYERS (block0+block1 × attn+ffn) 4-8× ABOVE the 0.05 KILL threshold by EP3, with monotonic growth EP1→EP3 (e.g. block1/ffn_fc2: 0.038 → 0.330 → 0.403). FFN > attn dominance (≈1.5×), block1 > block0 (productive paired asymmetry, H30 closure precedent). Best test_VP 3.794% at 20% budget is **only +0.151pp above floor 3.643%** — the closest of any non-merge-winning Wave 30/31 hypothesis. Send-back recipe: full 18h `SENPAI_TIMEOUT_MINUTES=1100 --epochs 13 --lr-cosine-t-max 13 --train-volume-points 65536 --vol-points-schedule "0:16384:3:32768:6:49152:9:65536"` with step-indexed kill thresholds at 10864/32594/65228 + EP6 val_abupt < 6.5% gate (H26 NPCA reference). VRAM safety: peak ~75-85GB projected at vp=65536 (within 97.9GB Blackwell PRO 6000).

2. **PR #1192 alphonse H45 ANCHOR-CROSSCHAN-DEC mid-EP3 — STRONGEST cross-channel mechanism signal in Wave 31 to date.** Run `lhivsp6j` mid-EP3 step 28,756 (88% through EP3, heartbeat 0.2 min ago, 4.77h runtime). Per-channel out_weight_norm: cp=0.006, τx=0.006, τy=**0.030** (5.2× cp), τz=**0.023** (3.9× cp). **τz/τx ratio = 3.91 ≫ 1.3 mech gate** — module is loading up exactly on the WSS channels it was designed to correct. val_abupt 7.76% mid-EP3, converging on H26 NPCA's EP3 trajectory. EP3 gate landing ~22:30Z.

3. **PR #1190 frieren H44 YAW-AUGMENTATION mid-EP6 — val_VP 3.687% ALREADY below baseline EP13 final (3.80%).** Run `6scw4nto` mid-EP6 step ~58,200 (10.35h runtime, heartbeat 0.2 min ago). **val_VP descent has front-loaded by ~7 epochs vs baseline**. Trajectory: 5.898 (EP2) → 4.064 (EP3, **−31%**) → 3.806 (EP4) → 3.738 (EP5) → 3.687 (mid-EP6). Distance to floor 3.643%: **0.044pp**. Combined with per-axis WSS finding (τz NOT touched by yaw rotation is now AHEAD of baseline by 0.131pp despite no direct mixing — NOVEL cross-channel rotational-prior regularization) and τz/τx mean ratio shift down 0.029 to 1.510 (stable mid-band, vs baseline EP6 1.539). **5th vol_p floor crossing candidate in flight, FIRST via data augmentation axis.** Continue to natural budget cutoff ~05:40Z May 19.

### Fleet status — 8/8 WIP, ZERO idle students (updated 02:30Z)

| Student | PR | H | Status @ 02:30Z | Latest val_abupt | Key signal |
|:--|---:|:--|:--|---:|:--|
| **alphonse** | **#1192** | H45 CROSSCHAN-DEC | 🔥🔥 EP4 step 32.8k, **GATE 1 MASSIVELY POSITIVE τz/τx out_norm 9.75, GATE 2 DECOUPLED (output ratio 1.52 in band)** | **6.960% (EP3)** | weight asymmetry / output band-break decoupling — Wave 31 structural insight |
| askeladd | #1198 | H50 COORDSLICE | 🟢 mid-EP2 step 14.1k (2.15h runtime), **L0-INVERSION CONFIRMED** (L0 inter_slice_cos 0.453 LOWEST, L4 0.647 HIGHEST — opposite of H33's failure pattern) | 28.90% (EP1 cold-start, normal) | coordinate-conditioned anchors delivering L0-targeted positional disambiguation |
| **edward** | **#1196** | **H48 TAU-Y-EQUALIZE** | 🔥🔥🔥🔥 EP4 step 43.6k (7.0h runtime), **τz/τx mean 0.401 PERSISTING ACROSS 4 EPOCHS (drift +0.0006 pp/1k flat)**, val_abupt 6.597% on potential merge trajectory | **6.597% (EP4)** | extreme band-break stable, possible merge candidate at budget cutoff EP10-EP11 |
| frieren | #1200 | H52 NPCA×YAW-AUG | NEW — PR #1200 assigned 02:15Z; H44 CLOSED (test_SP 3.8058% BREACH, test_VP 3.608% 6th crossing, val_abupt 6.340% MISS); H52 stacks --use_local_frame_proj + --yaw-aug-theta-max 5.0 | — | mechanistically orthogonal stack: NPCA rotation-equivariant soften H44's SP tax while preserving vol_p + WSS gains |
| nezuko | #1194 | H47 V-DEPTH | SENT BACK 21:40Z — re-running 18h full budget. Pickup pending | 6.846% (6h terminal) | mechanism alive (block residuals 4-8× over KILL), budget truncated |
| tanjiro | **#1202** | H53 CP-LOSS-WEIGHT | NEW — PR #1202 assigned 02:50Z (replaces malformed #1201 on dl24-prefixed branch); H36 CLOSED (test_VP 3.4780% 7th crossing −0.1650pp DEEPEST, test_SP 3.7169% BREACH, val_abupt 6.2379% MISS); H53 single arm: `--cp-loss-weight 2.0` (orthogonal-arm discipline with H48's `--tau-y-loss-weight 1.0`) | — | H48 TAU-Y-EQUALIZE precedent: single loss-weight flag causes persistent trajectory deflection; cp channel index 0 hardcoded 1.0 in surface_channel_weights |
| thorfinn | #1197 | H49 SDORTH-FULL | NEW — 13-ep H46 confirmation. Pickup pending | — | full-budget test of FIRST test τz/τx mean deflection 1.431 |
| fern | #1199 | H51 NPCA-SSFL-SLICES192 | NEW — H35 closure follow-up: slices=128→192 + ema-decay=0.999→0.9999. Pickup pending | — | use the 2.3× variance room H35 created (n_out_band 17/34) |

### vol_p floor crossing tally — 7 confirmed (updated 02:30Z)

| # | Hypothesis | Mechanism axis | test_vol_p | Δ vs floor | Status |
|---|---|---|---:|---:|---|
| 1 | H31 WALLDIST | encoder-input feature | 3.488% | −0.155pp | MERGED |
| 2 | H26 NPCA | encoder-input feature | 3.607% | −0.036pp | MERGED |
| 3 | H46 SDORTH PathB | decoder weight init | (3-ep gap) | — | closed (PathB) |
| 4 | H33 SLICEPE | encoder slice-PE additive | 3.522% | −0.121pp | closed |
| 5 | H35 NPCA+SSFL | stack: encoder + spectral-loss | 3.585% | −0.058pp | closed 00:15Z |
| 6 | H44 YAW-AUG | data augmentation rotation | 3.608% | −0.035pp | **CLOSED 02:15Z** — test_SP 3.8058% BREACH, val_abupt 6.340% MISS |
| 7 | **H36 ANCHOR-SLICE-QUERIES** | **anchor query modulation** | **3.4780%** | **−0.1650pp** | **CLOSED 02:30Z — DEEPEST vol_p crossing in Wave 31 history; test_SP 3.7169% BREACH, val_abupt 6.2379% MISS** |

5 of 7 closed crossings have val_abupt FAIL on merge dim; 6 of 7 have test_SP BREACH — **vol_p floor is consistently easier than both merge gate and SP floor**. H36 at 3.4780% is the deepest crossing by −0.123pp vs H31's previous record. test_SP remains 0/7 crossings beyond our baseline (3.577%). **SP floor is the binding unsolved gate.**

### Imminent decisions (next 8h)

1. **tanjiro H36 terminal SENPAI-RESULT ~07:30Z May 19** — **LIVE MERGE CANDIDATE** val_abupt 6.245% +0.12pp above gate, val_VP 3.629% below val floor, descending; budget cutoff ~06:10Z + test eval
2. **frieren H44 terminal SENPAI-RESULT ~05:40Z May 19** — 6th vol_p floor crossing watch, currently +0.044pp above
3. **alphonse H45 EP6 gate ~04:00Z May 19** — hard kill at val_abupt > 6.5%; early-positive at val_abupt < 6.35% → CONFIRM run + paper-facing eval at terminal
4. **edward H48 EP3-EP6** — τz/τx mean 0.40 persistence; if holds through EP13 + test → **major Wave 31 structural finding**
5. **askeladd H50 EP3 gate ~04:30Z** — first val_abupt under merge-gate-trajectory check + L0 inter_slice_cos decrease
6. **nezuko H47 18h rerun pickup confirmation** — wait for student to relaunch
7. **thorfinn H49 pickup confirmation** — full-budget H46 confirmation
8. **fern H51 pickup confirmation** — slices=192 + ema-decay=0.9999 (capacity expansion + EMA fix)

### Wave 31 priority ranking (post-00:15Z fleet sweep)

1. **H48 TAU-Y-EQUALIZE** (edward) — IN-FLIGHT mid-EP3, **τz/τx mean 0.400 PERSISTING (25× larger shift than H46)**, all 34/34 cars outside band, single-flag training-time mechanism
2. **H44 YAW-AUGMENTATION** (frieren) — IN-FLIGHT mid-EP6, val_VP 0.044pp from floor; novel rotational-prior cross-channel regularization
3. **H45 ANCHOR-CROSSCHAN-DEC** (alphonse) — IN-FLIGHT mid-EP3, strongest cross-channel out_weight_norm asymmetry (τz/τx 3.91)
4. **H47 V-DEPTH** (nezuko) — sent back for full 18h, mechanism positive, decoder-depth axis FIRST
5. **H49 SDORTH-FULL** (thorfinn) — full-budget H46 confirmation; could yield Wave 31's first single-model paper-facing breakthrough
6. **H51 NPCA-SSFL-SLICES192** (fern, NEW) — H35 closure follow-up: variance room + EMA fix; potentially closes merge gap
7. **H50 COORDSLICE** (askeladd) — DAB-DETR analogue, tests L0-dominance + better test_SP
8. **H36 ANCHOR-SLICE-QUERIES** (tanjiro) — slice-query Q modulation

### Time-tracking note

Real-clock time of this invocation: 22:18Z. Previous logged invocation: 21:25Z. **Frieren H44** mid-EP6 reading is the most consequential development of the past 0.9h — val_VP descent has pushed past baseline-EP13 territory and is approaching the floor. Next ~3h will see alphonse H45 EP3 gate landing, fern H35 EP9-EP10 reading (likely the day's most critical floor watch), and possibly nezuko H47 18h rerun launch.

---

## Previous invocation actions (2026-05-18 ~21:25Z) — ASKELADD H33 SLICEPE CLOSED EP13-EMA (23RD WAVE 30 DE on merge dimension, 🏆 **4TH TEST_VOL_P FLOOR CROSSING −0.121pp + NOVEL L0-DOMINANCE STRUCTURAL INSIGHT**); ASKELADD REASSIGNED H50 COORDSLICE (PR #1198, **COORDINATE-CONDITIONED SLICE IDS / DAB-DETR ANALOGUE** — replaces H33's free-floating learnable PE with 3D-centroid-derived slice anchors); WAVE 31 PRIORITY SHIFT: test_SP becomes the binding gate (4 vol_p crossings via 3 mech classes vs 0 test_SP crossings beyond baseline)

### Headline updates (21:25Z)

1. **PR #1187 askeladd H33 SLICEPE CLOSED — 23rd Wave 30 DE on merge dimension, BUT 4th test_vol_p floor crossing.** Terminal EP13 EMA (run `u58fwoym`, 838.16 min / 1100 min budget, clean completion): val_abupt **6.472%** FAIL +0.346pp, test_SP **3.793% FLOOR BREACH** +0.216pp, **test_vol_p 3.522% PASS −0.121pp ✅** (joins H31, H26, H46 Path B as 4th vol_p crossing). test_abupt 5.960% (+0.116pp), test_WSS 6.856% (+0.129pp), test τz/τx 1.487 (in band — no deflection). Descent saturated mid-EP7 with slope decay −0.0035 → −0.0004 pp/1k steps (9× slowdown).

2. **Wave 30/31 test_vol_p floor crossing tally now at 4 — but test_SP remains the binding merge gate.** Vol_p crossings via 3 mechanism classes (encoder-input feature, encoder-additive PE, decoder weight init) vs 0 test_SP crossings beyond baseline. **Wave 31 hypothesis design should prioritize surface-pressure-specific mechanisms** (per-vertex weighting, surface decoder restructuring, surface-side asymmetry correction). The vol_p pathway has 4 demonstrated routes — diminishing returns from additional vol_p crossing attempts unless they also unlock test_SP.

3. **H33 produced first cold-start fade with EXPLICIT per-layer L0-DOMINANCE diagnosis.** `inter_slice_cos` terminal: L0=**0.0851** (largest by 3-10×), L1=0.019, L2=0.029, L3=−0.001, L4=0.028. Predicted depth-monotonic (L0 < L1 < ... < L4) — actual pattern is L0 >> L4 ≈ L2 > L1 ≈ L3 (NON-MONOTONIC, L0-DOMINANT). slice_pe parameters auto-grew σ=0.020 → σ=0.150 (7.5×, model rediscovered O(1/√D) ≈ 0.088 emergently). **Novel insight**: L0 attends directly to raw geometric input and benefits most from per-slice positional disambiguation; deeper layers prefer content-routing. Reshapes Wave 31 hypothesis design toward L0-targeted mechanisms.

4. **PR #1198 askeladd H50 COORDSLICE ASSIGNED — coordinate-conditioned slice IDs (DAB-DETR analogue).** Replaces H33's free-floating learnable slice_pe with slice anchors derived from **physical 3D coordinates of each slice's centroid** at runtime. References: DAB-DETR (ICLR 2022, anchor boxes as queries), Anchor DETR (AAAI 2022, anchor-based query PE), Conditional DETR (ICCV 2021). Mechanism prediction: physically-grounded slice anchors at L0 should produce cleaner mechanism than auto-grown free-floating IDs, with potentially better test_SP behavior. Single new flag `--use-coord-slice-pe`. Full 13-ep budget, same recipe as H33 v2. Init scale O(1/√D_head) = 0.088 per H33 lesson.

### Wave 30 closure tally — UPDATED: **23 dead ends** on merge dimension + 3 mechanism wins + 1 outlier merged (H18)

| Closure # | Hypothesis | Tier | Mechanism contribution |
|---:|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape | none |
| 8 | H23 Mean Teacher | Training-reg | none |
| 9 | H18 area-weighted MSE (merged outlier) | Per-vertex position | mean + std shift via area weighting |
| 10 | H21 per-component heads | Decoder capacity | none |
| 11 | H25 ALGP | Aux head | EP1 deflection only |
| 12 | H27 PRLP | Train-eval space | none |
| 13 | H32 DIFFATTN | Attention | magnitude collapse (anti-mechanism) |
| 14 | H28 SAM | Optimizer | none |
| 15 | H24 GSTS | Encoder slice-temp | none |
| 16 | H29 SSFL | Frequency loss | spectral_loss descended only |
| 17 | H18d τz-only area | Channel-decoupled position | INVERSE band-break diagnostic |
| 18 | H31 WALLDIST | Encoder-input feature | 🏆 1st test_vol_p floor crossing −0.155pp |
| 19 | H26 NPCA | Encoder-input local-frame | 🏆🏆 2nd test_vol_p crossing + canonical std-spread mech win |
| 20 | H30 V2S xattn | Encoder-fusion (V→S) | EP1 deflection 1.428; V2S/S2V 5.3× asymmetry diagnostic |
| 21 | H34 OUTHEAD | Surface decoder per-channel aux | ANTI-DIRECTION τ_x 3.4× larger than τ_z |
| 22 | H46 SDORTH Path B (5-ep) | Surface decoder weight INIT | 🏆🏆🏆 3rd mech win: FIRST test τz/τx mean deflection (1.431) + path-dependent attractor proof |
| **23** | **H33 SLICEPE (13-ep)** | **Encoder slice-PE additive** | **🏆 4th test_vol_p floor crossing −0.121pp + novel L0-dominance per-layer mechanism diagnosis** |

### Wave 30/31 test_vol_p floor crossing tally — 4 distinct mechanism classes

| # | Hypothesis | Axis | val_abupt | test_vol_p | Merged? |
|---:|---|---|---:|---:|:--|
| 1 | H31 WALLDIST | Encoder-input log-SDF | 6.176% | **−0.155pp** | ✅ MERGED |
| 2 | H26 NPCA | Encoder-input local-frame | improved | **−0.035pp** | ✅ MERGED |
| 3 | H46 SDORTH Path B | Decoder weight init | 6.868% (3-ep gap) | floor breach (3-ep) | ❌ FLOOR BREACH on screening |
| **4** | **H33 SLICEPE** | **Encoder slice-PE additive** | **6.472%** | **−0.121pp** | ❌ NOT-A-MERGE (val_abupt + test_SP) |

**Pattern observation: vol_p floor crossable via 3 mech classes; test_SP floor unyielding beyond baseline.** Wave 31 should target test_SP-specific mechanisms.

### Three orthogonal Wave 30/31 mechanism wins now characterized (unchanged from 20:20Z)

| Mech win | Axis | val std | test std | test mean | test_vol_p |
|---|---|---:|---:|---:|---|
| H26 NPCA | Encoder-input local-frame | 0.259 | 0.132 | 1.467 (preserved) | 🏆 −0.035pp |
| H31 WALLDIST | Encoder-input log-SDF | ~0.25 | ~0.10 | 1.470 (preserved) | 🏆 −0.155pp |
| H46 SDORTH | Decoder weight init | 0.194 (EP3 only) | 0.112 | 1.431 ⭐ deflected | ❌ on 3-ep budget; H49 full-budget pending |

### Fleet status (8/8 active)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| **askeladd** | **#1198** | **H50 COORDSLICE** | **NEW — coordinate-conditioned slice IDs (DAB-DETR analogue). Pickup pending** | — |
| thorfinn | #1197 | H49 SDORTH-FULL | NEW — full 13-ep confirmation of H46 mechanism. Pickup pending | — |
| edward | #1196 | H48 TAU-Y-EQUALIZE | RUNNING — launched ~18:25Z, mid-EP4-EP5 by now | — |
| nezuko | #1194 | H47 V-DEPTH | RUNNING — EP1-EP3 zero-init shedding | (early) |
| alphonse | #1192 | H45 ANCHOR-CROSSCHAN-DEC | RUNNING — EP1+ pending | — |
| tanjiro | #1191 | H36 ANCHOR-SLICE-QUERIES | RUNNING mid-EP4-EP5, mech-positive τz/τx std 0.155-0.188 | 6.779% (EP3) |
| frieren | #1190 | H44 YAW-AUGMENTATION | RUNNING mid-EP5-EP6, val_VP 3.806% novel −31% descent signal | 6.634% (EP4) |
| fern | #1189 | H35 NPCA-SSFL-STACK | RUNNING EP8+, fleet-peak τz/τx std **0.2464**, val_VP **3.738%** — 0.095pp from floor (CRITICAL mech-positive) | 6.377% (EP7) |

### Updated Wave 31 attack map (post-H33 closure, post-L0-dominance insight)

**Closed mechanism axes:**
- Loss-shape — closed (Wave 30)
- Per-vertex position weighting — closed
- Training regularization — closed
- Optimizer-space — closed
- Encoder slice-temp — closed
- Auxiliary head supervision — closed
- Decoder capacity / per-component split — closed
- Train-eval space alignment — closed
- Attention differentiation — closed
- Encoder-fusion (V→S) — closed
- Encoder-input enrichment — 2 mech wins (H26, H31), continued exploration via stacking
- Surface decoder weight INIT — 3rd mech win via path-dependent test mean deflection (H46), full-budget confirmation in flight (H49)
- **Encoder slice-PE additive (free-learnable)** — closed via H33 SLICEPE; L0-dominance mechanism diagnosis surfaced

**Open mechanism axes (in-flight + new):**
- **Coordinate-conditioned slice IDs** (H50 askeladd — DAB-DETR analogue, NEW) — replaces H33's free PE with physical 3D centroids
- **Surface decoder pre-projection structure** (H45 alphonse — cross-channel attention)
- **Surface decoder weight init FULL BUDGET** (H49 thorfinn — 13-ep confirmation)
- **Volume decoder interior capacity** (H47 nezuko — depth bump)
- **Loss weight equalization** (H48 edward — single flag)
- **Slice-query Q modulation** (H36 tanjiro — anchor DETR Q-only)
- **Data augmentation** (H44 frieren — z-yaw augmentation)
- **Mechanism stacking** (H35 fern — encoder + frequency, fleet-peak τz/τx std)

**Reserved for next idle slot:**
- **LinearNO drop-slice-attention** (arXiv:2511.06294) — HIGH-risk HIGH-reward Transolver simplification; reserved as Wave 31 high-impact swing if next attacker fails

### Imminent decisions (next 4h)

1. **fern H35 EP9-EP10** — CRITICAL val_VP floor watch (was 3.738% at EP7, only 0.095pp from floor 3.643%; if held at EP10+ → 5th vol_p crossing candidate AND first NPCA-stacking mech-win)
2. **frieren H44 EP6-EP7** — val_VP descent continuation (was 3.806% at EP4)
3. **tanjiro H36 EP6+** — τz/τx std persistence + test_SP gate
4. **edward H48 EP3-EP4** — single-flag τ-y-equalize mechanism gate
5. **alphonse H45 EP3+** — cross-channel xattn pickup confirmation
6. **nezuko H47 EP3-EP5** — V-DEPTH zero-init shedding check + descent gate
7. **thorfinn H49 pickup confirmation** — full-budget H46 confirmation (highest info-value follow-up)
8. **askeladd H50 pickup confirmation** — coordinate-conditioned slice IDs (DAB-DETR analogue, NEW)

### Wave 31 priority ranking (post-H33 closure)

1. **H49 SDORTH-FULL** (thorfinn) — full-budget confirmation of FIRST test-side mean deflection; could yield Wave 31's first paper-facing breakthrough
2. **H35 NPCA-SSFL-STACK** (fern) — IN-FLIGHT highest-mech-signal candidate; fleet-peak std 0.2464, val_VP 0.095pp from floor at EP7
3. **H50 COORDSLICE** (askeladd, NEW) — DAB-DETR analogue; tests whether physically-grounded slice anchors break the L0-dominance pattern + improve test_SP
4. **H44 YAW-AUGMENTATION** (frieren) — IN-FLIGHT novel data-aug mechanism (val_VP −31% descent)
5. **H45 ANCHOR-CROSSCHAN-DEC** (alphonse) — surface decoder pre-projection axis (re-interpret as trajectory perturbation per H46 finding)
6. **H47 V-DEPTH** (nezuko) — volume DECODER attack
7. **H48 TAU-Y-EQUALIZE** (edward) — single-flag training-time test
8. **H36 ANCHOR-SLICE-QUERIES** (tanjiro) — slice-query positional differentiation (Q modulation only)

### Time-tracking note

Real-clock time of this invocation is 21:25Z. Previous invocation logged 20:20Z. Fern H35 has been running ~1.5h since the 20:20Z snapshot — likely at EP9-EP10 by now. If val_VP held below floor at EP9+ → first NPCA-stacking mech-win Wave 31 confirmation. Watch fern H35 closely in next invocation. The askeladd H33 SLICEPE completion adds 4th test_vol_p crossing to the tally and surfaces the L0-dominance insight that motivated the H50 COORDSLICE assignment.

---

## Previous invocation actions (2026-05-18 ~20:20Z) — THORFINN H46 SDORTH CLOSED EP3-BEST PATH B 5-EP (22ND WAVE 30 DE on merge dimension, **🏆🏆🏆 3RD MECHANISM WIN: FIRST TEST τz/τx MEAN DEFLECTION 1.431 BELOW BAND ⭐ + PATH-DEPENDENT ATTRACTOR PROOF — DEFINITIVELY ANSWERS WAVE 30 STRUCTURAL QUESTION**); THORFINN REASSIGNED H49 SDORTH-FULL (PR #1197, **FULL 13-EPOCH CONFIRMATION OF H46 MECHANISM** — same 3-LOC code, full budget); WAVE 30 STRUCTURAL FINDING FINALLY ANSWERED (τz/τx band attractor lives in TRAINING TRAJECTORY, not weight values — orth init perturbation persists in TEST mean despite weight-level orth fully gradient-overwritten by EP1)

### Headline updates (20:20Z)

1. **PR #1193 thorfinn H46 SDORTH CLOSED — 22nd Wave 30 DE on merge dimension, BUT 3rd mechanism win (cleanest mechanism proof in Wave 30/31).** Terminal: val_abupt 6.868% (FAIL, 3-ep budget vs 13-ep baseline gap), test_abupt 6.595% (+0.751pp regression), test_SP 4.226% **FLOOR BREACH** +0.649pp, test_vol_p 3.917% **FLOOR BREACH** +0.274pp, test_WSS 7.594%. **FIRST EVER test-side τz/τx mean deflection in Wave 30/31: test mean = 1.431 (below band lower edge 1.44 ⭐)** — val mean rebounded to 1.490 (in band) but test mean held at 1.431. 23/50 (46%) test cars displaced. val std monotonic 0.085 → 0.163 → 0.194 (>0.15 mech gate). Path B 5-ep screening recipe; floor breaches attributed to 3-ep vs 13-ep budget gap.

2. **Wave 30 structural question DEFINITIVELY ANSWERED — path-dependent attractor proof.** `surface_proj_row/cos_max_abs` trajectory: step 0 = 2.3e-08 → EP1 0.143 → EP2 0.204 → EP3 **0.210** (2.1× baseline 0.098). Weight-level orth structure is **fully gradient-overwritten BY EP1** and continues drifting past baseline. Yet val std keeps growing and TEST mean stays below band. **The τz/τx band attractor is a TRAINING-TRAJECTORY attractor, NOT a fixed-point in weight space.** This reshapes the Wave 31 attack map: H45 (encoder pre-projection) / H47 (volume decoder depth) / H48 (loss weight) attacks now reinterpret as trajectory perturbations rather than weight-structure attacks.

3. **PR #1197 thorfinn H49 SDORTH-FULL ASSIGNED — full 13-epoch confirmation of H46 mechanism.** Same 3-LOC code change (`--use-surface-orth-init --surface-orth-init-std 0.02`), full 18h baseline-equivalent recipe (`--epochs 13 --lr-cosine-t-max 13 --vp-points-schedule 0:16384:3:32768:6:49152:9:65536`). Binary high-info binding question: does the test τz/τx mean deflection (1.431 at EP3) PERSIST at terminal EP13, or rebound to band? Secondary question: do floor breaches close with 10 additional epochs? If mean deflection persists → **first single-model paper-facing breakthrough on surface decoder mean-deflection axis**. If rebounds → confirms late-stage cosine-decay regularizes back to attractor.

### Wave 30 closure tally — UPDATED: 22 dead ends on merge dimension + 3 mechanism wins (H31 vol_p, H26 NPCA full test generalization, **H46 SDORTH first test-side mean deflection**) + 1 outlier merged (H18)

| Closure # | Hypothesis | Tier | Mechanism contribution |
|---:|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape | none |
| 8 | H23 Mean Teacher | Training-reg | none |
| 9 | H18 area-weighted MSE (merged outlier) | Per-vertex position | mean + std shift via area weighting |
| 10 | H21 per-component heads | Decoder capacity | none |
| 11 | H25 ALGP | Aux head | EP1 deflection only |
| 12 | H27 PRLP | Train-eval space | none |
| 13 | H32 DIFFATTN | Attention | none |
| 14 | H28 SAM | Optimizer | none |
| 15 | H24 GSTS | Encoder slice-temp | none |
| 16 | H29 SSFL | Frequency loss | spectral_loss descended only |
| 17 | H18d τz-only area | Channel-decoupled position | INVERSE band-break diagnostic |
| 18 | H31 WALLDIST | Encoder-input feature | 🏆 1st test_vol_p floor crossing |
| 19 | H26 NPCA | Encoder-input local-frame | 🏆🏆 2nd test_vol_p crossing + canonical std-spread mech win |
| 20 | H30 V2S xattn | Encoder-fusion (V→S) | EP1 deflection 1.428; V2S/S2V 5.3× asymmetry diagnostic |
| 21 | H34 OUTHEAD | Surface decoder per-channel aux | ANTI-DIRECTION τ_x 3.4× larger than τ_z |
| **22** | **H46 SDORTH Path B (5-ep)** | **Surface decoder weight INIT** | **🏆🏆🏆 3rd mechanism win: FIRST test τz/τx mean deflection (1.431 below band ⭐) + path-dependent attractor proof. Floor breach blocks merge ON SCREENING BUDGET ONLY — H49 SDORTH-FULL follow-up assigned** |

### Three orthogonal Wave 30/31 mechanism wins now characterized

| Mech win | Axis | val std | test std | test mean | test_vol_p |
|---|---|---:|---:|---:|---|
| H26 NPCA | Encoder-input local-frame | 0.259 | 0.132 | 1.467 (preserved) | 🏆 −0.035pp |
| H31 WALLDIST | Encoder-input log-SDF | ~0.25 | ~0.10 | 1.470 (preserved) | 🏆 −0.155pp |
| **H46 SDORTH** | **Decoder weight init** | **0.194** (EP3 only) | **0.112** | **1.431 ⭐ deflected** | ❌ on 3-ep budget; H49 full-budget check pending |

### Fleet status (8/8 active)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| **thorfinn** | **#1197** | **H49 SDORTH-FULL** | **NEW — full 13-ep confirmation of H46. Pickup pending** | — |
| edward | #1196 | H48 TAU-Y-EQUALIZE | RUNNING — just launched ~18:25Z, EP1 ~19:30Z | — |
| nezuko | #1194 | H47 V-DEPTH | RUNNING — EP1-EP3 zero-init shedding | (early) |
| alphonse | #1192 | H45 ANCHOR-CROSSCHAN-DEC | RUNNING — EP1 pending | — |
| tanjiro | #1191 | H36 ANCHOR-SLICE-QUERIES | RUNNING mid-EP3-EP4, mech-positive τz/τx std 0.155-0.188 | 6.779% (EP3) |
| frieren | #1190 | H44 YAW-AUGMENTATION | RUNNING mid-EP4-EP5, val_VP 3.806% novel −31% descent signal | 6.634% (EP4) |
| askeladd | #1187 | H33 SLICEPE v2 | RUNNING mid-EP6, primary mech FALSIFIED (depth pattern backwards), val_VP 3.754% descent toward floor | 6.475% (mid-EP6) |
| fern | #1189 | H35 NPCA-SSFL-STACK | RUNNING EP7+, fleet-peak τz/τx std **0.2464**, val_VP **3.738%** — 0.095pp from floor (CRITICAL mech-positive) | 6.377% (EP7) |

### Updated Wave 31 attack map (post-H46 structural finding)

**Closed mechanism axes (Wave 30 + H46):**
- Loss-shape in ALL forms — closed
- Per-vertex position weighting — closed
- Training regularization — closed
- Optimizer-space — closed
- Encoder slice-temp — closed
- Auxiliary head supervision — closed
- Decoder capacity / per-component split — closed
- Train-eval space alignment — closed
- Attention differentiation — closed
- Encoder-input enrichment — 2 mech wins via std-spread + mean preservation (H26, H31)
- Encoder-fusion (V→S) — closed
- **Surface decoder weight INIT — 🏆🏆🏆 3rd mech win via path-dependent test mean deflection (H46)**

**Open mechanism axes (Wave 31 in-flight + new):**
- **Surface decoder pre-projection structure** (H45 alphonse — cross-channel attention) — RE-INTERPRET as trajectory perturbation per H46 finding
- **Surface decoder weight init FULL BUDGET** (H49 thorfinn — 13-ep confirmation, NEW)
- **Volume decoder interior capacity** (H47 nezuko — depth bump)
- **Loss weight equalization** (H48 edward — single flag)
- **Slice-query architecture** (H33 askeladd; H36 tanjiro)
- **Data augmentation** (H44 frieren — first attack on this tier, val_VP novel signal)
- **Mechanism stacking** (H35 fern — encoder + frequency, fleet-peak τz/τx std)

### Imminent decisions (next 4h)

1. **fern H35 EP8-EP9** — CRITICAL val_VP floor watch (currently 3.738%, only 0.095pp from floor 3.643%)
2. **frieren H44 EP5-EP6** — val_VP descent continuation (currently 3.806% at EP4)
3. **askeladd H33 EP7-EP8** — val_VP trajectory (currently 3.754% mid-EP6, slow descent)
4. **tanjiro H36 EP6+** — τz/τx std persistence check
5. **edward H48 EP1-EP3** — single-flag mechanism gate
6. **alphonse H45 EP1+** — cross-channel xattn pickup
7. **nezuko H47 EP3** — V-DEPTH zero-init shedding check
8. **thorfinn H49 pickup** — full-budget H46 confirmation (highest info-value follow-up)

### Wave 31 priority ranking (post-H46 mechanism win)

1. **H49 SDORTH-FULL** (thorfinn) — full-budget confirmation of FIRST test-side mean deflection (H46 path-dependent attractor proof); could yield Wave 31's first paper-facing breakthrough
2. **H35 NPCA-SSFL-STACK** (fern) — IN-FLIGHT highest-mech-signal candidate; fleet-peak std 0.2464, val_VP 0.095pp from floor
3. **H44 YAW-AUGMENTATION** (frieren) — IN-FLIGHT novel data-aug mechanism (val_VP −31% descent)
4. **H45 ANCHOR-CROSSCHAN-DEC** (alphonse) — surface decoder pre-projection axis (re-interpret as trajectory perturbation per H46 finding)
5. **H47 V-DEPTH** (nezuko) — volume DECODER attack
6. **H48 TAU-Y-EQUALIZE** (edward) — single-flag training-time test
7. **H36 ANCHOR-SLICE-QUERIES** (tanjiro) — slice-query positional differentiation
8. **H33 SLICEPE v2** (askeladd) — primary mech falsified but val_VP continuing

### Time-tracking note

Real-clock time of this invocation is 20:20Z. Previous invocation logged 18:06Z. Fern H35 has been running through this entire interval and likely crossed multiple critical thresholds (val_VP floor watch). Per-PR status report at 19:09Z (posted to issue #1056) showed H35 at EP7 with τz/τx std 0.2464 and val_VP 3.738%. Watch fern H35 closely in next invocation — could be the first Wave 31 single-model merge candidate.

---

## Previous invocation actions (2026-05-18 ~18:06Z) — TANJIRO H36 STALE_WIP CLEARED MID-EP3 (mechanism-positive at EP2: tau_zx_ratio_std=0.155 ALREADY ABOVE EP3 falsifier threshold 0.15); BACKFILL: EDWARD H34 OUTHEAD CLOSED 16:35Z by prior advisor invocation (21ST Wave 30 DE / advisor-killed mid-EP6 / ANTI-DIRECTION τ_x 3.4× larger than τ_z aux mean falsifies head-side rank-coupling hypothesis); EDWARD ASSIGNED H48 TAU-Y-EQUALIZE (PR #1196 — single-flag `--tau-y-loss-weight 1.5→1.0`, hypothesis derived from H34 anti-direction finding)

### Headline updates (18:06Z)

1. **PR #1191 tanjiro H36 ANCHOR-SLICE-QUERIES stale_wip CLEARED — mechanism-positive trajectory mid-EP3.** Run `vu93lzgc` at step 26,932 (mid-EP3, 3.99h, heartbeat <1 min), all 8 ranks tight (step spread 28). **EP2 reading: val_abupt 7.616%, tau_zx_ratio_per_car_std = 0.155 ALREADY OVER the EP3 mechanism falsifier threshold (0.15)** — variance-class signal materializing one epoch earlier than gate. anchor_mod_abs_mean ramped 0.080 → 0.632 (~8×) across EP1→EP2, anchor_pairwise_dist_mean stable at 2.156. n_outside_band 21/34 (EP1) → 11/34 (EP2) — redistribution sharpening. **tau_zx_ratio_mean = 1.498 stays INSIDE [1.44, 1.55] band attractor**, consistent with Wave 30 finding that band MEAN is fixed by surface decoder Linear(512,4) projection — H36 producing variance class WITHOUT mean shift (encoder-side mechanism axis). Continuation: watch EP3 gate ~16:30Z (~5min stale, gate ETA actually 18:30Z now after correcting time stamp). H36 EP2 signal parallels H26 NPCA closure pattern — pre-watch for val_SP merge-blocker + test-side fade.

2. **PR #1188 edward H34 OUTHEAD CLOSED — 21st Wave 30 DE (retroactive, advisor-killed mid-EP6).** Closed 16:35Z by previous advisor invocation. Run `iw2ommjz` 7.36h runtime, terminal EP5 val_abupt 6.542% (descending 6.867% → 6.627% → 6.542%) but mechanism falsified at EP3. **Aux head asymmetry τ_z/τ_x abs_mean = 0.298** (predicted >1.5) — **ANTI-DIRECTION**: τ_x aux head 3.4× larger than τ_z, opposite of rank-coupling hypothesis prediction. τz/τx ratio 1.531 STUCK in [1.44, 1.55] band at EP3. **Per-channel head capacity axis decisively closed** — surface decoder bottleneck is NOT missing per-channel capacity. The H45 (pre-projection) + H46 (weight init) attacks on remaining surface decoder axes are now the live front.

3. **PR #1196 edward H48 TAU-Y-EQUALIZE ASSIGNED — derived from H34 anti-direction finding.** Hypothesis: the τ_x>τ_z aux head asymmetry from H34 suggests τ_x is gradient-favored over τ_z, which may stem from `--tau-y-loss-weight 1.5` crowding out τ_z's share of the surface loss gradient. **Single flag change**: `--tau-y-loss-weight 1.5 → 1.0` (standalone test — never been tried in isolation). Step-indexed kill thresholds (10864, 32594, etc per Wave 30 lesson). Zero code change, zero OOM risk. EP3 mechanism gate: std(τz/τx) ≥ 0.15 OR mean(τz/τx) < 1.44 at any checkpoint.

### Wave 30 closure tally — UPDATED FINAL: 21 dead ends + 2 mechanism wins (H31 vol_p, H26 NPCA) + 1 outlier merged (H18)

| Closure # | Hypothesis | Tier | Mechanism contribution |
|---:|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape | none |
| 8 | H23 Mean Teacher | Training-reg | none |
| 9 | H18 area-weighted MSE | Per-vertex position | merged outlier |
| 10 | H21 per-component heads | Decoder capacity | none |
| 11 | H25 ALGP | Aux head | EP1 deflection only |
| 12 | H27 PRLP | Train-eval space | none |
| 13 | H32 DIFFATTN | Attention | none |
| 14 | H28 SAM | Optimizer | none |
| 15 | H24 GSTS | Encoder slice-temp | none |
| 16 | H29 SSFL | Frequency loss | spectral_loss descended only |
| 17 | H18d τz-only area | Channel-decoupled position | INVERSE band-break diagnostic |
| 18 | H31 WALLDIST | Encoder-input feature | 🏆 1st test_vol_p floor crossing |
| 19 | H26 NPCA | Encoder-input local-frame | 🏆🏆 2nd test_vol_p crossing + test generalization |
| 20 | H30 V2S xattn | Encoder-fusion (V→S) | EP1 deflection 1.428; V2S/S2V 5.3× asymmetry diagnostic |
| **21** | **H34 OUTHEAD** | **Surface decoder per-channel aux** | **ANTI-DIRECTION (τ_x 3.4× larger than τ_z): rank-coupling hypothesis FALSIFIED; capacity axis closed** |

### Fleet status (8/8 active)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| **edward** | **#1196** | **H48 TAU-Y-EQUALIZE** | **NEW — single-flag `--tau-y-loss-weight 1.5→1.0` derived from H34 anti-direction. Pickup pending** | — |
| nezuko | #1194 | H47 V-DEPTH | NEW — volume decoder interior capacity, 2 vol-only blocks after trunk, zero-init residual. Pickup pending | — |
| thorfinn | #1193 | H46 SDORTH | NEW — surface decoder init axis attack, 3 LOC change. Pickup pending | — |
| alphonse | #1192 | H45 ANCHOR-CROSSCHAN-DEC | NEW — surface decoder pre-projection cross-channel attn. Pickup pending | — |
| tanjiro | #1191 | H36 ANCHOR-SLICE-QUERIES | RUNNING mid-EP3, mechanism-positive at EP2 (tau_zx_std 0.155 > 0.15 gate). EP3 gate ~18:30Z | 7.616% (EP2) |
| frieren | #1190 | H44 YAW-AUGMENTATION | RUNNING — z-axis yaw, `wave31_h44_yaw_augmentation` | — |
| askeladd | #1187 | H33 SLICEPE v2 | RUNNING — EP4 6.541% fleet-mid, EP6 ETA ~16:30Z (slipped earlier; check actual state) | 6.541% (EP4) |
| fern | #1189 | H35 NPCA-SSFL-STACK | RUNNING — `wave31_h35_npca_ssfl_stack` | 7.805% (EP2) |

### Wave 31 mechanism-axis attack matrix (post-H34 closure)

**Surface decoder structural attacks** (3 orthogonal — all in-flight, pickup pending):

| Hypothesis | Student | Axis | LOC | Mechanism class | Status |
|---|:--|:--|---:|:--|:--|
| **H34 OUTHEAD** (CLOSED) | edward | POST-projection per-channel aux | ~30 | ✗ anti-direction falsification | DE |
| H45 CROSSCHAN-DEC | alphonse | PRE-projection cross-channel attn | ~50 | cross-channel attention | pickup pending |
| H46 SDORTH | thorfinn | PROJECTION WEIGHT INIT | 3 | orthogonal row init | pickup pending |
| **H48 TAU-Y-EQUALIZE** | edward | Loss weight equalization (training-time) | 0 (flag) | gradient capacity allocation | pickup pending |

**Volume pathway attacks** (input-vs-decoder split — 2 input-axis CLOSED with mechanism wins, 1 decoder-axis NEW):

| Hypothesis | Student | Axis | Mechanism class |
|---|:--|:--|:--|
| H31 (closed) WALLDIST | alphonse | Encoder-input (log-SDF) | 🏆 1st test_vol_p floor crossing −0.155pp |
| H26 (closed) NPCA | thorfinn | Encoder-input local-frame | 🏆🏆 2nd test_vol_p crossing −0.035pp + test generalization |
| H47 V-DEPTH | nezuko | Volume DECODER interior capacity | +2 vol-only TransolverBlocks after trunk |
| H35 stack | fern | Encoder-input + frequency-loss stack | mechanism stacking experiment |

### Tanjiro H36 EP2 mechanism reading — first Wave 31 IN-FLIGHT mechanism-positive signal

EP2 measurements (n=34 val cars):
- **tau_zx_ratio_per_car_std = 0.155** ✅ over EP3 falsifier threshold 0.15 (CRITICAL)
- tau_zx_ratio_mean = 1.498 (in [1.44, 1.55] band — Wave 30 finding holds: encoder-side mechanism class)
- n_outside_band = 11/34 (32%) vs 21/34 (62%) at EP1 — redistribution sharpening
- anchor_mod_abs_mean = 0.632 (~8× EP1 reading) — modulator firing
- anchor_pairwise_dist_mean = 2.156 stable (no collapse, no explosion)

**Comparison to H26 NPCA closure**: H26 EP1 std 0.092 → terminal val std 0.259 → test std 0.132 (24/50 cars outside band) + mean preserved at 1.467 ≈ baseline 1.473 → crossed test_vol_p floor. H36 EP2 std 0.155 is BETTER than H26's val-EP-equivalent — but the H26 fade pattern (test_SP floor breach due to surface decoder mean-preservation) may still cap H36. Watch EP3 gate + val_SP trajectory.

### Wave 31 priority ranking (post H34 closure + H36 EP2 mech-positive)

1. **H36 ANCHOR-SLICE-QUERIES** (tanjiro) — EP2 mechanism-positive, encoder-side variance-class signal LIVE; EP3 gate ~18:30Z (binding falsifier)
2. **H46 SDORTH** (thorfinn) — cleanest single-variable test of surface decoder weight init; 3 LOC; pickup pending
3. **H45 ANCHOR-CROSSCHAN-DEC** (alphonse) — surface decoder pre-projection axis; pickup pending
4. **H47 V-DEPTH** (nezuko) — first Wave 31 volume DECODER attack; pickup pending
5. **H48 TAU-Y-EQUALIZE** (edward) — single-flag training-time test derived from H34 closure (NEW)
6. **H44 YAW-AUGMENTATION** (frieren) — first data-aug attack
7. **H33 SLICEPE v2** (askeladd) — slice-token PE; EP6 gate due
8. **H35 NPCA-SSFL-STACK** (fern) — first mechanism-combination experiment

### Imminent decisions (next 4h)

1. **tanjiro H36 EP3 gate ~18:30Z** — binding mechanism falsifier (std ≥ 0.15 expected to hold given EP2 0.155); val_abupt < 9.5% AND SP < 5.5% kill bars
2. **askeladd H33 v2 EP6** — was due ~16:30Z, check actual state and post next gate
3. **frieren H44** — EP1 telemetry due
4. **alphonse/thorfinn/nezuko/edward pickup** — 4 new assignments waiting (H45/H46/H47/H48)

### Time-tracking note

The previous invocation's commit timestamp (15:30Z) was authored when the actual time was ~16:00Z+ — there was clock drift in my time-stamping. This invocation operates at 18:06Z real-clock. The state file now uses real-clock timestamps. The check-human-issues skill execution at 18:05Z autonomously created the H48 assignment from its forked context — a legitimate completion of the previous-advisor closure of #1188 (which left edward idle at 16:37Z). The forked-skill action is logged transparently here.

---

## Previous invocation actions (2026-05-18 ~15:30Z) — NEZUKO H30 V2S XATTN CLOSED EP13 (20TH WAVE 30 DE, **NO mechanism-class win** but FIRST WAVE 30 QUANTITATIVE MEASUREMENT OF ARCHITECTURAL-FUSION ASYMMETRY: V2S out_proj.max_abs 0.187 vs S2V 0.988 = **5.3× S2V dominance**); NEZUKO REASSIGNED H47 V-DEPTH (PR #1194, **FIRST WAVE 31 ATTACK ON VOLUME DECODER INTERIOR CAPACITY** — complement to H31/H26 vol_p crossings); WAVE 30 NOW COMPLETE AT 20 DE + 2 MECHANISM WINS + 1 OUTLIER MERGED

### Headline updates (15:30Z)

1. **PR #1184 nezuko H30 V2S xattn CLOSED — 20th Wave 30 dead end, NO mechanism-class win.** Terminal: val_abupt 6.362% (FAIL +0.236pp), test_abupt 6.091% (FAIL +0.247pp), test_SP 3.866% **FLOOR BREACH** +0.289pp, test_vol_p 3.781% **FLOOR BREACH** +0.138pp, test_WSS 6.976% (+0.249pp regression). Test τz/τx 1.462 **STAYS INSIDE band attractor** — the H18 watch-item-3 hypothesis (val-side fade + test-side survival) did NOT replicate. EP1 τz/τx 1.428 was deepest Wave 30 EP1 deflection from architectural-fusion path but faded to band by test. **10th cold-start fade in Wave 30** (joining the architectural-fusion axis as the 6th distinct mechanism axis showing the same fade pattern).

2. **Canonical V2S vs S2V asymmetry diagnostic adopted (NEW Wave 31 metric).** First Wave 30 quantitative measurement of architectural-fusion capacity asymmetry: forward-direction (S2V baseline) out_proj.max_abs 0.988 vs reverse-direction (V2S new) 0.187 — **5.3× S2V dominance**. The supervised loss landscape rewards S2V's information channel much more than V2S's. Adopted as canonical Wave 31 sublayer-biting diagnostic. Helps distinguish capacity-limited (narrow productive direction) vs information-limited (gradient flow weak) failures.

3. **PR #1194 nezuko H47 V-DEPTH ASSIGNED — first Wave 31 attack on volume decoder interior capacity.** Adds 2 dedicated volume-only TransolverBlocks AFTER the shared trunk, BEFORE the `volume_out` projection. Zero-init residual output projections (mlp.fc2 + attn.out_proj) → bit-exact baseline at step 0. ~30 LOC model.py change, ~5M extra params. Mechanism rationale: H31 + H26 both crossed test_vol_p floor via ENCODER-INPUT enrichment (richer features into volume pathway). H47 tests the COMPLEMENT — deeper volume DECODER capacity to exploit those features further. EP3 binary mechanism gate (canonical V2S/S2V asymmetry diagnostic precedent from H30 closure): BOTH blocks' attn AND ffn out_proj max_abs > 0.05 (else KILL). Memory watch: vp=65536 may climb 74 GB → 85-90 GB. Target: test_vol_p ≤ 3.55%, comparison H31 3.488% + H26 3.608%.

### Wave 30 closure tally — FINAL: 20 dead ends + 2 mechanism wins (H31 vol_p, H26 NPCA mean-preserved-std-spread + test generalization) + 1 outlier merged (H18)

| Closure # | Hypothesis | Tier | Mechanism contribution |
|---:|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape | none |
| 8 | H23 Mean Teacher | Training-reg | none |
| 9 | H18 area-weighted MSE | Per-vertex position | τz mean-shift artifact (merged test-side as outlier) |
| 10 | H21 per-component heads | Decoder capacity | none |
| 11 | H25 ALGP | Aux head | EP1 deflection only |
| 12 | H27 PRLP | Train-eval space | none |
| 13 | H32 DIFFATTN | Attention | none |
| 14 | H28 SAM | Optimizer | none |
| 15 | H24 GSTS | Encoder slice-temp | none |
| 16 | H29 SSFL | Frequency loss | spectral_loss descended only |
| 17 | H18d τz-only area | Channel-decoupled position | INVERSE band-break diagnostic |
| 18 | H31 WALLDIST | Encoder-input feature | 🏆 1st test_vol_p floor crossing |
| 19 | H26 NPCA | Encoder-input local-frame | 🏆🏆 2nd test_vol_p floor crossing + canonical mechanism win with test-side generalization (24/50 cars outside band) |
| **20** | **H30 V2S xattn** | **Encoder-fusion (V→S)** | **EP1 deflection 1.428 (deepest Wave 30); faded to test 1.462 in band; NEW V2S/S2V 5.3× asymmetry diagnostic** |

### Wave 30 structural conclusion (post-H30 — definitive)

H30 V2S xattn triangulates with H31 + H26 + H18d (all closed today) into a unified Wave 30 finding:

1. **Encoder-input axis** (H26/H31): CAN cross test_vol_p floor but CANNOT break test τz/τx band attractor (mean fixed at 1.46-1.47)
2. **Encoder-fusion axis** (H30 V2S xattn): produces deepest EP1 break-signal in Wave 30 but fades fully and does not cross either floor
3. **Loss-shape axis** (H10b/H11b/H12/H16/H16b/H20/H22/H23/H29/H18d): closes uniformly, never breaks the band on test
4. **Output-head axis** (H21 per-component, H25 ALGP, H27 PRLP): closes uniformly
5. **Optimizer axis** (H28 SAM, H24 GSTS): closes uniformly

**The surface decoder Linear(512, 4) projection's row-coupling is the structural cause of the τz/τx band attractor.** Confirmed by 20 Wave 30 closures + 2 mechanism wins + 1 outlier merged. All Wave 31 attacks on surface decoder are now in-flight (H34 post-projection, H45 pre-projection, H46 weight init).

### Fleet status (8/8 active)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| **nezuko** | **#1194** | **H47 V-DEPTH** | **NEW — volume decoder interior capacity, 2 vol-only blocks after trunk, zero-init residual. Pickup pending** | — |
| thorfinn | #1193 | H46 SDORTH | NEW — surface decoder init axis attack, 3 LOC change. Pickup pending | — |
| alphonse | #1192 | H45 ANCHOR-CROSSCHAN-DEC | NEW — first Wave 31 surface decoder structural attack (cross-channel attn before projection). Pickup pending | — |
| tanjiro | #1191 | H36 ANCHOR-SLICE-QUERIES | RUNNING — `wave31_h36_anchor_slice_queries`, mid-EP1 | — |
| frieren | #1190 | H44 YAW-AUGMENTATION | RUNNING — z-axis yaw, `wave31_h44_yaw_augmentation`, mid-EP1 | — |
| askeladd | #1187 | H33 SLICEPE v2 | EP4 6.541% fleet-mid post-EP3, EP6 ETA ~16:30Z (stale_wip cleared 14:55Z) | 6.541% |
| edward | #1188 | H34 OUTHEAD v2 | EP1 27.4%, EP2 7.83%, EP3 verdict ~14:15Z (long overdue, watch) | 7.830% (EP2) |
| fern | #1189 | H35 NPCA-SSFL-STACK | EP2 step 24553 val_abupt 7.805% — mid-EP3 | 7.805% (EP2) |

### Surface decoder attack matrix (Wave 31 — three orthogonal attacks on Linear(512,4) projection)

| Hypothesis | Student | Axis | LOC | Mechanism class |
|---|:--|:--|---:|:--|
| H34 OUTHEAD | edward | POST-projection (capacity) | ~30 | per-channel MLP residual |
| H45 CROSSCHAN-DEC | alphonse | PRE-projection (representation) | ~50 | cross-channel attention |
| H46 SDORTH | thorfinn | PROJECTION WEIGHT INIT (initial condition) | 3 | orthogonal row init |

### Volume pathway attack matrix (Wave 31 — orthogonal to surface decoder attacks)

| Hypothesis | Student | Axis | Mechanism class |
|---|:--|:--|:--|
| H31 (closed) WALLDIST | alphonse | Encoder-input feature (log-SDF) | 🏆 1st test_vol_p floor crossing −0.155pp |
| H26 (closed) NPCA | thorfinn | Encoder-input local-frame | 🏆🏆 2nd test_vol_p crossing −0.035pp + test generalization |
| **H47 V-DEPTH** | **nezuko** | **Volume DECODER interior capacity** | **+2 vol-only TransolverBlocks after trunk (zero-init residual)** |
| H35 NPCA-SSFL stack | fern | Encoder-input + frequency-loss stack | mechanism stacking experiment |

H47 is the COMPLEMENT to H31/H26 — those enrich the input; H47 enriches the processing. Forms a clean 2x2 matrix: input-vs-decoder × surface-vs-volume pathway.

### Wave 31 priority ranking (post H30 closure)

1. **H46 SDORTH** (thorfinn) — cleanest single-variable test of Wave 30 structural finding; 3 LOC
2. **H45 ANCHOR-CROSSCHAN-DEC** (alphonse) — surface decoder pre-projection axis; mechanism-class-novel
3. **H47 V-DEPTH** (nezuko) — first Wave 31 attack on volume decoder; vol_p mechanism-positive axis (NEW)
4. **H44 YAW-AUGMENTATION** (frieren) — first data-augmentation attack ever; orthogonal to all architectural attacks
5. **H36 ANCHOR-SLICE-QUERIES** (tanjiro) — slice-query positional differentiation, sister to H33
6. **H35 NPCA-SSFL-STACK** (fern) — first mechanism-combination experiment

### Imminent decisions (next 4h)

1. **askeladd H33 v2 EP6 ~16:30Z** — mechanism continuation gate (slipped from 14:50Z; current EP4 6.541%, slope −0.030 pp/1k steps)
2. **edward H34 v2 EP3** — long overdue from 14:15Z; needs check (kill bar 9.5%, mech bar 8.5%)
3. **alphonse H45 pickup** — surface decoder pre-projection attack
4. **thorfinn H46 pickup** — smallest LOC Wave 31 attack (3 LOC)
5. **nezuko H47 pickup** — volume decoder interior capacity (NEW)

### V2S/S2V asymmetry diagnostic — NEW canonical Wave 31 metric

Adopted from H30 closure: for any future architectural-fusion sublayer, track out_proj.max_abs comparison vs baseline path. Helps distinguish:
- **Capacity-limited failure** (narrow productive direction, like H30 V2S) — grad/param ratio high, but max_abs low
- **Information-limited failure** (gradient flow weak, sublayer never engages) — grad/param ratio low

H47 V-DEPTH will use the same diagnostic — both blocks' attn AND ffn out_proj max_abs > 0.05 at EP3 = sublayer biting (else KILL = info-limited fade).

---

## Previous invocation actions (2026-05-18 ~14:30Z) — THORFINN H26 NPCA CLOSED EP13 (19TH WAVE 30 DE, **2ND test_vol_p FLOOR CROSSING + CANONICAL MECHANISM WIN WITH TEST-SIDE GENERALIZATION**); THORFINN REASSIGNED H46 SDORTH (PR #1193, **SURFACE-DECODER INIT AXIS ATTACK** — smallest LOC attack on Wave 30 structural finding); WAVE 30 STRUCTURAL FINDING REFINED (surface decoder Linear(512,4) PROJECTION MEAN-PRESERVES τz/τx regardless of encoder representation diversity)

### Headline updates (14:30Z)

1. **PR #1177 thorfinn H26 NPCA CLOSED — 19th Wave 30 dead end + 2nd test_vol_p floor crossing.** Terminal: val_abupt 6.3462% (FAIL +0.220pp), test_abupt 6.0276% (FAIL +0.184pp), test_SP 3.8048% **FLOOR BREACH** +0.228pp, test_vol_p 3.6079% **FLOOR PASS** −0.035pp ✅ **2ND test_vol_p floor crossing in Wave 30** (joining H31 −0.155pp earlier today), test_WSS 6.946% (+0.219pp). Mechanism val std(τz/τx) 0.259 (~13× baseline), **test std 0.132 (~6.6× baseline), 24/50 (48%) test cars outside [1.40, 1.60]** — first Wave 30 mechanism to GENERALIZE to held-out test set. min test ratio 1.136 / max 1.831. But mean(τz/τx) **unchanged** at test (1.467 ≈ baseline 1.473).

2. **Wave 30 structural finding REFINED post-H26.** The surface decoder's `Linear(512, 4)` final projection **mean-preserves τz/τx regardless of encoder-input representation diversity**. Encoder-input axis (H26 NPCA + H31 WALLDIST) can produce per-car σ-level distribution structure but cannot shift body-averaged mean. The mean is fixed by the projection row-coupling. This is the canonical Wave 30 documentation locking the next attack axis to the surface decoder itself.

3. **PR #1193 thorfinn H46 SDORTH ASSIGNED — surface-decoder init axis attack.** Initialize the `Linear(512, 4)` final projection's 4 row vectors to be mutually orthogonal (via `nn.init.orthogonal_`), Kaiming-magnitude-matched. **3 LOC change** to model.py. Mechanism: test if the τz/τx band attractor is set by INITIAL CONDITION of projection row-coupling, or is gradient-driven beyond init. Three falsifiable outcomes: (a) attractor breaks AND orthogonality persists → init-set, permanent fix, (b) attractor breaks then re-emerges → 9th-cold-start-fade pattern from a new initial condition, (c) attractor unchanged → init axis decisively closed. Mechanism-class novel: NONE of 19 closed Wave 30 hypotheses or 5 in-flight Wave 31 runs attack the surface decoder WEIGHT INITIALIZATION axis.

### Wave 30 closure tally: 19 dead ends + 1 outlier merged (H18) + 2 mechanism wins (H31 vol_p, H26 NPCA mean-preserved-std-spread)

| Closure # | Hypothesis | Tier | Mechanism contribution |
|---:|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape | none |
| 8 | H23 Mean Teacher | Training-reg | none |
| 9 | H18 area-weighted MSE | Per-vertex position | τz mean-shift artifact (merged test-side as outlier) |
| 10 | H21 per-component heads | Decoder capacity | none |
| 11 | H25 ALGP | Aux head | EP1 deflection only |
| 12 | H27 PRLP | Train-eval space | none |
| 13 | H32 DIFFATTN | Attention | none |
| 14 | H28 SAM | Optimizer | none |
| 15 | H24 GSTS | Encoder slice-temp | none |
| 16 | H29 SSFL | Frequency loss | spectral_loss descended only |
| 17 | H18d τz-only area | Channel-decoupled position | INVERSE band-break diagnostic |
| 18 | H31 WALLDIST | Encoder-input feature | 🏆 1st test_vol_p floor crossing |
| **19** | **H26 NPCA** | **Encoder-input local-frame** | **🏆🏆 2nd test_vol_p floor crossing + canonical mechanism win with test-side generalization (24/50 cars outside band, std 6.6× baseline at test)** |

### Fleet status (8/8 active)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| **thorfinn** | **#1193** | **H46 SDORTH** | **NEW — surface decoder init axis attack, 3 LOC change. Pickup pending** | — |
| alphonse | #1192 | H45 ANCHOR-CROSSCHAN-DEC | NEW — first Wave 31 surface decoder structural attack (cross-channel attn before projection). Pickup pending | — |
| tanjiro | #1191 | H36 ANCHOR-SLICE-QUERIES | RUNNING — `wave31_h36_anchor_slice_queries`, mid-EP1 | — |
| frieren | #1190 | H44 YAW-AUGMENTATION | RUNNING — z-axis yaw, `wave31_h44_yaw_augmentation`, mid-EP1 | — |
| askeladd | #1187 | H33 SLICEPE v2 | EP3 6.871% MARGINAL — continuing to EP6 ~14:50Z | 6.871% |
| edward | #1188 | H34 OUTHEAD v2 | EP1 27.4%, EP2 7.83%, mid-EP3 step 26118 — EP3 verdict ~14:15Z | 7.830% (EP2) |
| nezuko | #1184 | H30 V2S xattn | EP6 step 65212 val_abupt 6.364%, projected NOT-A-MERGE. Budget ends ~16:05Z | 6.364% |
| fern | #1189 | H35 NPCA-SSFL-STACK | EP2 step 24553 val_abupt 7.805% — mid-EP3 | 7.805% (EP2) |

### Research map update (post H26 closure — Wave 30 structurally complete)

**Closed mechanism axes (Wave 30 final):**
- Loss-shape in ALL forms (spatial, per-vertex, per-channel, frequency-domain)
- Per-vertex position weighting (coupled H18 + decoupled H18d)
- Training regularization (H23)
- Optimizer-space (H28 SAM)
- Encoder slice-temp (H24)
- Auxiliary head supervision (H25)
- Decoder capacity / per-component split (H21)
- Train-eval space alignment (H27)
- Attention differentiation (H32 DIFFATTN)
- **Encoder-input enrichment** (H31 + H26 — both proved this axis crosses test_vol_p floor but CANNOT shift surface decoder mean)

**Open mechanism axes (Wave 31 architectural attacks on surface decoder):**
- **Surface decoder pre-projection structure** (H45 alphonse — cross-channel attention)
- **Surface decoder post-projection capacity** (H34 edward — per-channel MLPs)
- **Surface decoder WEIGHT INIT** (H46 thorfinn — orthogonal row init, NEW)
- **Slice-query architecture** (H33 askeladd SLICEPE; H36 tanjiro ANCHOR)
- **Data augmentation** (H44 frieren — first attack on this tier)
- **Mechanism stacking** (H35 fern NPCA-SSFL; future H26+H31 vol_p deep candidate)

### Surface decoder attack matrix (Wave 31 organized)

| Hypothesis | Student | Axis | LOC | Mechanism class |
|---|:--|:--|---:|:--|
| H34 OUTHEAD | edward | POST-projection (capacity) | ~30 | per-channel MLP residual |
| H45 CROSSCHAN-DEC | alphonse | PRE-projection (representation) | ~50 | cross-channel attention |
| **H46 SDORTH** | **thorfinn** | **PROJECTION WEIGHT INIT (initial condition)** | **3** | **orthogonal row init** |

Three orthogonal attacks on the same structural object (surface decoder projection). H46 is the cleanest single-variable test: pure initial-condition perturbation, no architecture change, no capacity change, no input change.

### Wave 31 priority ranking (post H26 closure)

1. **H46 SDORTH** (thorfinn) — cleanest single-variable test of Wave 30 structural finding; 3 LOC
2. **H45 ANCHOR-CROSSCHAN-DEC** (alphonse) — surface decoder pre-projection axis; mechanism-class-novel
3. **H44 YAW-AUGMENTATION** (frieren) — first data-augmentation attack ever; orthogonal to all architectural attacks
4. **H36 ANCHOR-SLICE-QUERIES** (tanjiro) — slice-query positional differentiation, sister to H33
5. **H35 NPCA-SSFL-STACK** (fern) — first mechanism-combination experiment

### Imminent decisions (next 4h)

1. **askeladd H33 v2 EP6 ~14:50Z** — mechanism continuation gate (mech-positive bar 6.50%)
2. **edward H34 v2 EP3 ~14:15Z** — kill bar 9.5%, mech bar 8.5%
3. **nezuko H30 V2S terminal ~16:05Z** — already projected NOT-A-MERGE
4. **alphonse H45 pickup** — any time; surface decoder pre-projection attack
5. **thorfinn H46 pickup** — any time; smallest LOC Wave 31 attack (3 LOC)

---

## Previous invocation actions (2026-05-18 ~14:20Z) — ALPHONSE H31 WALLDIST CLOSED EP13 (18TH WAVE 30 DE, **FIRST test_vol_p FLOOR CROSSING IN WAVE 30**); ALPHONSE REASSIGNED H45 ANCHOR-CROSSCHAN-DEC (PR #1192, **FIRST SURFACE-DECODER STRUCTURAL ATTACK** — missing axis after Wave 30 mechanism-class lock); WAVE 30 STRUCTURAL FINDING LOCKED (τz/τx ATTRACTOR LIVES IN SURFACE DECODER RESIDUAL — 9 cold-start fades across 5 mechanism axes)

### Headline updates (14:20Z)

1. **PR #1185 alphonse H31 CLOSED — 18th Wave 30 dead end with MECHANISM WIN.** Terminal: val_abupt 6.1735% (FAIL +0.0475pp), test_abupt 5.9014% (FAIL +0.057pp), test_SP 3.7536% **FLOOR BREACH** +0.177pp, test_vol_p 3.4880% **FLOOR PASS** −0.155pp ✅ **FIRST test_vol_p floor crossing in Wave 30 history**, test_WSS 6.799% (+0.072pp regression vs baseline). Volume side mechanism CONFIRMED: log-SDF gives encoder uniform sensitivity across boundary-layer regimes, vol decoder reads directly through short composition path. Surface side REJECTED: τz/τx 1.470 statistically identical to baseline 1.473 — 9th cold-start fade in Wave 30.

2. **Wave 30 structural finding LOCKED.** After 18 closures + 1 mechanism-win (H26 NPCA EP3 only): **the [1.44, 1.55] τz/τx band attractor lives in the SURFACE DECODER's residual representation, NOT in any encoder-input/loss/optimizer/regularization/aux-head pathway.** Evidence: 9 cold-start fades (H18/H20/H24/H25/H26 Path B/H29/H30 V2S/H18d/H31) across 5 different mechanism axes. The next attack class MUST target surface decoder structure.

3. **PR #1192 alphonse H45 ANCHOR-CROSSCHAN-DEC ASSIGNED — first Wave 31 surface-decoder structural attack.** Cross-channel attention OVER the 4 output dimensions {cp, τ_x, τ_y, τ_z} BEFORE the final `Linear(512, 4)` projection. Each channel becomes a query attending to others' representations. Zero-init final attention layer → bit-exact baseline at step 0. ~50 LOC model.py only. EP3 falsifier: val_abupt ≤ 7.0% AND τz/τx ≤ 1.45 with std ≥ 0.10 AND crosschan_residual_asymmetry_ratio > 1.5. Mechanism-class-novel: none of the 18 closed Wave 30 hypotheses or 5 in-flight runs attack the surface decoder's pre-projection representation. Theoretical motivation: if `Linear(512, 4)` rows are rank-coupled for τ_x and τ_z, decoupling-via-input-orthogonalization (NeRF/DETR/SpiderSolver pattern) lets each channel project from orthogonal vectors.

### Wave 30 closure tally: 18 dead ends + 1 mechanism win (H26 NPCA EP3 only)

| Closure # | Hypothesis | Tier | Mechanism contribution |
|---:|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape | none |
| 8 | H23 Mean Teacher | Training-reg | none |
| 9 | H18 area-weighted MSE | Per-vertex position | τz mean-shift artifact (merged test-side as outlier) |
| 10 | H21 per-component heads | Decoder capacity | none |
| 11 | H25 ALGP | Aux head | EP1 deflection only |
| 12 | H27 PRLP | Train-eval space | none |
| 13 | H32 DIFFATTN | Attention | none |
| 14 | H28 SAM | Optimizer | none |
| 15 | H24 GSTS | Encoder slice-temp | none |
| 16 | H29 SSFL | Frequency loss | spectral_loss descended only |
| 17 | H18d τz-only area | Channel-decoupled position | INVERSE band-break diagnostic |
| **18** | **H31 WALLDIST** | **Encoder-input feature** | **🏆 FIRST test_vol_p floor crossing** |

### Fleet status (8/8 active)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| **alphonse** | **#1192** | **H45 ANCHOR-CROSSCHAN-DEC** | **NEW — first surface-decoder structural attack. Pickup pending** | — |
| tanjiro | #1191 | H36 ANCHOR-SLICE-QUERIES | RUNNING — `wave31_h36_anchor_slice_queries`, mid-EP1 | — |
| frieren | #1190 | H44 YAW-AUGMENTATION | RUNNING — z-axis yaw, `wave31_h44_yaw_augmentation`, mid-EP1 | — |
| thorfinn | #1177 | H26 NPCA 18h | EP7 step 67932 val_abupt 6.346%, val_WSS 7.170%, val_VP 3.750%. EP13 ~17:00Z | 6.346% |
| askeladd | #1187 | H33 SLICEPE v2 | EP3 6.871% MARGINAL — continuing to EP6 ~14:50Z | 6.871% |
| edward | #1188 | H34 OUTHEAD v2 | EP1 27.4%, EP2 7.83%, mid-EP3 step 26118 — EP3 verdict ~14:15Z | 7.830% (EP2) |
| nezuko | #1184 | H30 V2S xattn | EP6 step 65212 val_abupt 6.364%, projected NOT-A-MERGE. Budget ends ~16:05Z | 6.364% |
| fern | #1189 | H35 NPCA-SSFL-STACK | EP2 step 24553 val_abupt 7.805% — mid-EP3 | 7.805% (EP2) |

### Research map update (post H31 closure — Wave 30 structurally complete)

**Closed mechanism axes (Wave 30 final):**
- Loss-shape in ALL forms (spatial, per-vertex, per-channel, frequency-domain)
- Per-vertex position weighting (coupled H18 + decoupled H18d)
- Training regularization (H23)
- Optimizer-space (H28 SAM)
- Encoder slice-temp (H24)
- Auxiliary head supervision (H25)
- Decoder capacity / per-component split (H21)
- Train-eval space alignment (H27)
- Attention differentiation (H32 DIFFATTN)
- **Encoder-input enrichment** (H31 — but proved this axis CAN move vol_p floor)

**Open mechanism axes (Wave 31):**
- **NPCA/coordinate-representation** (H26 mechanism win; H35 fern stacking)
- **Slice-query architecture** (H33 askeladd SLICEPE; H36 tanjiro ANCHOR)
- **Data augmentation** (H44 frieren — first attack on this tier)
- **Decoder structure** (H34 edward OUTHEAD post-projection; H30 nezuko V2S xattn cross-stream)
- **Surface decoder cross-channel structure** (H45 alphonse — NEW, the missing axis)

### Wave 31 priority ranking (post H31 closure)

1. **H45 ANCHOR-CROSSCHAN-DEC** (alphonse) — directly attacks the locked Wave 30 finding (surface decoder pre-projection axis); mechanism-class-novel
2. **H44 YAW-AUGMENTATION** (frieren) — first data-augmentation attack ever; orthogonal to all architectural attacks
3. **H36 ANCHOR-SLICE-QUERIES** (tanjiro) — slice-query positional differentiation, sister to H33
4. **H35 NPCA-SSFL-STACK** (fern) — first mechanism-combination experiment

### Imminent decisions (next 4h)

1. **askeladd H33 v2 EP6 ~14:50Z** — mechanism continuation gate (mech-positive bar 6.50%)
2. **edward H34 v2 EP3 ~14:15Z** — kill bar 9.5%, mech bar 8.5%
3. **nezuko H30 V2S terminal ~16:05Z** — already projected NOT-A-MERGE
4. **thorfinn H26 18h terminal ~17:00Z** — baseline-beat candidate
5. **alphonse H45 pickup** — any time; first surface-decoder structural attack

---

## Previous invocation actions (2026-05-18 ~12:30Z) — FRIEREN H44 IMPLEMENTATION DEVIATION RESOLVED (z-axis yaw correct, PR template had coord-convention bug); H44 RUN HEALTHY MID-EP1 (step 7551, runtime 67min); ADVISOR TEMPLATE LESSON LOCKED (DrivAerML uses z=vertical, NOT y)

### Headline updates (12:30Z)

1. **PR #1190 frieren H44 IMPLEMENTATION DEVIATION RESOLVED.** Frieren detected PR template's `apply_yaw_augmentation` formula mixed (x, z) which is **pitch in DrivAerML coords, not yaw**. Empirical verification from run_1: x=streamwise (asym), **y=lateral mirror-symmetric ±1.01**, **z=vertical asym ground-to-roof**. Implementation correctly switched to rotation about z mixing (x↔y) for coords/normals and (tau_x↔tau_y) for targets. Mechanism preserved: per-rotation tau_x variance indirectly disrupts τz/τx band attractor (tau_x denominator varies even within same car). Cross-referenced alphonse PR #937 precedent (`Rz = [[cy,-sy,0],[sy,cy,0],[0,0,1]]`). Run health: group `wave31_h44_yaw_augmentation`, 8 ranks alive, rank 0 `6scw4nto` mid-EP1 step 7551/10864, all ranks tight (7547-7564 = no DDP divergence), `yaw_aug_theta_max=5` confirmed.

2. **ADVISOR LESSON LOCKED — DrivAerML coordinate convention.** Saved to memory `reference_drivaerml_coords.md`: **z = vertical, NOT y**. Yaw = rotation about z mixing (x, y) and (tau_x, tau_y); pitch = rotation about y mixing (x, z) and (tau_x, tau_z); roll = rotation about x. All future geometric-augmentation / slice-position / coordinate-aware PR templates MUST explicitly name the axis in DrivAerML convention AND name channel columns being rotated. Cost on this round: ~1h frieren think-time + 2 PR comments to resolve.

## Previous invocation actions (2026-05-18 ~12:00Z) — TANJIRO H18D CLOSED EP13 (17TH WAVE 30 DE, CHANNEL-DECOUPLED AREA WEIGHT FALSIFIER: val_abupt 6.319%, τz/τx INVERSE ABOVE-BAND, test_SP FLOOR BREACH); TANJIRO REASSIGNED H36 ANCHOR-SLICE-QUERIES (PR #1191, DETR-LINE ARCHITECTURE ATTACK)

### Headline updates (12:00Z)

1. **PR #1183 tanjiro H18d CLOSED — 17th Wave 30 dead end.** Terminal EP13: val_abupt 6.319% (+0.193pp above baseline, per 05:05Z gate "val_abupt > 6.20% → not-a-merge"), test_SP 3.856% FLOOR BREACH, test_vol_p 3.637% marginal floor PASS, test_WSS 7.126% above goal. **τz/τx INVERSE signature**: val 1.633 (above band, only above-band run in Wave 30) → test 1.528 (back inside band). **Decisive channel-decoupled falsifier**: H18's band-break (1.418) was a tied-loss-budget artifact, NOT τz-physics. Decoupling τz-only released cp/τx/τy from starvation, τz drifted ABOVE band (mirror-image). **Per-vertex area weighting entire axis DEAD in both coupling directions (H18 coupled, H18d decoupled).**

2. **PR #1191 tanjiro H36 ANCHOR-SLICE-QUERIES ASSIGNED.** First architecture-side spatial-awareness attack. Adds learned 3D anchor positions A ∈ R^{128×3} to slice queries: `q_s' = q_s + MLP_anchor(PE(A_s))`. Zero-init final MLP layer = bit-exact baseline at step 0. ~60 LOC model.py only. EP3 falsifier: std(τz/τx) ≥ 0.15 AND anchor diversity ≥ 0.10. Same mechanism class as H26 NPCA (variance-break) but from query-modulation side rather than coordinate-transform side. SpiderSolver/DAB-DETR theoretical precedent.

### Wave 30 closure tally: 17 dead ends + 1 mechanism win (H26 NPCA) + 6 in-flight (Wave 30/31)

| Tier | Count |
|---|---:|
| Loss-shape (all forms) | 9 |
| Per-vertex position / training regularization | 5 |
| Decoder capacity / aux heads / optimizer | 3 |
| **TOTAL closures** | **17 + H18 merged test-side** |

### Fleet status (8/8 active)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| tanjiro | #1191 | H36 ANCHOR-SLICE-QUERIES | RUNNING — `wave31_h36_anchor_slice_queries`, mid-EP1 | — |
| frieren | #1190 | H44 YAW-AUGMENTATION | RUNNING — z-axis yaw, `wave31_h44_yaw_augmentation`, mid-EP1 | — |
| **alphonse** | **#1185** | **H31 WALLDIST** | **EP6.25 step 67932 val_abupt 6.176% (0.050pp above baseline), val_WSS 6.942% (best in fleet), val_VP 3.652% (~0.009pp above floor). 18h budget runs out ~17:00Z at ~EP9. Projected end val_abupt ~6.08% = BELOW baseline. val_SP 4.160% may breach test floor 3.577% — merge risk** | **6.176%** |
| thorfinn | #1177 | H26 NPCA 18h | EP7 step 67932 val_abupt 6.346%, val_WSS 7.170%, val_VP 3.750%. EP13 ~17:00Z | 6.346% |
| askeladd | #1187 | H33 SLICEPE v2 | EP3 6.871% MARGINAL (passes kill bar 7.50%, misses mech-pos 6.50%) — continuing | 6.871% |
| edward | #1188 | H34 OUTHEAD v2 | EP1 27.4%, EP2 7.83%, mid-EP3 step 26118 — EP3 verdict ~14:15Z | 7.830% (EP2) |
| nezuko | #1184 | H30 V2S xattn | EP6 step 65212 val_abupt 6.364% (+0.238pp baseline), val_WSS 7.207% (2nd in fleet), val_VP 3.827% (above floor). Budget ends ~16:05Z at ~EP7 — projected NOT-A-MERGE | 6.364% |
| fern | #1189 | H35 NPCA-SSFL-STACK | EP2 step 24553 val_abupt 7.805%, val_WSS 8.851% — mid-EP3 | 7.805% (EP2) |

### Research map (current)

**Closed tiers (never try again):**
- All loss-shape forms: per-vertex, per-channel, frequency-domain (H29 SSFL), spatial
- Per-vertex position weighting: both coupled (H18) and decoupled (H18d)
- Training regularization (H23)
- Optimizer-space (H28 SAM)

**Open tiers (active attacks):**
- **NPCA/coordinate-representation** (H26 thorfinn — only mechanism win; H35 fern stacking)
- **Slice-query architecture** (H33 askeladd SLICEPE EP3 verdict IMMINENT; H36 tanjiro ANCHOR — more expressive)
- **Data augmentation** (H44 frieren — first attack on this tier)
- **Decoder structure** (H34 edward OUTHEAD; H30 nezuko V2S xattn)

### Imminent decisions (next 4h)

1. **askeladd H33 v2 EP3 ~13:30Z** — mechanism gate (τz/τx ≤ 1.42 + slice_pe spread ≥ 0.05). Critical: if H33 passes, H36 is HIGH priority complement; if fails, slice-query positional-encoding axis is narrowed.
2. **thorfinn H26 EP13 ~14:30Z** — BASELINE-BEAT candidate (linear projection 5.95%)
3. **tanjiro H18d EP13 ~14:25Z** — ALREADY COMPLETE (this invocation, closed)
4. **alphonse H31 EP13 ~15-16Z** — fleet-lead, val_WSS-best

---

## Previous invocation actions (2026-05-18 ~11:30Z) — FRIEREN H29 SSFL CLOSED EP13 (16TH WAVE 30 DE, 1ST FREQUENCY-DOMAIN FALSIFIER: val_abupt 6.3538% / τz/τx NEVER BROKE BAND / ALL TEST FLOORS BREACHED); FRIEREN REASSIGNED H44 YAW-AUGMENTATION (PR #1190, FIRST DATA-AUGMENTATION ATTACK IN HISTORY)

### Headline updates (11:30Z)

1. **PR #1182 frieren H29 SSFL CLOSED — 16th Wave 30 dead end.** Terminal EP13: val_abupt 6.3538% (fleet-low but +0.228pp above baseline), test_abupt 6.1578% FAIL, test_SP 3.8617% FLOOR BREACH, test_vol_p 3.7667% FLOOR BREACH, test_WSS 7.0874% FAIL. τz/τx trajectory: 1.385 (cold-start break) → 1.484 (EP2 band re-entry) → 1.532 (terminal). **Primary falsifier NOT met: τz/τx never broke below 1.42.** Spectral_loss descended 99% cleanly (mechanism alive) while τz/τx held band — proves the band attractor is NOT spatial-frequency-mediated. **Loss-shape tier DECISIVELY CLOSED: 7 per-vertex loss-shape + H23 regularization + H29 frequency-domain = 9 loss-shape closures.**

2. **PR #1190 frieren H44 YAW-AUGMENTATION ASSIGNED — first data-augmentation attack in DrivAerML history.** Hypothesis: 400 training cars are all yaw=0; model exploits left-right symmetry and learns a coupled τz/τx representation. Random yaw θ ~ U[-5°, +5°] rotates input coordinates, normals, AND vector targets (τx, τz) consistently. ~40 LOC train.py only, no model changes, `--yaw-aug-theta-max 0.0` is bit-exact baseline recovery. EP3 falsifier: std(τz/τx) per-car ≥ 0.15 (augmentation must INCREASE per-car variance). EP3 kill: std < 0.05 or val_abupt > 9.5%. θ_max can reduce to 2.5° if EP2 > 6.80%.

### Wave 30 final closure count: 16 dead ends + 1 mechanism win (H26 NPCA) + 6 in-flight

| Tier | Closures | Examples |
|---|---:|---|
| Loss-shape (spatial/vertex) | 7 | H10b/H11b/H12/H16/H16b/H20/H22 |
| Training-regularization | 1 | H23 |
| Per-vertex position / decoder / aux-head / train-eval | 4 | H18/H21/H25/H27 |
| Optimizer | 1 | H28 SAM |
| Encoder slice-temp | 1 | H24 GSTS |
| **Frequency-domain loss-shape** | **1** | **H29 SSFL** |
| **TOTAL closures** | **15 dead ends** | + H18 outlier merged test-side |

### Fleet status (8/8 active)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| **frieren** | **#1190** | **H44 YAW-AUG** | **NEW — first data-aug attack. Pickup pending** | — |
| alphonse | #1185 | H31 WALLDIST | EP5.2 6.22% fleet-lead. EP13 ~15-16Z | **6.22%** |
| thorfinn | #1177 | H26 NPCA 18h | EP5.1 6.437%, variance mechanism. EP13 ~14:30Z | 6.437% |
| askeladd | #1187 | H33 SLICEPE v2 | Mid-EP2, τz/τx=1.432, slice_pe diff'ing. EP3 ~13:30Z | — |
| edward | #1188 | H34 OUTHEAD v2 | EP1 ETA passed (~10:30Z). Awaiting EP1 read | — |
| tanjiro | #1183 | H18d | EP9 6.325%, τz/τx 1.636 INVERSE. EP13 ~14:25Z | 6.325% |
| nezuko | #1184 | H30 V2S xattn | EP5 6.42%, continuing to terminal | 6.42% |
| fern | #1189 | H35 NPCA-SSFL-STACK | v2 launched, pre-EP1. EP1 ~11Z | — |

### Research map update (post H29 closure)

**Definitively closed tiers (never try again on this benchmark):**
- Loss-shape in any form (spatial weighting, per-vertex, frequency-domain, per-channel scalars)
- Training-regularization (mean teacher)
- Optimizer-space (SAM provably flat in band landscape)

**Open tiers with positive signals:**
- **Representation/geometry augmentation**: H26 NPCA is the ONLY mechanism win. Wave 31 must be NPCA-class or data-augmentation (H44 frieren).
- **Decoder structure**: H34 OUTHEAD (edward) and V2S xattn (nezuko) still in-flight.
- **Data augmentation**: H44 frieren — first attack ever on this tier. If it works, opens combination with NPCA (H44+NPCA = H46 or later).
- **Mechanism combinations**: H35 NPCA-SSFL-STACK (fern) — SSFL mechanism was alive in H29, even though it didn't break the band standalone. Combined with NPCA variance, the interaction may differ.

### Critical near-term watch (next 6h chronological)

1. **edward H34 v2 EP1 read** — EP1 should have landed ~10:30Z; need student post
2. **fern H35 v2 EP1 ~11Z** — safety check
3. **askeladd H33 v2 EP3 ~13:30Z** — mechanism gate (τz/τx ≤ 1.42 + slice_pe spread > 0.05)
4. **tanjiro H18d EP13 ~14:25Z** — terminal, val_abupt knife-edge for sub-baseline
5. **thorfinn H26 EP13 ~14:30Z** — terminal, BASELINE-BEAT candidate
6. **frieren H44 pickup** — any time (simple ~40 LOC implementation)
7. **alphonse H31 EP13 ~15-16Z** — fleet-lead, val_WSS-best in fleet

---

## Previous invocation actions (2026-05-18 ~09:30Z) — EDWARD H34 V2 LAUNCHED (corrected `--kill-thresholds`, step 3399 healthy); FERN H35 V2 LAUNCHED (after v1 self-kill, smoke crash fixes); TANJIRO H18D EP9 MARGINAL 6.325% (continuing to EP13); ASKELADD H33 V2 MID-EP2 healthy (EP1 τz/τx=1.432 + slice_pe DIFFERENTIATING); ISSUE #1056 WSS REPLY POSTED

### Headline updates (09:30Z)

1. **Edward H34 OUTHEAD v2 LIVE** — run `iw2ommjz`, step 3399, _runtime 0.52h, kill_thresholds correctly step-indexed (`32594:val_abupt<9.5,32594:val_SP<5.5`). EP1 val ETA ~10:30Z. Mechanism diagnostic at EP1: aux head asymmetry_ratio τz/τx (was 0.46 INVERTED on v1 — could be warmup-half-LR artifact). EP3 reading decisive.

2. **Fern H35 NPCA-SSFL-STACK v2 LIVE** — run `7zkdf9xv`, step 467, _runtime 0.07h. Path to v2: smoke v1 (`m042d0nj`) crashed at step 5249 → smoke v2 (`uwkejqqf`) orthofix crashed at step 252 → main v1 (`cat0c0eb`) self-killed at EP1 (27.2% — same kill_thresholds pitfall) → v2 (`7zkdf9xv`) with correct step-indexed gate `25000:val_abupt<9.5`. First Wave 31 mechanism-stack experiment in-flight. EP1 val ETA ~11Z.

3. **Tanjiro H18D EP9 landed: val_abupt 6.325% MARGINAL** (between PASS ≤6.25 and KILL >6.35). val_SP 4.072% also marginal. **CONTINUE TO EP13 per 05:05Z directive standing.** Critical data: **val_VP 3.648% AT test_vol_p floor (3.643%)** — H18 had val→test gap −0.158pp on vol_p, projecting test_vol_p ~3.49% solid floor PASS. **τz/τx 1.636 INVERSE signature** (above-band, only Wave 30 run that's bypassed the [1.44, 1.55] attractor entirely). EP13 projection 6.16-6.25% — knife-edge for sub-baseline; even if no merge, mechanism diagnostic is unique.

4. **Askeladd H33 SLICEPE v2 HEALTHY mid-EP2** (run `u58fwoym`, step 15653, _runtime 2.31h). EP1 reading: **τz/τx = 1.432 — just under band lower edge** (strong cold-start break). **slice_pe DIFFERENTIATING layer-by-layer** (inter_slice_cos: L0=0.0026 → L4=0.0201, deeper layers more differentiated as hypothesis predicts). Strongest EP1 mechanism reading of Wave 30 (both surface τ-channel AND position-identity signals firing). EP3 verdict ~13:30Z — if τz/τx stays <1.42 with active slice_pe spread, H33 becomes Wave 30's second mechanism-positive (after H26 NPCA).

5. **Issue #1056 WSS reply posted at 09:30Z** — Morgan complained "you haven't given a single WSS test metric here!!" Replied with explanation that test_WSS is only available at EP13 terminal + complete val_WSS table for 8 in-flight runs (best val_WSS: alphonse H31 WALLDIST 7.035%, frieren H29 SSFL 7.166%, nezuko H30 7.303%, thorfinn H26 7.305%). Committed to always including val_WSS + τz/τx ratio in future status updates.

### Fleet status (8/8 active)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| fern | #1189 | H35 NPCA-SSFL-STACK | **v2 LIVE** step 467, kill_thresholds correct. EP1 ~11Z | — (pre-EP1) |
| alphonse | #1185 | H31 WALLDIST | **FLEET-LEAD** EP5.2 6.22%, val_vol_p best-in-fleet. EP6-EP7 imminent | **6.22%** |
| thorfinn | #1177 | H26 NPCA 18h | EP5.1 6.437% (consistent with prior 6.541% EP5 report). 6h screening run finished 6.97% | 6.437% |
| askeladd | #1187 | H33 SLICEPE v2 | Mid-EP2 step 15653, τz/τx=1.432, slice_pe diff'ing. EP3 ~13:30Z | — (mid-EP2) |
| edward | #1188 | H34 OUTHEAD v2 | **v2 LIVE** step 3399, kill_thresholds correct. EP1 ~10:30Z | — (pre-EP1) |
| frieren | #1182 | H29 SSFL | EP6.2 6.36%, slope flat (saturating). EP9-EP13 watch | 6.36% |
| tanjiro | #1183 | H18d | **EP9 6.325% MARGINAL**, val_VP 3.648% AT floor, τz/τx 1.636 INVERSE. EP13 ~14:25Z | 6.325% |
| nezuko | #1184 | H30 V2S xattn | EP5.0 6.42%, continuing to terminal | 6.42% |

### Critical near-term watch (next 6h chronological)

1. **edward H34 v2 EP1 ~10:30Z** — safety + asymmetry_ratio re-read (was 0.46 INVERTED on v1)
2. **fern H35 v2 EP1 ~11Z** — safety (need val_abupt < 35%, no nonfinites)
3. **frieren H29 EP13 ~13Z** — terminal verdict, test eval, fleet-low candidate
4. **askeladd H33 v2 EP3 ~13:30Z** — mechanism gate (slice_pe + τz/τx)
5. **tanjiro H18d EP13 ~14:25Z** — terminal, val_abupt sub-baseline knife-edge
6. **thorfinn H26 EP13 ~14:30Z** — terminal, BASELINE-BEAT candidate, std(τz/τx) critical
7. **alphonse H31 EP13 ~15-16Z** — fleet-lead val_abupt, val_WSS-best run, terminal

**Three baseline-beat candidates in this window: alphonse H31 (6.22% at EP5.2, slope strong), thorfinn H26 (6.437% EP5.1 + variance mechanism break), tanjiro H18d (knife-edge but only above-band τz/τx — orthogonal mechanism).**

---

## Previous invocation actions (2026-05-18 ~09:00Z) — EDWARD H34 OUTHEAD self-killed by `--kill-thresholds` step-prefix misconfig (RELAUNCH INSTRUCTED with step-indexed gates); ASKELADD H33 v2 self-corrected and HEALTHY at mid-EP2; HUMAN STATUS REPLY POSTED to issue #1056

### Headline updates (09:00Z)

1. **PR #1188 edward H34 OUTHEAD self-killed at EP1** by same `--kill-thresholds` syntax pitfall that hit askeladd #1187 earlier. EP1 metrics clean (val_abupt 26.78%, normal cold-start), aux heads grew correctly (cp 0.246, τx 0.218, τy 0.156, τz 0.101 — all >0, gradients flowing). **Asymmetry ratio τz/τx = 0.46 — INVERTED direction at EP1** (predicted >1.5 if head-side rank-coupling hypothesis right). Could be EP1 warmup artifact (only second half saw full LR) or genuine inverse signal. EP3 reading is decisive. Relaunch v2 instructions posted with step-indexed gates (`32594:val_abupt<9.5,32594:val_SP<5.5`).

2. **Root cause locked: `--kill-thresholds` PREFIX IS GLOBAL STEP, NOT EPOCH.** Both H33 v1 and H34 v1 self-killed at EP1 because `1:val_abupt<9.5` means "from step 1 onward" (always-on, evaluated at first end-of-epoch val log at step 10864). Wave 30 lesson #8 locked. Memory saved at `feedback_kill_thresholds_step_indexed.md`. Correct step-indexed gates for 10864 steps/epoch recipe:
   - EP3 binding gate: `32594:val_abupt<9.5,32594:val_SP<5.5`
   - Optional EP1 cold-start fence (only for H32-style catastrophe detection): `10864:val_abupt<35,10864:val_vol_p_mae<60`

3. **Askeladd H33 SLICEPE v2 SELF-CORRECTED and running healthy** — v2 run `u58fwoym` launched at 06:58Z, currently mid-EP2 at step 14598 (val_abupt 29.63%), no early-kill triggered. Askeladd updated the threshold syntax independently before my 07:04Z relaunch advice. EP3 verdict ETA ~13:30Z.

4. **Human status reply posted to issue #1056** (Morgan asked at 07:44Z about tay/dl24 progress + timeout fix + longer-budget hand-off candidates). Reply at 07:50Z covered: timeout RESOLVED, Wave 30 tally (15 closures + 1 mechanism win H26 + 1 fleet-low H29), Wave 31 launched (H35 to fern), 3 dl24 hand-off candidates (H42 NPCA-WARMSTART top pick, H35 30-epoch variant, H41 NPCA-192-SLICES).

### Fleet status (8/8 active, EP3/EP6 gate batch impending)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| fern | #1189 | H35 NPCA-SSFL-STACK | Assigned 07:30Z, pickup pending |
| alphonse | #1185 | H31 WALLDIST | EP4 6.417%, val_vol_p 3.780% best-in-fleet. EP6 ~08:30Z (vp=49152) |
| thorfinn | #1177 | H26 NPCA Path A | EP5 6.541%, linear EP13 5.95%. EP6 ~08:30Z (std reading critical) |
| askeladd | #1187 | H33 SLICEPE v2 | Mid-EP2, run `u58fwoym`, healthy. EP3 verdict ~13:30Z |
| **edward** | **#1188** | **H34 OUTHEAD** | **v1 self-killed EP1. RELAUNCH INSTRUCTED with corrected step-indexed `--kill-thresholds`. v2 EP1 ETA ~10:30Z** |
| frieren | #1182 | H29 SSFL | EP7 6.4349% fleet-low. EP9-EP13 watch |
| tanjiro | #1183 | H18d | EP8 6.376%, τz/τx 1.647 above-band. EP9 ~09:00Z |
| nezuko | #1184 | H30 V2S xattn | EP4 6.576%, continuing to terminal |

### Key lessons added to Wave 30 → 31 transfer

7. **--kill-thresholds prefix is GLOBAL STEP, not epoch index** (lesson #8). Use step-indexed gates: 10864 = EP1 end, 32594 = EP3 end, 65228 = EP6 end. NEVER write `1:` or `3:` as "from EP1/EP3" gates. Saved as memory + propagated to advisor PR template defaults.

### Outstanding actions (next 8h chronological)

1. **alphonse H31 EP6 ~09:00Z** — val_vol_p floor-cross watch (vp=49152 curriculum)
2. **thorfinn H26 EP6 ~09:00Z** — std(τz/τx) reading critical (mean drifted 1.525→1.547)
3. **tanjiro H18d EP9 ~09:00Z** — mid-trajectory gate verdict
4. **edward H34 v2 launch + EP1 ~10:30Z** — with corrected gates
5. **askeladd H33 v2 EP3 ~13:30Z** — mechanism gate (slice_pe metrics critical)
6. **tanjiro H18d EP13 ~14:25Z** — terminal verdict (no-merge expected, test-side novelty)
7. **thorfinn H26 EP13 ~14:30Z** — terminal verdict (BASELINE-BEAT CANDIDATE, linear projection 5.95%)
8. **frieren H29 EP13 ~14-15Z** — terminal verdict (FLEET-LOW + test eval required)
9. **edward H34 v2 EP3 ~16-17Z** — asymmetry_ratio reading decisive
10. **fern H35 pickup + launch** — any time (cherry-pick H26 + H29)

---

## Previous invocation actions (2026-05-18 ~07:30Z) — FERN H24 GSTS CLOSED EP13 (15TH WAVE 30 DEAD END, NOT-A-MERGE: val_abupt 6.325% / test_vol_p FLOOR PASS) + FERN ASSIGNED H35 NPCA-SSFL-STACK (FIRST WAVE 31 HYPOTHESIS)

### Headline updates (07:30Z)

1. **Fern H24 GSTS terminal EP13: NOT-A-MERGE.** val_abupt 6.325% (+0.199pp FAIL), test_abupt 6.040% (+0.196pp FAIL), test_SP 3.831% FLOOR FAIL, test_WSS 6.953% FAIL, test_vol_p 3.610% FLOOR PASS (only floor cleared). test τz/τx 1.466 — inside band, no break. **15th Wave 30 dead end, 11th cold-start mean-shift fade.** Mechanism diagnostic: geom_temp MLP gradient-active but learned near-uniform softening (per-region delta only +0.0056) — lesson for Wave 31: per-vertex modulators need anti-uniform-collapse regularization.

2. **PR #1189 H35 NPCA-SSFL-STACK assigned to fern (FIRST WAVE 31)** — directly stacks H26 NPCA variance-break + H29 SSFL loss-reshape in single training run. Mechanism-distinct from all 15 closed dead ends. EP3 gate: std(τz/τx) ≥ 0.15 AND val_abupt ≤ 6.50%. EP13 success: val_abupt < 6.126% AND std ≥ 0.15. Branch instructs cherry-picking from `thorfinn/h26-...` and `frieren/h29-...`.

### Wave 30 final closure count: 15 dead ends + 1 retry-in-flight (H26 Path A)

| Closure # | Hypothesis | Mechanism class | Closure stage |
|---:|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape / per-vertex | EP3 fade |
| 8 | H23 Mean Teacher | Training-regularization | EP3 KILL |
| 9 | H18 area-weighted MSE | Per-vertex position | OUTLIER merged test-side |
| 10 | H21 per-component heads | Decoder capacity | EP3 DEAD |
| 11 | H25 ALGP | Aux head | EP3 KILL FADE |
| 12 | H27 PRLP | Train-eval space | EP3 KILL |
| 13 | H32 DIFFATTN | Attention mechanism | EP1 catastrophic |
| 14 | H28 SAM | Optimizer | EP2 fade + crash, advisor KILL relaunch |
| **15** | **H24 GSTS** | **Encoder slice-temp** | **EP13 NOT-A-MERGE (terminal)** |

Pattern locked: **11/12 mean-shift attacks faded into [1.44, 1.55] by EP3.** Only H26 NPCA (variance-break) survives mechanism gate. Wave 31 design principle: bias variance/spread + loss-reshape over mean-shift.

### Fleet status (8/8 active)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| **fern** | **#1189** | **H35 NPCA-SSFL-STACK** | **NEW — first Wave 31. Assigned 07:30Z. Cherry-pick from thorfinn H26 + frieren H29** |
| alphonse | #1185 | H31 WALLDIST | EP4 6.417%, val_vol_p 3.780% best-in-fleet. EP6 ~08:30Z (vp=49152 bump) |
| thorfinn | #1177 | H26 NPCA Path A | EP5 6.541%, linear EP13 5.95% (baseline-beat candidate). EP6 ~08:30Z |
| askeladd | #1187 | H33 SLICEPE v2 | Relaunch instructed (corrected `--kill-thresholds`). v2 EP1 ~08:30Z |
| edward | #1188 | H34 OUTHEAD | Pickup pending (assigned 06:00Z). EP3 ~16-17Z |
| frieren | #1182 | H29 SSFL | EP7 6.4349% fleet-low. EP9-EP13 watch |
| tanjiro | #1183 | H18d | EP8 6.3762% descent decelerating, τz/τx 1.647 above-band. CONTINUE TO EP13 ~14:25Z |
| nezuko | #1184 | H30 V2S xattn | EP4 6.576%, continuing to terminal for test-side |

### Outstanding actions (next 12h chronological)

1. **alphonse H31 EP6 ~08:30Z** — val_vol_p floor-cross watch
2. **thorfinn H26 EP6 ~08:30Z** — std(τz/τx) reading critical (mean drifted 1.525→1.547)
3. **askeladd H33 v2 EP1 ~08:30Z** — relaunch safety check
4. **tanjiro H18d EP9 ~09:00Z** — mid-trajectory gate verdict
5. **fern H35 pickup + launch** — any time (cherry-pick implementation ~40 LOC)
6. **askeladd H33 v2 EP3 ~13:30Z** — mechanism gate
7. **edward H34 pickup + EP1 safety check** — any time
8. **tanjiro H18d EP13 ~14:25Z** — terminal verdict (no-merge expected, test-side novelty)
9. **thorfinn H26 EP13 ~14:30Z** — terminal verdict (BASELINE-BEAT CANDIDATE)
10. **frieren H29 EP13 ~14-15Z** — terminal verdict (FLEET-LOW + test eval required)

### Wave 31 hypothesis queue (idle students next)

H35 (fern, ASSIGNED) → H42 NPCA-WARMSTART (when thorfinn H26 checkpoint available) → H41 NPCA-192-SLICES → H44 YAW-AUGMENTATION-NPCA → H36 ANCHOR-SLICE-QUERIES → H43 HEADS-DOUBLED → H37 WAVELET-SURFACE-LOSS

---

## Previous invocation actions (2026-05-18 ~07:00Z) — THORFINN H26 EP5 val_abupt 6.5414% (linear EP13 projection 5.95%, ON TRACK TO BEAT BASELINE) + ASKELADD H33 SELF-KILLED EP1 BY MISCONFIGURED `--kill-thresholds` (RELAUNCH INSTRUCTED, MECHANISM PRESERVED); WAVE 31 IDEAS BANK COMMITTED

### Headline updates (07:00Z)

1. **Thorfinn H26 Path A EP5 val_abupt 6.5414%** (W&B `gokysken` runtime 7.05h, step 43,813). Trajectory: EP3 6.91% → EP4 6.615% → EP5 6.541% (slope −0.074pp last epoch). **Linear projection EP13 = 5.95%, beats baseline 6.126% by 0.18pp**. τz/τx mean drifted up 1.525→1.540→1.547 (top of band) — std at EP4-5 NOT yet reported by student; this is the key remaining mechanism question. Stale_wip false positive #3 — check-in posted, run is fully healthy.

2. **Askeladd H33 SLICEPE SELF-KILLED at EP1** by misconfigured `--kill-thresholds "1:val_abupt<9.5"` (active from step 1, when EP1 baseline cold-start val_abupt is 20-28%). **Mechanism is alive** — EP1 val_vol_p_mae = 31.46 vs baseline 32.86 (slightly BETTER, no H32-style failure). EP1 surface metrics +6-9pp slower than baseline, expected for additive PE. Relaunch instructed with corrected `--kill-thresholds "3:val_abupt<9.5,3:val_SP<5.5"` (active from EP3). Run-v2 ETA EP1 ~08:30Z, EP3 verdict ~13:30Z.

3. **Wave 31 ideas bank committed** to `research/RESEARCH_IDEAS_2026-05-18_06:00.md` (researcher-agent delivered 11 mechanism-distinct candidates H35-H45). 4 immediate-assignment top picks: H35 NPCA-SSFL-STACK, H42 NPCA-WARMSTART, H41 NPCA-192-SLICES, H44 YAW-AUGMENTATION-NPCA — all build on H26 NPCA variance-break mechanism proof. Ready for student pickup as Wave 30 PRs close.

### Wave 30 mechanism scoreboard @ 07:00Z

| Hypothesis | Mechanism status | Accuracy status | EP13 projection |
|---|:--|:--|---:|
| **thorfinn H26 NPCA** | **MECHANISM PASSED** (EP3 std 0.228) | EP5 6.541%, slope −0.074 | **5.95% (linear, beats baseline)** |
| **frieren H29 SSFL** | spectral_loss saturated EP6+ (intended) | EP7 6.4349% **fleet-low** | 5.95-6.10% (conservative−optimistic) |
| **alphonse H31 WALLDIST** | volume pathway preserved | EP4 6.417%, val_vol_p 3.780% best-in-fleet | uncertain (val_vol_p approaching floor) |
| **nezuko H30 V2S xattn** | volume_p improvement at EP4 | EP4 6.576% (val fade), H18 test-side hope | won't beat val baseline |
| **tanjiro H18d** | mechanism locked (channel-decoupled) | EP6 6.457%, CONTINUE TO EP13 | 6.07% (Scenario B) |
| **fern H24 GSTS** | encoder slice-temp saturated | EP11 6.328%, slope ~0 | 6.30-6.33% (won't beat) |
| **askeladd H33 SLICEPE** | EP1 vol pathway intact (mechanism preserved) | RELAUNCHING v2 | EP3 verdict ~13:30Z |
| **edward H34 OUTHEAD** | pickup pending (assigned 06:00Z) | not started | EP3 verdict ~16-17Z |

### Fleet status (8/8 active)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| fern | #1174 | H24 GSTS | EP11 6.328% saturated. EP13 ETA ~09:00Z |
| alphonse | #1185 | H31 WALLDIST | EP4 6.417%, val_vol_p 3.780% best-in-fleet. Continuing |
| **thorfinn** | **#1177** | **H26 NPCA Path A** | **EP5 6.541%, linear EP13 5.95%. Stale_wip false positive #3 — check-in posted 07:05Z. τz/τx std at EP4-5 KEY remaining diagnostic** |
| **askeladd** | **#1187** | **H33 SLICEPE** | **SELF-KILLED EP1 by misconfigured `--kill-thresholds`. Mechanism preserved. RELAUNCHING with corrected `3:...` thresholds** |
| edward | #1188 | H34 OUTHEAD | Pickup pending (assigned 06:00Z) |
| frieren | #1182 | H29 SSFL | EP7 6.4349% fleet-low, ACK posted 06:00Z. EP9-EP13 watch |
| tanjiro | #1183 | H18d | CONTINUE TO EP13. EP9 gate ~09:00Z |
| nezuko | #1184 | H30 V2S xattn | EP4 6.576%, continuing to terminal for test-side |

### Outstanding actions (next 12h chronological)

1. **alphonse H31 EP6 ~08:30Z** — val_vol_p crossing floor? curriculum bump vp=49152
2. **askeladd H33 SLICEPE EP1 v2 ~08:30Z** — safety check with corrected thresholds
3. **tanjiro H18d EP9 ~09:00Z** — mid-trajectory gate (val_abupt ≤6.25% / val_SP ≤4.05% PASS)
4. **fern H24 EP13 ~09:00-10:00Z** — terminal verdict
5. **thorfinn H26 EP6 ~08:30Z** — vp=49152 curriculum bump; key std(τz/τx) reading
6. **thorfinn H26 EP13 ~14:30Z** — terminal verdict (linear projection 5.95%, BASELINE-BEAT candidate)
7. **frieren H29 EP13 ~14-15Z** — terminal verdict, test eval required
8. **edward H34 OUTHEAD launch + EP1 safety check** — pickup any time
9. **askeladd H33 EP3 v2 ~13:30Z** — mechanism gate verdict
10. **Wave 31 student-pickup queue**: H35/H42/H41/H44 ready when Wave 30 PRs close

### Key Wave 30 lessons (locked, unchanged from 06:00Z)

1. Mean-shift attacks fade, variance-break attacks survive (10/10 vs 1/1 evidence)
2. Loss reshape drives val_abupt within band (H29 fleet-low)
3. Volume pathway sensitive to input features and cross-modal fusion (H30/H31)
4. Slice-token magnitude critical at init (H32 design rule)
5. Band attractor geometrically FLAT in parameter space (H28 SAM negative result)
6. Head-side rank coupling UNTESTED (H34 OUTHEAD just assigned)
7. **NEW: `--kill-thresholds` syntax pitfall**: `N:metric<value` activates at STEP N (or EPOCH N), NOT meaning "step N+1 onwards". EP3 survival gates must be set to N=3 (epoch-indexed) or step-equivalent for the kill-eval window to start at EP3, not EP1.

---

## Previous invocation actions (2026-05-18 ~06:00Z) — H28 SAM CLOSED (14TH DEAD END / 10TH COLD-START FADE) + H34 OUTHEAD ASSIGNED TO EDWARD; FRIEREN H29 SSFL EP7 = WAVE 30 FLEET-LOW 6.4349%; THORFINN H26 PATH A MECHANISM GATE PASSED DECISIVELY

### Headline updates

1. **PR #1179 edward H28 SAM CLOSED ~06:00Z** — 14th Wave 30 dead end. Pod-crashed mid-EP3 at step 31,979 (hostname change, not training failure). Student auto-relaunched at 04:27Z with same params; advisor KILLED the relaunch based on EP1→EP2 fade-into-band signal (τz/τx 1.4406 → 1.4999 = Reading A pattern). 10th mean-shift cold-start fade of Wave 30. **Key negative result**: SAM mechanism fired healthy (cos g·ĝ 0.86→0.92, perturbed_loss > clean correctly) but converged to a flat basin INSIDE the band attractor — proves the band is geometrically FLAT in parameter space, not a sharpness trap. Optimizer-space axis CLOSED for ρ=0.05 mean-shift regime.

2. **PR #1188 edward H34 OUTHEAD assigned ~06:00Z** — per-channel auxiliary output heads (head-side rank-coupling attack). Hypothesis: 4-channel surface output `Linear(512, 4)` shares a near-rank-1 mode for τ_x/τ_z → all encoder-side attacks (10 closures) inherit this projection coupling. H34 adds 4 small zero-init residual MLPs (one per channel) for independent representational capacity. ~30 LOC `model.py` + `train.py`. EP3 gate: val_abupt ≤8.5%, τz/τx ≤1.42 (mechanism break), aux_head/tau_z/last_layer_norm > 0 (mechanism alive), aux_head/tau_z÷tau_x > 1.5 (mechanism asymmetry).

3. **Frieren H29 SSFL EP7 = WAVE 30 FLEET-LOW val_abupt 6.4349%** — descending faster than fern H24 at equivalent stage (slope −0.076pp EP6→EP7 vs fern's −0.041pp at EP6→EP7). spectral_loss plateau is interpreted as mechanism saturation, not failure. EP13 projection: 5.95-6.10% (clears baseline 6.126%). val_SP 4.257% and val_vol_p 3.874% both need final-epoch EMA+cosine compression. **Strongest accuracy story of Wave 30** alongside thorfinn H26 mechanism story.

4. **Thorfinn H26 Path A EP3 mechanism gate PASSED DECISIVELY** — std(τz/τx) per-car = **0.228** (4.56× over gate 0.05, 11.4× over baseline 0.02). 13/34 cars outside [1.40, 1.60] band. val_abupt 6.91% EP3 descending. **First Wave 30 mechanism proof that the band attractor is breakable via variance/spread (not mean drift).**

5. **Wave 30 mean-shift attack class EXHAUSTED**: 10 cold-start fades observed (H18, H20, H24, H25, fern H24, alphonse H25, edward H28, frieren H29, nezuko H30, alphonse H31). Of these, H29 shows that fade-into-band does NOT preclude best-in-fleet val_abupt achievement, and H26 NPCA Path A is the only ATTACK to escape the band via variance not mean.

### Floor disease — new Wave 30 mechanism map

After 14 closures + 4 surviving mechanism wins, the attractor map looks like:

| Mechanism type | Effect on band [1.44, 1.55] | Examples |
|---|---|---|
| **Mean-shift attack** (encoder, loss, optimizer, augmentation) | Fades into band by EP2-EP3 | 10/10 Wave 30 closures |
| **Variance/spread attack** (per-car local coords, NPCA-style) | Sustains spread breakout | thorfinn H26 (std 10× baseline) |
| **Loss reshape** (spectral, multi-frequency) | Drives absolute val_abupt down WITHIN band | frieren H29 fleet-low 6.4349% |
| **Volume pathway boost** (input feature, cross-modal) | Drives val_vol_p toward floor | alphonse H31 (3.780% EP4), nezuko H30 (3.914%) |
| **Head-side decoupling** (per-channel aux heads) | UNTESTED — H34 just assigned | edward H34 OUTHEAD live |

### Fleet status (8/8 active — H34 EDWARD NEW)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| fern | #1174 | H24 GSTS encoder slice-temp | EP11 val_abupt 6.328% saturated (EP10→EP11 slope −0.004pp). Won't beat baseline. EP13 ETA ~09:00Z |
| alphonse | #1185 | H31 WALLDIST log-SDF input | EP4 6.417%, val_vol_p 3.780% best-in-fleet (+0.137pp above floor). 9th cold-start fade. Continuing |
| thorfinn | #1177 | H26 NPCA Path A | **EP3 MECHANISM GATE PASSED** (std 0.228, 13/34 cars outside band). val_abupt 6.91% EP3, descending. EP6 ~09:30Z |
| askeladd | #1187 | H33 SLICEPE | Launched 05:16Z. Smoke checks excellent (vol magnitude 1.024× baseline, no H32 failure mode). Student flagged σ=0.02 may be below gradient signal floor (literature concern, σ≈0.088 recommended). EP1 ETA ~07:00-07:30Z |
| **edward** | **#1188** | **H34 OUTHEAD** | **NEW — assigned 06:00Z. Per-channel aux MLP output heads, zero-init bit-exact baseline. Head-side rank-coupling attack** |
| frieren | #1182 | H29 SSFL | **EP7 6.4349% = WAVE 30 FLEET-LOW**. Slope steeper than fern at equivalent stage. EP13 ETA later today. Continue |
| tanjiro | #1183 | H18d channel-decoupled τz | CONTINUE TO EP13. EP9 gate ~09:00Z (val_abupt ≤6.25% / val_SP ≤4.05% PASS) |
| nezuko | #1184 | H30 V2S xattn | EP4 6.576%, τz/τx 1.522 (val-side fade matching H18 watch-item-3). Continuing to terminal for test-side hope |

### Closed Wave 30 directions (14 confirmed dead ends + 1 outlier + 1 surviving mechanism win)

| # | Hypothesis | Tier | Key result |
|---|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape / per-vertex | All dead |
| 8 | H23 Mean Teacher EMA | Training-regularization | KILL (21.36% EP3) |
| 9 | H18 area-weighted MSE | Per-vertex position | OUTLIER: test τz/τx 1.418 ★ + test_vol_p PASS, but miss val/SP |
| 10 | H21 per-component heads | Decoder capacity | DEAD — capacity not the bottleneck |
| 11 | H25 ALGP aux local-grad | Mean-shift encoder via aux task | KILL EP3 FADE — objective-disconnected |
| 12 | H27 PRLP per-component rel-L2 loss | Train-eval space | KILL EP3 — per-car normalisation INVERSELY re-weights gradient. Train-eval space axis CLOSED. |
| 13 | H32 DIFFATTN differential attention | Attention mechanism | KILL EP1 — subtractive SDPA destroys slice-token magnitude (0.31×) → volume_p_mae catastrophic. Attention-mechanism axis CLOSED. |
| **14** | **H28 SAM optimizer-sharpness** | **Optimizer-space** | **KILL EP2 via Reading A fade (1.4406→1.4999). Pod-crashed mid-EP3. RELAUNCH KILLED BY ADVISOR. Key finding: band attractor is geometrically FLAT in parameter space (SAM cos g·ĝ 0.86→0.92 stable). Optimizer-space axis CLOSED for ρ=0.05 mean-shift regime.** |
| **Surviving** | **H26 NPCA local frame** | **Encoder INPUT geometric transformation** | **PATH A EP3 MECHANISM GATE PASSED (std 0.228 = 4.56× gate). First Wave 30 variance-break proof. Awaiting EP6 accuracy interim.** |

### Key fleet diagnostic — Wave 30 lessons crystallized

1. **Mean-shift attacks fade, variance-break attacks survive** (10/10 vs 1/1 evidence at EP3 gate)
2. **Loss reshape can drive val_abupt within band** (frieren H29 fleet-low) — band-breaking and best-val are SEPARATE achievements
3. **Volume pathway is sensitive to input features and cross-modal fusion** (alphonse H31 + nezuko H30 both approaching floor)
4. **Slice-token magnitude is critical at init** (H32 lesson: <0.5× → volume pathway catastrophe; H33 SLICEPE designed safe)
5. **Band attractor is GEOMETRICALLY FLAT in parameter space** (H28 SAM negative result rules out sharpness-trap mechanism)
6. **Head-side rank coupling UNTESTED** — H34 OUTHEAD is the first attack on this axis

### Outstanding actions (next 12h chronological)

1. **askeladd H33 SLICEPE EP1 safety check ~07:00-07:30Z** — val_volume_p_mae must be normal cold-start (5-10), NOT catastrophic like H32 (30+). slice_pe/global/abs_mean should be growing slowly from σ=0.02 init
2. **tanjiro H18d EP9 ~09:00Z** — mid-trajectory gate (val_abupt ≤6.25% / val_SP ≤4.05% PASS)
3. **thorfinn H26 Path A EP6 ~09:30Z** — decisive accuracy interim — does mechanism-proven hypothesis beat baseline?
4. **fern H24 GSTS EP13 ~10:00-11:00Z** — terminal verdict (val saturated at 6.328%, unlikely to beat baseline 6.126%)
5. **alphonse H31 EP6 ~08:30Z** — val_vol_p crossing floor? curriculum bump to vp=49152
6. **nezuko H30 EP6+ continued descent** — H18 pattern watch (test-side survival hope)
7. **frieren H29 SSFL EP9-13** — final EP descent, EP13 test eval required
8. **edward H34 OUTHEAD pickup + smoke + launch** — pending student poll cycle
9. **researcher-agent: Wave 31 ideas writing to /research/RESEARCH_IDEAS_2026-05-18_06:00.md** — background, 8-12 hypotheses

### Wave 31 design principles (locked from Wave 30 lessons)

- **Bias toward variance/spread attacks** over mean-shift attacks
- **Combine surviving mechanisms** (H26 NPCA + H29 SSFL stacking)
- **Frequency-domain and physics-informed losses** as next major axis
- **Anchor-conditioned slice queries** (DAB-DETR/SpiderSolver line — askeladd lit notes flagged)
- **No magnitude-destructive architectural changes** (H32 design rule)
- **Output-head explorations** if H34 OUTHEAD lands signal

---

## Previous invocation actions (2026-05-18 ~05:00Z) — H32 DIFFATTN CLOSED (13TH DEAD END) + H33 SLICEPE ASSIGNED TO ASKELADD; TANJIRO H18d CONTINUE TO EP13; 8TH COLD-START FADE (ALPHONSE H31 EP2)

### Headline updates

1. **PR #1186 askeladd H32 DIFFATTN CLOSED ~05:00Z** — 13th Wave 30 dead end. Both V1 (canonical paper) and V2 (minimal PR pseudocode) killed at EP1 (val_abupt ~28-30%, vol_p_mae 31-36). Failure mode diagnostic: subtractive SDPA at init destroys slice-token magnitude (0.31× baseline) → volume pathway catastrophic (no residual fallback for volume tokens). Surface/WSS metrics near-normal. Attention-mechanism axis CLOSED. Critical design rule for Wave 30 remainder: any modification that multiplies slice-token magnitude by <0.5× at init will break volume pathway.

2. **PR #1187 askeladd H33 SLICEPE assigned ~05:00Z** — learnable per-slice positional embedding `P ∈ R^{H × S × D_head}`, zero-init (σ=0.02), added to slice_tokens BEFORE SDPA. Directly safe from H32 failure mode (additive, small init, no magnitude destruction). Hypothesis: slice tokens lack explicit positional identity → band attractor's dominant mode can saturate all slices equally. With slice PE, model can route τz-specific physics to dedicated slices. ~15 LOC `model.py`. EP1 safety check: val_volume_p_mae must be normal cold-start range (5-10, not 30+).

3. **Tanjiro H18d #1183 CONTINUE TO EP13** — EP6 val_abupt 6.457% in marginal zone. Scenario B linear extrap → EP13 6.07% (clears baseline). EP9 mid-trajectory gate (~09:00Z): val_abupt ≤6.25% + val_SP ≤4.05% continue; >6.35%/>4.25% kill. Full test eval regardless at EP13.

4. **8th cold-start fade confirmed: alphonse H31 WALLDIST EP2 τz/τx 1.529** (EP1 was 1.376 = deepest of Wave 30). EP3 ETA ~05:00-05:15Z.

5. **Frieren H29 SSFL EP3 landed 6.6493%** — best absolute EP3 of Wave 30. Student verdict pending. spectral_loss=0 in summary raises mechanism-active question (flagged in check-in).

6. **Thorfinn H26 Path A EP2 mechanism signal growing**: std(τz/τx) = 0.185 at EP2 (vs 0.092 EP1, approaching Path B's EP3 0.216). EP3 ~04:43Z decisive.

### Floor disease localization — REFINED after H26 mechanism proof

H26 has now produced the **first evidence that the τz/τx band attractor is encoder-breakable** — but accuracy is still bounded. This SEPARATES the question:

- **Question 1 (Wave 30 has been chasing): Can the band attractor be broken?** → H26 NPCA: YES (std 10× baseline, decisively passes any spread-break gate). Wave 30 finding: band attractor IS structural, IS breakable, but is broken by LOCAL FRAME PROJECTION on input, not by attention/loss/optimizer manipulation.

- **Question 2 (newly opened): Does breaking the band attractor BEAT THE BASELINE on accuracy?** → H26 Path B is non-conclusive (budget-truncated). H26 Path A full-budget retry IS the test. If it beats baseline, the question is answered. If not, the band-break is necessary but insufficient.

### Fleet status (8/8 active)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| fern | #1174 | H24 GSTS encoder slice-temp | EP7 val landing shortly (was 6.375% mid-EP7 at 01:53Z). 5th cold-start fade (τz/τx 1.543). Gate: if EP7 <6.30% continue, ≥6.35% kill |
| alphonse | #1185 | H31 WALLDIST log-SDF input channel | EP2 landed: val_abupt 7.416%, τz/τx 1.529 = 8th cold-start fade (EP1 was 1.376). EP3 ETA ~05:00-05:15Z |
| thorfinn | #1177 | H26 NPCA full 18h retry Path A | EP2 std 0.185 (mechanism growing). EP3 ~04:43Z decisive gate |
| **askeladd** | **#1187** | **H33 SLICEPE learnable slice PE** | **NEW — assigned 05:00Z. Additive zero-init PE per slice. Safe from H32 volume_p_mae failure mode. EP1 safety check + EP3 gate** |
| edward | #1179 | H28 SAM optimizer-sharpness | EP2 val 8.804%, EP3 imminent (~03:10-03:30Z). cos 0.87. Near 9.5% KILL threshold |
| frieren | #1182 | H29 SSFL frequency-domain loss | **EP3 landed 6.6493% — BEST EP3 of Wave 30**. τz/τx 1.501 in band (fade). spectral_loss=0 concern flagged. Student verdict pending |
| tanjiro | #1183 | H18d channel-decoupled τz-only area weight | EP6 6.457% marginal → **CONTINUE TO EP13** granted. EP9 gate ~09:00Z May 18. Mechanism locked: band-break is channel-coupled tied-budget effect |
| nezuko | #1184 | H30 V2S xattn cross-modal | EP2 val 7.595%, τz/τx 1.500 (band drift from EP1 1.428). EP3 ~04:00Z |

### Closed Wave 30 directions (13 confirmed dead ends + 1 outlier + 1 retry-in-flight)

| # | Hypothesis | Tier | Key result |
|---|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape / per-vertex | All dead |
| 8 | H23 Mean Teacher EMA | Training-regularization | KILL (21.36% EP3) |
| 9 | H18 area-weighted MSE | Per-vertex position | OUTLIER: test τz/τx 1.418 ★ + test_vol_p PASS, but miss val/SP |
| 10 | H21 per-component heads | Decoder capacity | DEAD — capacity not the bottleneck |
| 11 | H25 ALGP aux local-grad | Mean-shift encoder via aux task | CRASHED + EP3 FADE — objective-disconnected |
| **12** | **H27 PRLP per-component rel-L2 loss** | **Train-eval space** | **KILL EP3 — per-car normalisation INVERSELY re-weights gradient. Train-eval space axis CLOSED.** |
| **13** | **H32 DIFFATTN differential attention** | **Attention mechanism** | **KILL EP1 — subtractive SDPA destroys slice-token magnitude (0.31×) → volume_p_mae catastrophic. Attention-mechanism axis CLOSED. Design rule: no modification that reduces slice-token magnitude <0.5× at init.** |
| **Retry** | **H26 NPCA local frame** | **Encoder INPUT geometric transformation** | **PATH-A 18H RETRY IN-FLIGHT. MECHANISM PROVEN (std 10×). EP3 ~04:43Z.** |

### Key fleet diagnostic refined after H26 mechanism proof

- **Band attractor IS encoder-breakable** via input local-frame projection (Gram-Schmidt tangent frame from normals) — H26 NPCA proven mechanism
- **Compressed budget masks accuracy potential** — H26 Path B EP3 6.97% with monotone descent does NOT preclude full-budget EP13 beating baseline (linear extrap 4× slowdown → 5.97%)
- **Per-channel mechanism types in Wave 30** are now:
  - Mean-shift below band (cold-start fade): fern H24 1.395→1.514, alphonse H25 1.377→1.512, frieren H29 1.385 EP1 only, tanjiro H18 1.412→1.489
  - Spread-break (sustained at EP3+): thorfinn H26 std 0.101→0.216 (test 0.120)
  - Lower-band-edge sitting: edward H28 EP1 1.4406
  - Above-band (channel-decoupling inverse signal): tanjiro H18d EP1 1.634

### Outstanding actions (next 12h chronological)

1. **thorfinn H26 Path A EP3 ~04:43Z May 18** — full-budget accuracy test for mechanism-proven hypothesis (std 0.185 at EP2, growing)
2. **alphonse H31 WALLDIST EP3 ~05:00-05:15Z** — 8th cold-start fade in progress, EP3 decisive
3. **nezuko H30 V2S EP3 ~04:00Z** — EP2 τz/τx 1.500 band drift (fade?)
4. **frieren H29 SSFL student EP3 verdict** — 6.6493% best EP3 of Wave 30, mechanism question (spectral_loss=0?)
5. **edward H28 SAM EP3 ~03:10-03:30Z** — 8.804% EP2, near KILL at 9.5%
6. **fern H24 EP7 gate** — 6.375% mid-EP7, ep7 decisive
7. **tanjiro H18d EP9 ~09:00Z** — mid-trajectory gate for CONTINUE-granted run
8. **askeladd H33 SLICEPE pickup + launch** (any time; ~15 LOC, zero-init safe)
9. **thorfinn H26 Path A EP6 ~09:30Z** — decisive interim for baseline-beating accuracy
10. **tanjiro H18d EP13 ~14:25Z** — full terminal test eval

**13 confirmed dead ends + 1 retry-in-flight. Plateau Protocol active. Critical design rule established: slice-token magnitude must be preserved at init. H26 NPCA remains STRONGEST hypothesis (mechanism proven, full-budget in flight). H33 SLICEPE is safest novel architectural attack designed around DIFFATTN diagnostic.**

---

## Earlier invocation actions (2026-05-17 ~22:50Z) — H25 ALPHONSE CRASHED + EP3 FADE = 11TH DEAD END; H30 ASSIGNED; floor disease localized to encoder feature CONTENT

### Headline updates

1. **PR #1176 alphonse H25 ALGP CLOSED TERMINAL NOT-A-MERGE pending student terminal SENPAI-RESULT** — run `pvdjrlx4` crashed at 22:12Z step 33,427 (after EP3 val landed at step 32,592). EP3 verdict: val_abupt 7.142% FAIL +1.016pp vs baseline 6.126%; τz/τx 1.5117 = **6th confirmed cold-start fade** (EP1 deepest signal 1.3774 → EP3 1.5117 rebounded into band [1.44, 1.55]). Mechanism worked perfectly (aux_loss −53%, aux_pred_std 1000× growth) but objective-disconnected from τ_z floor. Engineering note: 22:14Z health check posted "healthy" status reading stale W&B summary; correction posted 22:50Z.

2. **6th cold-start fade in Wave 30** — H18 (1.412→1.489), H20 (1.401→1.523), H23 (1.426→1.866 KILL), H24 (1.395→1.514), and now H25 (1.377→1.512). DEEPEST EP1 mean-shift in Wave 30 STILL faded. Conclusion: deeper EP1 mean-shifts are warmup-dynamics artifacts, NOT representational changes that survive epochs. **CLOSED axis: mean-shift τz/τx encoder manipulation via aux task / saliency MLP / curvature loss / temperature.**

3. **Floor disease NOW localized to encoder feature CONTENT** — not decoder capacity (H21 closed), not mean-shift encoder manipulation (H25 + H24 + H18 + H20 + H23 closed), not channel-coupled position weighting (H18 closed). Remaining attack class: **explicit physical features in input** (wall-distance log-SDF, separation-line auxiliary, pressure-gradient input) AND cross-modal feature fusion (H30 V2S in flight). Alphonse next assignment will likely be WALLDIST (log-SDF input channel for volume tokens) — direct attack on encoder information content per H21 diagnosis.

### Fleet status (8/8 active after H30 assignment, 7/8 after alphonse close pending)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| fern | #1174 | H24 GSTS encoder slice-temp | EP3 MARGINAL (τz/τx 1.5145 in-band rebound). EP6 binding gate ~02:00Z May 18 — decisive val_abupt baseline-crossing test |
| **alphonse** | **#1185** | **H31 WALLDIST (NEW)** | **#1176 CLOSED 23:10Z. H31 PR #1185 assigned ~23:22Z. log-SDF (volume_x col 3 already loaded) → log(|sdf|+1e-3) as 5th channel ~12 LOC. EP3 gate ~14:30Z May 18** |
| thorfinn | #1177 | H26 NPCA local-frame input aug | **PATH-B TERMINAL 23:18Z: mechanism PASSED decisively (std(τz/τx) 0.02→0.216 = 10× baseline, 14/34 val cars + 20/50 test cars outside band) — accuracy FAILED (val 6.97% vs 6.126%) BUT only 3 of 5 epochs ran (budget cut) AND trajectory still descending −0.088%/1k_steps. SENT BACK 23:38Z for full 18h Path A retry (epochs 13, vol-schedule 0:16384:3:32768:6:49152:9:65536). Linear extrapolation conservative 4× slowdown → EP13 val_abupt 5.97% would BEAT baseline. STRONGEST UNFINISHED HYPOTHESIS IN WAVE 30.** |
| askeladd | #1178 | H27 PRLP train-loss in eval space | training, EP1 τz/τx 1.523 IN-BAND (expected — floor mechanism not band-attractor). EP3 gate ~23:46Z (val_SP ≤ 4.10% beating H11b EP3 4.368%) |
| edward | #1179 | H28 SAM optimizer-sharpness | EP1: τz/τx 1.4406 lower band-edge (first optimizer-space signal). EP3 ~00:10–02:10Z May 18 |
| frieren | #1182 | H29 SSFL frequency-domain loss | EP1 mechanism PASS, val async ~23:45Z |
| tanjiro | #1183 | H18d channel-decoupled τz-only area weight | training, EP1 ETA ~23:04Z (overdue, agent reports step 14,984 at 22:58Z) |
| nezuko | #1184 | H30 V2S xattn (volume→surface cross-modal) | just assigned 21:50Z, awaiting pickup |

### Closed Wave 30 directions (11 confirmed dead ends + 1 outlier)

| # | Hypothesis | Tier | Key result |
|---|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape / per-vertex | All dead |
| 8 | H23 Mean Teacher EMA | Training-regularization | KILL (21.36% EP3) |
| 9 | H18 area-weighted MSE | Per-vertex position | OUTLIER: test τz/τx 1.418 ★ + test_vol_p PASS, but miss val/SP |
| 10 | H21 per-component heads | Decoder capacity | DEAD — mechanism proved, capacity not the bottleneck |
| **11** | **H25 ALGP aux local-grad** | **Mean-shift encoder via aux task** | **CRASHED + EP3 FADE — mechanism alive but objective-disconnected** |

### KEY FLEET DIAGNOSTIC: Floor disease localization (after H25 closure)

- **NOT decoder capacity** (H21 confirmed)
- **NOT per-vertex loss-reweighting** (7 attacks confirmed)
- **NOT training-regularization** (H23 confirmed)
- **NOT channel-coupled position-weighting** (H18 confirmed)
- **NOT mean-shift saliency MLP** (H24 partial fade — MARGINAL, not survive)
- **NOT mean-shift aux gradient prediction** (H25 deepest EP1 signal still faded)
- **NOT (likely) coordinate frame transformation** (pending H26 thorfinn EP3)

Surviving attack axes:
- **Encoder INPUT physical features** (alphonse WALLDIST next — log-SDF) — DIRECTLY targets H21 + H25 closure diagnosis
- **Cross-modal encoder fusion** (H30 nezuko V2S, in flight)
- **Spatial-frequency gradient** (H29 frieren SSFL — frequency-domain loss)
- **Optimization landscape sharpness** (H28 edward SAM — optimizer-space)
- **Train-eval space mismatch** (H27 askeladd PRLP — proxy loss)
- **Encoder INPUT local frame** (H26 thorfinn NPCA — coordinate augmentation, different mechanism class spread-break)

### Band attractor — fleet-wide pattern (6 confirmed cold-start fades + 1 test-survivor + 1 spread-break)

| Run | EP1 τz/τx | EP3 τz/τx | Verdict |
|:--|---:|---:|:--|
| H18 tanjiro (CLOSED) | 1.357 | 1.489 | val-fade BUT test 1.418 survived ★ |
| H24 fern (MARGINAL) | 1.395 | 1.514 | Cold-start fade |
| H25 alphonse (CRASHED, CLOSED) | **1.377 ★ deepest** | **1.512** | **6th cold-start fade** |
| H26 thorfinn (SPREAD-break) | std=0.101 | TBD | Different mechanism class |
| H28 edward SAM | **1.4406 lower edge** | TBD | First optimizer-space signal |

### Outstanding actions (next 12h chronological)

1. **alphonse terminal SENPAI-RESULT** (any time now) — close PR #1176, assign WALLDIST
2. **tanjiro H18d EP1** (~23:04Z, overdue) — channel-decoupled τz-only area weight gate
3. **askeladd H27 EP3 gate ~23:46Z** — val_SP ≤ 4.10% beating H11b EP3 4.368%
4. **thorfinn H26 EP3 mechanism gate ~23:48Z** — std + outside-band count
5. **frieren H29 EP1 val ~23:45Z** — async val_budget_minutes=90
6. **edward H28 EP3 ~00:10–02:10Z May 18** — optimizer-space τz/τx<1.42 + val_abupt<6.00%
7. **tanjiro H18d EP3 ~01:31Z May 18** — channel-decoupling
8. **fern H24 EP6 ~02:00Z May 18** — decisive val_abupt baseline-crossing test
9. **nezuko H30 V2S pickup** — awaiting student poll

**11 confirmed dead ends. Plateau Protocol active. Wave 30 attack tier diversity now spans 7 axes: per-vertex/loss-shape (CLOSED), regularization (CLOSED), decoder capacity (CLOSED), mean-shift encoder (CLOSED), encoder INPUT features (NEW — WALLDIST queued), cross-modal fusion (H30 IN-FLIGHT), optimizer-space (H28 IN-FLIGHT), frequency-domain loss (H29 IN-FLIGHT), train-eval-space loss (H27 IN-FLIGHT), coordinate-frame input (H26 IN-FLIGHT spread-break).**

---

## Earlier invocation actions (2026-05-17 ~21:50Z) — H21 CLOSED (10th dead end, decoder-capacity-not-the-bottleneck), H30 V2S xattn assigned to nezuko

### Headline updates

1. **PR #1171 nezuko H21 per-component output heads CLOSED TERMINAL NOT-A-MERGE / 10TH WAVE-30 DEAD END** — val_abupt 6.493% (+0.367pp miss), test_abupt 6.119% (+0.275pp miss), ALL FOUR test floors breach. test τz/τx 1.4391 (marginal band-edge, not a real break). **DECISIVE DIAGNOSIS**: decoder capacity is NOT the τ_z bottleneck. H21 mechanism worked perfectly (cleanest gradient-decoupling in fleet history, 11/13 buckets τz > τy > τx > cp, τ_z head absorbed 22% more param mass) yet still missed. The encoder features fed to output heads lack sufficient τ_z information — adding more decoder capacity cannot fix insufficient encoder representation.

2. **PR #1184 nezuko H30 V2S XATTN ASSIGNED** — Volume-to-Surface cross-attention: surface tokens (Q) attend to volume tokens (K/V), inserted AFTER existing surf-to-vol xattn. Volume tokens encode off-body flow physics (separation, wake, recirculation) that DETERMINES τ_z boundary-layer separation. Zero-init out_proj guarantees baseline recovery. Directly motivated by H18 evidence (S2V propagated better-fitted τz to vol_p). EP3 gate: τz/τx ≤ 1.42 AND val_abupt ≤ 8.5%. ~40 LOC change in model.py + small train.py flag.

3. **4 independent EP1 band-break signals active** — fern H24 τz/τx 1.395 (mean-shift), alphonse H25 τz/τx 1.377 (mean-shift), thorfinn H26 std=0.101 + 17/34 outside band (spread-break), edward H28 τz/τx 1.4406 (band-edge, first optimizer-space signal). All 4 are in different mechanism classes.

### Fleet status (8/8 active after H30 assignment)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| fern | #1174 | H24 GSTS encoder slice-temp | EP3 landed (W&B: val_abupt 6.791%, τz/τx 1.514 IN-BAND REBOUND — MARGINAL continue, pending student SENPAI-RESULT post) |
| alphonse | #1176 | H25 ALGP auxiliary local-gradient | training, EP3 ~23:00Z (needs-rebase deferred) |
| thorfinn | #1177 | H26 NPCA local-frame input aug | EP1 MECHANISM FIRING (std=0.101, 17/34 outside band), EP3 gate ~23:48Z |
| askeladd | #1178 | H27 PRLP train loss in eval space | EP3 gate ~23:46Z (post-v2-sqrtfix) |
| edward | #1179 | H28 SAM optimizer-sharpness | EP1: τz/τx 1.4406 lower band-edge (4th EP1 signal, first optimizer-space), EP3 ~00:10-02:10Z May 18 |
| frieren | #1182 | H29 SSFL frequency-domain loss | EP1 mechanism PASS (spectral_loss −75%, 1.1% share), EP1 val ~21:35Z |
| tanjiro | #1183 | H18d channel-decoupled τ_z-only area weight | launched 20:46Z, EP1 ~22:21Z |
| **nezuko** | **#1184** | **H30 V2S cross-attention** | **just assigned** |

### Closed Wave 30 directions (10 confirmed dead ends + 1 outlier)

| # | Hypothesis | Tier | Key result |
|---|---|---|:--|
| 1-7 | H10b/H11b/H12/H16/H16b/H20/H22 | Loss-shape / per-vertex | All dead |
| 8 | H23 Mean Teacher EMA | Training-regularization | KILL (21.36% EP3) |
| 9 | H18 area-weighted MSE | Per-vertex position | OUTLIER: test τz/τx 1.418 ★ + test_vol_p PASS, but miss val/SP |
| **10** | **H21 per-component heads** | **Decoder capacity** | **DEAD — mechanism proved, capacity not the bottleneck** |

### KEY FLEET DIAGNOSTIC: Floor disease localization (after H21 closure)

H21 closure eliminates decoder-capacity as the bottleneck. Surviving attack axes:
- **Cross-modal encoder fusion** (H30 nezuko V2S — NEW attack class, encoder feature content)
- **Spatial-frequency gradient** (H29 frieren SSFL — frequency-domain loss)
- **Optimization landscape sharpness** (H28 edward SAM — optimizer-space)
- **Backbone representation** (H25 alphonse ALGP — aux gradient task)
- **Train-eval space mismatch** (H27 askeladd PRLP — proxy loss)
- **Encoder INPUT local frame** (H26 thorfinn NPCA — coordinate augmentation)
- **Encoder slice temperature / geometry saliency** (H24 fern GSTS — MARGINAL at EP3)
- **Position-weighting channel-decoupled** (H18d tanjiro — evidence-following)
- **NOT decoder capacity** (H21 confirmed)
- **NOT per-vertex loss-reweighting** (7 attacks confirmed)
- **NOT training-regularization** (H23 confirmed)
- **NOT channel-coupled position-weighting** (H18 confirmed)

### Band attractor — fleet-wide pattern (4 active EP1 band-break signals + 1 test-survivor + 1 spread-break)

| Run | EP1 τz/τx | Notes |
|:--|---:|:--|
| H18 tanjiro (CLOSED) | 1.357 | Test-surviving: test 1.418 ★ |
| H24 fern (MARGINAL) | 1.395 | EP3 1.514 IN-BAND REBOUND — cold-start fade |
| **H25 alphonse** | **1.377** | EP3 gate ~23:00Z |
| **H26 thorfinn** | 1.427 mean, std **0.101** | SPREAD-break: 17/34 outside band, EP3 ~23:48Z |
| **H28 edward SAM** | **1.4406 lower edge** | First optimizer-space signal, EP3 ~02:00Z May 18 |

### Outstanding actions (next 8h chronological)

1. **fern H24 EP3 gate ~21:25Z** — band-break primary (τz/τx ≤ 1.40 CONTINUE; 1.40-1.60 MARGINAL; > 1.60 KILL)
2. **alphonse H25 EP3 gate ~23:00Z** — band-break secondary
3. **askeladd H27 EP3 gate ~23:46Z** — loss-normalization-space (val_SP ≤ 4.10% CONTINUE)
4. **edward H28 EP3 gate ~00:00-02:00Z May 18** — optimizer-space (τz/τx < 1.42 AND val_abupt < 6.00%)
5. **frieren H29 EP3 gate ~01:30-02:30Z May 18** — frequency-space (τz/τx ≤ 1.42 AND spectral_loss descending)
6. **tanjiro H18d EP3 gate ~6h after launch** — channel-decoupling (τz/τx ≤ 1.42 AND val_SP ≤ 4.50%)
7. **thorfinn H26 Path B 5-ep gate (~3-4h after launch)** — geom band-break in compressed schedule

**FIVE EP3 gates land in the next 12 hours** — the single highest-density decision point of Wave 30.

---

## Earlier invocation actions (2026-05-17 ~19:40Z) — H23 KILLED, H29 SSFL assigned — Wave 30 tier-shift to frequency-domain loss

### Headline updates

1. **PR #1173 frieren H23 Mean Teacher CLOSED TERMINAL KILL / 8TH WAVE-30 DEAD END / MECHANISM-WORKS-BUT-NET-COST-EXCEEDS-NET-BENEFIT** — EP3 val_abupt 21.36% triggers pre-specified KILL gate (>11.0%). Trajectory slope collapse 94% (EP2→EP3: −15.64pp → −0.95pp); EP13 linear projection ~11.9% best case, never approaches baseline. Mechanism alive (consistency_loss descending, student-teacher gap −71%, zero nonfinite) — failure at **objective-composition level**: regularizer constrains optimization manifold, preventing supervised loss from finding productive descent. τz/τx drifted to 1.866 (vs ~1.46 baseline band — WRONG DIRECTION). **8 of 8 Wave 30 loss-shape + training-regularization attacks now dead.** NOT pursuing parameter-space follow-ups (λ=0.02, warmup 5000+, σ=0.003, EMA 0.9995) — would consume 36-54h for diminishing-returns variants of confirmed failure mode.

2. **PR #1182 frieren H29 SSFL ASSIGNED — FIRST FREQUENCY-DOMAIN LOSS ATTACK in DrivAerML history** — Spectral Surface Loss with Streamwise Frequency Upweighting. Sort surface tokens by streamwise z-coord, apply `torch.fft.rfft` along sorted-N, compute frequency-weighted MSE with linear ramp `linspace(1.0, hf_weight=2.0, N//2+1)` upweighting Nyquist 2× vs DC. **F-principle grounding** (Xu et al. 2020): GD provably learns low-frequency first; high-frequency upweighting counteracts the bias. **Physical motivation**: separation events (windshield/A-pillar/mirror wakes) are LOCALIZED HIGH-SPATIAL-FREQUENCY in streamwise coords; MSE dilutes their gradient across smooth-region tokens. Recipe: `--lambda-spectral 0.1 --spectral-hf-weight 2.0 --spectral-channels wss` (τx/τy/τz only — cp is smooth). ~70 LOC train.py only. λ=0.0 = exact baseline recovery. EP3 falsifiable gate: `τz/τx ≤ 1.42 AND val_abupt ≤ 8.5% AND train/spectral_loss DESCENDING`. KILL if `val_abupt > 9.5%` OR `τz/τx > 1.55` OR `train/spectral_loss flat`. Baseline-compute (NO forward doubling unlike H28). Composable with H27 PRLP (loss-space orthogonal) and H28 SAM (optimizer-space orthogonal) for H30.

3. **Plateau Protocol tier-shift activated for the second time in 2 hours** — Wave 30 now spans 6 attack tiers: INPUT (H24 GSTS, H26 NPCA), OUTPUT-HEAD (H21), REPRESENTATION COUPLING (H25 ALGP), TRAIN-EVAL SPACE (H27 PRLP), TRAINING DYNAMICS-OPTIMIZER (H28 SAM), TRAINING DYNAMICS-LOSS-FREQUENCY (H29 SSFL — NEW). Only remaining loss-shape arm: tanjiro H18 area-weighted MSE.

### Fleet status (8/8 active after H29 assignment)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| **frieren** | **#1182** | **H29 SSFL — FIRST frequency-domain loss attack** | **just assigned (spectral gradient rebalancing)** |
| edward | #1179 | H28 SAM — first optimizer-space attack | training, EP3 gate ~00:00-02:00Z May 18 |
| askeladd | #1178 | H27 PRLP — train loss in eval space | EP3 gate ~23:46Z (post-bf16-NaN fix) |
| fern | #1174 | H24 GSTS encoder slice-temperature | EP1 τz/τx 1.395 (FIRST band-break signal of Wave 30) — EP3 gate ~21:25Z |
| nezuko | #1171 | H21 per-component output heads | EP10+ vol-65536 curriculum |
| tanjiro | #1163 | H18 area-weighted MSE | EP12+ stalled 6.582% (only remaining loss-shape arm) |
| alphonse | #1176 | H25 ALGP auxiliary local-gradient prediction | in-flight, needs-rebase notice (low priority) |
| thorfinn | #1177 | H26 NPCA encoder input local-frame aug | Path B 5-epoch compressed (pod 360-min budget bug) |

### Closed Wave 30 directions (now 8 confirmed dead ends spanning 2 tiers)

1. Per-vertex error reweighting (H18[earlier-iter], H20) — rel_L2 normalization erases absolute-residual gains
2. Static Huber on τ (H16, H16b) — frac_in_L1 decays 40× over training
3. Bounded-exp output activation (H10, H10b) — 73%/27% split structural in encoder
4. Output tangent-frame reparameterization (H17) — convergence gap too large
5. Charbonnier on cp + MAE-aux (H22) — cp-MSE saturation NOT the disease
6. Input-channel gating (H11b) — gate works mechanically but floor disease is DOWNSTREAM of input
7. Per-vertex τ-magnitude weighted MSE (H12) — structurally biases cp gradient; monotonic regression in α; τz/τx unchanged
8. **Mean Teacher EMA self-distillation (H23)** — 94% slope collapse; mechanism alive but regularizer kills supervised descent direction; τz/τx drifted to 1.866 (wrong direction)

**Strong consensus tier 1**: rel_L2 metric geometry erases gain from absolute-residual per-vertex reweighting under DrivAerML — axis SATURATED (7/7).
**Strong consensus tier 2**: training-regularization scheme stabilizes mechanism but breaks objective composition — axis CLOSED (1/1 with high information content).

### KEY FLEET DIAGNOSTIC: Floor disease localization (after H23 closure)

After 8 closed Wave 30 attempts spanning 2 tiers, the floor-breach disease is strongly localized to one of these surviving axes:
- **Spatial-frequency gradient imbalance** (H29 frieren SSFL — NEW frequency-domain tier)
- **Optimization landscape sharpness** (H28 edward SAM — optimizer-space)
- **Output-head / per-head gradient paths** (H21 nezuko per-component MLPs)
- **Backbone representation coupling cp/τ** (H25 alphonse ALGP)
- **Train-eval space mismatch** (H27 askeladd PRLP)
- **Encoder INPUT content** (H24 fern GSTS, H26 thorfinn NPCA)
- **NOT input-channel gating** (H11b confirmed)
- **NOT cp-loss-shape** (H22 confirmed)
- **NOT per-vertex loss-reweighting** (7/7 confirmed)
- **NOT training-side regularization** (H23 confirmed)

### Band attractor — fleet-wide pattern (FIRST band-break signal at fern H24 EP1)

| Run | EP1 τz/τx | EP_terminal τz/τx | Outcome |
|:--|---:|---:|:--|
| H10b fern | — | 1.530 | converged into band |
| H11b askeladd | 1.466 | 1.556 | converged INTO band |
| H12 edward | — | 1.476 | inside band (no shift) |
| H18 tanjiro | 1.412 (transient) | 1.482 | faded into band |
| H20 alphonse | 1.401 (transient) | 1.523 | faded into band |
| H21 nezuko | — | 1.529 | inside band |
| H22 thorfinn | — | (terminal pending offline test eval) | premise dead at EP3 |
| H23 frieren | 1.426 | 1.866 (EP3 KILL) | drifted ABOVE band (wrong direction) |
| **H24 fern** | **1.395 (LIVE)** | **EP3 gate ~21:25Z** | **FIRST true band-break signal** |

Two axes NOT yet tested:
- spectral content of gradient signal (H29 NEW)
- (formerly: optimizer dynamics — now in-flight via H28)

### Outstanding actions watch (next invocation)

1. **fern H24 EP3 gate (~21:25Z)** — τz/τx ≤ 1.40 = band-break confirmed CONTINUE; 1.40-1.60 MARGINAL; > 1.60 KILL. EP1 1.395 is strongest band-exit signal in Wave 30 history.
2. **askeladd H27 EP3 gate (~23:46Z)** — val_SP ≤ 4.10% CONTINUE; 4.10-4.40 MARGINAL; > 4.40 + τz/τx in band = KILL. Post-bf16-NaN fix (forward bf16, loss/grad fp32) verified clean.
3. **edward H28 EP3 gate (~00:00-02:00Z May 18)** — τz/τx < 1.42 AND val_abupt < 6.00% = band-break confirmed; KILL if val_abupt > 6.50% OR τz/τx > 1.55.
4. **frieren H29 EP1 launch confirmation** — verify `train/spectral_loss` in [0.001, 1.0]; if > 10, halve λ; if < 1e-4, increase hf_weight.
5. **frieren H29 EP3 gate (~6-7h after launch)** — τz/τx ≤ 1.42 AND val_abupt ≤ 8.5% AND `train/spectral_loss` descending = CONTINUE.
6. **thorfinn H26 Path B 5-ep gate (~3-4h after launch)** — geom band-break diagnostic in compressed schedule.
7. **tanjiro H18 EP12+ stale_wip check-ins** — long job; pod healthy expected.
8. **Three EP3 gates land in next 8 hours** — fern (~21:25), askeladd (~23:46), edward (~00:00-02:00). H29 will land ~01:30-02:30Z.

---

## Earlier invocation actions (2026-05-17 ~17:55Z) — H12 closed, H28 SAM assigned — Wave 30 tier-shift to optimizer-space

### Headline updates

1. **PR #1151 edward H12 CLOSED TERMINAL NEGATIVE / STRUCTURALLY-BIASED / 7TH WAVE-30 DEAD END** — Both arms regressed. Arm A (α=0.3, best): val_abupt 6.290% (+0.164pp FAIL), test_WSS 6.952% (+0.225pp FAIL), test_SP 3.816% (+0.239pp FLOOR BREACH), test_vol_p 3.620% (held −0.023pp), τz/τx 1.476 (NO MECHANISM SHIFT). Arm C (α=0.7) skipped — monotonic regression confirmed at 2 points; further α-sweep would only deepen the failure. **H12a (channel-decoupled) NOT pursued** — τ-weighting itself produces no band-break per τz/τx 1.476, so decoupling cp doesn't fix what isn't broken in τ direction. **7 of 7 Wave 30 per-vertex/per-token loss-shape attacks now dead at floor breach gate** (H10b, H11b, H12, H16, H16b, H20, H22) — per-vertex loss-reweighting axis fully closed against rel_L2 metric geometry.

2. **PR #1179 edward H28 SAM ASSIGNED — FIRST OPTIMIZER-SPACE ATTACK IN WAVE 30** — Sharpness-Aware Minimization (Foret et al. ICLR 2021) two-pass perturb-recompute-restore wrapper around Lion. `--sam-rho 0.05` default, ~60 LOC in target/train.py (SAMLion wrapper + DDP `no_sync()` first-pass + standard second-pass). zero-init at ρ=0 = exact baseline recovery. EP3 falsifiable gate: `τz/τx < 1.42 AND val_abupt < 6.00%` (band break required). KILL gate: `val_abupt > 6.50% OR τz/τx > 1.55`. ~36h wall-clock (2× forward passes), `SENPAI_TIMEOUT_MINUTES=2200`. **Information-value justification**: a definitive negative result rules out the entire optimizer-trajectory class of explanations for the τz/τx band attractor — load-bearing observation for Wave 31 design. Either SAM breaks the band (paradigm shift: opens optimizer-tuning direction) OR rules out optimizer-space decisively (redirects to representation H24/H26).

3. **Tier-shift activated for Wave 30** — Plateau Protocol applied: 7 consecutive loss-shape closures triggered escalation from loss-layer attacks to optimizer-dynamics attacks. Wave 30 now spans 5 attack tiers simultaneously: INPUT (H24, H26), OUTPUT-HEAD (H21), REPRESENTATION COUPLING (H25), TRAIN-EVAL SPACE (H27), TRAINING DYNAMICS (H23 EMA, H28 SAM), with H18 the only remaining loss-shape arm.

### Fleet status (8/8 active after H28 assignment)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| **edward** | **#1179** | **H28 SAM — FIRST optimizer-space attack** | **just assigned (training dynamics)** |
| askeladd | #1178 | H27 PRLP — train loss in eval space | just assigned (loss normalization axis) |
| fern | #1174 | H24 GSTS encoder slice-temperature | EP1 expected soon |
| nezuko | #1171 | H21 per-component output heads | EP7 6.557% slope re-acceleration |
| tanjiro | #1163 | H18 area-weighted MSE | EP8.9 6.635% (only remaining loss-shape arm) |
| frieren | #1173 | H23 Mean Teacher self-distillation | EP2 22.31% MARGINAL, EP3 gate ~17:55Z |
| alphonse | #1176 | H25 ALGP auxiliary local-gradient prediction | in-flight, needs-rebase notice posted (low priority) |
| thorfinn | #1177 | H26 NPCA encoder input local-frame aug | just assigned |

### Closed Wave 30 directions (now 7 confirmed dead ends — per-vertex loss-reweighting axis FULLY CLOSED)

1. Per-vertex error reweighting (H18[earlier-iter], H20) — rel_L2 normalization erases absolute-residual gains
2. Static Huber on τ (H16, H16b) — frac_in_L1 decays 40× over training
3. Bounded-exp output activation (H10, H10b) — 73%/27% split structural in encoder
4. Output tangent-frame reparameterization (H17) — convergence gap too large
5. Charbonnier on cp + MAE-aux (H22) — cp-MSE saturation NOT the disease
6. Input-channel gating (H11b) — gate works mechanically but floor disease is DOWNSTREAM of input
7. **Per-vertex τ-magnitude weighted MSE (H12)** — structurally biases cp gradient; monotonic regression in α; τz/τx unchanged

**Strong consensus**: rel_L2 metric geometry (per-car per-component normalization) erases gain from absolute-residual per-vertex reweighting under DrivAerML — axis is saturated, 7/7 confirmation.

### KEY FLEET DIAGNOSTIC: Floor disease localization (refined after H12 closure)

After 7 closed Wave 30 attempts, the floor-breach disease is strongly localized to:
- **Output-head / per-head gradient paths** (H21 nezuko per-component MLPs, H27 askeladd train loss in eval space)
- **Backbone representation coupling cp/τ** (H25 alphonse ALGP)
- **Train-eval space mismatch** (H27 askeladd directly attacks)
- **Optimization landscape sharpness** (H28 edward SAM — NEW tier)
- **Encoder INPUT content** (H24 fern GSTS, H26 thorfinn NPCA)
- **NOT input-channel gating** (H11b confirmed)
- **NOT cp-loss-shape** (H22 confirmed)
- **NOT per-vertex loss-reweighting** (7/7 confirmed)

### Band attractor — fleet-wide pattern (7 independent confirmations)

| Run | EP1 τz/τx | EP_terminal τz/τx | Outcome |
|:--|---:|---:|:--|
| H10b fern | — | 1.530 | converged into band |
| H11b askeladd | 1.466 | 1.556 | converged INTO band |
| **H12 edward** | — | **1.476** | inside band (no shift, monotonic in α) |
| H18 tanjiro | 1.412 (transient) | 1.482 | faded into band |
| H20 alphonse | 1.401 (transient) | 1.523 | faded into band |
| H21 nezuko | — | 1.529 | inside band |
| H22 thorfinn | — | (terminal pending offline test eval) | premise dead at EP3 |

Pattern strengthens: the [1.44, 1.55] τz/τx attractor is the geometric signature of either a sharp optimization basin (H28 SAM tests this) or a representation-structural floor (H24/H26 test this). One axis NOT yet tested: encoder INPUT content (H24, H26 now in-flight). One axis tier NEW: optimizer dynamics (H28).

### Outstanding actions watch (next invocation)

1. **frieren H23 EP3 (~17:55-18:05Z)** — decisive Mean Teacher KILL/CONTINUE — val_abupt ≤ 8.5% PASS, ≤ 11% MARGINAL, > 11% KILL
2. **fern H24 EP1 launch confirmation** — verify GSTS smoke smoke pre-run and EP1 token
3. **alphonse H25 / thorfinn H26 / askeladd H27 / edward H28 EP1 launch confirmations** — 4 fresh PRs to monitor
4. **nezuko H21 EP10 (~18:30Z)** — vol 65536 curriculum bump test
5. **edward H28 EP3 gate (~6-8h from launch, ~00:00-02:00Z May 18)** — KEY band-break test; if τz/τx < 1.42, paradigm shift
6. **edward H28 first SAM smoke 200 steps before main run** — verify DDP no_sync correctness, throughput ~50% baseline
7. **tanjiro H18 + edward[was-H12]** — terminal SENPAI-RESULT collection complete; edward now H28

---

## Latest invocation actions (2026-05-17 ~17:30Z) — H11b closed, H27 PRLP assigned

### Headline updates

1. **PR #1167 askeladd H11b CLOSED TERMINAL NOT-A-MERGE / VAL-AND-WSS-BEAT-BUT-FLOOR-BREACH** — first sustained Wave 30 baseline beat (val_abupt 6.057%, test_abupt 5.8147%, test_WSS 6.6322% all beat baseline) BUT both floors breached (test_SP 3.7179% +0.141pp, test_vol_p 3.6773% +0.034pp). Gate mechanism worked (mean_abs 0→0.537, 3 channels saturated to |g|>0.7) but floor disease is downstream of input. **5th confirmed dead end in Wave 30**. Stackable as substrate.

2. **PR #1178 askeladd H27 PRLP ASSIGNED** — Per-Component Relative-L2 Proxy Loss. Replace normalized-space MSE training loss with per-car per-component rel_L2 in DENORMALIZED (physical) space — making train objective structurally identical to eval metric. Direct attack on confirmed dual-space mismatch (trainer_runtime.py: training MSE in normalized space vs eval rel_L2 in physical space). ~60 lines. **First Wave 30 attack on loss NORMALIZATION SPACE** (orthogonal to all 6 closed loss-shape attacks). EP3 falsifiable gate: `val_SP ≤ 4.10%` (strictly below H11b EP3 4.368%). Explicitly composable with H11b for future stacking. Targets test_SP/test_vol_p floor breach directly.

3. **frieren H23 EP2 = 22.31% MARGINAL** (≤25% gate, >20% pass) — cold-start drag resolved EP1 37.95% → EP2 22.31% (−15.64pp drop). Mean Teacher mechanism active and not blocking training. `consistency_loss` descending (slope −0.0078 per 1k steps). budget 1010min confirmed. **EP3 (~17:55-18:05Z) is decisive** — if val_abupt ≤ 8.5%, clean PASS; ≤ 11% MARGINAL; > 11% KILL.

4. **PR #1176 alphonse H25 rebase notice posted** — branch needs rebase before final merge (tay advanced with 17:00Z + 16:40Z commits to research/ docs only, alphonse's training code untouched). Action: NONE while training; rebase after terminal SENPAI-RESULT. Low priority.

### Fleet status (8/8 active after H27 assignment)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| **askeladd** | **#1178** | **H27 PRLP — train loss in eval space** | **just assigned (loss normalization axis)** |
| fern | #1174 | H24 GSTS encoder slice-temperature | EP1 expected soon |
| edward | #1151 | H12 τ-magnitude weighted MSE | EP9.6 stalled 6.290% |
| nezuko | #1171 | H21 per-component output heads | EP7 6.557% slope re-acceleration |
| tanjiro | #1163 | H18 area-weighted MSE | EP8.9 6.635% |
| frieren | #1173 | H23 Mean Teacher self-distillation | EP2 22.31% MARGINAL, EP3 gate ~17:55Z |
| alphonse | #1176 | H25 ALGP auxiliary local-gradient prediction | in-flight, needs-rebase notice posted (low priority) |
| thorfinn | #1177 | H26 NPCA encoder input local-frame aug | just assigned |

### Closed Wave 30 directions (now 6 confirmed dead ends)

1. Per-vertex error reweighting (H18, H20) — rel_L2 normalization erases absolute-residual gains
2. Static Huber on τ (H16, H16b) — frac_in_L1 decays 40× over training
3. Bounded-exp output activation (H10, H10b) — 73%/27% split structural in encoder
4. Output tangent-frame reparameterization (H17) — convergence gap too large
5. Charbonnier on cp + MAE-aux (H22) — cp-MSE saturation NOT the disease
6. **Input-channel gating (H11b)** — gate works mechanically but floor disease is DOWNSTREAM of input

### KEY FLEET DIAGNOSTIC: Floor disease localization narrowing

After H11b closure, the floor-breach disease is increasingly localized to:
- **Output-head / per-head gradient paths** (H27 attack: train loss in eval space, H21 attack: separate MLPs)
- **Backbone representation coupling cp/τ** (H25 ALGP attacks this)
- **Train-eval space mismatch** (H27 attacks this directly)
- **NOT input-side** (H11b confirmed)
- **NOT cp-loss-shape** (H22 confirmed)

### Band attractor — fleet-wide pattern (6 independent confirmations)

| Run | EP1 τz/τx | EP_terminal τz/τx | Outcome |
|:--|---:|---:|:--|
| H10b fern | — | 1.530 | converged into band |
| H11b askeladd | 1.466 | **1.556** | converged INTO band |
| H12 edward | — | 1.553 | inside band |
| H18 tanjiro | 1.412 (transient) | 1.482 | faded into band |
| H20 alphonse | 1.401 (transient) | 1.523 | faded into band |
| H21 nezuko | — | 1.529 | inside band |

**The [1.44, 1.55] collapse band is now confirmed as a fleet-wide attractor for ALL input/output/loss-shape attacks. Future attacks must target either (a) the encoder mechanism (H24 GSTS), (b) backbone representation content (H25 ALGP), (c) encoder input feature content (H26 NPCA), or (d) the loss normalization space (H27 PRLP).**

### Outstanding actions for next-invocation watch

- frieren #1173 EP3 gate (~17:55-18:05Z) — decisive Mean Teacher kill/continue
- fern #1174 H24 EP1 (any minute now)
- alphonse #1176 H25 + thorfinn #1177 H26 launch confirmations
- nezuko #1171 EP10 (~18:30Z) — vol 65536 curriculum bump test
- edward #1151 + tanjiro #1163 — terminal SENPAI-RESULT expected at EP13

## Previous invocation actions (2026-05-17 ~17:00Z) — H22 closed, H26 NPCA assigned

### Headline updates

1. **PR #1172 thorfinn H22 CLOSED TERMINAL-NEUTRAL** — Charbonnier-cp + MAE-aux. Mechanism wired correctly but **decisively falsified at EP3**:
   - cp_charb/cp_mae ratio 0.97 → effective loss is `1.5·|e|` (MAE-equivalent, smooth-L1 transition not engaged at ε=1e-3)
   - val_SP @ EP3 = 4.405% **MATCHES** H10b 4.422% and H11b 4.368% — NOT steeper. cp descent under MAE-equivalent is identical to MSE
   - Critical side effect: τ-channel REGRESSION (cp loss contribution 4.9× larger than τ at EP3) → val_abupt +0.70pp WORSE than H11b, val_WSS +0.99pp WORSE
   - test_SP 4.079% (+0.502pp floor breach), test_vol_p 4.139% (+0.496pp floor breach), test_abupt 6.849% (+1.005pp baseline regression)
   - **NOT-STACKABLE** — premise dead. Floor-preservation via cp-side loss reformulation now CLOSED.
   - Student's pre-launch `printenv SENPAI_TIMEOUT_MINUTES=360.0` doc is the cleanest end-to-end budget bug evidence to date.

2. **PR #1177 thorfinn H26 NPCA ASSIGNED** — Normal-Projected Coordinate Augmentation. Lightweight encoder-input feature augmentation: Gram-Schmidt local tangent frame `[t1, t2]` from existing normals, project global position onto `[p·n, p·t1, p·t2]`, append 3 scalar features to `project_surface_features` input (4→7 dims). **Parameter-free local frame** (deterministic Gram-Schmidt), zero-init of new weight columns ensures baseline-identical start. **First Wave 30 attack on encoder INPUT content** (orthogonal to H24 slice-temperature, H25 backbone-aux-loss). Targets band attractor `[1.44, 1.55]` at the most upstream lever. EP3 falsifiable gate: `std(τz/τx | per-val-car) ≥ 0.04` AND ≥ 1/34 val cars outside [1.40, 1.60]. NOT same failure as H17 (output reparam, convergence gap) — H26 keeps output basis global. Literature anchors: Intrinsic Vector Heat Network (ICML 2024), Beyond Canonicalization (ICLR 2025).

3. **frieren H23 EP2 PENDING (~17:15Z)** — runtime 4.25h, EP~1.3. Earlier W&B check confused EP1 (37.95%) with EP2; EP2 has not been reported yet. EP2 gate: ≤20% PASS / ≤25% MARGINAL / >25% KILL.

4. **edward H12 STALL CONFIRMED** — val_abupt 6.290% at EP9.6, slope −0.005%/1k (essentially flat). tau_mag_weight mean=0.858, p95=1.465, max=2.283 (weights too concentrated near 1.0, mechanism not engaging). val_vol_p marginal at +0.031pp above floor. EP10 imminent.

5. **tanjiro H18 SLOW DESCENT** — val_abupt 6.635% at EP8.9, slope −0.015%/1k (3× faster than edward). val_vol_p only +0.018pp above floor — close to passing. τz/τx 1.482 still in band (cold-start break faded EP3 1.412 → EP9 1.482). EP10 imminent.

### Fleet status (8/8 active after H26 assignment)

| Student | PR | H | Status |
|:--|---:|:--|:--|
| askeladd | #1167 | H11b gated multi-scale input | EP~9, fleet best val 6.059% (only baseline crosser) |
| fern | #1174 | H24 GSTS encoder slice-temperature | just launched, identity-init verified |
| edward | #1151 | H12 τ-magnitude weighted MSE | EP9.6 stalled 6.290%, val_vol_p marginal floor |
| nezuko | #1171 | H21 per-component output heads | EP7 slope re-acceleration 6.557%, EP13 projection 6.08% |
| tanjiro | #1163 | H18 area-weighted MSE | EP8.9 6.635%, val_vol_p +0.018pp from floor, band re-collapsed |
| frieren | #1173 | H23 Mean Teacher self-distillation | EP1.3 awaiting EP2 gate (~17:15Z) |
| alphonse | #1176 | H25 ALGP auxiliary local-gradient prediction | just launched |
| **thorfinn** | **#1177** | **H26 NPCA local-frame input aug** | **just assigned (bold orthogonal — encoder input)** |

### Closed Wave 30 directions (now 5 confirmed dead ends)

1. Per-vertex error reweighting (H18, H20) — rel_L2 normalization erases absolute-residual gains
2. Static Huber on τ (H16, H16b) — frac_in_L1 decays 40× over training, single calibration insufficient
3. Bounded-exp output activation (H10, H10b) — 73%/27% mag/dir split structural in encoder, NOT output
4. Output tangent-frame reparameterization (H17) — convergence gap too large for 13ep at lr=9e-5
5. **Charbonnier on cp + MAE-aux (H22)** — cp-MSE saturation is NOT the disease, MAE-equivalent matches MSE

### KEY DIAGNOSTIC — H22 closure refines floor-breach diagnosis

Test_SP floor breach is NOT caused by cp gradient saturation. cp errors descend monotonically under MSE (H10b/H11b) and under MAE-equivalent (H22) — same trajectory. Therefore the test_SP disease is downstream of cp accuracy:
- Likely **structural in cp/τ representation coupling at backbone level** → H25 ALGP (in flight) addresses this
- Could be **rel_L2 metric weighting large-cp regions differently than training loss** → no current attack on this
- Per-channel grad-norm equalization is the natural future diagnostic

### Floor-breach pattern (snapshot @ 17:00Z)

| Student | PR | EP~ | val_abupt | val_SP | val_vol_p | τz/τx |
|---|---|---|---|---|---|---|
| askeladd H11b | #1167 | 9 | **6.059%** (#1) | 4.055% (+0.478) | 3.786% (+0.143) | 1.554 |
| nezuko H21 | #1171 | 7 | 6.557% | 4.289% (+0.712) | 3.841% (+0.198) | 1.539 |
| edward H12 | #1151 | 9.6 | 6.290% | 4.105% (+0.528) | 3.674% (+0.031 MARGINAL) | 1.554 |
| tanjiro H18 | #1163 | 8.9 | 6.635% | 4.327% (+0.750) | 3.661% (+0.018 MARGINAL) | 1.482 |

**Two runs (edward, tanjiro) have val_vol_p within +0.05pp of floor** — first within-noise approaches to floor pass since fern H10b's terminal pass. EP10-13 trajectory decides.

## Previous invocation actions (2026-05-17 ~16:40Z) — Mid-Wave-30 status update

### Headline updates

1. **thorfinn H22 SIGTERM @ 4.9h / EP3** — run `2y5zraax` finished early (360-min budget bug). val_abupt 7.110% at EP3 is **at-baseline-pace** (baseline #972 EP3 was ~7-8%) — NOT a mechanism failure, run was budget-killed before Charbonnier floor-preservation could take effect (typically EP8+). Requested student run offline test-eval on best-EP3 checkpoint to extract any signal. Closure decision pending eval results.

2. **askeladd H11b NEW FLEET BEST val_abupt 6.059%** (down from EP5.8's 6.076%, at 13.9h ≈EP9) — still ONLY active-fleet baseline crosser, descending slowly. val_SP 4.055% (vs floor 3.577%, +0.478pp BREACH) remains binding merge blocker. EP10 gate ~18:00Z.

3. **nezuko H21 EP7 slope RE-ACCELERATION** — vol curriculum advance 32768→49152 injected fresh descent: EP6→EP7 slope **−0.074pp/ep** (2.2× faster than EP5→EP6's −0.033pp/ep). EP7 val_abupt 6.557%, projected EP13 ~6.08% (below baseline 6.126%!). **Baseline beat back in play** if EP10 curriculum bump (65536 pts) maintains re-acceleration. Floor breach still binding (val_SP 4.289%, +0.712pp above floor at EP7).

4. **fern H24 launched cleanly** — identity init verified (t_v=1.000 exactly at step 0, `geom_temp_std` 1.21e-06 step 0 → 3.41e-05 at step 497), SENPAI_TIMEOUT_MINUTES=1100 confirmed via /proc, no budget bug. Smoke loss 4.1→2.1 over 500 steps. Full 8-GPU 13ep run started 16:26Z. Acknowledged fern's correct flag pruning (`--vector-decoupled-head --direction-cos-loss-weight 0.1` were H10/H10b artifacts, not in current `tay`).

5. **edward H12 stagnating** — val_abupt 6.290% at 13.4h (≈EP9-10), slope EP5.7→EP9-10 only −0.015pp/ep (severely flat). Projected NOT-A-MERGE. τz/τx 1.553 firmly in band — likely hitting same rel_L2 normalization blocker as H20 (per-token weighting can't beat absolute-residual erasure).

6. **tanjiro H18 stagnating** — val_abupt 6.676% at 11.2h (≈EP8-9). Cold-start band-break faded EP3 1.412 → EP4.8 1.467 → EP9 **1.478** (drifting up toward band center). Confirms fleet-wide pattern: ALL EP1-3 band-breaks are cold-start residuals that fade.

### Outstanding actions

- thorfinn #1172: awaiting offline test-eval on EP3 best checkpoint (Path B execution)
- frieren #1173: EP2 mechanism gate ~17:15Z (Mean Teacher kill-or-continue decision)
- edward #1151, tanjiro #1163: EP10 gates ~17:00Z (both projected NOT-A-MERGE)
- askeladd #1167: EP10 gate ~18:00Z (val_SP slope-watch for floor approach)
- nezuko #1171: EP10 gate ~18:30Z (vol curriculum 65536 bump — does slope hold?)
- fern #1174: EP1 expected ~17:30Z
- alphonse #1176: launch confirmation expected
- 4 status check-ins posted (#1172, #1167, #1151, #1163)

### Floor-breach pattern (snapshot @ 16:40Z)

| Student | PR | EP~ | val_abupt | val_SP | val_vol_p | τz/τx |
|---|---|---|---|---|---|---|
| **askeladd H11b** | #1167 | ~9 | **6.059%** (#1) | 4.055% (+0.478 floor) | 3.786% (+0.143 floor) | 1.554 |
| nezuko H21 | #1171 | 7 | 6.557% | 4.289% (+0.712 floor) | 3.841% (+0.198 floor) | 1.539 |
| edward H12 | #1151 | ~10 | 6.290% | 4.105% (+0.528 floor) | 3.674% (+0.031 floor) | 1.553 |
| tanjiro H18 | #1163 | ~9 | 6.676% | 4.349% (+0.772 floor) | 3.670% (+0.027 floor) | 1.478 |
| thorfinn H22 | #1172 | 3 (SIGTERM) | 7.110% | 4.405% | 4.166% | 1.502 |

**Critical pattern**: 4/5 active mid-fleet runs hover at val_vol_p within +0.03–0.20pp of floor (3.643%). This is the test_vol_p floor pass that fern H10b achieved at terminal (3.481%) — multiple ongoing runs are positioned to test it. Test_SP floor (3.577%) is breached by >+0.45pp in all 5 — the dominant merge blocker fleet-wide.

## Previous invocation actions (2026-05-17 ~16:20Z) — (1) **fern H10b CLOSED TERMINAL NOT-A-MERGE** (first test_vol_p floor pass 3.481%, 73%/27% split structural in encoder, H24 assigned); (2) **alphonse H20 CLOSED TERMINAL-NULL** (band-break cold-start artifact 1.401→1.523 by EP3, rel_L2 metric geometry kills focal gains, per-vertex reweighting direction CLOSED); (3) **fern assigned H24 GSTS** (PR #1174, encoder slice-temperature sharpening); (4) **alphonse assigned H25 ALGP** (PR #1176, auxiliary local-gradient prediction forcing backbone to encode spatially coherent τ_z — FIRST backbone-representation-content attack in Wave 30); fleet **8/8 active**

### Actions this invocation (16:20Z May 17)

- **ASSIGNED alphonse H25 (PR #1176)** — Auxiliary Local-Gradient Prediction (ALGP). AuxGradHead MLP (192→64→1, zero-init) attached to `surface_hidden [B, N_s, 192]` post-backbone; predicts KNN finite-difference ‖∇τ_z‖ per vertex (Option C: N_aux=4096 random vertices per batch, K=8, per-batch std normalization). Forces backbone to encode spatially coherent τ_z without changing output topology. **First Wave 30 attack on backbone representation content** (orthogonal to H24 encoder-slice-temp, and all loss/output attacks). λ_aux=0.1. AuxGradHead discarded at inference. EP3 gate: aux_loss decreasing monotonically AND τz/τx exits [1.40, 1.60] on ≥2 val cars. Fleet now 8/8 active.

### Actions this invocation (16:10Z May 17)

- **CLOSED PR #1164 fern H10b** as TERMINAL NOT-A-MERGE / MECHANISM-PASS / HYPOTHESIS-FALSIFIED / PARKED-STACKABLE. W&B verified (14.31h runtime, best EP12 EMA, all metrics exact). Decision: val_abupt 6.217% FAIL +0.091pp; test_SP 3.755% FLOOR FAIL +0.178pp; test_WSS 6.980% FAIL +0.253pp. Wins: **FIRST test_vol_p floor pass in active fleet** (3.481% < 3.643%, −0.162pp); best output-head band-edge break (test τz/τx 1.441). Decisive negative result: 73%/27% mag/dir split UNCHANGED from H10 despite bounded-exp matching GT distribution EXACTLY — **split is structural in encoder, not output activation**.
- **ASSIGNED fern H24 (PR #1174)** — Geometric-Saliency Slice-Temperature Sharpening (GSTS). Lightweight 3-layer MLP (4→16→1, zero-init) maps `[nx,ny,nz,area]` → per-vertex multiplicative temperature `t_v = exp(clamp(MLP, -3, 3))` applied to slice-assignment logits before softmax in `TransolverAttention.create_slices()`. Zero-init guarantees t_v=1.0 at init (exact Transolver identity). **First Wave 30 attack on encoder's slice-assignment mechanism.** Literature anchor: GeoTransolver (arXiv 2512.20399) achieves test_WSS 4.90% on DrivAerML via encoder geometry conditioning — only sub-baseline result in literature. EP3 gate: `geom_temp_std > 0.05` AND τz/τx exits [1.40, 1.60].
- **CLOSED PR #1170 alphonse H20** as TERMINAL-NULL. Mechanism executed cleanly (normalization invariant held, w_z_p95/mean=3.8, raw_w_z/raw_w_x crossover by EP2 confirming mechanism fired). BUT: (a) EP1-2 band-break (1.389→1.401) was cold-start residual artifact — reverted to 1.523 by EP3; (b) **rel_L2 metric geometry fundamentally fights absolute-residual reweighting** — metric normalizes by channel reference magnitude, erasing absolute τ_z gains; (c) easy-vertex damping at EP3 (raw_w_z≈0.03 → 97% vertices contribute nothing). **Per-vertex error-reweighting direction NOW CLOSED for DrivAerML rel_L2.** Pod effective SENPAI_TIMEOUT_MINUTES ~300 (budget bug).
- **DISPATCHED researcher-agent (background)** for alphonse H25 — bold orthogonal hypothesis needed given per-vertex error-reweighting direction is closed. Awaiting result.

### CRITICAL FLEET FINDING: No confirmed sustained band-break in Wave 30

Every apparent band-break (τz/τx < 1.44) in Wave 30 has faded back inside the [1.44, 1.55] band by EP3-5:

| Run | EP of break | τz/τx at break | τz/τx at EP3+ | Conclusion |
|:--|---:|---:|---:|:--|
| H18 tanjiro | EP3 | 1.412 (fleet-deepest) | 1.467 at EP4.8 | cold-start transient, fading |
| H20 alphonse | EP1-2 | 1.401 (fleet-deepest) | 1.523 at EP3 | cold-start residual artifact |

**The band attractor resets by EP3 for ALL mechanism types tested so far.** H24 (fern, encoder slice-temperature) is the first attack targeting the encoder's slice-assignment mechanism directly — theoretically the correct level to attack. If H24's `geom_temp_std` > 0.05 and τz/τx exits band at EP3, it would be the first confirmed sustained band-break in Wave 30.

### Current fleet (7/8 active, 1 idle pending H25)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| askeladd | #1167 | H11b gated multi-scale input | EP5.8 healthy, slope flat | **6.076% (#1) — BELOW baseline** |
| fern | **#1174** | **H24 GSTS — encoder slice-temp** | **just assigned** | **— (encoder attack)** |
| edward | #1151 | H12 τ-magnitude weighted MSE | EP5.7 descending | 6.350% |
| nezuko | #1171 | H21 per-component output heads | EP4.5 healthy | 6.631% |
| tanjiro | #1163 | H18 area-weighted surface MSE | EP4.8 healthy | 6.810% |
| thorfinn | #1172 | H22 Charbonnier-cp + MAE-aux | ~EP2 | — |
| frieren | #1173 | H23 Mean Teacher self-distillation | EP1 mech PASS | — |
| **alphonse** | **#1176** | **H25 ALGP — aux local-gradient pred** | **just assigned** | **— (backbone repr attack)** |

Previous update (14:50Z May 17) — (1) **askeladd H11b CROSSES BASELINE** EP5.8 val_abupt 6.076% (−0.05pp BELOW), slope dead → terminal 5.96-6.00% projection (decisive blocker: val_SP +0.496pp test_SP floor breach); (2) **fern H10b val_vol_p FLOOR PASS** EP6.25 val_vol_p 3.584% < 3.643% floor (FIRST val_vol_p floor pass in active fleet, stable below since EP4.5, mechanism-driven), but val_abupt plateaued at 6.217% (above baseline); (3) **alphonse H20 budget-bug CONFIRMED** SENPAI_TIMEOUT_MINUTES=360.0 → student plans EP3-kill at ~15:15Z + offline test-eval (would-be 4th truncated mechanism-PASS / baseline-FAIL); (4) **frieren H23 EP1 mechanism PASS** consistency_loss median 0.0649, student_minus_teacher mean_abs 0.115 (in advisor's (0.01, 0.5) window), `/proc/<pid>/environ` confirms SENPAI_TIMEOUT_MINUTES=1100 — frieren's pod has the override that bypasses the deployment bug; (5) **tanjiro H18 band-break FADING** τz/τx 1.412 EP3 → 1.467 EP4.8, alphonse H20 now sole sub-1.42 band-break holder. Fleet 8/8 active.

### Actions this invocation (14:50Z May 17)

- **POSTED check-in PR #1164 fern H10b**: 5th stale_wip false positive. Documented val_vol_p 3.584% < 3.643% floor (FIRST in fleet) — stable below floor for 2 epochs. val_abupt PLATEAUED at 6.217% (slope −0.002%/1k, dead). Terminal projection NOT-A-MERGE but **publishable test_vol_p floor pass** — recommended continue to EP13 as stackable signal for H22 floor-preservation companion.
- **POSTED check-in PR #1167 askeladd H11b**: 3rd stale_wip false positive. EP5.8 val_abupt 6.076% (−0.05pp BELOW baseline 6.126%) — **FIRST baseline crosser in Wave 30 active fleet**. Slope flat (−0.019pp per checkpoint). val_SP +0.496pp above floor → projected test_SP breach decisive merge blocker. Continue to EP13 for stackable diagnostic data.
- **POSTED check-in PR #1163 tanjiro H18 v2**: 4th stale_wip false positive. EP4.8 val_abupt 6.810%. **Band-break FADING**: τz/τx 1.412 (EP3 fleet-deepest) → 1.467 (EP4.8 converging back toward [1.44,1.55] band). Mechanism-induced band-breaks fade against upstream representation pressure. val_vol_p 3.711% lowest in active fleet (post-fern).
- **POSTED check-in PR #1151 edward H12**: 9th stale_wip false positive. EP5.7 val_abupt 6.350% — slope severely flattened (Δ −0.007%/1k). Terminal projection 6.10-6.18% (narrowest baseline beat possible). val_SP floor breach projected.
- **POSTED check-in PR #1170 alphonse H20**: launch confirm + REVISED EP3 KILL gate from val_abupt > 6.70% to band-break-weighted (τz/τx ≤ 1.42 AND val_abupt ≤ 9.0% → CONTINUE). Student CONFIRMED pod has SENPAI_TIMEOUT_MINUTES=360.0 bug — plans EP3 kill at ~15:15Z + offline test-eval before SIGTERM. Will preserve fleet-deepest persistent band-break (1.401 EP2).
- **NEW FINDING**: frieren H23 is the FIRST run in this session to confirm SENPAI_TIMEOUT_MINUTES=1100 made it through to torchrun child process — the command-line override DOES work. Possibly an intermittent pod-deployment behavior. Bug affects only some pods (thorfinn/frieren-earlier/alphonse=BAD; fern/edward/askeladd/frieren-now=GOOD).

### KEY FLEET DIAGNOSTIC: Baseline crossed but floor-breach pattern dominant

| Student | PR | EP | val_abupt | vs baseline | Floor risk |
|:--|---:|:--|---:|---:|:--|
| **askeladd** | **#1167** | **5.8** | **6.076%** | **−0.050pp BEAT** | val_SP +0.496pp BREACH |
| fern | #1164 | 6.25 | 6.217% | +0.091pp | val_vol_p **CLEARED**, val_SP +0.500pp BREACH |
| edward | #1151 | 5.7 | 6.350% | +0.224pp | val_SP +0.566pp BREACH |
| tanjiro | #1163 | 4.8 | 6.810% | +0.684pp | val_SP +0.860pp BREACH |
| nezuko | #1171 | 3.5 | 6.755% | +0.629pp | recovering |
| alphonse | #1170 | 2.3 | 10.67% | high cold-start | **band-break τz/τx 1.401** (sole) |
| thorfinn | #1172 | 1.3 | — | new | floor-preservation target |
| frieren | #1173 | 1.0 | — | new | bold lane (training dynamics) |

**3 of 8 in-flight runs project val_abupt baseline-beat by terminal** (askeladd confirmed, edward marginal, fern unlikely). ALL 3 project test_SP floor breach. **Floor-preservation is the merge-decisive blocker — H22 thorfinn (Charbonnier-cp) is the natural stacking partner.**

### KEY FLEET DIAGNOSTIC: Band-break source localization confirms upstream representation pressure

| Student | Layer attacked | EP | τz/τx | Outcome |
|:--|:--|:--|---:|:--|
| **alphonse** | per-vertex loss reweighting (focal) | EP2 | **1.401** | persistent break (only one) |
| tanjiro | per-vertex loss reweighting (area-weighted) | EP3 | 1.412 (faded to 1.467 EP4.8) | break FAILED to hold |
| askeladd | input gating (channel scaling) | EP5.8 | 1.553 | INSIDE band |
| fern | output head (bounded-exp magnitude) | EP6.25 | 1.530 | INSIDE band |
| edward | per-vertex magnitude weighting (α=0.3) | EP5.7 | 1.552 | INSIDE band |
| nezuko | per-component output topology | EP3.5 | 1.529 | INSIDE band |

Output/input/topology attacks ALL collapse INTO band by EP3-5. Only loss-side mechanisms attempt break — only focal-weighted MSE (alphonse H20) holds it persistent. **Band attractor is upstream of output heads**. For Wave 31: target encoder-layer representations directly, not output topology.

## Previous invocation (12:05Z May 17)

### Actions this invocation (12:05Z May 17)

- **CLOSED PR #1169 frieren H16b** as TERMINAL NOT-A-MERGE / Huber static-δ direction EXHAUSTED. Student posted `terminal=false, status=truncated_partial` SENPAI-RESULT with direct `/proc/<pid>/environ` verification that SENPAI_TIMEOUT_MINUTES=360 overrode the PR-specified 1100. Result at EP3.79: test_abupt 6.767% (+0.923pp), test_SP 4.547% (+0.970pp FLOOR BREACH), test_vol_p 4.635% (+0.992pp FLOOR BREACH). **Mechanism FADES structurally**: frac_in_L1 = 44%→27%→3.4%→1.1% over EP1→EP3.79 (40× decay). Combined with H16 (δ=1.0, frac_in_L1 = 0.014%, MSE-equivalent), the static-δ Huber direction is exhausted — a single calibration cannot track shrinking residual distribution. Dynamic-δ rejected in favor of bolder direction.
- **ASSIGNED frieren H23 (PR #1173)** — Mean Teacher self-distillation. Uses existing EMA model (`--use-ema --ema-decay 0.999`) as teacher; student forward on Gaussian-noise-augmented surface points (σ=0.01); consistency loss MSE(student_aug, teacher_unaug.detach()) with warmup λ → 0.1 over ~1 epoch. Tarvainen & Valpola 2017. Train.py-only ~50 lines. **Plateau Protocol bold-direction pick** — fundamentally different from MSE/Huber/quantile direction. Targets training dynamics, not output/loss/topology. Orthogonal to all 8 in-flight attacks.
- **POSTED #1056 budget-bug update**: 4 closures now budget-starved (H15b, H17 different cause, H19, H16b). frieren's `/proc/<pid>/environ` evidence is the cleanest proof. Posted runtime audit: GOOD pods (fern/edward/askeladd >7h+) vs BAD pods (thorfinn/frieren 271min cap). Requested human-team pod-manifest audit.
- **POSTED CHECK-IN PR #1167 askeladd H11b**: 2nd stale_wip false positive. EP6.5 val_abupt = **6.119%** — fleet #1, just below baseline 6.126%. floor breach risk severe (val_SP 4.099% +0.522pp, val_vol_p 3.826% +0.183pp).

### KEY FLEET DIAGNOSTIC: Huber-direction is closed, Mean Teacher opens consistency-regularization lane

Wave 30 has now exhausted the loss-form-on-τ direction:

| Direction | Attacks | Outcome |
|:--|:--|:--|
| Static Huber on τ | H16 (δ=1.0), H16b (δ=0.3) | EXHAUSTED — 2/2 fail, structurally limited |
| Magnitude-weighted MSE | H12 (edward), H13c (thorfinn closed) | in-flight (H12), 1 closed |
| Per-vertex focal | H20 (alphonse) | new launch |
| Charbonnier on cp | H22 (thorfinn) | new launch (floor preservation) |
| Area-weighted MSE | H18 (tanjiro v2, band-break) | in-flight |

The **consistency-regularization** lane (H23 Mean Teacher) is opened as the first attack on training dynamics rather than loss-form/output-head/topology. If H23 PR #1173 shows mechanism evidence (consistency loss → 0, student_minus_teacher_mean_abs decreasing), it validates a fresh lane for future Wave 31 attacks.

### Floor-breach mode still dominant

Test floors continue to be breached:
- H16b EP3: test_SP +0.97pp, test_vol_p +0.99pp (decisive NO-MERGE blocker)
- askeladd H11b val_SP at EP6.5: +0.522pp above floor (projected test_SP breach)
- fern H10b val_SP at EP7: +0.526pp above floor (projected test_SP breach)

H22 thorfinn (Charbonnier-cp + MAE-aux) is the dedicated floor-preservation attack. **It is the natural stacking partner** for both fern H10b and askeladd H11b winners if they clear val_abupt but breach floors.

### Current fleet (8/8 active)

| Student | PR | H | Status | Latest val_abupt |
|:--|---:|:--|:--|---:|
| askeladd | #1167 | H11b gated multi-scale input | EP6.5 healthy | **6.119% (#1)** |
| fern | #1164 | H10b bounded-exp magnitude | EP7.0 healthy | 6.245% (#2) |
| edward | #1151 | H12 τ-magnitude weighted MSE | ~EP6 approaching | 6.574% |
| nezuko | #1171 | H21 per-component output heads | EP2.8 cold-start recovery | 7.98% |
| tanjiro | #1163 | H18 area-weighted surface MSE v2 | EP3 band-break τz/τx=1.412 | 7.787% |
| alphonse | #1170 | H20 focal vertex loss | new launch | — |
| thorfinn | #1172 | H22 Charbonnier-cp + MAE-aux | new launch | — |
| **frieren** | **#1173** | **H23 Mean Teacher self-distillation** | **just assigned** | **— (bold direction)** |

Previous update (10:35Z May 17)

## Latest invocation actions (2026-05-17 ~10:35Z) — H19 thorfinn CLOSED NOT-A-MERGE / mechanism PARKED AS STACKABLE (budget-starved 3/13ep, SENPAI_TIMEOUT_MINUTES=360 bug); fleet leaders identified — fern H10b CO-LEADING at val_abupt 6.263% (EP4.5), askeladd H11b CO-LEADING at 6.194% (EP7); tanjiro H18 v2 DEEPEST τz/τx band-break in fleet history (EP3=1.412); H22 cp/SP floor-preservation lane being assigned to thorfinn (orthogonal to all 7 in-flight magnitude attacks)

### Actions this invocation (10:35Z May 17)

- **CLOSED PR #1168 thorfinn H19** as TERMINAL NOT-A-MERGE / mechanism PARKED AS STACKABLE. Mechanism PASS — VICReg fired 52% of 30,448 steps (no quiescence), std_τ_z lifted 0→0.110 (2.2× γ), batch mean |τ_z| reached GT scale 0.078. Probable band-break by MAE-ratio proxy (test τ_z MAE 0.0567 < GT |τ_z| 0.0793). But baseline FAIL all 4 axes + both floors at EP3-only training: val_abupt 6.998% (+0.872pp), test_abupt 6.670% (+0.826pp), test_SP 4.274% (+0.697pp floor), test_vol_p 3.934% (+0.291pp floor), test_WSS 7.671% (+0.944pp). Budget-starved 3/13 epochs by SENPAI_TIMEOUT_MINUTES=360 bug. 3rd "mechanism PASS, baseline FAIL, parked stackable" pattern after H15b alphonse + H17 nezuko.
- **FLAGGED fleet-level timeout bug** on PR #1163 (tanjiro H18 v2): if his pod has same 360min cap, his run hits timeout at 11:35Z mid-EP4. Asked tanjiro to verify `kubectl exec printenv SENPAI_TIMEOUT_MINUTES`. fern H10b is past 360min (currently 9.41h) so the bug is NOT fleet-wide — appears student-pod-specific.
- **HEAD-TO-HEAD LEADERS IDENTIFIED**: fern H10b at EP4.5 val_abupt 6.263% (+0.137pp above baseline, slope projects EP13 ~5.46% → 0.66pp BEAT) and askeladd H11b at EP7 val_abupt 6.194% (+0.068pp above baseline, EP4→EP7 slope −0.295pp/ep) — both projecting baseline beat at terminal but **both showing floor-breach risk** (val_SP 4.12% and 4.16% respectively vs test floor 3.577%; val_vol_p 3.60% and 3.88% vs test floor 3.643%).
- **TANJIRO H18 v2 BAND-BREAK at EP3**: val τz/τx = 1.412 — deepest band-break in fleet history (vs H6' 1.420 prior best). But val_abupt 7.787% (+1.66pp above baseline) and val_SP/vol_p severely above floors. Overriding EP3 KILL gate (was > 7.4% threshold) to continue to EP6 — band-break too research-valuable to lose. Will re-assess at EP6.
- **H22 hypothesis being assigned to thorfinn** — cp/surface_pressure floor-preservation lane. Designed to address the observed floor-breach failure mode in dl24 H10b, fern H10b, askeladd H11b, edward H12, and H19. Charbonnier loss on cp channel + auxiliary MAE on cp. Orthogonal to all 7 in-flight magnitude attacks.

### KEY FLEET DIAGNOSTIC: floor-breach is the NEW bottleneck

dl24 H10b LOCKED as SOTA-beat winner at 09:25Z with test_wss=6.68 ✅ BUT test_vol_p=4.17 (+0.53pp floor breach) and test_SP=3.86 (+0.28pp floor breach). My tay fleet leaders fern H10b and askeladd H11b are projecting the SAME floor-breach pattern at terminal — strong val_abupt but cp/SP/vol_p underweighted.

**The next merge gate isn't val_abupt or test_WSS — it's whether floors hold.** All 7 in-flight attacks target the magnitude bottleneck (τ_z), none directly protect cp/SP. H22 fills this gap.

Previous update (06:15Z May 17)

### Latest invocation actions (2026-05-17 ~06:15Z) — alphonse H15b CLOSED NOT-A-MERGE (mechanism PASS / baseline FAIL, parked); nezuko H17 CLOSED TERMINAL-NULL (KILLED EP3, hard-constraint cold-start gap ~11pp); 2 students idle (alphonse, nezuko) → researcher-agent dispatching for H20/H21 fresh axes

### Actions this invocation (06:15Z May 17)

- **CLOSED PR #1165 alphonse H15b** as NOT-A-MERGE. Mechanism CONFIRMED (EMA AHEAD of raw by +0.80pp at EP3 — clean inverse of H15's −9.54pp gap, EMA crossed at EP1). But absolute metrics regress on ALL 4 axes + both floors breach: val_ema 6.838% / test 6.636% (+0.792pp), test_WSS 7.636% (+0.909pp), test_SP 4.268% (FAIL +0.691pp floor), test_vol_p 3.924% (FAIL +0.281pp floor). Recipe budget-starved (3/13 epochs trained, cosine LR barely 6% along). **NEW DATAPOINT**: test τz/τx = 1.439 from EMA weight averaging — 2nd-closest band-edge break after H6' 1.420. Implies EMA implicitly decorrelates per-vertex τ predictions. H15 series PARKED as stackable mechanism for future winners.
- **CLOSED PR #1162 nezuko H17** as TERMINAL-NULL. Hard τ·n=0 by-construction tangent-frame implementation VERIFIED CORRECT (orthogonality residuals 2.51e-9 / 1.91e-8, max|τ·n| ≤ 2.4e-7 in fp32 — machine-zero). But trajectory PARALLEL to baseline (gap +12.83→+11.13→+11.34pp across EP1→EP3) — KILL at EP3 per PR's own gate. Cold-start representational gap: data-dependent output basis + reduced DOF in 2-channel head can't recover within 13ep lr=9e-5 budget. Soft-penalty (H6') is better experimental match in this budget; hard constraint requires longer warmup or transfer learning.
- **DISPATCHED researcher-agent (background)** for 2 fresh fleet-orthogonal hypotheses: H20 for alphonse (optimization-layer angle, different from EMA series), H21 for nezuko (representation-layer angle, different from tangent-frame). Output to `RESEARCH_IDEAS_2026-05-17_06:00.md`.

### KEY FLEET DIAGNOSTIC: 2nd band-edge break from optimization-layer (not architecture)

Wave 30 now has **two distinct band-break signals** at test τz/τx:

| Closure | Layer | Mechanism | test τz/τx | Verdict |
|---|---|---|---:|:--|
| H6' #1147 (closed) | output-loss layer | Soft τ·n=0 penalty | **1.420** (FIRST break) | NOT-A-MERGE SP floor |
| H15b #1165 (closed) | optimization layer | EMA weight averaging | **1.439** (2nd closest) | NOT-A-MERGE all axes |

**Research-direction implication**: Band-breaking is achievable from MULTIPLE layers, not just geometric-constraint or per-vertex-loss. EMA's implicit ensemble effect on per-vertex predictions hints that **prediction variance reduction across training trajectories has structural-decoupling content** — a fresh lens not yet exploited.

Previous update (06:00Z May 17)

### Latest invocation actions (2026-05-17 ~06:00Z) — frieren H16 CLOSED TERMINAL-NULL (δ=1.0 dormant, mechanism calibration failed, watchdog false-positive kill; reassigned to H16b δ=0.3 PR #1169); fern H10b EP3 PASS at 6.697% (CLOSEST to baseline of all in-flight attacks); frieren newly idle → H16b assigned

### Actions this invocation (06:00Z May 17)

- **CLOSED PR #1161 frieren H16** as TERMINAL-NULL. Pod killed by student-watchdog (false-positive label drift) at EP4 step 37k. EP3 val_abupt=6.894% (marginal-pass). Mechanism diagnostic: `frac_in_L1_region/tau_z = 0.014-0.018%` (expected 2.5%) — δ=1.0 dormant because heavy tail collapsed EP1→EP3 (max_abs 9.8→1.8σ). No test data. H16 was effectively MSE-on-τ. KEY LEARNING: Huber-on-τ requires post-EP1-distribution δ calibration.
- **ASSIGNED frieren H16b (PR #1169)**: δ=0.3 restart (one-flag change). EP1 mechanism gate: `frac_in_L1_region/tau_z ∈ (0.4, 0.8)` — if achieved, Huber IS active on the bulk residual distribution.
- **VERIFIED fern H10b (PR #1164)**: EP3 PASS at **val_abupt 6.697%** — currently CLOSEST to baseline 6.126% of all in-flight attacks. Val trajectory EP1→EP2→EP3: 31.80→7.42→6.70%. slope −0.072pp/1k steps. No magnitude-head diagnostic keys logged — asked student to add at EP6.

### KEY FLEET DIAGNOSTIC UPDATED: fern H10b leads the fleet at EP3

H10b at val_abupt 6.697% (EP3) is the closest any single-model in-flight Wave 30 attack has been to baseline 6.126% at this stage of training. If the EP2→EP3 improvement slope holds, EP6 projection ~6.0% which would beat baseline. **H10b is the top-priority watch-item this wave.**

Previous update (04:50Z May 17)

### Latest invocation actions (2026-05-17 ~04:50Z) — thorfinn H13c CRASHED CLOSED DEAD-END (val_abupt 8.608% at EP5–6 / +2.482pp above baseline; 2nd direction-saturation diagnostic confirms magnitude is the bottleneck; researcher-agent generating thorfinn's next hypothesis)

### Actions this invocation (04:50Z May 17)

- **REVIEWED + CLOSED PR #1158 (thorfinn H13c Lagemann cos+mag decoupling)** as DEAD-END. W&B verified: rank0 `1udny3hl` crashed at ~04:16Z (8.2h / 18h budget) with nonfinite=0 — OOM or orchestrator-kill. Latest val_abupt=8.608% at EP5–6 (+2.482pp WORSE than baseline). cos_sim=0.990 confirms direction saturation. Per-epoch slope flattening (50.8→16.1→10.8→9.56→8.61, geometric decay ~0.85×/ep → EP13 projection ~6.0% even with recovery). Floors already breached at EP5–6 (val_SP +0.575pp / val_vol_p +0.357pp).
- **POSTED CHECK-INS on stale_wip false positives:**
  - PR #1161 (frieren H16) — main run healthy at EP3.05/13, val_abupt=6.894% MARGINAL-PASS. **CRITICAL: `huber/frac_in_L1_region/tau_z = 0.014%` — 180× lower than expected ~2.5%. The tail-clipping mechanism is essentially inactive; H16 is silently behaving as MSE-on-τ. EP6 decisive on whether to re-run at δ=0.3.**
  - PR #1158 (thorfinn H13c) — closed instead.
  - PRs #1164 (fern H10b) and #1163 (tanjiro H18) — both healthy at EP~1.6–1.7, posted in prior invocation.

### KEY FLEET DIAGNOSTIC UPDATED: 2nd direction-saturation confirmation

Wave 30 now has **two independent** experimental confirmations that direction is essentially solved and magnitude is the bottleneck:

| Closure | Layer | cos_sim | mag/dir error split |
|---|---|---:|---|
| H10 #1148 (fern) | output model arch (vector-decoupled head) | 99.65% | **73% / 27%** explicit |
| H13c #1158 (thorfinn) | loss formulation (Lagemann cos+mag) | 99.0% | direction saturated (consistent) |

**Research-direction implication**: any future loss-level cosine-aware or direction-only attack is now ruled out. The bottleneck has moved one layer down to the magnitude regression head and/or the τ_z over-prediction collapse band. H10b (in flight) attacks magnitude; H17 (in flight) attacks the band geometrically.

### Wave 30 fleet — 7 active + 1 idle (thorfinn — H22 assignment pending); Wave 30 closed count: 22

| PR | Student | Axis | Status |
|---|---|---|---|
| #1171 | nezuko | H21 per-component independent output heads (4 separate MLPs for cp/τx/τy/τz; τz head deeper) | EP~2.2 (running, no val yet) |
| #1170 | alphonse | H20 focal vertex loss (dynamic per-vertex error-weighted MSE on τ channels, γ=0.5, γ_z=1.5×γ_x) | EP~0.25 (just launched 10:25Z) |
| #1169 | frieren | H16b Huber loss on τ channels δ=0.3 (calibrated bulk-reshape — restart from H16) | EP~2.3 val 7.839% (+1.71pp above baseline — EP3 gate at 11:45Z critical) |
| #1168 | ~~thorfinn~~ | ~~H19 VICReg batch-variance~~ — **CLOSED 10:35Z NOT-A-MERGE / PARKED STACKABLE** | budget-starved 3/13ep, mechanism PASS / baseline FAIL |
| #1167 | **askeladd** | **H11b gated multi-scale input (zero-init gate)** | **EP~7 val 6.194% (+0.068pp above baseline) — CO-LEADING; floor risk HIGH** |
| #1164 | **fern** | **H10b bounded-exp magnitude fix (softplus→clamp.exp)** | **EP~4.5 val 6.263% (+0.137pp above baseline) — CO-LEADING; floor risk HIGH** |
| #1163 | tanjiro | H18 per-vertex area-weighted surface MSE (v2 relaunch after watchdog SIGTERM v1) | EP~3 val 7.787% (+1.66pp above baseline) BUT val **τz/τx=1.412 DEEPEST BAND-BREAK** in fleet history ✦ |
| #1151 | edward | H12 τ-magnitude-weighted loss (sweep α=0.5 done, α=0.3 running) | a0p5 arm: NOT-A-MERGE (val 6.326%, floor breach); a0p3 EP~5.5 val 6.574% — likely NOT-A-MERGE (floor) |
| (pending) | thorfinn | **H22 cp/SP floor-preservation lane** (Charbonnier-cp + MAE_aux on cp) | **TO BE ASSIGNED 10:40Z** — orthogonal to all 7 in-flight magnitude attacks |

**8 simultaneous attacks on the magnitude bottleneck, attacking from 8 different causal angles**:
- Output head ARCHITECTURE: H10b (activation function), H21 (decoder topology)
- Loss formulation per-vertex: H12 (static |τ_gt| weights), H18 (area weights), H20 (dynamic |error| weights), H16b (δ=0.3 outlier clipping)
- Input feature gating: H11b (multi-scale)
- Batch statistics: H19 (VICReg variance hinge)

This is the best-controlled multi-pronged diagnostic pass we have run on a single bottleneck. Even partial wins from any angle will deeply update the causal map.

**Closed in Wave 30** (21): H1 #1139, H2 #1136, H3 #1138, H4 #1141, H5 #1137, H6 #1134 (mech-PASS), H7 #1140, H14 #1153 (diverged), H13 β=5 #1152 (diverged), H13b β=2 #1156 (diverged), H8 #1143 (FLAT NULL), H9' #1146 (NOT-A-MERGE SP floor), **H6' #1147 (NOT-A-MERGE SP floor; τz/τx=1.420 band-break signal)**, **H10 #1148 (NOT-A-MERGE; 73%/27% mag/dir diagnostic)**, **H15 #1155 (TIMEOUT-NULL)**, H15v1 #1122, **H11 #1150 (NOT-A-MERGE SP+vol_p floors; BEST single-model test_WSS 6.633% and test_abupt 5.809% on tay)**, **H13c #1158 (DEAD-END CRASHED; 2nd direction-saturation diagnostic)**, **H16 #1161 (TERMINAL-NULL δ=1.0 dormant; watchdog false-positive kill at EP4; KEY LEARNING: Huber-on-τ requires post-EP1 δ calibration)**, **H17 #1162 (TERMINAL-NULL KILLED EP3; hard-τ·n=0-by-construction implementation verified correct but cold-start gap ~11pp vs baseline; soft-penalty H6' is better budget-match)**, **H15b #1165 (NOT-A-MERGE; mechanism PASS — EMA AHEAD of raw +0.80pp at EP3, test τz/τx=1.439 band-edge — but baseline FAIL all axes + both floors; budget-starved 3/13 epochs)**.

### Causal map of τ_z bottleneck — updated after H10 and H6' diagnostics

Two orthogonal diagnostic signals define the research frontier:
1. **H6' (PR #1147)**: τ·n=0 constraint CAN reduce τ_z/τ_x (test=1.420, below band lower edge 1.44). Soft-penalty formulation failed on primary metrics but mechanism is real → H17 hard-constraint is the right test.
2. **H10 (PR #1148)**: 73% of WSS squared error is in MAGNITUDE, 27% in direction. Direction head saturated (99.65% cos-sim). Softplus floor prevents near-zero magnitude predictions → H10b bounded-exp is the targeted fix.

These two diagnostics are **orthogonal attacks on the same τ_z problem**: one fixes the geometric constraint (τ·n=0), the other fixes the magnitude-prediction regime near zero. The ideal model would pass both H17 and H10b, then stack.

| Layer | In-flight probes | Closed |
|---|---|---|
| **Architecture / input-feature** | **1 (H11b gated kNN — fix from H11)** | **8 closures** (H11 not broken, just floor regression from unscaled gate) |
| **Loss formulation (channel/vertex reweighting)** | 3 (H12 mag, H13c cos+mag, H18 area-weighted) | 4 (H13 β=5, H13b β=2, H6' soft penalty, H10 indirect) |
| **Loss formulation (outlier robustness)** | 1 (H16 Huber δ=1.0) | 0 |
| **Output reparameterization (hard constraints)** | 1 (H17 tangent-frame, strong prior from H6') | 0 |
| **Output projection (head architecture)** | 1 (H10b bounded-exp — top priority after 73%/27% diagnostic) | 1 (H10 softplus form failed) |
| **Optimization** | 1 (H15b EMA decay=0.999) | 2 (H14 lr confound, H15 TIMEOUT-NULL) |

### Primary research question now: can we unblock the 73% magnitude bottleneck?

**H10b is the single highest-priority test.** If `log_mag.clamp(min=-3, max=3).exp()` unblocks the near-zero magnitude regime (floor drops from 6.93 to 0.05 in normalized space), we expect:
- test_WSS to drop meaningfully (targeting < 6.600%)
- magnitude error fraction to fall below 50%
- τ_z/τ_x may respond if over-prediction is a magnitude artifact (τ_z is the "small" tangential component — consistently over-predicted in magnitude)

**H17 is the second highest-priority test** (nezuko in-flight). If hard tangent-frame output enforces τ·n=0 by construction and test τ_z/τ_x falls below 1.40, we have two orthogonal mechanism wins to stack.

### Next-idle assignment queue

1. **H10b + H17 stacking** — if both yield improvements, stack bounded-exp head in tangent-frame basis
2. **Adaptive vertex sampling on high-τ_z vertices** — alternative to loss reweighting, changes data not loss  
3. **Per-component output heads** — separate MLP per τ component (tests single-head capacity limitation)
4. **Focal MSE on per-vertex error** — error-space amplification focused on worst vertices
5. **Lookahead optimizer** — complementary to EMA, would stack with H15b
6. **Surface-normal-aligned coordinate frame for τ targets** — predict in (t1, t2, n) basis; overlaps with H17 but approaches from training-target angle not output-constraint angle

## ★ CRITICAL FLEET-WIDE FINDING (2026-05-16 ~19:00Z): lr=5e-4 confound on 4 Wave 30 PRs (preserved for reference)

The cascade of 4 Wave 30 divergences (#1153 H14, #1152 H13 β=5, #1156 H13b β=2, #1155 H15) all stem from a single advisor error: writing `--lr 5e-4` instead of canonical `--lr 9e-5` on the PR commands. Conflation of SOTA single-model PR #972 (`--lr 1e-4`) with the Wave 30 Lion reference. The canonical Wave 30 recipe is BASELINE.md "L=5 + surf→vol xattn ... Lion **lr=9e-5**, 13ep" and most-recent closed-clean Wave 30 PR (#1138 H3) used exactly that.

**Survey of finished runs with `lr=5e-4 lion ep=13 bs=4`: NONE** (H14 crashed, H15 diverging). This LR has never reached a stable trajectory under this config.

**Fleet-wide impact limited to those 4 PRs.** Verified all other in-flight Wave 30 PRs use lr=9e-5:
- #1143 frieren H8 ✓ (running cleanly at val_abupt 6.340%)
- #1146 nezuko H9' ✓
- #1147 tanjiro H6' ✓
- #1148 fern H10 ✓
- #1150 askeladd H11 ✓
- #1151 edward H12 ✓

**Process correction**: future loss-layer or optimization-layer probes MUST use `--lr 9e-5` (the canonical Lion reference) unless a sweep is explicitly testing LR.

## Latest invocation actions (2026-05-16 ~20:30Z) — frieren #1143 H8 mirror-aug TERMINAL FLAT NULL CLOSED (first data-distribution-layer attack closed — test τz/τx = 1.456 lands EXACTLY in collapse band; cleanly falsifies "data diversity" hypothesis; data-distribution layer EXHAUSTED on τ_z); frieren reassigned to H16 per-channel z-score normalization (PR #1161, output-side per-channel calibration probe); fern #1148 H10 vector-decouple PASSING EP10 GATE TWO EPOCHS EARLY at val_abupt=6.281%, direction_cos_loss=0.00355 (head saturated at 99.65% cos-sim, all residual error in magnitude) — most promising in-flight signal so far; askeladd #1150 EP8.3 healthy at 6.215% (false-positive stale_wip resolved)

### Actions this invocation (continued — Wave 30 ~20:30Z)

- **CLOSED PR #1143 (frieren H8 mirror-symmetry data aug)** as FLAT NULL — first dedicated data-distribution-layer attack on τ_z bottleneck; cleanly falsifies "data diversity" hypothesis. Terminal test τz/τx = 1.456 lands EXACTLY in [1.44, 1.47] collapse band of 6 closed model-side attacks. val_WSS 7.188% / test_WSS 7.001% (+0.274pp vs baseline 6.727%, FAIL merge gate); test_SP 3.822% (+0.245pp over floor 3.577%, FAIL floor); test_vol_p 3.578% PASS floor (cosmetic). Student diagnostic exemplary: math identity reasoning, complete implementation verification, per-channel ordering analysis, **explicit recommendation to shift toward output-side per-channel calibration / target normalization**. Data-distribution attack layer EXHAUSTED.
- **ASSIGNED PR #1161 (frieren: Wave 30 H16 Per-channel z-score normalization of surface targets)** — direct follow-up to frieren's H8 closure recommendation. Compute per-channel mean/std on train set offline (Welford's online algorithm), z-score targets in LOSS only (not architecture), denormalize at eval. Predictions remain in raw physical units; metric formulas unchanged. Mechanism: equalizes per-channel loss SCALE before static per-channel weights apply (vs GradNorm which dynamically balances LOSSES per gradient norm). **First output-side per-channel statistical calibration probe** in Wave 30. Includes 2-epoch SMOKE on 2 ranks for sanity (loss_per_channel_normed all O(0.1-1), ratio_normed_to_raw = 1/σ²_c per channel, val_abupt EP1 ≤ 33% warmup floor) before launching the 18h main run.
- **POSTED CHECK-IN on PR #1148 (fern H10 vector-decoupled output head)** — EP8 PASSES EP10 GATE TWO EPOCHS EARLY at val_abupt=6.281%; direction_cos_loss saturated at 99.65% cos-sim with GT → head has fully learned direction; ALL residual error is in MAGNITUDE reconstruction. **Most promising in-flight signal in Wave 30 so far.** Margin to baseline closed to 0.155pp. Watch-items posted: (1) test τz/τx — does H10 break the 1.44-1.47 collapse band? (FIRST mechanism that could), (2) post-hoc direction-vs-magnitude error split on best ckpt, (3) val_vol_p just crossed under floor 3.636% vs 3.643% (tight), (4) val_SP 4.107% may not clear test floor 3.577% (val→test gap ~0.5pp puts projected test_SP ~3.7-3.8%). Merge decision will hinge on test_WSS + test_SP simultaneously.
- **RESOLVED #1150 (askeladd H11 multi-scale kNN) stale_wip false positive** (4th time on this PR). Verified via W&B: state=running, heartbeat 2s, EP 8.3 / 13, val_abupt=6.215% (improved from 6.285% at 17:50Z), best_checkpoint updated. Pace rock-stable at 1.058 h/epoch. Terminal projection ~01:20Z May 17. Posted check-in with EP10/test τz/τx watch-items.

### Wave 30 fleet — 8 active + 0 idle (Wave 30 closed count: 11)

| PR | Student | Axis | Status |
|---|---|---|---|
| #1161 | frieren | H16 per-channel target z-score | **JUST ASSIGNED** (output-side calibration probe — first of its kind in Wave 30) |
| #1158 | thorfinn | H13c Lagemann cos+mag decoupling | Assigned ~17:30Z |
| #1155 | alphonse | H15 EMA / Polyak averaging v2 | EP1+ active at lr=9e-5 (relaunched 19:15Z) |
| #1151 | edward | H12 τ-magnitude-weighted loss | EP 3.23, val_abupt 6.78% (EP3 PASS); pace IMPROVED to 1.057h/ep — terminal ~03:50Z May 17 |
| #1150 | askeladd | H11 multi-scale kNN context | EP 8.3, val_abupt **6.215% IMPROVING**; terminal ~01:20Z May 17 |
| #1148 | fern | H10 vector-decoupled output head | **EP 8 PASSING — val_abupt 6.281%, direction-cos saturated** — most promising signal |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | EP3+ PASS at 6.904%, τz/τx 1.512 not breaking |
| #1146 | nezuko | H9' curvature input feature | EP11.6+, val_abupt 6.250% |

**Closed in Wave 30** (11): H1 #1139, H2 #1136, H3 #1138, H4 #1141, H5 #1137, H6 #1134 (mech-PASS), H7 #1140, H14 #1153 (diverged lr=5e-4), H13 β=5 #1152 (diverged lr=5e-4), H13b β=2 #1156 (diverged lr=5e-4 + formulation broken), **H8 #1143 (FLAT NULL — data-distribution layer EXHAUSTED on τ_z)**.

### Causal map of τ_z bottleneck — DATA-DISTRIBUTION LAYER NOW CLOSED + architecture exhausted + anisotropic-loss axis exhausted

| Layer | In-flight probes | Closed |
|---|---|---|
| **Architecture** | 0 | 6 widening + 1 mechanism-PASS (DEFINITIVELY exhausted) |
| **Loss** | 4 (H6' tan, H12 mag, H13c cos+mag, H15 EMA tangentially) | 2 (H13 β=5, H13b β=2 — anisotropic axis exhausted) |
| **Data-input/distribution** | 2 (H9' curv, H11 multi-scale kNN) | **1 (H8 mirror-aug — distribution-layer FALSIFIED for τ_z)** |
| **Output projection** | 1 (H10 vector-decouple — most promising) | 0 |
| **Output-side calibration** | **1 (H16 z-score NEW)** | 0 |
| **Optimization** | 1 (H15 EMA — relaunching at lr=9e-5) | 1 (H14 5× head LR — diverged lr=5e-4 confound) |

### NEW KEY INSIGHT — fern H10 EP8 readout

`direction_cos_loss = 0.00355` (99.65% cos-sim with GT WSS direction at EP8) means **H10's vector-decoupled head has fully learned the WSS DIRECTION; all residual error is in MAGNITUDE reconstruction**. This is the cleanest mechanistic dissection of the τ_z bottleneck we have so far. **If H10 terminal test τz/τx breaks out of the 1.44-1.47 collapse band, the τ_z bottleneck story rewrites toward "magnitude prediction is the lever, direction is already solved at fern's head architecture".** This will guide the priority of follow-up probes (focal magnitude loss > direction-related probes).

### Loss-layer attack fleet — now 4-strong probe

| PR | Probe | Direction | Mechanism if winning |
|---|---|---|---|
| #1147 (H6') | soft τ_pred·n=0 | suppress model's normal-component | model was over-predicting normal noise |
| #1151 (H12) | (\|τ_target\|/mean)^α weight | upweight high-magnitude vertices | long-tail magnitude under-learned |
| #1158 (H13c) | mag-MSE + λ·cos-loss decoupling | separate magnitude from direction | direction-error contaminates magnitude gradient |
| #1161 (H16) | per-channel target z-score | equalize per-channel loss scale | per-channel statistics imbalance below GradNorm's reach |

### Next-idle assignment queue (in priority order)

1. **Stacking experiments** — once Wave 30 terminals land (H10 fern likely first, ~02:00Z May 17), combine top winner with EMA + orthogonal axes
2. **Per-vertex offline difficulty weighting** — pre-compute baseline model's per-vertex error on train set, weight loss by inverse difficulty (complement to H12 which uses target magnitude not error)
3. **Layer-wise LR multipliers** (different from H14's head-LR) — geometric LR schedule across depth, lower for early layers
4. **AdamW for H14 retry** — clean optimizer-effect separation; AdamW handles wider LR range via second-moment normalization
5. **Capacity-matched H4 retry (z-slice-fraction=0.55)** — clean falsification of remaining slice-routing variant
6. **Focal MSE loss with γ on per-vertex error** — alternative loss attack (error-space amplification vs H12's target-space)
7. **τ_z residual from τ_x, τ_y** — predict τ_z as function of predicted τ_x, τ_y + surface geometry (cross-channel constraint)
8. **Spherical-harmonic WSS basis** — stronger H10 variant if H10 partial-wins
9. **Lookahead optimizer (Zhang et al. 2019)** — complementary to EMA
10. **Per-channel attention head** — separate cross-attention head dedicated to τ_z

## Older invocation actions (2026-05-16 ~19:00Z) — Two more closures: #1156 thorfinn H13b β=2 CLOSED (anisotropic-loss formulation broken at BOTH β=5 and β=2 — not just amplification cliff), #1155 alphonse H15 EMA RELAUNCHED (advisor confirmed lr=5e-4 mistake); thorfinn reassigned to H13c Lagemann cosine+magnitude decoupling (PR #1158) at correct lr=9e-5

### Actions this invocation (continued — Wave 30 ~19:00Z)

- **CLOSED PR #1156 (thorfinn H13b β=2 anisotropic loss)** — DIFFERENT failure mode from H13 β=5: corruption WITHOUT gradient explosion (clipping prevented nonfinites entirely, 0% post-warmup nonfinite_grad vs H13's 100%). But near-identical EP1 val_abupt across β=5 and β=2 (49.43% vs 47.51%) during low-LR warmup at 2.5e-5 — BEFORE any LR×amplification interaction can fire. **The anisotropic-loss FORMULATION ITSELF disrupts learning, not just the amplification factor.** Student diagnostic exemplary: math identity verified pre-launch, model_n/t ≈ 0.066-0.073 tracks GT_n/t ≈ 0.079-0.083 (model IS learning correct geometry), step-skip distribution table cleanly distinguishes corruption-without-grad-explosion from grad-explosion mode. H13 hypothesis "model under-fits GT normal-component" FALSIFIED.
- **SENT BACK PR #1155 (alphonse H15 EMA)** for relaunch at lr=9e-5. Student diagnostic identified the lr=5e-4 confound exactly: PR specified 5.5× the working reference. EMA implementation itself verified correct (skip on nonfinite_grad ✓, store/copy_to/restore mechanic ✓, ema warmup at EP1 raw=25.20% vs ema=81.39% as predicted). Live model is divergent at lr=5e-4 + Lion; EMA cannot smooth a divergent trajectory. Relabeled status:wip. Provided exact relaunch command matching H3 / PR #823 reference + EMA overlay. Answered all 3 student open questions: (1) lr=9e-5 confirmed; (2) train_surface_points=65536 to match H3; (3) ema_decay=0.9999 retained as the H15 hypothesis.
- **ASSIGNED PR #1158 (thorfinn: Wave 30 H13c Lagemann cosine+magnitude decoupling)** at lr=9e-5. Following thorfinn's queued plan post-#1156 closure. Splits τ loss into `mag_loss + λ·cos_loss` (λ=0.1, small weight on cosine direction loss). Opposite design DNA from H13: no per-vertex tangent/normal frame, no amplification, decoupling instead of rotation. Published precedent: Lagemann et al. arxiv 2507.22817 (AAA WSS). Builds on thorfinn's H13 diagnostic infra (`model_n_t_ratio` panel reused). Clean overlay on the known-stable training trajectory (lr=9e-5, vol-curriculum 16k→65k, surf-to-vol xattn).
- **ADVISOR ACKNOWLEDGMENT** to alphonse + thorfinn: explicit apology for the lr=5e-4 mistake on the H13/H13b/H14/H15 PR commands. Their fleet-wide intelligence catches saved further GPU waste.

### Wave 30 fleet — 8 active + 0 idle (Wave 30 closed count: 10)

| PR | Student | Axis | Status |
|---|---|---|---|
| #1143 | frieren | H8 mirror-symmetry data aug | EP13 in test eval (val_abupt 6.340%, near-terminal SENPAI-RESULT pending) |
| #1146 | nezuko | H9' curvature input feature | EP11.6+, val_abupt 6.250% (improving, EP10 PASS band) |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | EP3+ PASS at 6.904%, τz/τx 1.512 not breaking yet |
| #1148 | fern | H10 vector-decoupled output | EP1-EP2 active training |
| #1150 | askeladd | H11 multi-scale kNN context | EP2.5+, val_abupt 6.864% improving (EP3 PASS band) |
| #1151 | edward | H12 τ-magnitude-weighted loss | EP2.34, val_abupt 8.23% improving, pace 1.59h/ep budget-risk |
| #1155 | alphonse | H15 EMA / Polyak averaging | **RELAUNCHING at lr=9e-5** (just sent back) |
| #1158 | thorfinn | H13c Lagemann cos+mag decoupling | JUST ASSIGNED (replaces #1156) |

**Closed in Wave 30** (10): H1 #1139, H2 #1136, H3 #1138, H4 #1141, H5 #1137, H6 #1134 (mech-PASS), H7 #1140, H14 #1153 (diverged lr=5e-4), H13 β=5 #1152 (diverged lr=5e-4), **H13b β=2 #1156 (diverged lr=5e-4 + formulation broken)**.

### Causal map of τ_z bottleneck — architecture surface ABSOLUTELY EXHAUSTED (7 closures); amplification axis on normal-direction exhausted (3 closures: H6', H13 β=5, H13 β=2)

| Layer | In-flight probes | Closed |
|---|---|---|
| **Architecture** | 0 | 6 widening + 1 mechanism-PASS (DEFINITIVELY exhausted) |
| **Loss** | 4 (H6' tan, H12 mag, **H13c cos+mag NEW**, EMA tangentially) | 2 (H13 β=5, H13b β=2 — anisotropic axis exhausted) |
| **Data-input** | 3 (H8 mirror, H9' curv, H11 multi-scale) | 0 |
| **Output projection** | 1 (H10 vector decouple) | 0 |
| **Optimization** | 1 (H15 EMA smoothing — relaunching at lr=9e-5) | 1 (H14 5× head LR DIVERGED, lr=5e-4 confound) |

### Loss-layer attack fleet — now 4-strong probe of τ_z bottleneck direction

| PR | Probe | Direction | Mechanism if winning |
|---|---|---|---|
| #1147 (H6') | soft τ_pred·n=0 | suppress model's normal-component | model was over-predicting normal noise |
| #1151 (H12) | (\|τ_target\|/mean)^α weight | upweight high-magnitude vertices | long-tail magnitude under-learned |
| #1158 (H13c) | mag-MSE + λ·cos-loss decoupling | separate magnitude from direction | direction-error contaminates magnitude gradient |
| (closed H13/H13b) | β·MSE(τ_n_err) per-vertex anisotropy | upweight matching GT normal-component | (FALSIFIED — formulation breaks at all β) |

**Loss-layer attack now wedge of THREE published-precedented techniques** plus the queued H6' diagnostic. The cosine+magnitude split (H13c) is fundamentally different from H6' (suppress) and H13 (amplify) — it decouples rather than re-weighting.

## Older invocation actions (2026-05-16 ~17:45Z) — thorfinn #1152 H13 β=5 CLOSED (catastrophic warmup-boundary divergence; mirror image of H14; math identity PASS; GT τ_n/τ_t = 0.08 measured = fleet-wide intelligence); thorfinn reassigned to H13b β=2 (PR #1156)

### Actions this invocation

- **CLOSED PR #1152 (thorfinn H13 anisotropic β=5)** — catastrophic divergence at step 3793 (~168 steps after warmup boundary, LR jumping 2.5e-5 → 5e-4). Math identity verified PRE-launch (abs diff 0.00 at α=β=1.0). Pre-divergence trajectory: model n/t = 0.061 closely tracked GT n/t = 0.080 → mechanism IS engaging correctly during warmup. Root cause: per-vertex grad on normal component is 5× larger at β=5; LR×β interaction explodes at peak LR. From step 5000 onward, 100% of optimizer steps skipped via nonfinite_grad guard. Student diagnostic exemplary: math verification + step-level forensics + GT n/t = 0.08 measurement + 4 well-reasoned follow-up suggestions.
- **MIRROR-IMAGE OF H14 DIVERGENCE CONFIRMED**: both Wave 30 amplification-style attacks (#1153 H14 head_lr 5× and #1152 H13 β=5) crashed at the SAME EP1→EP2 LR jump. Fleet-wide intelligence: the standing 1-epoch warmup + lr=5e-4 recipe has very little safety margin for amplification-style attacks. Any future "amplify direction X by N×" probe needs N ≤ 2-3 OR extended warmup OR lower base LR.
- **CRITICAL DATA POINT for entire fleet**: GT τ_normal_to_tangent magnitude ratio = **0.08** — the normal component IS real signal (8% of tangent magnitude, not noise), but small enough that 5× amplification was clearly past the LR×β stability cliff. This informs both H6' (suppressing normal may remove real signal) and H13b's β-sweep direction.
- **ASSIGNED PR #1156 (thorfinn: Wave 30 H13b anisotropic loss at β=2)** — PR-queued follow-up. 2.5× reduction from divergent β=5; squarely in the stable band per GT n/t = 0.08 measurement. Same infrastructure (anisotropic loss + diagnostics already implemented in H13), 1-line config change. ~5h total run-time (thorfinn's throughput is ~23min/epoch, fastest in fleet). Clean falsification triangle: (a) trains+beats baseline → H13 PASS at softer weight; (b) trains+no improvement → H13 premise wrong; (c) also diverges → pivot to cosine+magnitude decoupling (Lagemann arxiv 2507.22817).

### Wave 30 fleet — 8 active + 0 idle (Wave 30 closed count: 9)

| PR | Student | Axis | Status |
|---|---|---|---|
| #1143 | frieren | H8 mirror-symmetry data aug | EP12.4 NEAR-TERMINAL, val_abupt 6.368%, τz/τx 1.539 |
| #1146 | nezuko | H9' curvature input feature | EP4.7+, val_abupt 6.381% (EP3 PASS), fastest pace 1.06h/ep |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | EP3+ PASS at 6.904%, τz/τx 1.512 not breaking yet |
| #1148 | fern | H10 vector-decoupled output | EP1-EP2 active training |
| #1150 | askeladd | H11 multi-scale kNN context | EP2.5+, val_abupt 6.864% improving (EP3 PASS band) |
| #1151 | edward | H12 τ-magnitude-weighted loss | EP2.34, val_abupt 8.23% improving, pace 1.59h/ep budget-risk |
| #1155 | alphonse | H15 EMA / Polyak averaging | EP1+ main arm running (smoke PASS) |
| #1156 | thorfinn | H13b anisotropic β=2 (follow-up) | JUST ASSIGNED |

**Closed in Wave 30** (9): H1 #1139, H2 #1136, H3 #1138, H4 #1141, H5 #1137, H6 #1134 (mech-PASS), H7 #1140, H14 #1153 (diverged), **H13 β=5 #1152 (diverged at warmup boundary, mirror image of H14)**.

### Causal map of τ_z bottleneck — architecture surface ABSOLUTELY EXHAUSTED (7 closures)

The Wave 30 attack-surface inventory at this point:

| Layer | In-flight probes | Closed |
|---|---|---|
| **Architecture** | 0 | 6 widening + 1 mechanism-PASS (DEFINITIVELY exhausted) |
| **Loss** | 3 (H6' tan, H12 mag, H13b aniso β=2) | 1 (H13 β=5 DIVERGED at warmup boundary) |
| **Data-input** | 3 (H8 mirror, H9' curv, H11 multi-scale) | 0 |
| **Output projection** | 1 (H10 vector decouple) | 0 |
| **Optimization** | 1 (H15 EMA smoothing) | 1 (H14 5× head LR DIVERGED at warmup boundary) |

**Two consecutive amplification-style divergences (H14 head_lr 5× and H13 β=5)** crashed at the SAME EP1→EP2 LR jump (2.5e-5 → 5e-4). The standing 1-epoch warmup + lr=5e-4 recipe has very little safety margin for amplification-style attacks. Any future "amplify direction X by N×" probe needs N ≤ 2-3 OR extended warmup OR lower base LR. H13b (β=2) and H15 (smoothing instead of amplifying) are the survivors with margin to spare.

### Loss-layer attack fleet — 3-strong probe of τ_z bottleneck direction

| PR | Probe | Direction | Mechanism if winning |
|---|---|---|---|
| #1147 (H6') | soft τ_pred·n=0 | suppress model's normal-component | model was over-predicting noise |
| #1151 (H12) | (\|τ_target\|/mean)^α weight | upweight high-magnitude vertices | long-tail magnitude under-learned |
| #1152 (H13) | β·MSE(τ_n_err) on GT-normal | upweight matching GT normal-component | model was under-predicting normal signal |

**H6' and H13 are direct symmetric opposites** — only one can be correct at the τ_z bottleneck. H12 is orthogonal to both (per-vertex magnitude weighting independent of direction). Together this 3-probe wedge tightly localizes which loss-layer reformulation breaks the structural τz/τx ceiling.

### Next-idle assignment queue (in priority order)

1. **Stacking experiments** — once Wave 30 terminals land, combine top winner with EMA (H15) + orthogonal axes
2. **H14b LR-sweep follow-up** (multiplier ∈ {1.5, 2.0, 3.0}) — pin down where the Lion-stability cliff lies; only run if H15 EMA result is also null and optimization-layer still needs probes
3. **AdamW for H14 retry** — student #1153 suggested this; AdamW handles wider LR range cleanly via second-moment normalization; clean separation between "5× is aggressive on this head" vs "Lion specifically cannot operate at 2.5e-3"
4. **Capacity-matched H4 retry (z-slice-fraction=0.55)** — clean falsification of remaining slice-routing variant
5. **Focal MSE loss with γ on per-vertex error** — alternative loss attack if H12 partial-wins (uses error magnitude not target magnitude)
6. **Spherical-harmonic WSS basis** — stronger H10 variant if H10 partial-wins
7. **Lookahead optimizer (Zhang et al. 2019)** — k inner steps + slow weight interpolation; complementary to EMA
5. **Curriculum on τ_z weight** — schedule tau_z_loss_weight 2.0→3.5 across epochs; cheap, additive
6. **Multi-head output (one per τ channel)** — alternative output-head attack if H10/H14 partial-win
7. **SAM optimizer** — sharpness-aware minimization (2× compute, may be infeasible in 18h budget)

## Older invocation actions (2026-05-16 ~11:50Z) — thorfinn #1138 CLOSED (5-of-5 model-side widening at terminal closure) → thorfinn reassigned to H13 tangent/normal anisotropic surface-loss decomposition (PR #1152); loss-layer attack fleet now 3-strong (H6'/H12/H13)

### Actions this invocation

- **CLOSED PR #1138 (thorfinn H3 soft normal-routing)** at terminal. W&B run `of1ur6fp` finished cleanly (best_epoch=12, EMA source, 14h+ training). Test: test_WSS=6.898% (+0.171pp FAIL), test_SP=3.709% (+0.132pp FLOOR BREACH), test_vol_p=3.462% (PASS −0.181pp), test τz/τx=1.452 (NULL collapse from val=1.536). Mechanism strongly engaged (slice entropy 0.96→0.36, `normal_slice_bias.param_norm` grew healthily) but engaged-but-neutral on τ_z. Surprising side-effect: vol_p beats baseline by 0.181pp. Student pod went idle at 10:31Z (Claude exited code=0 after iteration 451); training continued in background to finished state at 11:14Z but no Claude session was alive to post the terminal SENPAI-RESULT. Advisor pulled metrics from W&B summary as authoritative — benign harness orchestration artifact.
- **5-of-5 Wave 30 architecture-attack widening pattern at TERMINAL CLOSURE CONFIRMED**. H1/H2/H3/H5/H7 all closed with the engaged-but-neutral signature. Only #1141 alphonse H4 hard MoE routing remains in-flight on the architecture layer.
- **ASSIGNED PR #1152 (thorfinn: Wave 30 H13 Tangent/Normal Anisotropic Surface-Loss Decomposition)** — decompose per-vertex τ prediction into tangent (τ_t = τ − (τ·n)n) and normal (τ_n = (τ·n)n) components using surface normals from `surface_x[..., 3:6]`. Apply α_tangent=1, β_normal=5 to MSE. **Symmetric opposite of H6'**: H6' suppresses model's τ_pred·n; H13 explicitly upweights MATCHING the GT normal-component. Only one of (H6', H13) can be the correct direction at the τ_z bottleneck. Diagnostic `train/tau_normal_to_tangent_ratio` traces should clarify direction by EP3.

### Wave 30 fleet — 8 active + 0 idle (Wave 30 closed count: 6)

| PR | Student | Axis | Status |
|---|---|---|---|
| #1141 | alphonse | H4 hard MoE routing | in flight EP7+ ~6h to terminal |
| #1143 | frieren | H8 mirror-symmetry data aug | in flight EP3 marginal |
| #1146 | nezuko | H9' curvature input feature | in flight warmup |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | in flight warmup |
| #1148 | fern | H10 vector-decoupled output | in flight warmup |
| #1150 | askeladd | H11 multi-scale kNN context | in flight warmup |
| #1151 | edward | H12 τ-magnitude-weighted loss | in flight warmup |
| #1152 | thorfinn | H13 tangent/normal anisotropic loss | JUST LAUNCHED |

**Closed in Wave 30** (5 widening + 1 mechanism-PASS-absolute-FAIL): H2 #1136, H5 #1137, H6 #1134, H7 #1140, H1 #1139, H3 #1138.

### Loss-layer attack fleet — 3-strong probe of τ_z bottleneck location

| PR | Probe | Direction | Mechanism if winning |
|---|---|---|---|
| #1147 (H6') | soft τ_pred·n=0 | suppress normal component | model was over-predicting noise |
| #1151 (H12) | (\|τ_target\|/mean)^α weight | upweight high-magnitude vertices | long-tail magnitude under-learned |
| #1152 (H13) | β·MSE(τ_n_err) on GT-normal component | upweight matching GT normal component | model was under-predicting normal signal |

**H6' and H13 are direct symmetric opposites** — only one can be correct at the τ_z bottleneck. H12 is orthogonal to both (per-vertex magnitude weighting independent of direction). Combined, this 3-probe wedge tightly localizes which loss-layer reformulation breaks the structural τz/τx ceiling.

### Causal map of τ_z bottleneck — 5-of-5 architecture closures terminal-confirmed

- **DEFINITIVELY NOT at architecture layer**: 5-of-5 closed Wave 30 architecture attacks (H1/H2/H3/H5/H7) all show val widening 1.50-1.55 that collapses to test ~1.44-1.47 baseline band
- **Bottleneck IS at output head + input feature distribution + loss layer**:
  - 1 in-flight output-head probe (H10 vector decouple) — H6 closed mechanism PASS, H7 closed null
  - 3 in-flight data-level probes (H8 mirror, H9' curvature, H11 multi-scale)
  - 3 in-flight loss-layer probes (H6' tangent-penalty, H12 magnitude-weight, H13 anisotropic-decomp)
- **Architecture-attack remnant**: #1141 alphonse H4 hard MoE routing is the only architecture attack still in-flight; expected to show the same widening pattern but stronger mechanism engagement than H3 (hard routing > soft routing)

### Next-idle assignment queue (in priority order)

1. **Stacking experiments** — once Wave 30 terminals land, combine top winner with orthogonal axes
2. **Focal MSE loss with γ on per-vertex error** — alternative loss attack if H12 partial-wins (uses error magnitude not target magnitude)
3. **Spherical-harmonic WSS basis** — stronger H10 variant if H10 partial-wins
4. **Curriculum on τ_z weight** — schedule tau_z_loss_weight from 2.0→3.5 across epochs; cheap, additive
5. **Geodesic distance to sharp-edge feature** — alternative data-level input signal

## Older invocation actions (2026-05-16 ~10:10Z) — edward #1139 CLOSED (7-of-7 model-side widening) → edward reassigned to H12 τ-magnitude-weighted MSE loss (PR #1151); architecture attack surface definitively exhausted

### Actions this invocation

- **CLOSED PR #1139 (edward H1 cylindrical coords)** at terminal: test_WSS=7.049% (+0.322pp FAIL), test_SP=3.865% (+0.288pp FLOOR BREACH), test_vol_p=3.682% (+0.039pp FLOOR BREACH), test τz/τx=1.469 (NULL). The sincos pos_embed already provides a complete Fourier basis subsuming cylindrical decomposition. Clean falsification.
- **7-of-7 Wave 30 model-side widening pattern CONFIRMED**. Architecture-layer attack surface **definitively exhausted**.
- **ASSIGNED PR #1151 (edward: Wave 30 H12 τ-Magnitude-weighted MSE Loss)** — multiply per-vertex surface MSE by `(|τ_target_i| / batch_mean)^α`. Sweep α ∈ {0.3, 0.5, 0.7}. Direct attack on long-tail τ_z error distribution at the loss layer. Aligns training objective with rel_l2 evaluation metric (which inherently weights high-magnitude regions more).
- **Posted stale_wip check-ins on #1146 (nezuko H9') and #1147 (tanjiro H6')** — both just launched ~3-4h ago, pods healthy 1/1 Ready, warmup phase normal.

### Wave 30 fleet — 8 active + 0 idle

| PR | Student | Axis | Status |
|---|---|---|---|
| #1138 | thorfinn | H3 soft normal-routing | in flight EP11+ terminal expected ~1h |
| #1141 | alphonse | H4 hard MoE routing | in flight EP7+ ~6h to terminal |
| #1143 | frieren | H8 mirror-symmetry data aug | in flight early-EP |
| #1146 | nezuko | H9' curvature input feature | in flight warmup |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | in flight warmup |
| #1148 | fern | H10 vector-decoupled output | in flight warmup |
| #1150 | askeladd | H11 multi-scale kNN context | in flight warmup |
| #1151 | edward | H12 τ-magnitude-weighted loss | JUST LAUNCHED |

**Closed in Wave 30** (5 confirmed all model-side widening + 1 mechanism-PASS-absolute-FAIL via sledgehammer): H2 #1136, H5 #1137, H6 #1134, H7 #1140, H1 #1139.

### Causal map of τ_z bottleneck — updated 7-of-7

- **DEFINITIVELY NOT at architecture layer**: 7-of-7 model-side architecture attacks (H1/H2/H4/H5/H7) all show val widening that collapses to baseline on test
- **Bottleneck IS at output head + input feature distribution + loss layer**:
  - 3 in-flight output-head probes (H6'/H10/H7) — H7 closed null
  - 2 in-flight data-level probes (H8 mirror, H9' single-scale curvature)
  - 1 in-flight multi-scale data probe (H11)
  - **NEW**: 1 in-flight loss-layer probe (H12 magnitude-weighted)
- **Highest-EV remaining axes for terminal-wave winner**: H11 (multi-scale data) > H12 (loss-magnitude) > H9' (single-scale data) > H10 (output reparam) > H8 (mirror aug) > H6' (loss penalty). Ordering reflects: (1) dl24 cross-pollination confirming data-level signal works, (2) H6 mechanism PASS confirming output-head is bottleneck, (3) novelty of loss-layer attack relative to in-flight axes.

### Next-idle assignment queue (in priority order)

1. **Stacking experiments** — once Wave 30 terminals land, combine top winner with orthogonal axes
2. **Focal MSE loss with γ on per-vertex error** — alternative loss attack if H12 partial-wins (uses error magnitude not target magnitude)
3. **Spherical-harmonic WSS basis** — stronger H10 variant if H10 partial-wins
4. **Curriculum on τ_z weight** — schedule tau_z_loss_weight from 2.0→3.5 across epochs; cheap, additive
5. **Geodesic distance to sharp-edge feature** — alternative data-level input signal

## Older invocation actions (2026-05-16 ~09:40Z) — askeladd #1140 CLOSED (fleet leader stalled at EP13, 6-of-6 widening pattern) → askeladd reassigned to H11 multi-scale kNN-pooled context features (PR #1150)

### Actions this invocation

- **CLOSED PR #1140 (askeladd H7 normal-prediction aux head)** at terminal. Fleet leader at EP7-8 (val_abupt=6.222%) stalled at EP13 (val_abupt EMA=6.1975%, +0.071pp ABOVE baseline). Test: test_WSS=6.9018% (+0.175pp FAIL), test_SP=3.8246% (+0.248pp FLOOR BREACH), test_vol_p=3.5776% (PASS −0.065pp). Mechanism cleanly null: `aux_normal_cosine` converged to 0.999951 by step 10k → backbone already encodes full normal info, aux head had nothing to inject.
- **6-of-6 Wave 30 model-side widening pattern CONFIRMED**. Bottleneck definitively NOT at architecture layer.
- **ASSIGNED PR #1150 (askeladd: Wave 30 H11 Multi-scale kNN-pooled context features)** — direct upgrade of H9' (which is single-scale, k=16 NN-of-normals statistic). H11 computes 3 stats × 3 scales = 9 channels: cos_alignment, mean_area, mean_dist at k=4/16/64. Provides explicit multi-resolution geometric context that pure attention captures only implicitly. Cached per-case for fast reload. Strong Kaggle/PointNet++/FPN pedigree.

### Wave 30 fleet — 8 active in-flight + 0 idle = 12 axes attempted total in this wave (4 closed, 8 active)

| PR | Student | Axis | Status |
|---|---|---|---|
| #1138 | thorfinn | H3 soft normal-routing | in flight EP10+ |
| #1139 | edward | H1 cylindrical coords | in flight EP10+ |
| #1141 | alphonse | H4 hard MoE routing | in flight EP7+ |
| #1143 | frieren | H8 mirror-symmetry data aug | in flight EP1+ |
| #1146 | nezuko | H9' curvature input feature | in flight |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | in flight |
| #1148 | fern | H10 vector-decoupled output | in flight |
| #1150 | askeladd | H11 multi-scale kNN context | JUST LAUNCHED |

**Closed in Wave 30**: H2 #1136 (normal spectral), H5 #1137 (Y-arch), H6 #1134 (hard τ·n=0 PAPER-WORTHY), H7 #1140 (normal-aux).

### Causal map of τ_z bottleneck — updated 6-of-6

- **NOT at backbone**: H6 hard τ·n=0 broke τz/τx to 1.281 (falsified backbone-bottleneck)
- **DEFINITIVELY NOT at architecture layer**: 6-of-6 closed Wave 30 model-side axes (H1/H2/H4/H5/H7 in val widening; H6 mechanism break by sledgehammer) — none unlocked test τz/τx with absolute fidelity
- **Bottleneck IS at output head + input feature distribution**: 3 in-flight output-head probes (H6'/H10/H7) + 2 in-flight data-level probes (H8 mirror, H9' single-scale curvature) + this H11 (multi-scale data)
- **Strongest candidates for next terminal-wave winner**: H11 (multi-scale data) > H9' (single-scale data) > H10 (output reparam) > H8 (mirror aug) > H6' (loss penalty). Ordering is based on dl24 cross-pollination evidence (curvature mechanism real) + H6 mechanism PASS (output head is bottleneck location).

### Next-idle assignment queue (in priority order)

1. **Stacking experiments** — combine top Wave 30 winner with any orthogonal axis once terminals land
2. **Spherical-harmonic WSS basis** — predict WSS in a learned anisotropic local frame; stronger H10 variant if H10 partial-wins
3. **Curriculum on τ_z weight** — schedule tau_z_loss_weight from 2.0→3.5 across epochs; cheap, additive to any winner
4. **Focal MSE loss with γ on per-vertex error** — long-tail loss attack; alternative to H6' direction-based penalty
5. **Geodesic distance to feature** — precompute distance-to-nearest-sharp-edge per vertex; alternative data-level input signal

## Older invocation actions (2026-05-16 ~09:10Z) — fern #1137 CLOSED (5-of-5 model-side widening pattern, all 3 floors breached) → fern reassigned to H10 vector-length-decoupled WSS head (PR #1148)

### Actions this invocation

- **CLOSED PR #1137 (fern Y-arch dual-backbone)** at terminal: test_WSS=7.109% (+0.382pp FAIL), test_SP=3.931% (+0.354pp FLOOR BREACH), test_vol_p=3.673% (+0.030pp FLOOR BREACH), test τz/τx=~1.453 (NULL). Run hit OOM at EP7 boundary; EP6 EMA checkpoint clean for test eval. Branch cos_sim 0.17-0.20 (healthy), but τ_z reduced proportionally with τ_x/τ_y — task-interference hypothesis falsified.
- **5-of-5 Wave 30 model-side widening pattern CONFIRMED**. The bottleneck is **definitively NOT at the model architecture layer**. The unexplored attack surface narrows to: (1) output-head reformulation, (2) data distribution, (3) input features.
- **ASSIGNED PR #1148 (fern: Wave 30 H10 Vector-Length-Decoupled WSS Head)** — predict `(cp, dir_x, dir_y, dir_z, log_mag)` instead of Cartesian `(cp, τx, τy, τz)`. Reconstruct `τ = softplus(log_mag) * unit(dir)`; aux cos-sim loss on direction. Decouples direction from magnitude; orthogonal to all 10 in-flight axes (only experiment that reparametrizes the WSS output).

### Wave 30 fleet — 11 orthogonal attacks now in parallel

| PR | Student | Axis | Status |
|---|---|---|---|
| #1138 | thorfinn | H3 soft normal-routing | in flight EP10+ |
| #1139 | edward | H1 cylindrical coords | in flight EP10+ |
| #1140 | askeladd | H7 normal-aux head | in flight EP8 **fleet leader val_abupt=6.222%** |
| #1141 | alphonse | H4 hard MoE routing | in flight EP7 |
| #1143 | frieren | H8 mirror-symmetry data aug | in flight EP1+ |
| #1146 | nezuko | H9' curvature input feature | JUST LAUNCHED |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | JUST LAUNCHED |
| #1148 | fern | H10 vector-decoupled output | JUST LAUNCHED |

### Causal map of τ_z bottleneck — updated 5-of-5

- **NOT at backbone**: H6 hard τ·n=0 broke τz/τx to 1.281 (falsified backbone-bottleneck)
- **DEFINITIVELY NOT at architecture layer**: 5-of-5 Wave 30 model-side axes (H1/H2/H4/H5/H7) all show val widening that collapses to baseline on test
- **Bottleneck IS at output head + likely data distribution**: H6 hard projection works but loses fidelity; soft penalty (H6'), reparametrization (H10), and data injection (H8 mirror, H9' curvature) are the active probes
- **Three independent attacks now in flight on the output head**: H6' soft τ·n penalty (loss), H10 vector-decoupled (reparametrization), H7 normal-aux (gradient). Each tests a different mechanism for unlocking the τ_z bottleneck without losing absolute fidelity.

### Next-idle assignment queue (in priority order)

1. **Stacking experiments** — combine top Wave 30 winner (test_WSS < baseline) with H6'/H9'/H10/H8 once terminals land. Highest expected compound gain.
2. **kNN-pooled local context feature** — pool 16-NN surface features (normals, area, curvature) per token, append as 6-8 channels. Truly orthogonal to all 11 axes; data-level signal injection.
3. **Spherical-harmonic WSS basis** — predict WSS in a learned anisotropic local frame (not Cartesian, not pure polar). Stronger H10 variant if H10 partial-wins.
4. **Curriculum on τ_z weight** — schedule tau_z_loss_weight from 2.0→3.5 across epochs. Cheap, additive to any winner.
5. **GroupedSeparable positional encoding** — different sigma per axis (separable not just by axis but by axis group: xyz vs τ-magnitude axis). Targets the per-axis representation imbalance hypothesis.

## Older invocation actions (2026-05-16 ~08:45Z) — Double closure of nezuko #1136 (4-of-4 widening confirmed) + tanjiro #1134 (paper-worthy mechanism PASS); both students immediately reassigned to H9' (PR #1146) and H6' (PR #1147)

### Actions this invocation

- **CLOSED PR #1136 (nezuko H2 normal spectral encoding)** at terminal: test_WSS=6.928% (+0.201pp FAIL), test_SP=3.827% (+0.250pp FLOOR BREACH), test τz/τx=1.457 (NULL — essentially baseline). val widened 1.49→1.548 then collapsed on test. **4-of-4 Wave 30 model-side widening pattern confirmed** (H1/H2/H4/H7).
- **CLOSED PR #1134 (tanjiro H6 local-frame WSS head)** at terminal: paper-worthy mechanism PASS / absolute FAIL. test τz/τx=**1.281** — cleanest τ_z structural break across NINE prior mechanisms. BUT test_WSS=26.69% (+19.96pp absolute fail), SP+vol_p both breach by 1.4pp / 0.8pp due to hard projection removing 5–8% real GT normal-component signal. Falsifies backbone-bottleneck alternative; bottleneck IS at output head.
- **ASSIGNED PR #1146 (nezuko: Wave 30 H9' curvature-aware surface feature)** — port of dl24-tanjiro #1132 (test_WSS=6.609% reported on parallel branch). 8th surface channel via kNN-of-normals statistic κ=mean(1-cos(n_i,n_j)). FIRST input-feature attack on tay. Recipe: standard 18h x 13EP lion recipe + `--use-curvature-feature --curvature-knn 16`. Default off, baseline-safe.
- **ASSIGNED PR #1147 (tanjiro: Wave 30 H6' soft τ·n=0 penalty)** — direct follow-up to own H6 mechanism PASS. Loss-term `λ · E[(τ·n)²/|τ|²]` with sweep λ∈{0.05, 0.1, 0.25}. Restores 4-channel head freedom while preserving structural-break bias. HIGHEST expected-value unassigned slot.

### Wave 30 fleet — 10 orthogonal attacks now in parallel

| PR | Student | Axis | Status |
|---|---|---|---|
| #1137 | fern | H5 Y-architecture | in flight (slow descender ~2.5h/EP) |
| #1138 | thorfinn | H3 soft normal-routing | in flight EP6+ |
| #1139 | edward | H1 cylindrical coords | in flight EP6+ |
| #1140 | askeladd | H7 normal-aux head | in flight EP7+ **leading 6.222%** |
| #1141 | alphonse | H4 hard MoE routing | in flight EP7+ |
| #1143 | frieren | H8 mirror-symmetry data aug | in flight warmup→EP3 |
| #1146 | nezuko | H9' curvature input feature | JUST LAUNCHED |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | JUST LAUNCHED |

### Causal map of τ_z bottleneck — updated

- **NOT at backbone**: H6 hard τ·n=0 broke τz/τx to 1.281 (falsified backbone-bottleneck hypothesis)
- **NOT at architecture layer**: 4-of-4 Wave 30 model-side axes (H1/H2/H4/H7) all show val widening that collapses to baseline on test
- **Bottleneck IS at output head + data distribution**: hard projection works but loses fidelity → soft penalty (H6') is the natural next move
- **Cross-pollination corroborates**: dl24-tanjiro #1132 curvature mechanism produced real WSS gain — suggests data-level signal injection (H9' curvature, H8 mirror, H6' soft penalty) is the right direction

### Next-idle assignment queue (in priority order)

1. **Stacking H6' (winner λ) + H9' curvature** — if both win, stacking is the immediate compounding play
2. **Stacking H8 mirror-aug + H6' soft penalty** — if H8 lands, mirror-symmetry + soft tangency is theoretically complementary
3. **kNN-pooled local context feature** — pool 16-NN surface features per token, append as channel; orthogonal to all 10 axes
4. **Learned anisotropic basis** — vector-valued WSS prediction in a learned local frame (not fixed normal/tangent); soft regularization of basis orthonormality
5. **Curriculum on τ_z weight** — schedule tau_z_loss_weight from 2.0→3.5 across epochs; ramps the gradient pressure

## Older invocation actions (2026-05-16 ~07:35Z) — Fleet mid-flight progress snapshot; human confirmed dl24 fleet is real (separate advisor); 4-of-4 Wave 30 model-side axes confirming τ_z structural ratio is NOT at the model layer

### Mid-flight Wave 30 progress (val_abupt at 07:35Z W&B snapshot, no terminals yet)

| PR | Student | Axis | val_abupt | val τz/τx | Runtime | EP gate |
|---|---|---|---:|---:|---:|---|
| #1140 | askeladd | H7 normal-aux head | **6.222%** | ~1.54 | 12.4h | EP7-8 PASS, leading fleet |
| #1138 | thorfinn | H3 soft normal-routing | 6.334% | ~1.53 | 11.7h | EP6-7 PASS, slice-entropy 0.374 |
| #1139 | edward | H1 cylindrical coords | 6.334% | ~1.53 | 11.2h | EP6-7 PASS |
| #1141 | alphonse | H4 hard MoE routing | ~6.4% | ~1.54 | mid-EP7 | both partitions saturated 1.0 |
| #1136 | nezuko | H2 normal Fourier | 6.404% | 1.542 | 13.7h | EP7+ PASS |
| #1137 | fern | H5 Y-architecture | 6.523% | — | 12.0h | slow descender, EP6 borderline |
| #1143 | frieren | H8 mirror-symmetry aug | — | — | warmup | just launched ~04:55Z |
| #1134 | tanjiro | H6 local-frame τ·n=0 | **18.64%** | **1.351** | EP12 | mechanism break, terminal absolute fail |

### Emerging Wave 30 finding (4-of-4 model-side axes that have reached EP6+)

ALL clean absolute descent below baseline trajectory, BUT τz/τx uniformly widens 1.38 → ~1.54. **The structural τ_z bottleneck appears to be NOT at the model layer** — input frame (H1), input features (H2), attention routing soft+hard (H3+H4), gradient signal (H7) all improve aggregate metrics without breaking the structural ratio.

**The single exception is tanjiro H6 (τz/τx=1.351, decisively below 1.40)** — but hard architectural τ·n=0 enforcement throws away ~5–8% real normal-component signal in GT, making it incompatible with the absolute metric. Mechanism works, metric fails.

### Next decisive datapoint — frieren #1143 H8 mirror-symmetry data aug

The ONLY attack on the input distribution. Two possible outcomes:
- **If it moves τz/τx without metric loss** → bottleneck is data-level (symmetry-breaking), follow-up: stack with H1/H3/H7 winners
- **If it's a null** → bottleneck is structural, next move is H6' soft-τ·n=0 penalty (λ~0.1) — preserves absolute WSS while retaining directional constraint signal

### Human directive (07:29Z #1056)

"please ensure to report val and test (if available) scores" — commit to val+test in all future updates. Test only available at EP13 terminal SENPAI-RESULT; val per-epoch during training.

### Cross-pollination from dl24 advisor (07:59Z #1056)

The dl24 advisor (running parallel fleet on `drivaerml-long-20260504`) shared 5 terminal results. Two key findings:

1. **Curvature attention bias #1132 (dl24-tanjiro H5): test_WSS=6.609% (−0.118 under SOTA), test_τ_z=8.592% (−0.155 on dominant axis), val_abupt=6.168%.** Blocked by floor breaches (test_vol_p=3.955%, test_SP=3.651%) due to GradNorm starving w_vol_p. **The curvature feature attack IS a real WSS mechanism — and is the GAP in tay Wave 30 fleet.** Filed as the next-idle assignment candidate: **H9' "Curvature attention bias on tay" — port dl24-tanjiro #1132's curvature mechanism to tay where we use FIXED loss weights (no GradNorm starvation). Should beat SOTA on WSS without floor breaches.**

2. **GradNorm w_vol_p crushed to 0.0064 vs w_τ_z=2.318 (362× lower)** — explains every dl24 vol_p floor breach. tay fleet uses fixed loss weights so we don't share this failure mode. But it's the right diagnostic frame: **gradient-budget mismatch is the unifying explanation for both fleets** — dl24 starves vol_p, tay can't break τ_z without surrender. Both extremes (too soft = 4-of-4 widening, too hard = val_WSS=26%) confirm a Goldilocks zone for τ·n=0 type constraints.

**dl24 active runs to watch for cross-pollination:**
- #1135 frieren H6 wind-exposure attn bias zero-init — EP10 val_wss<7% first time in wave
- #1142 fern H7 surface_loss_weight=1.5 — val_vol_p=3.675% (just above floor)
- #1144 nezuko H8 Lion → AdamW Plateau Protocol — warmup
- #1145 tanjiro H9 curvature bias + GradNorm w_vol_p clamp ≥0.05 — direct attack on #1132 root cause

### Next-idle assignment queue (in priority order)

1. **H9' Curvature attention bias on tay** (port of dl24-tanjiro #1132 mechanism) — high confidence test_WSS<SOTA
2. **H6' Soft τ·n=0 penalty** (port of tanjiro #1134 mechanism with λ~0.1 soft constraint) — preserve absolute WSS while retaining directional constraint signal
3. **Stacking experiments** — combine top Wave 30 winner with H9' or H6' once terminals land

Once Wave 30 terminals start landing, deploy H9' on the first idle student.



## Latest invocation actions (2026-05-16 ~04:30Z) — frieren #1133 CLOSED terminal (NINTHFOLD structural ratio confirmation, magnitude decomposition cleanly falsified), Wave 30 fleet now 7-of-8 active + frieren idle pending reassignment; nezuko #1136 H2 mid-EP7 healthy descending but τz/τx widening (negative-mechanism / positive-absolute signal)

### Actions this invocation

- **CLOSED PR #1133 (frieren per-axis WSS mag decomp |τ_z|+||τ_xy||)** at terminal.
  - Test metrics: test_WSS=**6.853%** (+0.126pp miss), test_vol_p=**3.620%** PASS, test_SP=**3.837%** (+0.260pp FLOOR BREACHED), val_abupt=6.254%, test τz/τx=**1.469**.
  - **Fails 2/4 hard gates.** Clean mechanism falsification: mag_xy calibration ratio = 0.999 and mag_z calibration ratio = 1.001 by EP6, holding through EP13. mag_xy loss term EXCEEDED mag_z throughout training (1.07–1.34× ratio) — opposite of what τ_z-is-hard predicts. **Backbone represents |τ_z| just as easily as ||τ_xy||** — the bottleneck is *signed* τ_z, not magnitude encoding.
  - **NINTHFOLD structural ratio confirmation** of the 1.44–1.57 band. Loss-side reformulation provides no traction.

- **FALSE-POSITIVE check-in on #1136 (nezuko Wave 30 H2 normal Fourier features)**.
  - Run `lths1ujt` alive at step 59,267, mid-EP7, heartbeat fresh at 04:37Z.
  - Per-epoch val_abupt **monotonically descending**: 32.45% (EP1.3) → 7.55% (EP2.6) → 6.85% (EP3.9) → 6.60% (EP4.6) → 6.52% (EP5.2) → 6.47% (EP5.9) → 6.44% (EP6.3) → **6.43%** (EP6.7).
  - **HOWEVER** τz/τx is monotonically *increasing* (1.385 → 1.542) — normal Fourier features help all axes but the relative balance favors τ_xy over τ_z. **Interesting negative-mechanism / positive-absolute signal.**
  - Let run complete to EP13 for terminal — may still merge if test_WSS < 6.727% with both floors.

- **ASSIGNED PR #1143 (frieren: Wave 30 H8 mirror-symmetry data augmentation)** — EIGHTH Wave 30 attack axis, ORTHOGONAL to all 7 in-flight architectural axes.
  - **Hypothesis**: DrivAerML is zero-yaw zero-pitch → exact x-z mirror symmetry. Apply random 50% per-sample y-flip during training (flip surface_x y/ny, surface_y τ_y, volume_x y; cp + vol_p invariant). NOT an ensemble — single model, single forward pass at test time.
  - **Why this attacks τ_z**: forces τ_y representations to be sign-flip-equivariant, breaks any spurious τ_y/τ_z correlation that biases the structural ratio. Doubles effective dataset (400 → 800 orientation views).
  - **Theoretical basis**: Weiler & Cesa NeurIPS 2019 (exact symmetry → augmentation converges to equivariant solution); DrivAerML data paper documents exact symmetry.
  - **Implementation**: ~35 LOC across train.py (Config flag) + data/loader.py (new `mirror_collate` wrapper) + trainer_runtime.py (conditional collate_fn). Frozen dataclass `DrivAerMLCase` → `dataclasses.replace`. Validation/test loaders unchanged.
  - **Falsifiability**: WIN if test_WSS < 6.727% AND test τz/τx ≤ 1.40. Either result sharply updates the causal map of the τ_z bottleneck — first attack on the *input distribution*, all 9 prior attacks were on the model.

### Wave 30 fleet now 8-of-8 architectural attack axes active in parallel

| PR | Student | Mechanism | Attack layer |
|----|---------|-----------|-------------|
| #1134 | tanjiro | H6 local-frame WSS head (τ·n=0) | output decomposition |
| #1136 | nezuko | H2 normal Fourier features | input features |
| #1137 | fern | H5 Y-architecture dual-backbone | backbone split |
| #1138 | thorfinn | H3 normal-aligned slice groups (SOFT) | attention soft |
| #1139 | edward | H1 cylindrical coords (r,θ,z) | input coord frame |
| #1140 | askeladd | H7 normal-prediction aux head | gradient signal |
| #1141 | alphonse | H4 hard normal slice routing (MoE) | attention hard |
| **#1143** | **frieren** | **H8 mirror-symmetry augmentation** | **input distribution** |

**Eight orthogonal attacks. Eight students. Zero idle GPUs.**

### Ninthfold structural ratio confirmation — full table

| # | Mechanism | Lever | test τz/τx |
|---|-----------|-------|-----------:|
| 1 | thorfinn EMA 0.9995 (#1124) | temporal | 1.469 |
| 2 | nezuko spatial-prior α=10 (#1125) | sampling | 1.449 |
| 3 | fern surface_out depth=4 (#1126) | output capacity | 1.462 |
| 4 | edward per-channel heads (#1116) | output decoupling | 1.460 |
| 5 | thorfinn τ_z weight 3.0 (#1128) | loss weighting | ~1.44 |
| 6 | askeladd surface_loss warmup (#1127) | curriculum | ~1.52 |
| 7 | frieren mag-only #1121 | loss reform | 1.46 |
| 8 | alphonse SDF FAR-field α=2.0 (#1122) | volume sampling | 1.465 |
| **9** | **frieren mag-decomp #1133** | **loss reform v2** | **1.469** |

Loss-side, sampling-side, capacity-side, output-decoupling-side, volume-sampling-side, and mag-decomp-side mechanisms ALL converge to the structural band. Wave 30 architectural attacks (7 in-flight) + data-distribution attack (H8 #1143) are the entirety of the remaining frontier.

## Prior invocation actions (2026-05-15 ~21:20Z) — alphonse #1122 CLOSED (EIGHTHFOLD structural ratio confirmation), Wave 30 FLEET EXPANDED to 7-of-8 (H4 hard normal routing #1141 launched), stale_wip false-positives cleared on #1133 + #1140

### Actions this invocation

- **CLOSED PR #1122 (alphonse SDF FAR-field α=2.0)** at terminal.
  - Test metrics: test_WSS=**7.518%** (+0.792pp), test_vol_p=**4.524%** (+0.881pp floor regress), test_SP=**4.141%** (+0.564pp floor regress), val_abupt=6.698%.
  - **Fails 3/3 merge gates.** Volume-sampling-side mechanism cleanly REFUTED for fixing the τ_z structural bottleneck.
  - Test τz/τx = **1.465** — **EIGHTHFOLD** confirmation of the structural ratio band.
  - Note: this was a port test, not a SOTA attempt. PR #972 used the full SDF stratification stack (never landed on tay); FAR-field α=2.0 alone is insufficient to reproduce that result.

- **FALSE-POSITIVE check-ins on #1133 (frieren per-axis WSS mag decomp) + #1140 (askeladd Wave 30 H7)**.
  - #1133: run `5l9i6fjn` alive at step 46,522, runtime 7.66h (~EP6), val_abupt=**6.47%** (already below baseline 6.126% trajectory at this checkpoint). Strong signal.
  - #1140: run `e5ztxjc3` alive at step 14,751, runtime 2.19h (~EP2), val_abupt=25% (warmup expected). Diagnostic note: `train/normal_aux_loss` is ~0.001 (1000× smaller than predicted) — student asked to confirm whether logged value is weighted or raw at next epoch report.

- **Assigned PR #1141 (alphonse: Wave 30 H4 hard normal slice routing)** — SEVENTH Wave 30 attack axis, completes the soft↔hard sweep on attention routing.
  - **Hypothesis**: Partition `num_slices=128` budget into `num_slices_z = int(0.25 × 128) = 32` for z-normal surfaces (|n_z| ≥ 0.5) and `num_slices_xy = 96` for sides. Hard-route via pre-softmax `masked_fill(-inf)` — MoE-style. Volume tokens (no normals) retain all-slice access (baseline behavior).
  - **Theoretical basis**: Switch Transformer (Fedus 2022), Expert Choice (Zhou 2022) — hard routing > soft routing when problem has discrete structural modes (roof/underbody n_z≈±1 vs sides n_z≈0).
  - **Diagnostic**: `slice_capacity_utilization_z` and `slice_capacity_utilization_xy` should both approach 1.0 once routing engages. Pairs with thorfinn #1138 (soft) for a clean soft↔hard sweep on the same architectural layer.
  - **Falsifiability**: WIN if test τz/τx ≤ 1.40. Together with H3 result: if BOTH soft + hard routing fail to move τz/τx, the bottleneck is NOT at the attention layer.

### Wave 30 fleet status — SEVEN of EIGHT architectural attack axes now active in parallel

| PR | Student | Mechanism | Attack axis | LOC | EP/Status |
|----|---------|-----------|-------------|-----|-----------|
| #1134 | tanjiro | H6: local-frame WSS head (τ·n=0 by construction) | output decomposition | ~65 | EP3 gate due ~tomorrow AM |
| #1136 | nezuko | H2: normal spectral encoding | input features | ~35 | EP1.7 healthy |
| #1137 | fern | H5: Y-architecture dual-backbone | backbone split | ~80 | EP0-1 |
| #1138 | thorfinn | H3: normal-aligned slice groups (SOFT routing) | attention layer (soft) | ~50 | EP0-1 |
| #1139 | edward | H1: cylindrical coords (r, θ, z) | input coord frame | ~35 | EP0-1 |
| #1140 | askeladd | H7: normal-prediction aux head | gradient signal | ~80 | EP2 healthy |
| **#1141** | **alphonse** | **H4: hard normal slice routing (MoE-style)** | **attention layer (hard)** | **~70** | **EP0 (just launched)** |

**Seven mechanisms, six layers of the architecture stack, all targeting the τ_z bottleneck:**
- Input coord frame (H1) → Input features (H2) → Attention routing soft+hard (H3 + H4 pair) → Backbone split (H5) → Output decomposition (H6) → Gradient signal (H7)
- **H3 + H4 form a soft↔hard sweep** on the attention-routing axis — diagnostic pair.

Wave 30 reserve (1 of 8 still on the bench): H8 (contrastive orientation regularization). Reserved for next idle.

Wave 29 fleet still in flight (1 of 8):

| PR | Student | Mechanism | EP/Status |
|----|---------|-----------|-----------|
| #1133 | frieren | per-axis WSS mag decomp | ~EP6, val_abupt 6.47% |

**Zero idle. Eight students all running.**

### Eighthfold structural ratio confirmation — full table

| # | Mechanism | Lever | test τz/τx |
|---|-----------|-------|-----------:|
| 1 | thorfinn EMA 0.9995 (#1124) | temporal | 1.469 |
| 2 | nezuko spatial-prior α=10 (#1125) | sampling | 1.449 |
| 3 | fern surface_out depth=4 (#1126) | output capacity | 1.462 |
| 4 | edward per-channel heads (#1116) | output decoupling | 1.460 |
| 5 | thorfinn τ_z weight 3.0 (#1128) | loss weighting | ~1.44 |
| 6 | askeladd surface_loss warmup (#1127) | curriculum | ~1.52 |
| 7 | frieren mag-only #1121 | loss reform | 1.46 |
| **8** | **alphonse SDF FAR-field α=2.0 (#1122)** | **volume sampling** | **1.465** |

Sampling-side and loss-side exploration both exhausted. Wave 30 architectural attacks are the entirety of the remaining frontier.

### Next-highest-EV gates

| ETA | Event | Action |
|-----|-------|--------|
| ~next-day-AM | tanjiro #1134 EP3 gate | First Wave 30 verdict (H6 output decomposition) |
| ~next-day-AM | nezuko #1136 EP3 gate | Wave 30 H2 verdict (input features) |
| ~next-day-AM | fern #1137 EP3 gate | Wave 30 H5 verdict (backbone split) |
| ~next-day-PM | thorfinn #1138 / edward #1139 / askeladd #1140 EP3 gates | Wave 30 H3/H1/H7 verdicts (staggered) |
| ~next-day-PM | alphonse #1141 EP3 gate | Wave 30 H4 verdict (hard routing) |
| ~next-day-PM | frieren #1133 EP13 terminal | Last Wave 29 single-model candidate |

---

## Prior invocation actions (2026-05-15 ~21:00Z) — TRIPLE CLOSURE (thorfinn #1128, askeladd #1127, edward #1116), Wave 30 FLEET EXPANSION to 6-of-8 (thorfinn H3 #1138, edward H1 #1139, askeladd H7 #1140)

### Actions this invocation

- **CLOSED PR #1128 (thorfinn τ_z loss weight 3.0)** at terminal.
  - Test metrics: test_WSS=**6.938%** (+0.211pp miss), test_τ_z=−0.44pp absolute on target axis but test_τ_x +0.18pp / test_τ_y +0.19pp off-axis costs wiped the net gain. test_vol_p=**3.584%** PASS (−0.059pp under floor — isolated win). test_SP=**3.838%** (+0.261pp floor regress).
  - **Fails 3/4 merge gates.** Mechanism is real but pays for τ_z gain in τ_x and τ_y — the τ_z bottleneck is **not just loss-weight** but a structural ratio that loss-side levers can shift but not break.
  - Test τz/τx = **~1.44** — **ELEVENTH** confirmation of the structural ratio band.

- **CLOSED PR #1127 (askeladd surface_loss warmup curriculum)** at terminal.
  - Test metrics: test_WSS=**7.227%** (+0.500pp miss), test_τ_z=**9.293%** (+0.24pp REGRESS on the target axis), all floors regressed.
  - **Worst Wave 29 result of this batch.** Hypothesis cleanly falsified. Three independent loss-curriculum/shape attempts (#1127, #1109 spatial focal, #1110/#1118 OHEM) all negative — **loss-side exploration exhausted**.

- **CLOSED PR #1116 (edward per-channel WSS heads)** at terminal.
  - Test metrics: test_WSS=**6.900%** (+0.173pp vs PR #972 SOTA, beats no-SDF ceiling 6.989% by only 0.089pp), test_SP=**3.801%** (+0.224pp floor regress).
  - **Mechanism reproducible**: matched-budget 3-EP A/B showed −0.660pp test_WSS; full 18h showed −0.062pp val_WSS vs single-head. Per-head gradient decoupling confirmed (τ_z head pulls **1.57× more gradient** than τ_x head — physically expected).
  - Test τz/τx = **~1.46** — **TWELFTH** confirmation. Per-channel heads marked as **stackable mechanism** for future Wave 30 winners (decoupled per-axis output is a healthy primitive; just can't break the ceiling alone).

- **MECHANISTIC CONSOLIDATION (Wave 29 → Wave 30 pivot rationale)**:
  - 12 independent mechanisms (capacity uplifts, loss reshaping, EMA, sampling priors, depth, per-channel heads, τ_z weight escalation) all converge to test τz/τx in the **1.44–1.57 band** with val→test compression of ~0.085–0.10 units.
  - PR #1126 (fern depth=4) and PR #1100 (thorfinn slices=256) hit identical **no-SDF ceiling within 0.001pp** on test_WSS, test_vol_p, test_SP — TWO orthogonal capacity uplifts saturate together.
  - **Conclusion confirmed**: bottleneck is a **representation-axis bottleneck**, not capacity, not loss curriculum. Architectural mechanisms targeting normal/orientation handling are the open frontier.

- **Assigned PR #1138 (thorfinn: Wave 30 H3 normal-aligned slice groups)** — FOURTH Wave 30 architectural attack.
  - **Hypothesis**: Add learnable `Linear(3, num_heads × num_slices)` bias to TransolverAttention slice logits, scaled by `--normal-slice-alpha 0.5`. Zero-init so EP0 behavior preserved. Creates orientation-coherent slice token groups (upward-facing patches attract to "roof" slice, sideways patches to "side" slice).
  - **Theoretical basis**: PointBERT 2022, Point-MAE 2022, DGCNN 2019 — geometric attention with orientation-aware grouping outperforms purely feature-based grouping for shape-dependent outputs. Extends the original Transolver paper's "physics-informed slicing" principle from spatial to orientation grouping.
  - **Diagnostic**: slice-weight entropy (should drop 5–10% vs alpha=0 baseline if mechanism engages).

- **Assigned PR #1139 (edward: Wave 30 H1 cylindrical coords (r, θ, z))** — FIFTH Wave 30 attack (simplest).
  - **Hypothesis**: Replace Cartesian `(x, y, z)` with cylindrical `(r=√(x²+y²), θ=atan2(y,x), z)` before pos_embed/string_sep. The car has near-mirror symmetry across x-z plane; cylindrical around z makes `z` (the τ_z-relevant axis) a dedicated channel and explicitly separates horizontal layout from altitude.
  - **Theoretical basis**: Equivariant networks (SE(3)-Transformer, EGNN, Tensor Field Networks) consistently improve sample efficiency when input coords align with geometry's symmetry axis.
  - **Cheapest test in Wave 30** — ~35 LOC, one CLI flag `--use-cylindrical-coords`, definitive answer in 18h.

- **Assigned PR #1140 (askeladd: Wave 30 H7 normal-prediction aux head)** — SIXTH Wave 30 attack.
  - **Hypothesis**: Add `Linear(hidden_dim, 3)` aux head predicting input surface normal from each surface backbone-emitted feature; cosine-embedding aux loss weighted at 0.1×. Forces backbone to **retain orientation information** through the stack. Different from H2 (which adds normal Fourier info *into* the model): H7 attacks the **gradient signal** to make orientation legible at every surface token.
  - **Theoretical basis**: Self-supervised aux tasks (DINO, MAE, Point-MAE) consistently preserve representation quality. For τ_z specifically, recovering `n_z` from features makes downstream τ_z regression easier.
  - **Diagnostic**: `train/normal_aux_loss` should drop from ~1.0 (random) to <0.1 at EP13 if mechanism engages.

### Wave 30 fleet status — SIX of EIGHT architectural attack axes now active in parallel

| PR | Student | Mechanism | Attack axis | LOC | EP/Status |
|----|---------|-----------|-------------|-----|-----------|
| #1134 | tanjiro | H6: local-frame WSS head (τ·n=0 by construction) | output-side | ~65 | EP3 gate due ~tomorrow |
| #1136 | nezuko | H2: normal spectral encoding (StringSep on nx,ny,nz) | input-side features | ~35 | EP1-2 |
| #1137 | fern | H5: Y-architecture dual-backbone (cp vs WSS branches) | backbone split | ~80 | EP0-1 |
| **#1138** | **thorfinn** | **H3: normal-aligned slice groups (soft attention routing)** | **attention layer** | **~50** | **EP0 (just launched)** |
| **#1139** | **edward** | **H1: cylindrical coords (r, θ, z) input frame** | **input coord frame** | **~35** | **EP0 (just launched)** |
| **#1140** | **askeladd** | **H7: normal-prediction aux head (gradient signal regularizer)** | **gradient flow** | **~80** | **EP0 (just launched)** |

**Six mechanisms, six layers of the architecture stack, all targeting the τ_z bottleneck:**
- Input coord frame (H1) → Input features (H2) → Attention routing (H3) → Backbone split (H5) → Output decomposition (H6) → Gradient signal (H7)
- If ANY of these breaks the τz/τx ≤ 1.40 wall, the corresponding architectural layer is the structural bottleneck.

Wave 30 reserve (2 of 8 still on the bench): H4 (hard normal routing) and H8 (contrastive orientation regularization). Reserved as next-cohort assignments depending on which of H1/H2/H3/H5/H6/H7 succeeds.

Wave 29 fleet still in flight (3 of 8):

| PR | Student | Mechanism | EP/Status |
|----|---------|-----------|-----------|
| #1122 | alphonse | SDF FAR-field α=2.0 (only SDF stack) | EP10 truncate due |
| #1133 | frieren | per-axis WSS mag decomp | EP3+ alive |
| (PR #972 SOTA: val_abupt=6.126%, test_WSS=6.727%, test_vol_p=3.643%, test_SP=3.577%)| | | |

**Zero idle. Eight students all running.**

### Next-highest-EV gates (post triple-closure + Wave 30 expansion)

| ETA | Event | Action |
|-----|-------|--------|
| ~21:30Z | thorfinn #1138 EP1 smoke (slice-weight entropy logged) | Verify mechanism engages |
| ~21:30Z | edward #1139 EP1 smoke (no-NaN gradient) | Verify cylindrical transform safe |
| ~21:30Z | askeladd #1140 EP1 smoke (`train/normal_aux_loss` drops) | Verify aux head learns |
| ~next-day-AM | tanjiro #1134 EP3 gate | First Wave 30 H6 verdict |
| ~next-day-AM | nezuko #1136 EP3 gate | First Wave 30 H2 verdict |
| ~next-day-AM | fern #1137 EP3 gate | First Wave 30 H5 verdict |
| ~next-day-PM | thorfinn #1138 / edward #1139 / askeladd #1140 EP3 gates | Wave 30 H3/H1/H7 verdicts (staggered) |
| ~next-day | alphonse #1122 EP10 + test eval | SDF FAR-field verdict |

---

## Prior invocation actions (2026-05-15 ~19:45Z) — fern #1126 CLOSED terminal (decoder-depth hypothesis FALSIFIED, **no-SDF ceiling convergence finding**), Wave 30 third architectural experiment launched (PR #1137 H5 Y-architecture)

### Actions this invocation

- **CHECKED human GH issues** — all 4 open issues (#1056, #285, #618, #252) have advisor responses, no new human messages. Primary directive remains: test_WSS < 5.85% with floors test_vol_p ≤ 3.643%, test_SP ≤ 3.577%. NO MORE ENSEMBLES (explicit). Single-model breakthroughs only.

- **CLOSED PR #1126 (fern surface_out depth=4)** at terminal.
  - Test metrics: test_WSS=**6.9886%** (+0.262pp miss), test_vol_p=**3.6452%** (+0.001pp marginal floor regress, statistical tie), test_SP=**3.8335%** (+0.257pp floor regress), val_abupt=6.342%.
  - **Fails 3/4 merge gates.** Decoder-depth-bottleneck hypothesis cleanly FALSIFIED — val τz/τx ratio rose MONOTONICALLY EP1→EP12 (1.341 → 1.546), τ_z was the slowest axis at every epoch, no crossover even in vol_points=65536 regime.
  - Test τz/τx = **1.462** — 10th confirmation of the structural band.

- **CRITICAL MECHANISTIC FINDING from #1126**: This run reveals that **TWO independent capacity uplifts** converge to *exactly* the same no-SDF ceiling within sub-0.001pp:
  - test_WSS: 6.9886% (fern depth=4) vs 6.989% (thorfinn #1100 slices=256) — Δ = −0.0004pp
  - test_vol_p: 3.6452% (fern depth=4) vs 3.6442% (thorfinn #1100) — Δ = +0.001pp
  - test_SP: 3.8335% (fern depth=4) vs 3.8324% (thorfinn #1100) — Δ = +0.001pp

  **This is the strongest evidence yet that the bottleneck is a REPRESENTATION-AXIS bottleneck, not a capacity bottleneck.** Two independent capacity uplifts (backbone width AND decoder depth) hit identical walls. Wave 30's architectural pivot is exactly the right direction.

- **Assigned PR #1137 (fern: Wave 30 H5 Y-architecture dual-backbone)** — THIRD architectural experiment, runs in parallel with tanjiro #1134 H6 and nezuko #1136 H2.
  - **Hypothesis**: Split backbone after first encoder layer into parallel pressure-branch (cp, vol_p) and WSS-branch (τx, τy, τz) transformer stacks. Tests task-interference hypothesis: does shared backbone optimization favor pressure over WSS optimization, leaving τ_z as residual?
  - **Theoretical basis**: Pressure (potential/irrotational) and WSS (rotational/viscous) correspond to different physical modes. Cross-Stitch Networks (Misra 2016), Multi-Task Learning Survey (Vandenhende 2021) show Y-arch outperforms single-backbone multi-task models when tasks have structurally distinct optimal representations.
  - **Implementation**: ~80 LOC in `model.py`, single CLI flag `--y-arch-split-layer 1`. With 5 layers and split-at-1, total params ~34M (1.8× baseline ~17.4M). Expected throughput drop +30%.
  - **Branch separation diagnostic**: cos_sim between pressure-branch and WSS-branch surface features at EP3/EP10/EP13. <0.7 = branches diverging as intended; ~0.99 = branches collapsed (would need split_layer=2 retry).
  - **Falsifiability**: BIG WIN if test τz/τx ≤ 1.40. MERGE if test_WSS < 6.727% with both floors. INTERESTING NULL if branches diverge but metrics don't move (rules out task-interference). FLAT NULL if branches collapse (retry with split_layer=2/3).

### Wave 30 fleet status — THREE architectural attack axes now active in parallel

| PR | Student | Mechanism | Attack axis | EP/Status |
|----|---------|-----------|-------------|-----------|
| #1134 | tanjiro | H6: local-frame WSS head (τ·n=0 by construction) | output-side | EP0 (launched ~16:30Z) |
| #1136 | nezuko | H2: normal spectral encoding (StringSep on nx,ny,nz) | input-side | EP0 (launched ~19:00Z) |
| **#1137** | **fern** | **H5: Y-architecture dual-backbone (cp vs WSS branches)** | **backbone-side** | **EP0 (just launched)** |

If τ_z bottleneck is at the output → H6 wins. If at the input → H2 wins. If at shared backbone optimization → H5 wins. **Three orthogonal hypotheses, three independent students, parallel execution.**

Wave 29 mid-late fleet still in flight:

| PR | Student | Mechanism | EP/Status |
|----|---------|-----------|-----------|
| #1116 | edward | per-channel WSS output heads | terminal imminent |
| #1122 | alphonse | SDF FAR-field α=2.0 (only SDF stack) | EP10 truncate due |
| #1127 | askeladd | surface_loss warmup curriculum | terminal imminent |
| #1128 | thorfinn | τ_z loss weight 3.0 | terminal imminent (val_abupt 6.31% at EP9) |
| #1133 | frieren | per-axis WSS mag decomp | EP3+ (alive, checked) |

**Zero idle. Eight students all running.**

### Wave 30 architectural roadmap — three of eight active

| Rank | ID | Hypothesis | LOC | Risk | Status |
|------|----|------------|-----|------|--------|
| 1 | H6 | Local-frame WSS head (τ·n=0 by construction) | ~65 | LOW | **tanjiro PR #1134 ACTIVE** |
| 2 | H2 | Normal spectral encoding (Fourier basis on normals) | ~35 | LOW | **nezuko PR #1136 ACTIVE** |
| 3 | H5 | Y-architecture dual-backbone | ~80 | MEDIUM | **fern PR #1137 ACTIVE** |
| 4 | H3 | Normal-aligned slice groups (soft routing) | ~50 | MEDIUM | reserve |
| 5 | H4 | Hard normal routing (dedicated τz slice partition) | ~70 | MEDIUM | reserve |
| 6 | H1 | Cylindrical coords (r, θ, z) input | ~35 | LOW | reserve |
| 7 | H7 | Normal-prediction auxiliary head | ~80 | MEDIUM | reserve |
| 8 | H8 | Contrastive orientation regularization | ~80 | MEDIUM | reserve |

Full details in `research/RESEARCH_IDEAS_2026-05-15_18:00.md`.

### Next-highest-EV gates

| ETA | Event | Action |
|-----|-------|--------|
| ~20:00Z | thorfinn #1128 EP13 terminal | First merge-eligible Wave 29 single-model candidate |
| ~20:00Z | edward #1116 EP13 terminal | Per-channel heads verdict |
| ~20:45Z | alphonse #1122 EP10 + test eval | SDF FAR-field verdict |
| ~21:00Z | askeladd #1127 EP13 terminal | Surface-loss warmup curriculum verdict |
| ~tomorrow | tanjiro #1134 EP3 gate | First Wave 30 H6 verdict |
| ~tomorrow | nezuko #1136 EP3 gate | First Wave 30 H2 verdict |
| ~tomorrow | fern #1137 EP3 gate | First Wave 30 H5 verdict |

---

## Prior invocation actions (2026-05-15 ~19:00Z) — nezuko #1125 CLOSED terminal, Wave 30 second architectural experiment launched (PR #1136 H2 normal spectral encoding)

### Actions this invocation

- **CLOSED PR #1125 (nezuko spatial-prior α=10)** at terminal.
  - Test metrics: test_WSS=**7.106%** (+0.379pp miss vs 6.727%), test_vol_p=**3.634%** (PASS, fleet-best margin 0.009pp below floor), test_SP=**3.954%** (+0.377pp floor regress), val_abupt=6.390%.
  - **Fails 3/4 merge gates.** Spatial-prior α=10 was too aggressive vs the prior α=5 sweet spot. Test_SP +0.377pp is the dealbreaker.
  - Test τz/τx = **1.449** (val 1.549) — val→test compression of 0.10 units consistent with tanjiro #1124. 9th confirmation of the structural bottleneck.
  - **Signal preserved**: test_vol_p=3.634% is fleet-best on volume pressure — spatial-prior remains an orthogonal mechanism for vol_p improvement worth stacking later. α=5 remains the sweet spot.

- **POSTED check-in on PR #1133 (frieren per-axis mag decomp)** to clear `stale_wip` flag. W&B run `5l9i6fjn` verified alive at EP2.72, step 29,268, heartbeat 0.2 min — false-positive between-epoch silence. Open question to frieren on whether `mag_z_loss`/`mag_xy_loss`/`calib_ratio` in W&B summary are epoch-boundary-only logging (grads on aux heads are non-zero, suggesting the heads ARE training).

- **Assigned PR #1136 (nezuko: Wave 30 H2 normal spectral encoding)** — SECOND architectural experiment of Wave 30, runs in parallel with tanjiro #1134 H6.
  - **Hypothesis**: Surface normals (nx, ny, nz) currently pass through a single linear projection while positions (x, y, z) get full per-axis spectral basis via `StringSeparableEncoding`. Closing this obvious asymmetry — apply the same spectral encoding to normals — should improve orientation-conditional features, especially for τ_z (which depends most heavily on patch orientation).
  - **Theoretical basis**: Fourier features for directional quantities on the sphere are classical in physics. Recent geometric DL work (NequIP, Equiformer, Clifford Neural Layers) consistently shows spectral treatment of directional inputs outperforms linear projection on directional output quantities.
  - **Implementation**: ~35 LOC in `model.py`, single CLI flag `--normal-spectral-encoding`. Volume path unchanged (only SDF, no normals). Surface extras split into normals (indices 3:6) + area (index 6:7); spectral encoding applied to normals, area concatenated as-is.
  - **Falsifiability**: MAJOR WIN if test τz/τx ≤ 1.40. MERGE if test_WSS < 6.727% with both floors. FAIL if τz/τx stays > 1.47 (rules out input-side orientation encoding as the bottleneck).
  - **Orthogonality to H6**: H6 tests output-side decomposition (local-frame WSS head); H2 tests input-side representation (normal Fourier features). If BOTH fail, the bottleneck is in the middle (backbone attention) → H5 Y-architecture next.

### Wave 30 fleet status (2 architectural experiments active in parallel)

| PR | Student | Mechanism | Tier | EP/Status |
|----|---------|-----------|------|-----------|
| #1134 | tanjiro | H6: local-frame WSS head (τ·n=0 by construction) | output-side | EP0 (just launched) |
| #1136 | nezuko | H2: normal spectral encoding (StringSep on nx,ny,nz) | input-side | EP0 (just launched) |

Wave 29 mid-late fleet still in flight:

| PR | Student | Mechanism | EP/Status |
|----|---------|-----------|-----------|
| #1116 | edward | per-channel WSS output heads | ~EP10+ (terminal imminent) |
| #1122 | alphonse | SDF FAR-field α=2.0 (only SDF stack) | EP10 truncate due |
| #1126 | fern | surface_out depth=4 | ~EP10+ (terminal soon) |
| #1127 | askeladd | surface_loss warmup curriculum | ~EP8 |
| #1128 | thorfinn | τ_z loss weight 3.0 | terminal imminent (EP13) |
| #1133 | frieren | per-axis WSS mag decomp | EP2.72 |

**Zero idle. Eight students all running.**

### Wave 30 architectural roadmap progress

| Rank | ID | Hypothesis | LOC | Risk | Status |
|------|----|------------|-----|------|--------|
| 1 | H6 | Local-frame WSS head (τ·n=0 by construction) | ~65 | LOW | **tanjiro PR #1134 ACTIVE** |
| 2 | H2 | Normal spectral encoding (Fourier basis on normals) | ~35 | LOW | **nezuko PR #1136 ACTIVE** |
| 3 | H5 | Y-architecture dual-backbone (split cp vs WSS branches) | ~80 | MEDIUM | reserve for next idle |
| 4 | H3 | Normal-aligned slice groups (soft routing) | ~50 | MEDIUM | reserve |
| 5 | H4 | Hard normal routing (dedicated τz slice partition) | ~70 | MEDIUM | reserve |
| 6 | H1 | Cylindrical coords (r, θ, z) input | ~35 | LOW | reserve |
| 7 | H7 | Normal-prediction auxiliary head | ~80 | MEDIUM | reserve |
| 8 | H8 | Contrastive orientation regularization | ~80 | MEDIUM | reserve |

Full details in `research/RESEARCH_IDEAS_2026-05-15_18:00.md`.

### Next-highest-EV gates (post nezuko #1125 close + H2 launch)

| ETA | Event | Action |
|-----|-------|--------|
| ~20:00Z | thorfinn #1128 EP13 terminal | First merge-eligible Wave 29 single-model candidate (val_abupt currently 6.31%) |
| ~20:00Z | fern #1126 EP13 + test eval | Decoder-depth verdict (val_abupt 6.36% at EP9, new best) |
| ~20:00Z | edward #1116 EP13 terminal | Per-channel heads verdict |
| ~20:45Z | alphonse #1122 EP10 + test eval | SDF FAR-field verdict |
| ~21:00Z | askeladd #1127 EP13 terminal | Surface-loss warmup curriculum verdict |
| ~next day | tanjiro #1134 EP3 gate | First Wave 30 H6 architectural verdict |
| ~next day | nezuko #1136 EP3 gate | First Wave 30 H2 architectural verdict |

---

## Prior invocation actions (2026-05-15 ~16:30Z) — tanjiro #1124 CLOSED terminal, Wave 30 architectural pivot launched (PR #1134 H6 local-frame WSS head)

### Actions this invocation

- **CLOSED PR #1124 (tanjiro EMA decay 0.9995)** at terminal EP13.
  - Test metrics: test_WSS=**6.898%** (+0.171pp vs PR #972 6.727%), test_vol_p=**3.666%** (+0.023pp floor regress), test_SP=**3.811%** (+0.234pp floor regress), val_abupt=6.221%.
  - **Fails all 4 merge gates.** Slow-decay EMA hypothesis cleanly REFUTED (EMA-vs-raw Δ on τ_z peaked at EP3 +0.937pp and shrank to +0.021pp at EP13 — 98% shrinkage from peak; opposite of predicted growth).
  - Test τz/τx = **1.469** — 8th confirmation of the structural bottleneck pattern (val:test compression observed for first time).
  - Run quality high: `best_checkpoint/updated=1` at every recent epoch gate, pure monotonic descent.

- **Generated Wave 30 architectural roadmap** via researcher-agent. Output: `research/RESEARCH_IDEAS_2026-05-15_18:00.md` — 8 candidate architectural hypotheses, scored on Mechanism×Risk×EV.

- **Assigned PR #1134 (tanjiro: Wave 30 H6 local-frame WSS head)** — FIRST architectural experiment of Wave 30.
  - **Hypothesis**: Replace global Cartesian (τ_x, τ_y, τ_z) head with local-frame (τ_t1, τ_t2) head using orthonormal surface basis from Gram-Schmidt of surface normals. Reconstructs `τ_global = τ_t1·t1 + τ_t2·t2`, which automatically satisfies physics constraint **τ·n=0**.
  - **Theoretical basis**: WSS is by definition the tangential component of the wall stress tensor; τ·n=0 ALWAYS. Current Cartesian head doesn't enforce this and must learn it from data — the eightfold structural finding is consistent with the model being unable to learn this constraint reliably.
  - **Implementation**: ~65 LOC in `model.py`, single CLI flag `--local-frame-wss-head`, no loader changes (normals already in `surface_x[..., 3:6]`).
  - **Pre-flight diagnostic**: compute `mean(|τ·n|/|τ|)` on baseline predictions. Expect >0.01 (constraint violated). If <0.01, hypothesis is wrong (close immediately).
  - **Falsifiability**: MAJOR WIN if test_τz/τx ≤ 1.40 (first mechanism to break structural pattern). MERGE if test_WSS<6.727% with both floors. FAIL if val_τz/τx > 1.45 at EP13 (bottleneck deeper than output head).
  - **Falsification value**: if H6 fails, the bottleneck is NOT at the output head — points to backbone slice-attention (H3) or full backbone replacement (H5 Y-architecture) as next bets.

### Active fleet — Wave 29 (6 students still mid-late-EP) + Wave 30 (tanjiro starting)

| PR | Student | Mechanism | EP/Status |
|----|---------|-----------|-----------|
| #1116 | edward | per-channel WSS output heads | ~EP9-10 (slope shallow, val_abupt 6.34%) |
| #1122 | alphonse | SDF FAR-field α=2.0 (only SDF stack) | EP4 MARGINAL → EP10 truncate |
| #1125 | nezuko | spatial-prior α=10 | ~EP6 (val_abupt 6.40%) |
| #1126 | fern | surface_out depth=4 | ~EP9-10 (val_abupt 6.36% new best at EP9) |
| #1127 | askeladd | surface_loss warmup curriculum | ~EP6 (val_abupt 6.48%) |
| #1128 | thorfinn | τ_z loss weight 3.0 | ~EP9 (val_abupt 6.31%, ratio asymptoted 1.539) |
| #1133 | frieren | per-axis WSS mag decomp | EP1.32 (just launched) |
| **#1134** | **tanjiro** | **Wave 30 H6: Local-frame WSS head** | **EP0 (just launched)** |

**Zero idle.** Eight students all running.

### Wave 30 architectural roadmap (researcher-agent output, top-3)

| Rank | ID | Hypothesis | LOC | Risk | Status |
|------|----|------------|-----|------|--------|
| 1 | H6 | Local-frame WSS head (τ·n=0 by construction) | ~65 | LOW | **tanjiro PR #1134 ACTIVE** |
| 2 | H2 | Normal spectral encoding (give normals Fourier basis like positions) | ~35 | LOW | reserve for next idle |
| 3 | H5 | Y-architecture dual-backbone (split cp vs WSS branches) | ~80 | MEDIUM | reserve for next idle |

The remaining 5 ideas (H1, H3, H4, H7, H8) are in `research/RESEARCH_IDEAS_2026-05-15_18:00.md`.

### Next-highest-EV gates (post tanjiro #1124 close)

| ETA | Event | Action |
|-----|-------|--------|
| ~16:25Z (passed) | alphonse #1122 EP6 readout | Slope continuation; if hit, reassess EP10 truncate |
| ~18:00Z | thorfinn #1128 EP13 terminal | First merge-eligible single-model candidate of remaining fleet |
| ~19:30Z | fern #1126 EP13 + test eval | Decoder-depth verdict |
| ~17:30Z | frieren #1133 EP3 gate | mag_z_loss + mag_xy_loss separation |
| ~20:45Z | alphonse #1122 EP10 + test eval | SDF FAR-field verdict + budget-extension request |
| ~next day | tanjiro #1134 EP3 gate | First Wave 30 architectural verdict |

---

## Prior invocation actions (2026-05-15 ~15:10Z) — τ_z structural finding SEVENFOLD confirmed (alphonse EP4), tanjiro #1124 leading fleet

### Verified fleet metrics (2026-05-15 ~15:05Z, GraphQL + W&B parallel pulls)

| Rank | PR | Student | Mechanism | W&B run | EP | val_abupt | val_WSS | vol_p | SP | τz/τx | best_ckpt |
|------|----|---------|-----------|---------|----|-----------|---------|-------|----|-------|-----------|
| **1** | #1124 | tanjiro | EMA decay 0.9995 | `mw6d04kc` | 6.25 | **6.2499%** | **7.058%** | 3.706% | **4.119%** | 1.555 | ✅ updated every gate |
| 2 | #1128 | thorfinn | τ_z weight 3.0 | `uwqybod5` | 8 | 6.307% | 7.130% | 3.716% | 4.184% | 1.539 | asymptote |
| 3 | #1116 | edward | per-channel heads | `3ufrbxl6` | 9 | 6.340% | 7.154% | 3.805% | 4.163% | 1.551 | slope shallowing |
| 4 | #1126 | fern | surface_out d=4 | `gr9ht3h5` | 9 | 6.360% | 7.193% | (sync lag) | (sync lag) | 1.543 | new best at EP9 |
| 5 | #1125 | nezuko | spatial-prior α=10 | `rp1op3z6` | 8 | 6.470% | 7.248% | 3.727% | 4.267% | 1.548 | EP8 PASS |
| 6 | #1127 | askeladd | surface_loss warmup | `ag1dnelx` | ~8 | 6.485% | 7.323% | 3.824% | 4.266% | 1.559 | mid-curr |
| 7 | #1122 | alphonse | SDF FAR-field α=2.0 | `vvv84p32` | **4** | 6.886% | 7.668% | 4.602% | 4.431% | 1.526 | EP4 best |
| 8 | #1133 | frieren | per-axis mag decomp | `5l9i6fjn` | 1.32 | 31.55% (EP1) | 35.69% | 16.89% | 24.17% | 1.388 | EP1 healthy |

### CRITICAL: τ_z structural finding SEVENFOLD CONFIRMED — architectural pivot signal

Eight active mechanisms tested:

| Mechanism | EP | τz/τx | Verdict |
|-----------|----|-------|---------|
| EMA 0.9995 (tanjiro) | 6.25 | 1.555 | in band |
| τ_z weight 3.0 (thorfinn) | 8 | 1.539 (asymptoted) | in band |
| per-channel heads (edward) | 9 | 1.551 | in band |
| surface_out d=4 (fern) | 9 | 1.543 | in band |
| spatial-prior α=10 (nezuko) | 8 | 1.548 | in band |
| surface_loss warmup (askeladd) | 8 | 1.559 | in band |
| **SDF FAR-field α=2.0 (alphonse)** | **4** | **1.526** | **in band — 7th confirmation** |
| mag-only decomp (frieren #1121, closed) | 12 | 1.570 | in band — terminal |
| per-axis mag decomp (frieren #1133) | 1.32 | 1.388 → TBD | **8th and final loss/data-side test** |

**The τ_z/τ_x ratio converges to 1.50–1.57 across:**
- loss weighting (×3 vs ×1)
- sampling bias (spatial-prior + SDF FAR-field)
- output capacity (per-channel decoupled heads)
- decoder depth (surface_out d=2 → d=4)
- temporal averaging (EMA 0.999 vs 0.9995)
- magnitude calibration (frieren #1121 mag-only aux head)
- input weighting curriculum (askeladd surface_loss warmup)

**Conclusion**: τ_z bottleneck is **NOT** addressable by ANY data-side or loss-side intervention. The mechanism is backbone-representation-side. Once frieren #1133 (per-axis mag decomp, the 8th and final loss-side test) confirms or breaks this pattern, we commit to Wave 30 architectural experiments:

**Wave 30 architectural roster (proposed)**:
1. **Coordinate-system change**: 3D Cartesian (x,y,z) → cylindrical (r,θ,z) or vehicle-body frame (longitudinal/lateral/vertical). τ_z is "vertical wall-shear" — if the backbone is encoding all three axes in shared Cartesian features, a coordinate system aligned with the dominant flow direction would give τ_z its own preferred basis direction.
2. **Per-axis attention heads in the backbone**: split Transolver attention layers into per-axis sub-tensors after a specified layer, so τ_z gets dedicated attention rather than competing with τ_x/τ_y for shared head capacity.
3. **Dedicated τ_z encoder branch (Y-architecture)**: parallel branch from a mid-network feature layer that processes only τ_z magnitude prediction, with separate normalization and MLP depth.
4. **Mixture-of-Experts on the surface head**: K experts, each with output specialization on one axis or feature.

### Tanjiro #1124 = leading single-model candidate on no-SDF tay

- EP6.25 val_abupt=6.2499% (-0.076pp from EP5.75) with `best_checkpoint/updated=1` at every recent gate
- EP13 projection (conservative slope-shallowing): **val_abupt 5.88–6.03%**
- Would beat 6.126% baseline by 0.10–0.24pp
- Floor risk at val_SP=4.119% (frieren #1121 closed at val_SP=4.218% → test_SP=3.734% +0.157pp regress)
- **Critical request posted to tanjiro**: report EP12 best-EMA-checkpoint metrics specifically (not EP13 final)

### Alphonse #1122 truncation decision (just posted)

- EP10 truncation confirmed (cumulative 17h57m, ~20min safety margin)
- Test-eval +45min budget extension conditionally granted (single highest-EV SDF experiment)
- Standing instructions: report at EP6 (mid-vol=32k) and EP9 (end-vol=49k); interrupt me only if τz/τx <1.45

### Actions this invocation
- Posted EP4 truncation decision to alphonse #1122 (with SDF FAR-field τ_z confirmation)
- Pulled tanjiro #1124 + frieren #1133 W&B states
- Posted leadership-ack + EP12 best-EMA harvest instruction to tanjiro #1124
- Posted launch confirmation + mag_z/mag_xy diagnostic ask to frieren #1133
- Survey via GraphQL (REST API rate-limited until ~15:19Z)

### Next-highest-EV gates

| ETA | Event | Action |
|-----|-------|--------|
| ~15:08Z | edward #1116 EP10 val (first vol=65k epoch) | Watch slope reacceleration |
| ~15:35Z | thorfinn #1128 EP9-10 (vol curriculum bump) | Watch τ_z reduction at higher vol |
| ~16:00Z | tanjiro #1124 EP7 (advisor request) | val_abupt + val_SP + per-axis report |
| ~16:25Z | alphonse #1122 EP6 readout | Slope continuation check |
| ~17:30Z | frieren #1133 EP3 gate | mag_z_loss and mag_xy_loss separation diagnostic |
| ~18:00Z | thorfinn #1128 EP13 terminal | First merge-eligible candidate |
| ~19:00Z | tanjiro #1124 EP12 best-EMA harvest | **Highest merge-priority gate** |
| ~19:30Z | fern #1126 EP13 + test eval | Decoder-depth verdict |
| ~20:45Z | alphonse #1122 EP10 + test eval | SDF FAR-field verdict + budget-extension request |

---

## Prior invocation actions (2026-05-15 ~12:55Z) — Wave 29 mid-late EP fleet status, edward #1116 terminal imminent

### Verified fleet metrics from W&B (2026-05-15 ~12:50Z)

| Rank | PR | Student | Mechanism | W&B run | EP | val_abupt | val_WSS | vol_p | SP | τz/τx |
|------|----|---------|-----------|---------|----|-----------|---------|-------|----|-------|
| **1** | #1124 | tanjiro | EMA decay 0.9995 | `mw6d04kc` | ~6.2 | **6.228%** | **7.030%** | 3.704% | **4.110%** | 1.554 |
| 2 | #1128 | thorfinn | τ_z loss weight 3.0 | `uwqybod5` | ~5.45 | 6.307% | 7.130% | 3.716% | 4.184% | 1.539 |
| 3 | #1116 | edward | per-channel WSS heads | `3ufrbxl6` | **~12.6/13** | 6.340% | 7.150% | 3.810% | 4.160% | 1.551 |
| 4 | #1126 | fern | surface_out depth=4 | `gr9ht3h5` | ~9.22 | 6.360% | 7.193% | 3.762% | 4.218% | 1.543 |
| 5 | #1125 | nezuko | spatial-prior α=10 | `rp1op3z6` | ? | 6.404% | 7.248% | 3.727% | 4.267% | 1.548 |
| 6 | #1127 | askeladd | surface_loss warmup | `ag1dnelx` | ? | 6.485% | 7.323% | 3.824% | 4.266% | 1.559 |
| 7 | #1122 | alphonse | SDF FAR-field α=2.0 | `vvv84p32` | EP3 | 7.168% | 8.002% | 4.665% | 4.684% | 1.515 |
| 8 | #1133 | frieren | per-axis mag decomp | TBD | EP0 | (just launched) | — | — | — | — |

### Critical observations

**1. Edward #1116 terminal imminent** (~13:30–13:50Z): at EP12.6/13, walltime 675.7 min = 11.26h. Final EP13 + test eval expected within 30-50 min. Edward's val_abupt=6.34% is unlikely to beat the 6.126% baseline at terminal — slope has flattened. **Most likely outcome: close (no improvement) or send back for variation.**

**2. Tanjiro #1124 leads the fleet at val_abupt 6.228% at EP~6.2.** This is the slowest-EMA experiment in the fleet (EMA 0.9995 vs default 0.999, half-life 1386 vs 693 steps). Comparison to frieren #1121 terminal trajectory: frieren EP6 was 6.397%, terminal best-EMA EP12=6.073%. If tanjiro tracks similarly, terminal projection lands ~5.95–6.05% val_abupt — **would beat 6.126% baseline by 0.07–0.18pp**. Highest current single-model contender on no-SDF tay.

**3. τ_z/τ_x ratio confirmed SEVENFOLD-EIGHTFOLD structural** (now including: tanjiro EMA 1.554, thorfinn τ_z×3 1.539, edward per-channel 1.551, fern depth=4 1.543, nezuko spatial-prior 1.548, askeladd warmup 1.559, alphonse SDF FAR-field 1.515 at early EP3, frieren #1121 closed 1.570). Ratio converges to ~1.50–1.57 across ALL mechanisms. **τ_z bottleneck is NOT addressable by loss weighting, sampling, output capacity, EMA, magnitude calibration, or input-bias re-weighting (SDF FAR-field is the latest test).** Architectural pivot required if alphonse and frieren #1133 also confirm.

**4. Alphonse #1122 SDF FAR-field α=2.0 EP3 MARGINAL** at 7.168% val_abupt — already responded with budget-management guidance (prefer EP12 truncate over skip-eval) and τ_z/τ_x ratio monitoring ask for EP4. EP4 readout ~14:30Z is the cleanest mechanism test (vol curriculum bump from 16k→32k).

**5. Floor analysis (val→test mapping):** PR #972 baseline floors are test_vol_p ≤3.643%, test_SP ≤3.577%. Val→test compression typically ~0.10pp (frieren #1121 was val_vol_p=3.517% → test=3.545%, val_SP=4.218% → test=3.734%). Current fleet:
   - tanjiro val_SP=4.110% → test projection ~3.63% (close to floor)
   - All other students val_SP ≥4.16% → test projection ≥3.7% (above floor)
   - **Multiple runs at risk of test_SP floor regression at terminal.**

### Action this invocation
- Verified fleet state via parallel W&B pulls (tanjiro/nezuko/askeladd in one batch; thorfinn/fern/edward in three parallel agents).
- Updated state doc with current EP positions and metrics.
- Responded to alphonse #1122 EP3 MARGINAL with EP4 monitoring ask.
- Issue #1056 status posted at 12:53Z (via check-human-issues).
- Schedule ~35min wakeup for edward #1116 terminal.

### Next-highest-EV events (ordered by ETA)

| ETA | Event | Action |
|-----|-------|--------|
| ~13:30–13:50Z | **edward #1116 terminal** (EP13 + test eval) | Review terminal; merge if test_WSS<6.85% AND floors held, else close/back |
| ~14:30Z | **alphonse #1122 EP4 readout** (vol curriculum bump 16k→32k) | Monitor τz/τx; <1.49 = SDF FAR-field breaks structural pattern |
| ~14:30–15:00Z | thorfinn #1128 EP6/7 gate | Check slope; tanjiro EP6.5 reference |
| ~15:30Z | fern #1126 EP10/11 | Late-EP slope check |
| ~16:30–17:30Z | alphonse #1122 EP5/6 | Curriculum-shift convergence check |
| ~17:00–18:00Z | tanjiro #1124 EP10 | Best-EMA crossover prediction |
| ~18:00–20:00Z | tanjiro #1124 / thorfinn #1128 terminal | First merge-eligible single-model candidates |

---

## Prior invocation actions (2026-05-15 ~12:45Z) — Frieren #1121 closed terminal, reassigned to #1133 per-axis-mag decomp

### Actions this invocation

- **Closed PR #1121 (frieren mag-only decomp + 18h)** at terminal EP13.
  - Test metrics: test_WSS=**6.859%** (+0.132pp vs PR #972 SOTA, but **−0.137pp vs no-decomp #1078**), test_vol_p=3.545% PASS, **test_SP=3.734% (+0.157pp FLOOR REGRESS)** ❌, test_abupt=5.939%.
  - Val: val_abupt=**6.073%** (−0.053pp vs PR #972 6.126% baseline) — **first single-model val_abupt improvement on no-SDF tay** since the corrected split landed.
  - Methodology success: mag head perfectly calibrated (ratio 0.9993, mag_loss 0.0011, 4.4× tighter than #1112 EP3). λ_dir=0 confirmed throughout.
  - **Why close**: test_SP floor regression is a merge blocker; single-model winners must hold both floors. Methodology preserved as strong building block for stacking (most natural pairing: SDF FAR-field α=2.0 ←→ alphonse #1122).
  - **τ_z structural finding strengthened to SIXFOLD confirmation**: this is the 6th active mechanism (loss weight, sampling, output capacity, EMA, mag-only decomp, per-channel heads) converging to τz/τx ratio ~1.50–1.57 by EP5-10. EP9→EP10 τ_z reversal (+0.020pp) is the cleanest single-run instance. τ_z bottleneck is **NOT** addressable by these levers.

- **Assigned PR #1133 (frieren: per-axis WSS magnitude decomp + 18h)** — direct architectural attack on τ_z structural finding.
  - **Hypothesis**: split mag aux head into `surface_mag_z_aux` (predicts |τ_z|) and `surface_mag_xy_aux` (predicts ||τ_xy||₂) as SEPARATE heads. Tests whether mag-only's success was bandwidth-limited (single head must encode all three axes' magnitudes) vs. representational (backbone features can't carry τ_z).
  - **Loss**: `L = L_base + λ_mag_z * MSE(|τ_z_pred|, |τ_z_gt|) + λ_mag_xy * MSE(||τ_xy_pred||₂, ||τ_xy_gt||₂)`
  - **Asymmetric defaults**: λ_mag_z=0.1, λ_mag_xy=0.05 — emphasize τ_z bottleneck.
  - **CLI flags**: `--wss-decomp-method per-axis-mag --wss-decomp-lambda-mag-z 0.1 --wss-decomp-lambda-mag-xy 0.05`
  - **Win signal**: test_τ_z ≤ 8.50% (vs #1121's 8.873%, ≥0.37pp improvement). Reach: test_WSS<6.85% AND test_SP≤3.577% AND test_vol_p≤3.643% AND val_abupt≤6.20% → first single-model merge on tay since SDF stack.
  - **Falsifiability**: test_τ_z ≥ 8.80% would confirm τ_z bottleneck is BACKBONE-side (no aux-head decomp can rescue) and force pivot to coordinate-system or attention-mechanism changes.
  - 18h budget (`SENPAI_TIMEOUT_MINUTES=1100`), DDP 8 GPU, group `frieren-per-axis-mag-decomp`. Branch `frieren/per-axis-wss-mag-decomp-18h`.

### Active fleet (7 students still in WIP from Wave 29 + frieren just reassigned)

| PR | Student | Status |
|----|---------|--------|
| #1116 | edward | active — per-channel WSS output heads (τ_x/τ_y/τ_z), 18h |
| #1122 | alphonse | active — SDF FAR-field α=2.0 corrected mechanism |
| #1124 | tanjiro | active — EMA decay 0.9995, 18h |
| #1125 | nezuko | active — spatial-prior surface sampling α=10, 18h |
| #1126 | fern | active — deeper surface_out MLP (depth 2→4), 18h |
| #1127 | askeladd | active — surface_loss warmup curriculum, 18h |
| #1128 | thorfinn | active — τ_z loss weight 3.0, 18h |
| #1133 | frieren | NEW — per-axis WSS magnitude decomp, 18h |

**Zero idle.** Fleet remains at full 8 active.

### Highest-EV next event

- **alphonse #1122 EP3 gate** (~07:55Z if recipe held pace; verify W&B `vvv84p32` actual EP) — this is the corrected SDF FAR-field α=2.0 mechanism, the only SDF-stacked experiment in flight. Hit signal: ≤6.9% PASS / ≤7.2% MARGINAL. Largest expected uplift in the fleet.
- After alphonse EP3, monitor EP5 gates fanning in for fern/askeladd/edward/thorfinn/nezuko in 06:00–08:30Z window.

---

## Prior invocation actions (2026-05-15 ~06:30Z) — Wave 29 EP gate monitoring, fleet-wide τ_z structural finding confirmed

### Fleet-wide EP gate status (2026-05-15 ~06:30Z)

| PR | Student | W&B run | Current EP | Latest val_abupt | Latest WSS | τz/τx | vol_p | Gate Status |
|----|---------|---------|-----------|---------|---------|-------|-------|-------------|
| #1121 | frieren | `gljtmuvs` | EP8.67 | **6.0782%** | **6.8775%** | 1.570 | **3.527%** | EP8 PASS ✓ — LEADING RUN |
| #1122 | alphonse | `vvv84p32` | EP2.18 | 8.2300% | 9.0683% | 1.541 | 5.479% | EP3 gate pending (~163 min from 06:30Z) |
| #1124 | tanjiro | `mw6d04kc` | EP6.21 | 6.3963% | 7.2069% | 1.547 | 3.831% | EP6 MARGINAL (0.096pp above ≤6.3% PASS); EP7 gate: ≤6.3% PASS |
| #1125 | nezuko | `rp1op3z6` | EP5.19 | 6.7039% | 7.6024% | 1.516 | 3.897% | EP5 PASS (≤7.2%); EP8 gate pending |
| #1126 | fern | `gr9ht3h5` | EP4.88 | 6.6062% | 7.4646% | 1.519 | 3.924% | EP4 MARGINAL; EP5 gate imminent |
| #1127 | askeladd | `ag1dnelx` | EP4.91 | 6.7613% | 7.6589% | 1.526 | 3.966% | EP4 MARGINAL; EP5 gate imminent |
| #1116 | edward | `3ufrbxl6` | EP4.59 | 6.5968% | 7.4533% | 1.537 | 3.925% | EP4 PASS; EP5 gate approaching |
| #1128 | thorfinn | `uwqybod5` | EP4.14 | 6.5675% | 7.4273% | 1.513 | 3.880% | EP4 MARGINAL; EP5 approaching |

### CRITICAL FLEET-WIDE FINDING: τ_z bottleneck is STRUCTURAL

ALL τ_z-targeted interventions have FAILED to suppress τz/τx ratio. Every agent's ratio monotonically rises to ~1.50–1.57 by EP5-8 regardless of approach:
- nezuko α=10: 1.371→1.516 by EP5
- thorfinn τz_weight=3.0: 1.288→1.513 by EP4 (transient EP1 suppression only)
- edward per-channel heads: 1.400→1.537 by EP4
- frieren mag-only: 1.389→1.570 by EP8.5 (stabilizing)
- tanjiro EMA 0.9995: 1.454→1.547 by EP6

**Conclusion**: τ_z bottleneck is NOT addressable by loss weighting, sampling, or output capacity. Requires architectural solution targeting the τ_z representational bottleneck (e.g., coordinate system change, dedicated physics-informed τ_z head with orthogonal basis, or attention mechanism change).

### Gate comments posted this invocation
- **Frieren EP8 PASS** → EP10 gate: val_WSS ≤6.80% PASS / ≤6.85% MARGINAL / >6.85% KILL
- **Tanjiro EP6 MARGINAL** → EP7 gate: val_abupt ≤6.3% PASS / 6.3-6.5% MARGINAL / >6.5% KILL
- **Alphonse EP2 progress** → EP3 gate: ≤6.9% PASS / 6.9-7.2% MARGINAL / >7.2% KILL; vol_p ≤4.5%
- **Fern EP4 MARGINAL** → EP5 gate: ≤6.5% PASS / ≤6.8% MARGINAL / >6.8% KILL
- **Askeladd EP4 MARGINAL** → EP5 gate: ≤6.5% PASS / ≤6.8% MARGINAL / >6.8% KILL
- **Edward EP4 PASS** → EP5 gate: ≤6.5% PASS / ≤7.0% MARGINAL / >7.0% KILL
- **Thorfinn EP4 MARGINAL** → EP6 gate: ≤6.5% PASS / ≤6.8% MARGINAL / >6.8% KILL

---

## Latest invocation actions (2026-05-15 03:48–04:00Z) — Wave 29 full fleet confirmed active, all 8 students running

- **Closed PR #1123 (thorfinn τ_z dedicated subnet)** — zero student activity after 4+ hours, four advisor check-in messages unanswered. Pod confirmed idle (1/1 READY via kubectl). Hypothesis is sound but requires code implementation; reassigned pod to a zero-code-change experiment to eliminate implementation failure mode.
- **Assigned PR #1128 (thorfinn: τ_z loss weight escalation 2.0→3.0)** — pure CLI flag change `--tau-z-loss-weight 3.0`, no model code changes. Directly attacks dominant error axis (test_τ_z ≈ 9.05–10.1% across all no-SDF runs). Pass signal: τ_z/τ_x ratio at EP13 < 1.5 (down from ~1.6–1.7 baseline). Full 18h budget (SENPAI_TIMEOUT_MINUTES=1100). W&B run `uwqybod5` (group `tau-z-loss-weight-3p0`), launched 03:48:42Z.
- **thorfinn #1128 confirmed launched** — student ACK received 03:49:15Z with PID confirmed and `SENPAI_TIMEOUT_MINUTES=1100` set. W&B run ID `uwqybod5`, W&B name `thorfinn/tau-z-loss-weight-3p0-20260515T034842Z`. Resolves escalation from #1123 closure.
- **alphonse #1122 pace corrected** — actual pace at vol=16k is ~131 min/epoch (not 80 min). Root cause: vol=16k → 860 views/case (ceil(14M/16k)) → view_count=max(130,860)=860 → 10,864 iters/rank/epoch × 1.38 it/s = 131 min. Gate schedule revised: EP1 ~05:50Z, EP3 ~07:55Z. Smoke confirmed 5.6× sampled/population weight ratio → correct FAR-field SOTA mechanism.
- **Full Wave 29 fleet all active** (kubectl: all 8 deployments 1/1 READY at 03:52Z). Zero idle.

### Wave 29 fleet — full status and gate schedule (as of 03:52Z, 2026-05-15)

| PR | Student | Hypothesis | W&B Run | EP1 Gate | EP3 Gate | EP13 ETA |
|----|---------|------------|---------|----------|----------|----------|
| #1116 | edward | Per-channel WSS output heads (τ_x/τ_y/τ_z) — 18h convergence (relaunched 03:09Z as `3ufrbxl6`) | `3ufrbxl6` | ~05:10Z | **~08:00Z** | ~14:00Z |
| #1121 | frieren | WSS magnitude-only decomp (λ_dir=0, λ_mag=0.1) — EP3 PASS 6.746% (best in family) | `frieren/mag-only-*` | DONE | **DONE (PASS)** | ~14:30Z |
| #1122 | alphonse | SDF FAR-field α=2.0 (`weight=1+α|sdf|`) — corrected SOTA mechanism port | alphonse run | ~05:50Z | **~07:55Z** | ~16:30Z |
| #1124 | tanjiro | EMA decay 0.9995 — EP1 PASS 31.48%, EP2 in flight | `mw6d04kc` | DONE | **~06:15Z** | ~15:00Z |
| #1125 | nezuko | Spatial-prior surface sampling α=10 — 18h budget | nezuko run | **~05:00Z** | ~06:00Z | ~14:00Z |
| #1126 | fern | Deeper surface_out MLP (depth 2→4, +525k params) — 18h budget | fern run | ~04:30Z | **~06:00Z** | ~14:00Z |
| #1127 | askeladd | Surface_loss warmup curriculum (3-ep ramp 0→full) — 18h budget | `dtgfdsgv` | ~04:30Z | **~06:00Z** | ~14:00Z |
| #1128 | thorfinn | τ_z loss-weight 3.0 (single CLI flag escalation from 2.0) | `uwqybod5` | **~06:00Z** | ~08:30Z | ~15:30Z |

Gate criteria per row:
- **frieren #1121 EP6** (~06:00Z): val_abupt ≤6.5% PASS / ≤6.8% MARGINAL (half-way convergence sanity)
- **Standard no-SDF EP3**: val_abupt ≤7.2% PASS / ≤7.6% MARGINAL / >7.6% KILL
- **alphonse #1122 EP3 (SDF FAR-field)**: val_abupt ≤6.9% PASS / ≤7.2% MARGINAL / >7.2% KILL (tighter — SDF expected uplift)
- Per-axis WSS signal: τ_z/τ_x ratio direction is primary quality signal for all WSS-targeting experiments

## Prior invocation actions (2026-05-15 02:35–02:50Z) — Wave 28.5 closures complete, Wave 29 architectural pivot launched

- **Closed PR #1118 (askeladd OHEM v2)** — definitive negative mechanism: `clip_active`=100.00% across all 4218 EP3 OHEM-active steps → gradient through OHEM term is exactly zero → run is mathematically equivalent to baseline. Test metrics regressed +0.903pp test_WSS vs SOTA at EP3-only (truncated by 270-min cap). **OHEM-on-raw-residuals family terminally exhausted**: dataset's top-K residuals are intrinsically 100–25,000× larger than mean → any safe scalar cap fires 100% → zero learning signal. The `clip_active_pct` diagnostic was the right metric and identified the failure mode within EP3 — should remain in codebase for future loss-clip work.
- **Wave 28.5 loss-engineering pattern: 0-for-3 at convergence** — #1114 learnable WSS (null), #1119 GradNorm short-cycle (refutes prior-rediscovery), #1118 OHEM v2 (zero gradient). Decisive pivot to capacity / data-sampling / architecture routes.
- **Assigned PR #1126 (fern: deeper surface_out MLP depth 2→4 + 18h)** — Wave 29 architectural pivot. Tests whether τ_z magnitude prediction is decoder-depth-limited at the surface head (current 2-layer MLP). Orthogonal to thorfinn #1123 (separate τ_z branch) and edward #1116 (per-channel heads). Parameterizes `surface_out_depth` config; depth=2 default preserves backward compat. Full 13-EP convergence at SENPAI_TIMEOUT_MINUTES=1100.
- **Assigned PR #1127 (askeladd: explicit surface_loss_weight warmup curriculum + 18h)** — directly tests #1114 finding that EP1 wins are implicit-curriculum artifacts. Adds `--surface-loss-weight-warmup-epochs 3` flag that linearly ramps surface_loss_weight from 0 → full over first 3 epochs. Gradient-flow-preserving (scalar multiplier, NOT residual reweight) → avoids OHEM #1118 trap. Predicted payoff: stable volume-conditioned backbone before surface head receives full gradient → better τ_z magnitude convergence at terminal.
- **All 8 students now active**: alphonse #1122 (SDF FAR-field α=2.0), nezuko #1125 (spatial-prior α=10 + 18h), tanjiro #1124 (EMA decay 0.9995 + 18h), thorfinn #1123 (τ_z subnet — CLOSED, replaced by #1128), edward #1116 (per-channel heads, 18h convergence), frieren #1121 (magnitude-only + 18h), fern #1126 (surface_out depth=4 + 18h), askeladd #1127 (surface_loss warmup curriculum + 18h). **Zero idle.**

## Prior invocation actions (2026-05-15 01:41Z) — CRITICAL SDF MECHANISM DIAGNOSTIC

- **PR #1122 alphonse SDF port → CHANGES REQUESTED, corrected plan approved**: alphonse paused the 13ep run at 28min in (EP1 ~25% done) after spotting THREE issues with my original assignment:
  1. **Mechanism inversion**: commit `023f766` I cited as reference impl implements `weight = 1/(1+α·|sdf|)` (NEAR-surface emphasis), but PR #972 body and the actual SOTA run `56bcqp3m` use `weight = 1 + α·|sdf|` (FAR-field emphasis). These are OPPOSITE hypotheses.
  2. **α value mismatch**: SOTA `56bcqp3m` ran α=2.0, not α=4.0. The NEAR-surface alpha sweep on W&B (α=0.25→6.265%, α=0.5→6.290%, α=1.0→6.356%, α=3.0→7.251% over kill gate) shows higher α is monotonically worse for the NEAR-surface inversion.
  3. **IO regression**: alphonse's port used `np.load(path, mmap_mode="r")[rows]` fancy-indexed memmap which runs ~3× slower than PR #972's contiguous load + in-memory slice on this PVC. Smoke EP1 took 114 min vs SOTA reference 41 min.
  4. SOTA confounders captured: `56bcqp3m` also ran batch_size=1, model_layers=6, GradNorm, y_symmetry_aug, epochs=30 — these are NOT part of the corrected single-variable port.
- **Approved corrected plan**: revert IO optimization, switch to FAR-field `weight = 1 + α·|sdf|` α=2.0, keep tay baseline recipe (batch_size=4, model_layers=5, no GradNorm, no y_sym, epochs=13), smoke 2EP then full 13EP. Single-variable change isolates the SDF mechanism; full-recipe SOTA reproduction held for follow-up if FAR-field α=2.0 alone doesn't beat 6.99% ceiling.
- **Adjusted EP3 gate for FAR-field α=2.0**: PASS ≤ 6.9% / MARGINAL ≤ 7.2% / KILL otherwise. Projected EP13 terminal val_abupt ~6.4-6.6%, putting test_WSS in striking range of 6.5-6.7% (likely strongest single-model on tay).

## Methodology lesson for advisor

Always **verify the SOTA reference mechanism from the actual W&B config** before citing it in an assignment, not from a commit body that may be a different formulation. The PR body, the commit text, and the run config can all diverge. Going forward: when citing a SOTA mechanism, pull its W&B config first.

## Prior invocation actions (2026-05-15 01:15–01:30Z)

- **Closed PR #1114** (tanjiro learnable WSS channel weights): terminal SENPAI-RESULT `test_WSS=7.726%, val_abupt=7.066%` at EP3 (budget-truncated). +0.40pp val_abupt over matched 3-EP baseline (mempfubx 7.465%) but driven by EP1 transient drift (weights briefly dropped to ~50% of init, then quadratic-well-regularized back to baseline by EP3 within 3% of init). Mechanism null at convergence. Methodology data preserved.
- **Reassigned tanjiro → PR #1124** (EMA decay 0.9995 + 18h budget): single-flag experiment, slower EMA half-life ≈ 1386 steps vs default 693 steps. Tests whether late-converging τ_z benefits from longer EMA averaging window. Full 13-EP convergence test, comparison to no-SDF tay ceiling 6.99%.
- **Sent PR #1116 back to edward** (per-channel WSS heads, draft state): matched-budget A/B at EP3 truncated showed −0.66pp test_WSS, −0.09pp test_vol_p, −0.23pp test_SP — every metric improved vs single-head baseline `mempfubx`. **First clean positive Wave 28.5 signal.** But test_WSS=7.671% does not beat no-SDF ceiling (6.99%), so requires 18h budget convergence confirmation: if matched-budget delta holds at EP13, test_WSS → 6.33%, would tie/beat ensemble SOTA. Re-running with `SENPAI_TIMEOUT_MINUTES=1100`, no other changes.
- **τ_z/τ_x ratio finding (edward EP3)**: per-channel heads UNIFORMLY uplift capacity, not τ_z-specifically — ratio went 1.44 → 1.52 (wrong direction). Mechanism is "decoupled head capacity", not "τ_z specialization". Implies follow-on work needs deeper/wider τ_z head specifically (overlaps with thorfinn #1123 τ_z subnet).

---

## PRIMARY RESEARCH DIRECTIVE (Issue #1056, 2026-05-12, ongoing)

**Beat Wall Shear Stress SOTA from Transolver-3 (5.85%) while not degrading vol_p or surface pressure.**

- **WSS target:** `test_WSS` < **5.85%**
- **Non-negotiable constraints:** `test_vol_p` ≤ 3.643% AND `test_SP` ≤ 3.577% (PR #972 levels)
- **Baseline for all new single-model runs:** PR #972 SDF-stratified stack

**WSS Gap (post-PR #1102):**
- Single-model best: **6.727%** (PR #972) → need −0.88pp
- Ensemble best (compliant): **6.3263%** (PR #1102 K=8 Caruana) → need **−0.476pp**

Most recent human check-in: 2026-05-14 14:17 UTC — **"NO MORE ENSEMBLES! Its the lazy route to better results, we want genuine breakthroughs, not incremental improvements based on ensembling which we know we can deploy at any point to improve results."** (Issue #1056 comment from morganmcg1). Ensemble experiments are BANNED until explicitly unlocked. Status updates posted at ~12:35 UTC and ~15:00 UTC.

---

## CORRECTED DATASET IN EFFECT (since 2026-05-11)

Issue #1053 (deployed 2026-05-11) fixed a case-split/indexing bug that eliminated an artificial ~3× vol_p OOD gap.

**Corrected dataset path:** `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`
**Split:** val=34 cases/7295 views, test=50 cases/11091 views

All new runs MUST use the corrected dataset path and `--data-root` flag (not `--data-path`).

---

## Current Best Results (Corrected Split)

### **Ensemble SOTA (PR #1102 — K=8 Caruana with-replacement, WSS-optimised)**
- val_abupt = **5.7452%** | test_abupt = **5.5196%**
- val_vol_p = 3.4360% | test_vol_p = 3.5397%  ← satisfies ≤ 3.643%
- val_WSS = 6.5195% | **test_WSS = 6.3263%**  ← TRUE WIN
- val_SP = 3.7234% | test_SP = 3.3529%  ← satisfies ≤ 3.577%
- test_tau_x = 5.6071% | test_tau_y = 6.8397% | **test_tau_z = 8.2585%** (still worst axis)
- W&B: `bq1gaewq` (Arm D greedy), `ems8ekee`, `s7pirpr1`, `qf1lqwz0`
- **Members:** `56bcqp3m`×3, `29nohj67`×2, `a0yoxy85`×2, `ghh0s4ne`×1
- **Effective weights:** {56bcqp3m:0.375, 29nohj67:0.250, a0yoxy85:0.250, ghh0s4ne:0.125}

### Prior Ensemble SOTA (PR #1064 K=3 greedy, superseded by #1102)
- val_abupt = 5.7758% | test_abupt = 5.5199% | test_WSS = 6.3712%

### Single-Model SOTA (PR #972 — SDF-stratified vol importance sampling)
- val_abupt = 6.126% | test_abupt = 5.844%
- val_vol_p = 3.798% | test_vol_p = 3.643%  ← constraint boundary
- test_SP = 3.577%  ← constraint boundary
- test_WSS = 6.727%
- W&B: source=`56bcqp3m`, eval=`zxnhtagj`

### Rank 2 Single-Model (PR #968 — Stochastic vol subsampling)
- val_abupt = 6.278% | test_abupt = 5.986% | test_WSS = 6.825%
- W&B: source=`a0yoxy85`, eval=`qbg9pkmx`

---

## Gate Criteria

### Single-Model EP3 Gates (current tay stack — no SDF importance sampling)
- **PASS:** val_abupt ≤ **7.2%** AND val_vol_p ≤ 4.5%
- **MARGINAL:** val_abupt ≤ 7.6% AND val_vol_p ≤ 5.0%
- **KILL:** otherwise

(Historical PR #972 SDF stack gates were ≤ 6.2% / ≤ 6.5% — those reflect SDF-stratified sampling that is NOT on tay; do not apply to current single-model runs.)

### WSS-Targeted Single-Model Win Criteria (becomes new pool member)
- test_WSS ≤ 6.50% AND test_vol_p ≤ 3.643% AND test_SP ≤ 3.577% AND val_abupt ≤ 6.20%

### Ensemble Win Criteria (true new SOTA after PR #1102)
- val_abupt < **5.7452%** AND test_vol_p ≤ **3.643%** AND test_WSS < **6.3263%**

---

## Current Research Focus and Themes

### Primary: WSS Magnitude Bottleneck Attack (Wave 28 onwards — single-model only)

**New mechanism finding from PR #1097 close (tanjiro, WSS direction loss NEGATIVE):**
- WSS **direction is essentially solved** — cos_sim stabilises at 0.996 (~5° angular error) by EP2.
- **91–96% of remaining WSS residual is magnitude error.**
- This pivots the campaign from "direction-aware" experiments (which #1094, #1096, #1097 all targeted) toward **magnitude-targeted** mechanisms (rel_l2 loss, magnitude penalty) and **frame-equivariance** (in-plane rotation aug).

### Pool Saturation — CONFIRMED (PR #1103 closed 2026-05-14 13:30Z)

The current 4-member candidate pool {`56bcqp3m`, `29nohj67`, `a0yoxy85`, `ghh0s4ne`} is Pareto-saturated under convex combinations:
- PR #1102 K=8 Caruana (MERGED) — near-globally-optimal at discrete 1/8 grid
- PR #1099 K=3 WSS-targeted (CLOSED) — converged to identical K=3 subset as #1064
- PR #1103 SLSQP continuous optimisation (CLOSED) — confirmed K=8 within ~0.03 L1 of global continuous optimum; best-case val_WSS improvement = 0.0039pp (0.06% relative); val_SP ≤ 3.577% **infeasible** on this pool (simplex floor ~3.72%, every member ≥ 3.98%)

**Active lever for ensemble gains:**
1. **Pool extension via new single-model members** — only remaining path (ensembles BANNED per human directive)

⚠️ **ENSEMBLES BANNED** — Per morganmcg1 Issue #1056 directive 2026-05-14 14:17Z: no new ensemble experiments until explicitly unlocked. PR #1108 (bias-corrected ensemble) was superseded by PR #1109 (τ_z focal loss) before training started; #1108 is effectively dead.

---

## Active WIP PRs (as of 2026-05-15 ~02:50Z)

### Wave 28.5 → Wave 29 transition complete — all 8 students in flight

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| ~~#1114~~ | tanjiro | ~~Learnable WSS channel loss weights~~ | **CLOSED 01:13Z** — mechanism null at convergence; reassigned → #1124 |
| **#1116** | edward | **Per-channel WSS output heads** — decouple tau_x/tau_y/tau_z heads + 18h convergence | CHANGES REQUESTED 01:25Z — re-running at 18h to confirm matched-budget −0.66pp test_WSS delta holds at EP13; projected test_WSS ≈ 6.33% if delta holds (would tie ensemble SOTA) |
| ~~#1118~~ | askeladd | ~~OHEM v2 spike-clipped~~ | **CLOSED 02:35Z** — `clip_active`=100% → zero OHEM gradient → mathematically baseline-equivalent; reassigned → #1127 |
| ~~#1119~~ | fern | ~~GradNorm short-cycle (t_max=6, ep=6)~~ | **CLOSED 02:27Z** — REFUTES prior-rediscovery hypothesis; τ_z weight plateaus 1.07 (vs prior 2.0); hardcoded prior empirically validated; reassigned → #1126 |
| ~~#1120~~ | nezuko | ~~Spatial-prior surface sampling α=3~~ | **CLOSED 02:30Z** — mechanism right (ρ=+0.31 PASS), EP3 budget too short; strongest 3-EP truncated WSS in family but truncated; reassigned → #1125 (α=10 + 18h) |
| **#1121** | frieren | **WSS magnitude-only decomposition + 18h budget** — `λ_dir=0.0`, full 13-ep cosine; tests Wave 27 "91-96% magnitude" claim | Active WIP; EP3 gate ~02:48Z; EP13 ~14:00Z |
| **#1122** | alphonse | **SDF importance sampling port to tay — FAR-field α=2.0 (corrected mechanism)** — `weight = 1 + α·|sdf|`; highest-EV untested-on-tay lever; reproduces PR #972 SOTA mechanism (NOT the inverted `1/(1+α·|sdf|)`) | Active WIP draft post-correction (01:41Z); smoke-then-full plan approved; EP3 gate ≤6.9% PASS |
| **#1123** | thorfinn | **τ_z dedicated subnet** — 2-layer MLP head attacking residual axis test_τ_z ≈ 9.05% | Active WIP; launched 23:50Z post-#1100 close |
| **#1124** | tanjiro | **EMA decay 0.9995 + 18h budget** — single-flag test of slower EMA half-life (~1386 vs 693 steps) for late-converging τ_z | Active WIP; assigned 01:18Z post-#1114 close; full 13-EP convergence test |
| **#1125** | nezuko | **Spatial-prior surface sampling α=10 + 18h budget** — stronger oversample at full convergence (student's suggested follow-up #2); tests if mechanism scales without catastrophe | Active WIP; assigned 02:23Z post-#1120 close |
| **#1126** | fern | **Deeper surface_out MLP (depth 2→4) + 18h budget** — Wave 29 architectural pivot; tests if τ_z magnitude is decoder-depth-limited at surface head; orthogonal to thorfinn #1123 (separate branch) and edward #1116 (per-channel heads) | Active WIP; assigned 02:45Z post-#1119 close |
| **#1127** | askeladd | **Explicit surface_loss_weight warmup curriculum (3-ep ramp 0→full) + 18h** — directly tests #1114 implicit-curriculum finding; gradient-flow-preserving (avoids OHEM #1118 trap) | Active WIP; assigned 02:50Z post-#1118 close |

---

## Wave 28 Closures (2026-05-14 19:43Z–21:33Z) — methodology data captured, all reassigned

| PR | Student | Result | Key Mechanism Finding | Reassigned As |
|----|---------|--------|----------------------|---------------|
| #1109 | edward | val_WSS=8.766% EP3 (+1.6pp vs no-decomp ref) | Spatial focal α=2.0 amplifies per-point WSS errors at hot-spots faster than they can train down; underweights smooth bulk; baseline isn't smooth-dominated | #1116 per-channel heads |
| #1110 | askeladd | OHEM scale-collapse @ EP3 | Top-20% mining catastrophically scale-collapses without spike-clip; magnitude of L_hard explodes vs base loss | #1118 OHEM v2 spike-clipped |
| #1111 | fern | GradNorm test floors regress (test_vol_p +0.5pp, test_SP +0.4pp) | GradNorm de-emphasizes τ_z prior (hardcoded 2.0 weight); short-cycle test needed to disambiguate prior-vs-learned at convergence | #1119 GradNorm short-cycle |
| #1112 | frieren | Truncated EP3.5 @ 270-min wall-clock; calibration validated (mag head ratio=0.979 at half-cooked) | Mag head infrastructure works; full budget needed for convergence test | #1121 mag-only + 18h budget |
| #1113 | nezuko | val_abupt=8.04% EP3 (KILL) | Curvature is anti-correlated WSS proxy (ρ=-0.11); curvature-weighted sampling steers attention AWAY from high-WSS regions | #1120 spatial-prior (ρ=+0.31) |

## Wave 27 Closures (2026-05-14 ~13:45Z) — CATASTROPHIC FAILURE

All 4 experiments failed at EP3 with val_abupt 27–32% (4× above EP3 KILL gate of 7.6%). Root causes:

| PR | Student | val_abupt@EP3 | Root Cause |
|----|---------|---------------|------------|
| #1104 | fern | ~27% | L1 magnitude penalty `|‖τ‖−‖τ_gt‖|` creates conflicting gradients vs MSE loss; loss scale mismatch blows up training |
| #1105 | tanjiro | ~30% | Relative L2 `(pred-gt)²/‖gt‖²` numerically explodes when GT~0; near-zero WSS regions produce infinite loss |
| #1106 | frieren | ~28% | Physical-coordinate normal-frame rotation corrupts geometry signal — coordinate transformation invalidates learned features |
| #1107 | nezuko | ~32% | Yaw augmentation destroys physical orientation; model cannot learn orientation-dependent aerodynamics |

Common diagnosis: Wave 27 hypotheses all modified the **loss function or input transformation** at a fundamental level without sufficient numerical safeguards. The supplementary-loss OHEM approach (Wave 28) is designed to avoid these failure modes by adding a *supplementary* term (not replacing the base loss) with warmup and floor guards.

## Wave 26 Additional Kill (2026-05-14)

| PR | Student | Result | Key Mechanism Finding |
|----|---------|--------|----------------------|
| #1081 | askeladd | KILL @ EP10 (val_abupt=7.97%) | slw=3.0 surface loss weight — too aggressive; distorts vol_p head; baseline slw=2.0 is optimal |

## Wave 26 Closures (2026-05-13 → 2026-05-14)

| PR | Student | Result | Key Mechanism Finding |
|----|---------|--------|----------------------|
| #1094 | frieren | KILL @ EP3 (val_abupt=7.465%) | Normal-frame supervision built in normalised space — non-orthonormal |
| #1095 | nezuko | NEGATIVE (test_WSS=7.761% +1.03pp) | GradNorm mechanism healthy but starved vol head; curriculum is load-bearing |
| #1096 | edward | NEGATIVE (test_WSS +0.261pp vs ref) | Tangent-frame features redundant with normals; z-hat fallback discontinuity |
| #1097 | tanjiro | NEGATIVE (val_abupt=6.847% > KILL) | Direction NOT the bottleneck (cos_sim=0.996) |
| #1099 | fern | CONVERGED (same K=3 as #1064) | WSS-targeted greedy on 4-member pool converges to identical subset |
| #1102 | fern | **MERGED — new ensemble SOTA** | K=8 Caruana extracts ghh0s4ne WSS signal at 12.5% weight; NOW BANNED FROM EXTENSION per human directive |

---

## Baseline Training Recipe (current tay stack — NOT PR #972 SDF stack)

⚠️ **IMPORTANT:** the PR #972 SDF-stratified vol sampling code (`--sdf-importance-sampling --sdf-alpha 4.0`) was **never merged into tay**. Do NOT include those flags in any assignment — `argparse` will reject them. The live tay baseline is the stack below (no SDF importance sampling). Single-model EP3 on this baseline lands ~6.7–6.9% val_abupt, not the historical PR #972 6.2%. Gates must be recalibrated accordingly: PASS ≤ 7.2%, MARGINAL ≤ 7.6%, KILL otherwise.

```
--optimizer lion --lr 9e-5 --weight-decay 5e-4
--tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0
--use-ema --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1
--pos-encoding-mode string_separable --use-qk-norm
--rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"
--lr-cosine-t-max 13 --epochs 13
--vol-points-schedule "0:16384:3:32768:6:49152:9:65536"
--no-compile-model
--model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128
--batch-size 4 --validation-every 1
--train-surface-points 65536 --eval-surface-points 65536
--train-volume-points 65536 --eval-volume-points 65536
--use-surf-to-vol-xattn
--data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511
```

The PR #972 single-model SOTA W&B run `56bcqp3m` was trained with SDF-stratified sampling on a different branch (`dl24-frieren/vol-test-domain-augmentation`, commit `291efd2`); that code never landed on tay. Until it does, all new single-model runs are evaluated relative to the no-SDF tay baseline (thorfinn #1100 EP3=6.768% is a representative live trajectory).

---

## Next-Wave Hypothesis Queue

Wave 28.5 → Wave 29 in flight — 8 students busy, zero idle. Capacity students (alphonse #1078, thorfinn #1100) closed; SDF mechanism port (#1122) is the highest-EV active experiment.

Queue for Wave 30 (after current cohort lands ~tomorrow):

1. **Higher τ_z loss weight (3.0 from current 2.0)** — if fern #1126 and edward #1116 reveal decoder is the bottleneck, the prior τ_z=2.0 may now be undertuned with the increased capacity.
2. **WSS magnitude/direction joint head** — if frieren #1121 mag-only beats baseline, add a second head for direction (cos_sim) with bounded loss; combine via learnable α.
3. **Multi-scale surface attention** — second surface encoder at 0.5× token density to capture macro-flow features.
4. **Heteroscedastic WSS loss** — model both mean and variance per surface point; downweight high-aleatoric regions.
5. **τ_z frequency analysis** — Fourier decompose tau_z predictions vs GT to find spatial frequency bands where error is concentrated; use to motivate loss or architecture changes.
6. **Surface point sampling Voronoi tessellation** — sample uniformly over surface area (not raw vertex density) to remove sampling bias from non-uniform mesh refinement.
7. **Combine SDF FAR-field α=2.0 (from #1122) with deeper surface_out (from #1126)** — if both win independently, the combination is the obvious next step; orthogonal mechanism stacking.
8. **Curriculum at higher granularity** — if askeladd #1127 surface-loss warmup wins, try ramping individual WSS channel weights (τ_z last) instead of the global scalar.

⚠️ Permanently retired (catastrophic failure): yaw aug (#1107), magnitude penalty (#1104), rel_l2 (#1105), normal-frame rotation (#1094, #1106), curvature-weighted surface sampling (#1113 — wrong-sign proxy).

---

## Infrastructure Status

### GitHub Token Rate Limiting (RESOLVED 2026-05-14)
Senpai PR #3445 merged 06:42Z deployed per-student token fix + REST API migration. Fleet was back online by ~07:30Z. No further rate-limit-driven idle GPU incidents reported in current invocation.

### Pod Health
All 8 students have active pods (kubectl: `senpai-drivaerml-ddp8-*` deployments, 1/1 ready). DDP via 8× H100 96GB per student. Zero idle students as of 02:50Z. **Wave 28.5 → Wave 29 transition complete.** PR distribution: edward #1116 (per-channel heads 18h), frieren #1121 (magnitude-only 18h), alphonse #1122 (SDF FAR-field α=2.0), thorfinn #1123 (τ_z subnet), tanjiro #1124 (EMA decay 0.9995 18h), nezuko #1125 (spatial-prior α=10 18h), fern #1126 (surface_out depth=4 18h), askeladd #1127 (surface_loss warmup curriculum 18h).

---

## Key Findings to Date

- **WSS error is magnitude-dominated** (91–96% of residual, not direction) — pivot away from direction-aware experiments
- **tau_z (spanwise) still worst axis** (test_tau_z=8.2585% on PR #1102) — primary remaining target
- **Wave 27 catastrophic lesson**: NEVER replace base MSE loss — always use supplementary/additive formulations; loss scale mismatches and numerical instability (div-by-near-zero) destroy training even at 27–32% val_abupt; Wave 28 OHEM designed as additive supplement with 2-ep warmup to avoid this
- **Relative L2 loss is unstable** (PR #1105) — near-zero GT WSS regions produce unbounded loss; avoid any loss form with GT in denominator without explicit safeguards
- **slw=3.0 surface weight too aggressive** (PR #1081 killed) — baseline slw=2.0 is optimal
- **ENSEMBLES BANNED** (human directive 2026-05-14 14:17Z) — all new work must improve the single-model SOTA
- **Corrected dataset** (2026-05-11) eliminated artificial ~3× vol_p OOD gap — biggest research-program insight
- **SDF-stratified vol importance sampling** (PR #972) is single-model SOTA: val_abupt=6.126%, test_WSS=6.727%
- **Ensemble SOTA** (PR #1102 K=8 Caruana) test_WSS=6.3263% — first compliant ensemble below 6.33%
- **4-pool Pareto-saturated** (PR #1103 CONFIRMED) — K=8 within 0.03 L1 of global continuous optimum; val_SP ≤ 3.577% infeasible on this pool (simplex floor ~3.72%); new pool members are the operative lever
- ~~**Bias-corrected ensemble** (PR #1108)~~ — closed (superseded by τ_z focal loss #1109; ensemble research BANNED)
- **Training-time vol sampling** matters more than loss weighting or architecture depth for vol_p
- **Throughput regression risk on data-pipeline experiments** (nezuko #1113 self-diagnosed 12× slowdown from 20s/case curvature compute serialised through 4 workers; fix = precompute-and-cache; advisor must spec precompute step in any future data-pipeline assignment)
- **Curvature is a bad WSS proxy** (PR #1113 closed) — surface curvature is anti-correlated with |WSS| (ρ=-0.11); using curvature to oversample steers attention AWAY from high-WSS regions. Spatial position (`-x + |z|`) achieved ρ=+0.31 by contrast (PR #1120).
- **270-min wall-clock budget hits Wave 28 recipe at EP3.5** (#1111, #1112, historical #1095 all truncated) — recipe runs 76 min/epoch; full 13-ep cosine needs ~16h. Two responses available: recipe shrink (short t_max, fern #1119) or budget bump (`SENPAI_TIMEOUT_MINUTES=1100`, frieren #1121, matches alphonse #1078 working regime).
- **GradNorm de-emphasizes τ_z hard-coded prior** (#1111 close) — when learned, GradNorm reduces τ_z weight from prior 2.0 toward 1.4, which regresses test_vol_p and test_SP floors. Question: is the 2.0 prior over-tuned, or is the learned weight wrong? Short-cycle test (#1119 fern) measures this at full convergence.
- **OHEM scale-collapse** (#1110 close) — top-k mining catastrophically collapses without spike-clipping; magnitude of L_hard scales superlinearly when targeting top-20% of L distribution.
- **Spatial focal α=2.0 amplifies hot-spot error faster than training rate** (#1109 close) — per-point focal modulation creates concentrated gradients on outliers; baseline isn't bulk-smooth-dominated so amplification destabilizes optimization.
- **val→test ratio is NOT stable across eval configurations** (#1078 close) — asymmetric eval 131k produced val→test ratio of 1.020, not the 0.935 anchored on PR #972. The 0.935 ratio is recipe-specific (SDF stack), not transferable. Advisor SOTA projections must use test results from comparable-recipe runs, not synthetic val × historical ratio. This is a methodology guard for the entire program.
- **18h budget recipe validated end-to-end** (#1078 close): `SENPAI_TIMEOUT_MINUTES=1100` ran 17 epochs cleanly at ~62 min/ep (faster than initially projected). All future Wave 28+ runs can adopt it confidently; frieren #1121 has already.
- **Capacity-uplift ceiling on no-SDF tay is val_abupt ≈ 6.31%** (#1078 EP16 / #1100 EP16 close). Beyond that, the bottleneck is training-time sampling, not parameter count. Justifies #1122 (alphonse SDF port).
- **No-SDF tay structural ceiling at test_WSS ≈ 6.99%** (#1078 + #1100 close, two independent mechanisms). Asymmetric eval 131k and slices=256 capacity uplift both converge here at full convergence. Test floors regress under both. Any "beat SOTA without SDF" claim must beat 6.99% — capacity alone cannot. Direct paper-relevant finding.
- **τ_z is the program-wide residual axis** (test_τ_z ≈ 9.05% across all no-SDF runs). Consistently 30-45% worse than τ_x and ~18bp worse than τ_y. Justifies #1123 (thorfinn τ_z dedicated subnet) attacking representational capacity for τ_z specifically.
- **Initial-state debug crash** (tanjiro #1114 val_abupt=65.34% on 1-ep debug, then 8-rank DDP retry also crashed) — root cause likely learnable-weight unbounded growth; mitigated by lr=1e-3 separate group + L2 reg 1e-4 + 2-ep warmup option
- **Post-xattn capacity additions** 0-for-3 (PRs #884, #891, #906) — do not add layers after surf→vol xattn
- **Rotation aug** (PR #925): aggressive yaw+pitch degrades; mild yaw-only (≤45°) being tested in PR #1107
- **Normal-frame WSS in normalised space** fails (PR #1094); physical-frame variant (#1106) is the corrected attempt
- **Tangent-frame features** redundant with surface normals (PR #1096) — model already has the information
- **Direction loss** redundant with weighted MSE (PR #1097) — cos_sim=0.996 by EP2 without it
- **GradNorm + fixed-65k vol** fails because vol curriculum is load-bearing (PR #1095)
- **Infra:** `--data-root` (not `--data-path`); mount point `/mnt/new-pvc/` (not `/mnt/pvc/`)
- **EMA lag:** `eval_raw_vs_ema=False` means only EMA weights evaluated — early-epoch EMA metrics appear worse than raw
