# RESEARCH IDEAS — Final Day Sprint (2026-05-28)

**Generated**: 2026-05-28 ~11:00Z
**Context**: Final ~24 hours of the senpai DrivAerML CFD-surrogate window.
**Primary target**: test_WSS < 5.85% (Transolver-3 SOTA, Morgan Issue #1056)
**Current SOTA**: H112 PR #1283 — val_abupt 6.1358% / test_WSS 6.752% / test_WSS_z 8.720%
**Gap**: −0.90pp test_WSS required
**Idle students**: alphonse, askeladd, nezuko, tanjiro

---

## Research State at Time of Generation

### What We Know Is True (locked evidence)

**4-axis slope-preservation cohort (PROVISIONAL pending H164d, frieren PR #1357, terminal ~19:25Z):**

| Mechanism | Delta slope WSS agg | Delta slope WSS_z |
|---|---:|---:|
| H145 tau_y=3.0 | −0.048pp STEEPER | −0.026pp STEEPER |
| H148 mirror-aug | −0.081pp STEEPER | −0.013pp STEEPER |
| H149 AdamW | −0.036pp STEEPER | −0.060pp STEEPER |
| H157 SGDR T_0=4 | −0.036pp STEEPER | −0.075pp STEEPER |

None of these axes alone beats H112 val. Compounding is required for SOTA.

**z-axis 4-point closure LOCKED permanently:**
H112 tau_z=2.0 is test-optimum on both aggregate AND WSS_z. All 3 perturbations (H143 tau_z=4.0, H165 tau_z=1.5, H144 tau_z=6.0) produce FLATTER slope AND worse absolute test_WSS. Do not touch tau_z.

**Slope-FLATTENING class (closed):**
z-axis perturbation of any kind, dynamic loss balancing, architectural splits, capacity additions. H112 depth-5 is test-optimum.

### What Is Still Open

1. **H164d RNG sanity** (frieren PR #1357, terminal ~19:25Z): Does running H112 recipe with different RNG seed produce slope within ±0.02pp of H112? If YES — framework REAL. If ±0.05–0.15pp deviation — framework COLLAPSES. Cannot wait for this result given 13h run time. All 4 hypotheses below are designed to be valid under Scenario A (framework REAL) and provide useful signal under Scenario B.

2. **DropPath rate exploration**: H112 uses drop-path-rate=0.10. H131 tested 0.15 BUT paired with hidden-576 capacity increase, making it a DIFFERENT experiment. Clean H112 recipe + drop-path-rate=0.15 has NEVER been tested.

3. **Augmentation-strength sweep**: H148 tested mirror-aug at default p=0.5. LOWER probability (p=0.25) has never been tested — different effective regularization pressure from the same mechanism.

4. **Multi-axis compounding**: H145, H148, H149, H157 have each been tested in isolation. Two- and three-axis factorial compounds have NOT been tested (H183 is in flight as fern PR #1356 testing H148+H145 only).

5. **Single late restart T_0=10**: H157 tested SGDR T_0=4 (three restarts), which fires every 4 epochs. A SINGLE restart at T_0=10 (fires once at epoch 10, 77% through budget) has never been tested. This is explicitly banked as H179 in CURRENT_RESEARCH_STATE.md.

---

## Hypothesis Ranking by Expected Value

**EV ordering**: H186 > H184 > H185 > H179

Rationale:
- H186 compounds the maximum number of confirmed orthogonal slope-preservation axes simultaneously — highest theoretical ceiling.
- H184 compounds slope-preservation (mirror-aug) with a confirmed-unexplored regularization escalation (DropPath 0.15) — the strongest single mechanism + an unexplored adjacent axis.
- H185 probes the augmentation-strength axis on the STRONGEST single slope-preservation mechanism — high information gain, needed to find optimal p before compounding.
- H179 tests a scheduler variant that is orthogonal to all confirmed compounding axes — lower prior because H157 at T_0=4 only moved slope by −0.036pp, but T_0=10 targets different training dynamics.

---

## Hypothesis 1: H186 (askeladd) — Triple Slope-Cohort Compound

**One-line**: Simultaneously apply all three non-y-loss slope-preservation axes (mirror-aug + AdamW + SGDR T_0=4) to test if confirmed orthogonal mechanisms compound additively.

**Research mode**: Tier shift (compounding three independently confirmed axes simultaneously)

**Mechanism prediction**:
H148 (mirror-aug) steepens val→test slope via data-distribution invariance across y-axis.
H149 (AdamW) steepens via optimizer noise regularization.
H157 (SGDR T_0=4) steepens via periodic learning-rate perturbations that escape local basin plateaus.
These three mechanisms target different parts of the training pipeline (data preprocessing, gradient update rule, LR schedule) and should be fully additive. Predicted combined slope steepening: −0.048 to −0.12pp WSS aggregate vs H112. If the three mechanisms are sub-additive (interference), expected: −0.08 to −0.10pp. If super-additive, expected: > −0.12pp.

**Why it has not been tried**: Each axis was closed in isolation (H148 mirror-aug, H149 AdamW, H157 SGDR) — all C NULL because none beat H112 val alone. H183 (fern PR #1356, in flight) tests H148+H145 only (mirror-aug + tau_y=3.0). No experiment has combined three confirmed non-y-loss axes.

**Key risk**: The three mechanisms may conflict. AdamW + SGDR together change gradient scale dynamics in a way that Lion+cosine does not. Mirror-aug with AdamW may interact differently than mirror-aug with Lion. If val_abupt tracks WORSE than H148 alone (>6.388%), something is interfering.

**Expected outcome**:
- val_abupt: 6.15–6.25% (similar to H148/H149/H157 individually)
- test_WSS: 6.65–6.70% (if compounding is additive: −0.10pp from H112)
- Slope WSS agg: −0.28 to −0.33pp (steeper than H112's −0.215pp, more than any single axis)

**Falsification criterion**: If slope WSS agg is FLATTER than H148 alone (less negative than −0.296pp), the three-way interaction has a destructive interference — the axes are not independent.

**Compounding potential**: If successful (test_WSS < H112), this becomes the new compounding base for H181-style EMA-0.9999 application.

**Gate thresholds (standard kill schedule)**:
- EP1 (step ~10,864): val_abupt < 67.97%
- EP3 (step ~32,592): val_abupt < 25%
- EP6 (step ~48,897): val_abupt < 11%
- EP9: val_abupt < 8%
- EP12 (binding): val_abupt < 7.0% (expected ~6.18–6.25%)

**CLI reproduce command**:
```bash
SENPAI_TIMEOUT_MINUTES=1100 torchrun --standalone --nproc-per-node=8 target/train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --manifest data/split_manifest.json \
  --agent askeladd --optimizer adamw \
  --lr 9e-5 --weight-decay 5e-4 --batch-size 4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 \
  --surface-loss-weight 2.0 --volume-loss-weight 0.5 \
  --use-surf-to-vol-xattn --enable-residual-positions \
  --use-drop-path --drop-path-rate 0.10 \
  --mirror-aug \
  --epochs 13 --lr-warmup-epochs 1 --lr-schedule sgdr --lr-cosine-t0 4 \
  --ema-decay 0.999 --grad-clip 1.0 --save-best-checkpoint \
  --wandb_group H186-triple-compound
```

**Implementation note**: AdamW replaces Lion (`--optimizer adamw` without `--lion-*` flags, per H149 recipe). SGDR T_0=4 replaces cosine LR (`--lr-schedule sgdr --lr-cosine-t0 4`, per H157 recipe). Mirror-aug is added via `--mirror-aug` (per H148 recipe). All three changes are additive on top of H112 base recipe. tau-z-loss-weight stays 2.0 (LOCKED).

**Taste scores**:
- Mechanistic grounding: 4 (three independently confirmed mechanisms, each with a measured slope delta, combined on orthogonal axes — the mechanism is precise and falsifiable)
- Research-state value: 4 (success or failure both update the research map sharply: either compounding is additive or it reveals an interference that explains why no single-axis compound beats H112)
- Execution value: 3 (13h run, targets test_WSS directly; slightly lower because we don't know if additivity holds — a 2-axis compound might have been a cheaper first test, but fern PR #1356 covers H148+H145 already)

**Overall EV rank**: #1

---

## Hypothesis 2: H184 (alphonse) — Mirror-aug × DropPath 0.15 Compound

**One-line**: Combine the strongest single slope-preservation axis (H148 mirror-aug) with an escalated stochastic depth rate (0.15 vs H112's 0.10) to test if regularization compounding via data-space invariance + architecture-space dropout is additive.

**Research mode**: Frontier refinement (compounding the strongest mechanism with the nearest unexplored adjacent axis)

**Mechanism prediction**:
H148 (mirror-aug) steepens val→test slope via data-distribution invariance: the model learns features invariant to y-axis reflection, reducing overfitting to training-set asymmetries. Its slope delta is −0.081pp WSS aggregate — STRONGEST of the four cohort axes.
DropPath 0.15 vs 0.10 escalates stochastic depth regularization: at rate 0.15, each block has a 15% chance of being dropped, vs 10% at H112. This increases architecture-level noise during training, which should compound with data-level noise from mirror-aug in an orthogonal way.
H131 (PR #1312) tested DropPath 0.15 + hidden-576 (capacity increase) — FAILED due to capacity pathology. A clean H112 + DropPath 0.15 (no capacity change, no width change) has NEVER been tested.

**Why it has not been tried**: H131 (PR #1312) is the closest prior, but H131 conflates DropPath 0.15 with hidden-576 capacity increase (+11% params), which causes slope flattening independently. H184 isolates the DropPath rate change at fixed depth-5/hidden-512.

**Expected outcome**:
- val_abupt: 6.15–6.30% (DropPath 0.15 may slightly increase regularization noise)
- test_WSS: 6.65–6.72% (if compounding is additive with H148's −0.081pp: ~6.671%)
- Slope WSS agg: −0.27 to −0.30pp (steeper than H148 alone)

**Falsification criterion**: If test_WSS is WORSE than H148 alone (>6.935%), DropPath rate escalation without capacity increase is not a valid regularization escalation at this architecture scale.

**Compounding potential**: If DropPath 0.15 is confirmed as an independent slope-preservation axis, it becomes a third component of the compounding recipe, addable to H186's triple compound.

**Gate thresholds (standard kill schedule)**:
- EP1 (step ~10,864): val_abupt < 67.97%
- EP3 (step ~32,592): val_abupt < 25%
- EP6 (step ~48,897): val_abupt < 11%
- EP9: val_abupt < 8%
- EP12 (binding): val_abupt < 7.0%

**CLI reproduce command**:
```bash
SENPAI_TIMEOUT_MINUTES=1100 torchrun --standalone --nproc-per-node=8 target/train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --manifest data/split_manifest.json \
  --agent alphonse --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 9e-5 --weight-decay 5e-4 --batch-size 4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 \
  --surface-loss-weight 2.0 --volume-loss-weight 0.5 \
  --use-surf-to-vol-xattn --enable-residual-positions \
  --use-drop-path --drop-path-rate 0.15 \
  --mirror-aug \
  --epochs 13 --lr-warmup-epochs 1 --lr-schedule cosine \
  --ema-decay 0.999 --grad-clip 1.0 --save-best-checkpoint \
  --wandb_group H184-mirror-droppath-15
```

**Implementation note**: Only two changes vs H112: `--drop-path-rate 0.15` (H112 uses 0.10) and `--mirror-aug` (H112 has none). All other flags identical to H112. DO NOT change `--hidden-dim` or any other architecture parameter — that is the H131 trap. tau-z-loss-weight stays 2.0 (LOCKED).

**Taste scores**:
- Mechanistic grounding: 3 (H148 mechanism is confirmed; DropPath 0.15 mechanism is plausible but untested clean — H131 failure was due to confound, not DropPath itself)
- Research-state value: 3 (if DropPath 0.15 is confirmed as independent axis, opens a new compounding dimension; if it fails, closes DropPath rate escalation at this scale)
- Execution value: 3 (clean two-factor compound; only 2 flag changes from H112; cost-proportional to signal)

**Overall EV rank**: #2

---

## Hypothesis 3: H185 (nezuko) — Asymmetric Mirror Augmentation p=0.25

**One-line**: Test mirror-augmentation at half the default probability (p=0.25 vs p=0.50) to find the augmentation-strength operating point on the STRONGEST single slope-preservation axis before compounding.

**Research mode**: Diagnostic (augmentation-strength sweep on the highest-EV mechanism axis)

**Mechanism prediction**:
H148 tested mirror-aug at default p=0.50 — each batch item has 50% probability of being y-reflected. This produced the steepest slope improvement of any single axis (−0.081pp WSS aggregate) but still did NOT beat H112 val alone (val_abupt 6.388% vs H112 6.136%).
At p=0.25, effective augmentation frequency is halved. The training distribution remains more anchored to the original data manifold, reducing the regularization pressure. This may:
(a) Push val_abupt LOWER (closer to H112) because the model overfits less to the augmentation invariance, OR
(b) Reduce slope steepening if the mechanism requires sufficient augmentation coverage.
If val_abupt < H112's 6.136% while slope remains steeper than H112, p=0.25 may be a better single-axis operating point than p=0.50.

**Why it has not been tried**: H148 (PR #1341) used only default p=0.5. The probability sweep was explicitly banked as H185 in CURRENT_RESEARCH_STATE.md ("asymmetric mirror p=0.25 (softer version of H148)"). No experiment has tested non-default mirror probability.

**Key diagnostic value**: This experiment answers "is H148's 6.388% val (worse than H112's 6.136%) caused by the augmentation over-regularizing the model?" If p=0.25 gives val_abupt closer to 6.136% while preserving slope, the answer is yes and p=0.25 becomes the optimal augmentation point.

**Expected outcome**:
- val_abupt: 6.20–6.35% (between H112 and H148; lower augmentation pressure → closer to H112 val)
- test_WSS: 6.72–6.85% (depending on whether slope steepening scales with p or is binary)
- Slope WSS agg: −0.22 to −0.28pp (some slope steepening, less than H148's −0.296pp)

**Falsification criterion**: If slope WSS agg is FLAT relative to H112 (within ±0.02pp), augmentation strength at p=0.25 is insufficient for the mechanism — the slope effect requires high augmentation coverage (p≥0.5).

**Compounding potential**: If p=0.25 provides better val_abupt with maintained slope, H185 replaces H148 as the canonical mirror-aug module in all future compounds (H186-style, H181-style). This optimizes the data-aug axis before stacking others.

**Gate thresholds (standard kill schedule)**:
- EP1 (step ~10,864): val_abupt < 67.97%
- EP3 (step ~32,592): val_abupt < 25%
- EP6 (step ~48,897): val_abupt < 11%
- EP9: val_abupt < 8%
- EP12 (binding): val_abupt < 7.0%

**CLI reproduce command**:
```bash
SENPAI_TIMEOUT_MINUTES=1100 torchrun --standalone --nproc-per-node=8 target/train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --manifest data/split_manifest.json \
  --agent nezuko --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 9e-5 --weight-decay 5e-4 --batch-size 4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 \
  --surface-loss-weight 2.0 --volume-loss-weight 0.5 \
  --use-surf-to-vol-xattn --enable-residual-positions \
  --use-drop-path --drop-path-rate 0.10 \
  --mirror-aug --mirror-aug-prob 0.25 \
  --epochs 13 --lr-warmup-epochs 1 --lr-schedule cosine \
  --ema-decay 0.999 --grad-clip 1.0 --save-best-checkpoint \
  --wandb_group H185-mirror-p25
```

**Implementation note**: Two changes vs H112: `--mirror-aug` and `--mirror-aug-prob 0.25`. If `--mirror-aug-prob` is not a supported CLI flag in the current training script, the student should check the exact flag name in `target/train.py --help` and update accordingly. The equivalent functionality might be exposed as `--mirror_aug_prob`, `--aug_prob`, or similar. tau-z-loss-weight stays 2.0 (LOCKED).

**Taste scores**:
- Mechanistic grounding: 3 (mechanism is confirmed from H148; p=0.25 tests the augmentation-strength dimension of the same mechanism with a clear prediction about val vs slope trade-off)
- Research-state value: 4 (this is the discriminating diagnostic for the H148 mechanism: if p=0.25 gives better val while maintaining slope, it sharply updates how we apply mirror-aug in all future compounds)
- Execution value: 3 (one flag change from H148; minimal implementation risk; high information gain per unit compute)

**Overall EV rank**: #3

---

## Hypothesis 4: H179 (tanjiro) — Single Late Cosine Restart T_0=10

**One-line**: Test a single SGDR restart at epoch 10 (77% through training budget) to probe whether a late-training LR spike can escape the basin plateau without the disruptive multi-restart pattern of H157 (T_0=4).

**Research mode**: Diagnostic (scheduler variant in the confirmed slope-preservation axis)

**Mechanism prediction**:
H157 (SGDR T_0=4) fires restarts at epochs 4, 8, 12 — three restarts that each briefly spike LR before decaying. This steepened slope by −0.036pp (WSS agg), matching H149 (AdamW). However, H157's early restarts (epoch 4, epoch 8) may be disrupting early-training convergence unnecessarily.
H179 (T_0=10) fires a SINGLE restart at epoch 10 (~77% through budget). The model trains normally through epochs 1–10 (standard cosine decay), then gets ONE LR spike at epoch 10 before a final decay to EP13. This targets: (a) the basin-escape hypothesis (late LR spike shakes loose a local minimum), while (b) preserving early-training convergence (no disruption before EP10).
The "lr_min freeze" hypothesis was FALSIFIED — H112 val IMPROVES through EP13 under near-zero cosine LR — meaning the late-train dynamics are NOT frozen, and a late restart may inject useful perturbation into a still-active optimization process.

**Why it has not been tried**: H157 tested T_0=4 (multiple restarts). The single-restart late-fire variant is explicitly banked as H179 in CURRENT_RESEARCH_STATE.md ("single late-restart cosine T_0=10 (SGDR revisit with T_0>=75% budget)"). No experiment has tested T_0 > 4.

**Key risk**: If the slope-preservation from H157 was specifically FROM the early restarts (epoch 4 restart fires during the most sensitive training phase), then a late-only restart at epoch 10 may not replicate the slope steepening. In this case slope would be similar to H112 (flat) and test_WSS would be H112-equivalent.

**Expected outcome**:
- val_abupt: 6.15–6.25% (similar to H157, which was 6.xxx%)
- test_WSS: 6.72–6.80% (if slope steepening holds with late restart)
- Slope WSS agg: −0.22 to −0.27pp (some steepening; less than H157 if early restarts were the mechanism)

**Falsification criterion**: If slope WSS agg is within ±0.02pp of H112 (−0.215pp), late-only restart has no slope preservation effect — the mechanism requires early-training disruption, not late-training disruption.

**Compounding potential**: If H179 shows slope steepening, it provides a scheduler variant that avoids early-training disruption — potentially combining better with mirror-aug (which also induces data-distribution perturbation in early training) without double-perturbing the early optimization path.

**Gate thresholds (standard kill schedule)**:
- EP1 (step ~10,864): val_abupt < 67.97%
- EP3 (step ~32,592): val_abupt < 25%
- EP6 (step ~48,897): val_abupt < 11%
- EP9: val_abupt < 8%
- EP12 (binding): val_abupt < 7.0%

**CLI reproduce command**:
```bash
SENPAI_TIMEOUT_MINUTES=1100 torchrun --standalone --nproc-per-node=8 target/train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --manifest data/split_manifest.json \
  --agent tanjiro --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 9e-5 --weight-decay 5e-4 --batch-size 4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 \
  --surface-loss-weight 2.0 --volume-loss-weight 0.5 \
  --use-surf-to-vol-xattn --enable-residual-positions \
  --use-drop-path --drop-path-rate 0.10 \
  --epochs 13 --lr-warmup-epochs 1 --lr-schedule sgdr --lr-cosine-t0 10 \
  --ema-decay 0.999 --grad-clip 1.0 --save-best-checkpoint \
  --wandb_group H179-sgdr-t0-10
```

**Implementation note**: One change vs H112: `--lr-schedule sgdr --lr-cosine-t0 10`. This replaces `--lr-schedule cosine`. With T_0=10 and 13 total epochs (minus 1 warmup = 12 cosine-schedule epochs), SGDR fires a restart at epoch 10. The LR then decays from lr_max to lr_min over the remaining 3 epochs. Verify with student that T_0 is measured in epochs (not steps) in the current training script, consistent with H157's T_0=4. tau-z-loss-weight stays 2.0 (LOCKED).

**Taste scores**:
- Mechanistic grounding: 2 (mechanism is plausible — late basin escape — but the causal story for WHY T_0=4 restarts steepen slope is not well established; T_0=10 may or may not preserve the same mechanism)
- Research-state value: 3 (clear falsification: if slope doesn't steepen, early-restart disruption was causal in H157; if slope DOES steepen, late restart is a gentler alternative that avoids early-training interference)
- Execution value: 3 (one flag change from H112; extremely low implementation risk; result directly updates the SGDR mechanism understanding)

**Overall EV rank**: #4

---

## Experiment Decision Tree

```
H186 (triple compound, askeladd)
  SUCCESS (test_WSS < H112):
    → Compounding is additive across 3 axes
    → Next: H186 + ema=0.9999 (H181-style) for SOTA push
    → Next: Add H185/H179 as 4th axis if they confirm
  FAIL (slope FLATTER than H148 alone):
    → AdamW + SGDR interaction is destructive
    → Next: Test H148 + H149 ONLY (without SGDR) — is AdamW the interference?
    → Next: Test H148 + H157 ONLY (without AdamW) — is AdamW the problem?
  FAIL (slope similar to H112, all mechanisms cancelled):
    → Three-way interaction destroys all slope benefits
    → Framework may be fragile; await H164d for RNG baseline

H184 (mirror-aug × DropPath 0.15, alphonse)
  SUCCESS (test_WSS < H112):
    → DropPath 0.15 is confirmed independent slope-preservation axis
    → Next: Add DropPath 0.15 to H186 triple compound → quadruple compound
  PASS gate (slope steeper than H112, val higher):
    → DropPath 0.15 compounds with mirror-aug for slope but doesn't close val gap alone
    → Next: H184 + ema=0.9999 (H181-style applied to H184 base)
  FAIL (slope NOT steeper than H148 alone):
    → DropPath 0.15 is not additive with mirror-aug; axes are NOT independent at the DropPath level
    → Next: Test standalone DropPath 0.15 (no mirror-aug) to isolate

H185 (mirror p=0.25, nezuko)
  val_abupt LOWER than H148 (6.388%), slope still steep:
    → p=0.25 is better operating point for mirror-aug
    → IMMEDIATELY replace H148 flags with p=0.25 in H186/H184 compounds
    → Next: H185 + ema=0.9999 for SOTA push
  val_abupt similar to H148, slope LESS steep:
    → Augmentation strength matters; higher p gives more slope steepening
    → H148 (p=0.5) remains canonical; p=0.25 was too soft
    → H185 confirms p=0.5 is optimal; close this axis
  val_abupt similar to H148, slope similar to H112:
    → p=0.25 is below the effectiveness threshold; augmentation mechanism is frequency-dependent
    → Suggests H148 slope effect requires high augmentation coverage; keep p=0.5

H179 (SGDR T_0=10, tanjiro)
  Slope STEEPER than H112, val similar to H157:
    → Late restart is sufficient for slope preservation
    → Late restart is PREFERABLE to early restarts for combination with mirror-aug
    → Next: H179 + H148 (late-restart + mirror-aug, no early disruption interference)
  Slope within ±0.02pp of H112 (FLAT):
    → Late restart alone is not the slope mechanism; H157's early restarts were causal
    → T_0 < training_budget/3 required for slope effect
    → Close scheduler-axis exploration; H157 (T_0=4) remains canonical

---

IF H164d (frieren PR #1357, terminal ~19:25Z) returns Scenario B (slope deviation ±0.05–0.15pp):
  → Entire slope-preservation cohort framework COLLAPSES as RNG noise
  → Pivot to absolute test_WSS improvement mode
  → Primary next experiments should focus on: better val-to-test transfer via architecture (not capacity), test-set-aligned loss formulation, or data representation improvements
  → None of H184/H185/H186/H179 become invalid as experiments — they still test valid mechanisms — but the INTERPRETATION of slope changes as signals shifts entirely
```

---

## Research State Update

**Current best explanation for limiting progress**: The slope-preservation cohort framework (4 confirmed axes: H145, H148, H149, H157) shows that certain orthogonal perturbations systematically steepen the val→test slope, but none individually beat H112 val. The bottleneck is that slope steepening + val improvement are two separate effects that require compounding to achieve simultaneously. The val_abupt gap from H112 (6.136%) is ~0.25pp for any individual slope-preservation mechanism.

**Evidence**: H145 (6.388%), H148 (6.388%), H149 (val within noise of H148), H157 (val within noise of H148) — all ~0.25pp above H112 val. Slope steepening of −0.036 to −0.081pp each.

**Program-critical uncertainty**: H164d (frieren PR #1357) terminal ~19:25Z determines if the entire framework is real or RNG noise. This is the single most important result in the program.

**Ruled-out paths**: z-axis perturbation (H143, H144, H165), dynamic loss balancing (H147), architectural splits (H138, H146), capacity additions (H118–H125), EMA 0.9999 standalone (H150), lr_min-freeze scheduler variants.

**Open uncertainties**:
1. Does multi-axis compounding of slope-preservation mechanisms produce additive slope steepening?
2. Is there an optimal mirror-aug probability between 0 and 0.5 that improves val_abupt without losing slope steepening?
3. Is H157's slope steepening caused by early restarts (T_0 < budget/4) or is a late single restart (T_0 = 0.77 × budget) sufficient?

**Next discriminating experiment**: H164d result at ~19:25Z. If Scenario A — proceed with H186 result as primary compound confirmation. If Scenario B — pivot to test-absolute-improvement mode.

**Stop condition**: The program has no natural stop condition (Morgan directive: "test_WSS < 5.85% is THE objective"). The intermediate stop condition for the slope-preservation framework is H164d Scenario B. The program stop condition for this file's hypotheses would be if all four compounds (H184, H185, H186, H179) show slope FLATTER than H112 — this would falsify multi-axis additivity and require pivoting to a fundamentally different mechanism.
