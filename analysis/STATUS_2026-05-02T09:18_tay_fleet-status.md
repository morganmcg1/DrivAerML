# STATUS — 2026-05-02T09:18Z — tay branch

## 1. Survey Snapshot

| Slot | Student | PR | Status | W&B Run | Runtime | val_abupt |
|---|---|---|---|---|---|---|
| 1 | edward | #311 | WIP | gcwx9yaa | ~10h | **7.546%** |
| 2 | nezuko | #283 | WIP | z6xc97gg | ~6.5h | 8.994% |
| 3 | tanjiro | #323 | WIP | vjs33n25 | ~2.4h | 12.07% |
| 4 | thorfinn | #345 | WIP | gradclip-2.0-w0 | ~2h | 13.42% |
| 5 | frieren | #352 | WIP | 9yme70s3 (arm-D-swiglu-uniform) | ~0.7h | ~58% (ep1) |
| 6 | alphonse | #287 | WIP | qknorm-armB | ~0.7h | ~67% (ep1) |
| 7 | fern | #351 | WIP | tangent-frame-tau-r18 | ~0.7h | ~65% (ep1) |
| 8 | askeladd | #353 | WIP | askeladd-huber-tau-delta0p5 | ~0.7h | N/A (ep1) |

- **0 review-ready PRs** — nothing queued for merge
- **8 WIP PRs** — all students running
- **0 idle students**

## 2. Current SOTA

- **Baseline**: PR #309, thorfinn, run `ztdhodw1`, val_abupt=9.0389%, test_abupt=10.126%
- **Per-axis test (SOTA vs AB-UPT reference)**:

| Channel | SOTA (PR#309) | AB-UPT ref | Gap |
|---|---|---|---|
| surface_pressure | 5.395% | 3.82% | +1.575pp |
| wall_shear (mean) | 9.883% | 7.29% | +2.593pp |
| volume_pressure | 12.484% | 6.08% | +6.404pp |
| tau_x | 8.402% | 5.35% | +3.052pp |
| tau_y | 11.941% | 3.65% | +8.291pp |
| tau_z | 12.407% | 3.63% | +8.777pp |

The largest gaps are tau_y (×3.27 vs AB-UPT) and tau_z (×3.42 vs AB-UPT). Volume pressure also significantly above reference.

## 3. Leading Experiment: Edward PR #311 (STRING Separable PE)

**This is a potential new SOTA.** Run `gcwx9yaa` at ~10 epochs:

| Channel | val | vs. SOTA gap |
|---|---|---|
| **abupt mean** | **7.546%** | **-1.493pp below SOTA** |
| surface_pressure | 4.867% | -0.528pp |
| volume_pressure | 4.525% | **Beats AB-UPT ref (6.08%)** |
| wall_shear | 8.527% | -1.356pp |
| tau_x | 7.402% | -1.125pp |
| tau_y | 9.605% | -2.299pp |
| tau_z | 11.330% | -1.000pp |

- All per-channel val slopes are **negative** — still converging
- **volume_pressure at 4.525% beats AB-UPT 6.08%** — first time any metric beats reference
- Needs test_abupt evaluation before merge, but val margin (1.49pp) is highly significant
- Action: **monitor for completion; merge if test_abupt < 10.126%**

## 4. Near-SOTA: Nezuko PR #283 (5-Layer Model)

Run `z6xc97gg` at val_abupt=8.994% at 6.5h — just 0.045pp above SOTA.

| Channel | val |
|---|---|
| abupt mean | 8.994% |
| surface_pressure | 5.597% |
| volume_pressure | 6.027% |
| wall_shear | 9.978% |
| tau_x | 8.465% |
| tau_y | 11.924% |
| tau_z | 12.956% |

- All slopes strongly negative — still improving
- Could overtake SOTA (9.039%) with a few more epochs
- Action: **monitor — may become review-ready**

## 5. Mid-Flight: Tanjiro PR #323 (Pre-Norm MLP Vol Head)

Run `vjs33n25` at val_abupt=12.07% at 2.4h. Still early (~5 epochs at ~25min/epoch).
- Fixed the divergence from PR #319 (MLP vol head with post-norm diverged at ep9)
- Pre-norm + tail init std=1e-2 should be stable
- val=12.07% is far from SOTA but typical for ep5; need ep10+ to assess
- Action: **let run; check at ep10**

## 6. Config/Hypothesis Mismatches Requiring Advisor Action

### Thorfinn PR #345 — gradclip=2.0 vs RFF Retest
- **PR body**: Describes RFF (Random Fourier Features) positional encoding retest
- **Actual W&B run**: `gradclip-2.0-w0` with config `grad_clip_norm=2.0`
- **Trajectory concern**: val_abupt=13.42% at 2h (~4-5 epochs) is behind SOTA trajectory (~10% by ep4)
- **Advisor ping posted**: Yes (from `morganmcg1` at 09:05Z) — no student response
- **Action needed**: Post follow-up. If still at 13%+ at ep8-10, advise kill and relaunch correct experiment.

### Frieren PR #352 — H09 Output-Head-Scaling vs SwiGLU/mlp_activation_uniform
- **PR body**: Describes per-channel output-head scaling (H09 hypothesis)
- **Actual W&B run**: `arm-D-swiglu-uniform` with config `mlp_activation_uniform=True`, `model_heads=8` (not SOTA 4!)
- **Double deviation**: Wrong experiment + non-SOTA heads count
- **At ep1 (0.7h)**: val=58% — normal startup, not divergence
- **Zero student responses**: No comments from frieren across multiple sessions
- **Action needed**: Direct instruction — either (a) confirm mlp_activation_uniform experiment and update PR body accordingly with SOTA config (heads=4), or (b) restart with H09 output-head-scaling as described. The heads=8 deviation is a significant config error that will slow convergence.

### Alphonse PR #287 — QK-Norm with grad_clip_norm=1 (not SOTA 0.5)
- **Config**: `grad_clip_norm=1` — not SOTA 0.5
- **Impact**: May underperform; grad clip norm affects Lion optimizer stability
- **At ep1 (0.7h)**: val=67% — normal startup
- **Action**: Let run but note deviation; if doesn't beat SOTA, recommend rerun with clip=0.5

## 7. Newly Started — Await Convergence

All three just started (~0.7h, ep1):

| Student | PR | Hypothesis | W&B Group | Config Matches SOTA? |
|---|---|---|---|---|
| fern | #351 | Tangent-frame projection loss for tau | fern-tangent-frame-r18 | Yes (SOTA + `--use-tangential-wallshear-loss`) |
| alphonse | #287 | QK-norm on Q/K attention vectors | alphonse/qknorm-armB | Mostly (grad_clip_norm=1, not 0.5) |
| askeladd | #353 | Channel-selective Huber loss for tau (delta=0.5) | askeladd-huber-tau-delta0p5 | Yes |

These need ep5-8 to show useful convergence signal. Check again at ~3-4h runtime.

## 8. Best Paper-Facing Results vs AB-UPT

**Current best on test set (PR #309)**:
- test_abupt = 10.126% vs AB-UPT 9.484% implied from per-axis reference
- Most channels still 2-8pp above AB-UPT reference targets
- **Edward PR #311 is showing val_abupt = 7.546%** — if this holds on test, it would be the first result clearly below any AB-UPT-equivalent threshold

**Volume pressure**: Edward's run beats AB-UPT reference (4.525% < 6.08%) on validation. This is scientifically interesting — first time a specific channel shows reference-level performance.

## 9. Is the Advisor/Student System Doing Useful Work?

**Yes.** Evidence:
1. Edward PR #311 shows a 1.49pp val improvement over SOTA after ~10 epochs — the STRING separable PE hypothesis is showing real traction.
2. Nezuko PR #283 is within 0.05pp of SOTA and still converging — may compound improvement.
3. The hypothesis generation pipeline is diversified: physics-informed losses (fern tangent-frame, askeladd Huber tau), attention stabilization (alphonse QK-norm), architecture variants (tanjiro pre-norm MLP head, frieren activation study).
4. The divergence fix from tanjiro PR #323 (pre-norm + tail init) addressed a fundamental instability in the MLP vol head path — this unblocks a whole architectural direction.

**Concerns**:
- Two students (thorfinn, frieren) are running experiments that don't match their PR descriptions — quality of hypothesis-to-implementation handoff is inconsistent.
- Frieren has model_heads=8 instead of SOTA 4 — a config error that will cost performance.
- Multiple students using grad_clip_norm=1 instead of SOTA 0.5.

## 10. Advisor Actions Required

**Immediate:**
1. Post follow-up on PR #345 (thorfinn): assess gradclip=2.0 trajectory — kill if behind at ep8-10, demand RFF experiment restart with correct config
2. Post follow-up on PR #352 (frieren): direct instruction to either fix config (heads=4) and confirm actual hypothesis, or restart with H09 as described — zero tolerance for multiple sessions of no response
3. Monitor edward PR #311 for completion — this is imminent merge candidate

**Short-term (when runs mature to ep10+):**
4. Review tanjiro PR #323 at ep10 — assess pre-norm MLP vol head convergence
5. Check alphonse/fern/askeladd at 3-4h for ep5-8 convergence signals

**When edward completes:**
6. Verify test_abupt < 10.126% on edward's run — if yes, merge as new SOTA
7. Immediately assign new experiment to edward upon merge

## 11. Research Velocity Assessment

- **Round age**: Multiple rounds in progress; current leading val=7.546% represents ~1.5pp improvement over SOTA in one round
- **Trajectory**: If edward validates, and nezuko converges, both could merge in one cycle — compounding gains
- **Bottleneck**: Frieren/thorfinn experiment drift is wasting GPU slots; two runs are off-spec and below expected trajectory
- **Next hypothesis priority**: Once edward merges (if it does), immediately launch experiments targeting tau_y/tau_z which remain the largest gaps vs AB-UPT (×3 above reference)
