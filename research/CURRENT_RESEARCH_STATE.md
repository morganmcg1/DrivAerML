# SENPAI Research State
- **Updated:** 2026-05-01 11:45 UTC
- **Branch:** `yi`
- **Baseline:** PR #99 (fern), `abupt_axis_mean_rel_l2_pct = 10.69`, W&B run `3hljb0mg`

---

## Most Recent Research Direction from Human Researcher Team

**Issue #18 (Morgan, 2026-04-30T20:29:19Z — overarching directive):**
> Stop incremental tuning. Rip out the model architecture and try completely new approaches. Students can handle radical departures from the reference train.py as long as logging/validation/checkpointing are maintained.

**Morgan's ordered priority list:**
1. Surface-tangent frame wall-shear prediction — 4× wall shear y/z error is a coordinate frame mismatch
2. Perceiver-IO backbone replacement — ~3× faster per epoch = more epochs within budget
3. asinh/log target normalization — wall shear spans 4 decades (tested, **NEGATIVE — tail suppression problem**)
4. Physics-informed RANS constraint — soft divergence-free penalty on predicted velocity at volume points
5. 1-cycle LR schedule with higher peak (tested, **NEGATIVE — budget starvation + stability ceiling at ~6.5e-4**)

---

## Current Baseline

| Metric | yi best | AB-UPT | Gap |
|---|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **10.69** | — | — |
| `surface_pressure_rel_l2_pct` | **6.97** | 3.82 | 1.8× |
| `wall_shear_rel_l2_pct` | **11.69** | 7.29 | 1.6× |
| `volume_pressure_rel_l2_pct` | **7.85** | 6.08 | 1.3× |
| `wall_shear_x_rel_l2_pct` | **10.17** | 5.35 | 1.9× |
| `wall_shear_y_rel_l2_pct` | **13.73** | 3.65 | **3.8×** |
| `wall_shear_z_rel_l2_pct` | **14.73** | 3.63 | **4.1×** |

**The wall_shear_y/z gap at 3.8-4.1× AB-UPT is the primary research target.**

---

## Active WIP PRs (as of 2026-05-01 11:45 UTC)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#199** | stark | Smooth tangent-frame wall-shear prediction (continuous e_x-projection frame, 2D output) | JUST ASSIGNED |
| **#200** | emma | Wall-shear magnitude/direction decomposition loss (log-mag + cosine direction) | JUST ASSIGNED |
| **#198** | senku | Stochastic Weight Averaging (SWA) free gain (r12) | WIP, no data yet |
| **#197** | gilbert | k-NN local surface attention for tau_y/z gap (r12) | WIP, no data yet |
| **#196** | edward | Lion optimizer sweep vs AdamW (r12) | WIP, no data yet |
| **#193** | thorfinn | Curvature-biased surface point sampling | WIP, 1 comment only |
| **#192** | tanjiro | asinh target normalization for tau_y/z (separate from frieren's approach) | WIP, 4-seed relaunch w/ warmup-500 |
| **#191** | haku | 1-cycle LR max=1e-3 (corrected epoch-limited schedule) | WIP, A_tuned running |
| **#184** | kohaku | FiLM with identity/zero-init (DiT-style) | **arm B (lr=4e-4) sole survivor, ep2≈99%, ETA ~14:25 UTC** |
| **#183** | fern | Omega-bank frequency sweep | **3 arms healthy (A2/C3/D3); mw=100 FALSIFIED; ep2 results ~12:00-12:20 UTC** |
| **#171** | norman | Snapshot ensemble with cyclic LR | **V2 running (okm6uoea, eta_min=5e-5 + clip=0.5); ETA ~17:30 UTC** |
| **#168** | askeladd | Normal-consistency penalty λ∈{0.01,0.05,0.10} | **λ=0.10 (gawdh7ah) ep1=17.103; clip=0.5 relaunches for λ=0.01/0.05 in flight** |
| **#165** | chihiro | mlp_ratio=8 hardened (3-seed) | **seed1337 ep3=11.92 (NOT beating 10.69); clip=0.5 go/no-go at 12:20/13:30 UTC** |
| **#164** | alphonse | 8L/256d + 1cycle LR recovery | WIP |
| **#152** | violet | 14-dim analytic geometry moment conditioning | WIP |
| **#151** | nezuko | L/R symmetry augmentation (tau_y gap) | WIP |
| **#123** | frieren | asinh/log wall-shear target normalization | **Critical finding: asinh-1.0 trades metric for stability; arms A/D/B(v3p1) final results ~12:10-13:51 UTC** |

---

## Current Research Themes

### Theme 1: Coordinate-Frame Hypothesis for tau_y/z Gap (HIGHEST PRIORITY)
**Root question:** Is the 4× tau_y/z error a coordinate-frame problem? AB-UPT achieves equal error on tau_x/y/z (~3.6) while we have 10/14/15. The asymmetry exists in our model but not in AB-UPT.

- **#199 stark (NEW):** Full tangent-frame prediction (smooth e_x-projection frame). Addresses Morgan's #1 directive. Different from PR #121 (Duff ONB discontinuity) — uses continuous frame with fallback at poles.
- **#168 askeladd:** Normal-consistency penalty (soft tangentiality constraint from the other direction)
- **#200 emma (NEW):** Magnitude/direction decomposition loss — separate log-mag and cosine-direction losses to decouple scale from alignment learning
- **#192 tanjiro:** asinh normalization applied specifically to y/z channels

### Theme 2: Target Representation for Heavy-Tail Distributions
- **#123 frieren (near completion):** PARTIAL RESULT — asinh-1.0 is NEGATIVE (suppresses tail learning signal). log1p (arm D) and asinh-0.5 (arm B) still viable; final results ~12:10-13:51 UTC.
- **Key finding:** Cannot suppress tail to gain stability — the gap lives in the tail. Need a way to handle heavy-tailed targets WITHOUT compression.

### Theme 3: Optimizer Stability Infrastructure (CONFIRMED FLEET-WIDE GAP)
**Four independent confirmations (frieren #123, fern #183, askeladd #168, chihiro #165):** PR #169's NaN-skip only catches non-finite gradients. Large-but-finite spikes (165, 252, 2.2M) bypass it, corrupt Adam m/v after clipping.
- Needed: magnitude-based grad-skip (pre_clip_norm > N × running_median, or abs threshold)
- Also needed: finite-but-pathological loss guard (abort if train_loss > 5× running_median for sustained steps)
- **This should be an infrastructure PR** — candidate to assign to the next available thorfinn/infra-capable student

### Theme 4: FiLM Geometry Conditioning (Near Complete)
- **#184 kohaku:** FiLM stability characterized. 5/5 at lr=5e-4 dead regardless of init_scale. Stability axis is lr alone. Arm B (lr=4e-4) sole survivor, completing ~14:25 UTC.
- **Key finding:** FiLM requires lr ≤ 4e-4, which may conflict with the lr=5e-4 optimum. The volume_pressure signal from PR #126 Arm-3 (vp=7.05 vs 7.85 baseline) is real — FiLM may still help volume even if it can't close the tau_y/z gap.

### Theme 5: Training Budget Efficiency
- **#171 norman:** Snapshot ensemble with cyclic LR. V1 diverged at epoch 3 (50× LR jump); V2 has 10× ratio + clip=0.5, running cleanly, ETA ~17:30 UTC.
- **#196 edward:** Lion optimizer (round 12) — Lion typically needs ~3× lower LR than AdamW
- **#198 senku:** SWA — free post-train gain from averaging model weights across last epochs
- **#197 gilbert:** k-NN local attention — addresses spatial locality for tau_y/z

### Theme 6: Architecture Exploration (Round 12)
- **#191 haku:** 1cycle LR (corrected — calibrated to actual epoch budget)
- **#164 alphonse:** 8L/256d depth with 1cycle (time-limited recovery — must beat PR #144 ep4 val=12.69)

---

## Key Structural Findings Accumulated

### Stability
- **Fleet-wide instability mechanism (CONFIRMED):** Large-but-finite grad spikes bypass PR #169's NaN-skip. clip_grad_norm=1.0 normalizes direction but preserves poison vector → Adam m/v corruption. Fix needed: magnitude-based skip.
- **FiLM stability axis is LR, not init_scale:** scale=0.001 delays divergence 6× vs scale=1.0 but cannot prevent it at lr=5e-4. lr=4e-4 is the stability boundary.
- **mw=100 (omega-bank) structurally untenable:** 3 independent attempts all diverged. Low max_wavelength → highly compressed Fourier features → fragile at lr=5e-4.
- **LR ceiling at ~6.5e-4:** OneCycleLR peaks ≥ 6.5e-4 diverge regardless of warmup schedule.
- **W_y=W_z > 2.0 overfits tau_y/z:** W=3 (PR #66) scores 13.18 vs 12.74 for W=2. Static W=2 is the sweet spot.

### Target Representation
- **asinh-1.0 trades metric for stability (frieren #123):** Suppresses gradient explosions by compressing the tail, but the gap to AB-UPT lives in the tail. Stability ≠ metric improvement.
- **Heavy-tail is real:** Wall shear spans 4 decades. The tail (high-|τ| separation regions) dominates the rel_L2 numerator AND denominator. Compression hurts.

### Architecture
- **Depth >> Width:** 6L/256d (4.73M) dominates 4L/512d (12.7M). 8L/256d time-limited but promising.
- **Multi-scale hierarchy failed (PR #150):** Receptive field is not the lever for tau_y/z gap.
- **384d in bf16 is unstable:** QK-norm + fp32 attention required (PR #170 gilbert testing).

### Loss/Data Representation
- **Per-axis wallshear upweighting:** W_y=W_z=2 is optimal (PR #66). W=3+ overfits, W=1.5- underdirects.
- **Normal-consistency λ ranking inversion:** λ=0.10 most stable, λ=0.01 most unstable — larger constraint enforces stronger out-of-plane avoidance, bounding the squared-dot spike.
- **Tangent-frame prediction (Duff ONB):** Discontinuous at t1.x sign-flip → incompatible with non-gauge-equivariant Transolver. Use smooth e_x-projection frame instead.
- **Coord-normalization wrong lever (PR #143):** global-scale breaks meter-calibrated omega bank; per-axis causes volume-token explosions.

### Optimizer
- **EMA m/v desynchronization:** Mid-run schedule changes → second-moment coupling issues. Start Adam with training-time weights.
- **Uniform surface_sw amplifies already-upweighted channels.** Per-component weight is the right knob.

---

## Near-Term Priority Queue

1. **Await and merge #184 (kohaku FiLM, arm B) ~14:25 UTC** — if val_abupt < 10.69
2. **Review #123 (frieren) final results ~12:10-13:51 UTC** — close based on best arm
3. **Monitor #165 (chihiro)** — seed42 go/no-go at step 6783 (~12:20 UTC)
4. **Review #183 (fern)** — cross-arm ep2 table at ~12:30 UTC
5. **Infrastructure PR** — magnitude-based grad-skip (assign to thorfinn when free)
6. **Perceiver-IO backbone** (Morgan's #2 directive) — not yet assigned; needs careful scoping
7. **Physics-informed RANS** (Morgan's #4 directive) — not yet assigned

---

## Key Constraints
- Training budget: ~270 min training + ~90 min val/test = 360 min total (~3-4 epochs at 6L/256d)
- VRAM: 96 GB per GPU; 6L/256d at bs=8 uses ~75 GB
- Gradient clipping: clip_grad_norm=1.0 standard; clip=0.5 used for stability-sensitive experiments
- LR warmup: 500 steps (1e-5 start) now standard for experiments with elevated initial gradients
- Students have 4 GPUs each
