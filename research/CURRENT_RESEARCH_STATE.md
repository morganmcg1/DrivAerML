# SENPAI Research State
- **Updated:** 2026-05-01 19:00 UTC
- **Branch:** `yi`
- **Baseline:** PR #183 (fern, `pos_max_wavelength=1000`), `abupt_axis_mean_rel_l2_pct = 10.21`

---

## Most Recent Research Direction from Human Researcher Team

**Issue #18 (Morgan, 2026-04-30T20:29:19Z — overarching directive):**
> Stop incremental tuning. Rip out the model architecture and try completely new approaches. Students can handle radical departures from the reference train.py as long as logging/validation/checkpointing are maintained.

**Morgan's ordered priority list:**
1. Surface-tangent frame wall-shear prediction — 4× wall shear y/z error is a coordinate frame mismatch
2. **Perceiver-IO backbone replacement** — ~3× faster per epoch = more epochs within budget (**ASSIGNED #212 noam**)
3. asinh/log target normalization — tested, **NEGATIVE — tail suppression problem (PR #123)**
4. Physics-informed RANS constraint — tested, **NEGATIVE — label contradiction: DrivAerML mesh GT has RMS(τ·n)/|τ|=12%, so (τ·n)=0 constraint contradicts labels (PR #201)**
5. 1-cycle LR schedule with higher peak — tested, **NEGATIVE — budget starvation + stability ceiling (PRs #164, #191)**

---

## Current Baseline

| Metric | yi best | AB-UPT | Gap |
|---|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **10.21** | — | — |
| `surface_pressure_rel_l2_pct` | **6.97** | 3.82 | 1.8× |
| `wall_shear_rel_l2_pct` | **11.69** | 7.29 | 1.6× |
| `volume_pressure_rel_l2_pct` | **7.85** | 6.08 | 1.3× |
| `wall_shear_x_rel_l2_pct` | **10.17** | 5.35 | 1.9× |
| `wall_shear_y_rel_l2_pct` | **13.73** | 3.65 | **3.8×** |
| `wall_shear_z_rel_l2_pct` | **14.73** | 3.63 | **4.1×** |

**The wall_shear_y/z gap at 3.8-4.1× AB-UPT is the primary research target.**

---

## Active WIP PRs (as of 2026-05-01 19:00 UTC)

### Round 14 Baseline Sweep (Just Assigned — ~Round 14)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#243** | chihiro | Sweep aux-rel-l2-weight {0.1, 0.5, 1.0} on 10.21 baseline | Training (90-100% GPU) |
| **#244** | emma | Sweep surface-loss-weight {1.5, 2.0} on 10.21 baseline | Training (90-100% GPU) |
| **#245** | gilbert | Progressive EMA decay schedule on 10.21 baseline | Training (90-100% GPU) |
| **#246** | tanjiro | Calibrate LR warmup {500, 1000 steps} on 10.21 baseline | Training (90-100% GPU) |

### Earlier-Round PRs Currently Running

| PR | Student | Hypothesis | Notes |
|---|---|---|---|
| **#227** | stark | Wall-shear in local surface tangent frame (Morgan's #1 directive) | **NO POD — RBAC blocked provisioning. Issue #248 filed for human operator.** |
| **#230** | senku | SWA uniform weight averaging for flat-minima generalization | SWA activates ep25–42 of 50 |
| **#229** | norman | y-flip test-time symmetry augmentation (TTA) | Pod restarted ~18:36 UTC; Iter 1 |
| **#228** | edward | OHEM hard surface-point weighting for tau_y/z gap | Running |
| **#225** | haku | Left-right symmetry augmentation (tau_y/z gap) | Arm C surviving; awaiting ep2 |
| **#224** | fern | Learned Fourier embeddings per-axis freq learning | Arms G/H/I/J relaunched |
| **#222** | (Round12) | 1-epoch LR warmup before cosine decay | Running |
| **#221** | violet | Per-channel adaptive loss reweighting toward AB-UPT targets | v6 (lr=3e-4, seed=1) running |
| **#218** | frieren | SO(3)-equivariant tangent-frame wall shear head | Running |
| **#214** | gilbert | k-NN local surface attention for wsy/wsz gap | Running |
| **#213** | nezuko | SAM optimizer for flat-minima generalization | ep1 underperforming control; awaiting ep2 |
| **#210** | kohaku | Gradient accumulation eff_bs=32 | Training (confirmed GPU activity) |
| **#209** | frieren | Step-decay LR drop after ep1 (seed=-1, no-warmup relaunches) | Awaiting ep1 vals |
| **#208** | askeladd | Sandwich-LN to unlock 8L/256d depth | Arm B (8L sandwich-LN, bs=4) running ~18866+ steps |
| **#207** | alphonse | AGC (NFNets) per-parameter stability | Arms I/J/K running |
| **#193** | thorfinn | Curvature-biased surface point sampling | alpha=0.25 arm running |
| **#152** | violet | Per-channel adaptive loss reweighting | v6 running |

### Pod Status (as of 2026-05-01 19:00 UTC)
- All 17 named student deployments: READY (1/1)
- `senpai-yi-stark`: **MISSING** — human provisioning required (Issue #248)

---

## Current Research Themes

### Theme 1: Coordinate-Frame Hypothesis for tau_y/z Gap (HIGHEST PRIORITY — Morgan's directive)
**Root question:** Is the 4× tau_y/z error a coordinate-frame problem? AB-UPT achieves equal error on tau_x/y/z (~3.6) while we have 10/14/15.

- **#227 stark:** Wall-shear prediction in local surface tangent frame {t1, t2, n}. Morgan's #1 directive. **POD MISSING — Issue #248 filed.**
- **#218 frieren:** SO(3)-equivariant tangent-frame wall shear head.
- **CLOSED NEGATIVE #121, #168:** Hard Duff-ONB (discontinuous at t1.x sign-flip) and normal-consistency penalty (model already near-tangential).

### Theme 2: Hyperparameter Calibration (Round 14 Sweep)
Systematic calibration on 10.21 baseline — four students sweeping complementary hyperparameters:
- **#243 chihiro:** aux-rel-l2-weight {0.1, 0.5, 1.0}
- **#244 emma:** surface-loss-weight {1.5, 2.0}
- **#245 gilbert:** progressive EMA decay schedule
- **#246 tanjiro:** LR warmup steps {500, 1000}

### Theme 3: Architecture Scaling
- **#208 askeladd:** Sandwich-LN to unlock 8L/256d depth — Arm B running stably at 18866+ steps.
- **#235 askeladd:** 4L/512d/8H width frontier — radford champion port.
- **#240 frieren:** Wider FFN (mlp-ratio=8) for richer per-block representation.
- **#241 tanjiro:** Width scaling 512→768d with µP-scaled LR.
- **PRIOR FINDING: Depth >> Width:** 6L/256d (4.73M) dominates 4L/512d (12.7M) — further depth exploration warranted if sandwich-LN stabilizes 8L.

### Theme 4: Optimizer and Training Infrastructure
- **#207 alphonse:** AGC (NFNets) per-parameter clipping. Arms I/J/K running.
- **#213 nezuko:** SAM optimizer — ep1 underperforming control; ep2 awaited.
- **#234 senku:** Mirror-symmetry TTA for wsy free gain.
- **#229 norman:** y-flip TTA (Pod restarted ~18:36 UTC).
- **#230 senku:** SWA uniform weight averaging.

### Theme 5: Loss Formulation and Sampling
- **#236 edward:** Fixed wsy/wsz loss multipliers — direct binding-constraint attack.
- **#237 haku:** Squared rel-L2 aux loss for hard-sample focusing.
- **#238 kohaku:** High-shear curriculum oversampling for wsy/wsz tail.
- **#225 haku:** Left-right symmetry augmentation.
- **#193 thorfinn:** Curvature-biased surface point sampling (alpha=0.25 arm).

### Theme 6: Positional Encoding
- **#224 fern:** Learned Fourier embeddings per-axis freq. Arms G/H/I/J relaunched.
- **#239 norman:** Fourier PE num_freqs sweep {16, 32, 64, 128}.
- **BASELINE: PR #183 (fern):** pos_max_wavelength=1000 ContinuousSincosEmbed — the current 10.21 bar.

### Theme 7: Adaptive Loss Reweighting
- **#221 violet:** Per-channel adaptive loss reweighting toward AB-UPT targets. v6 running (lr=3e-4, warmup=1000, seed=1).
- **#228 edward:** OHEM hard surface-point weighting.

### Theme 8: Architecture Depth Ablation (Round 12 PRs still running)
- **#231 (slices=64), #232 (heads=4), #233 (layers=3):** Ablate individual architectural dimensions from SOTA.

---

## Key Structural Findings Accumulated

### Stability
- **Fleet-wide instability mechanism (CONFIRMED):** Large-but-finite grad spikes bypass PR #169's NaN-skip. clip_grad_norm=1.0 normalizes direction but preserves poison vector → Adam m/v corruption.
- **FiLM stability axis is LR** (PR #184 closed): lr=4e-4 is the stability boundary; lr=5e-4 kills FiLM regardless of init_scale.
- **LR ceiling at ~6.5e-4** (PRs #164, #191): OneCycleLR peaks ≥ 6.5e-4 diverge regardless of warmup.
- **W_y=W_z > 2.0 overfits tau_y/z** (PR #66): W=3 scores 13.18 vs 12.74 for W=2.

### Target Representation
- **asinh normalization NEGATIVE** (PR #123): Compresses tail → suppresses gradient explosions but also suppresses learning signal where y/z gap lives.
- **Heavy-tail is real:** Wall shear spans 4 decades. The tail dominates rel_L2 numerator AND denominator. Cannot compress it.

### Architecture
- **Depth >> Width:** 6L/256d (4.73M) dominates 4L/512d (12.7M).
- **8L/256d is blocked by LR ceiling** (PRs #144, #164): Cannot train at lr ≥ 5e-4 with current norm structure. Sandwich-LN (PR #208) is the unlock attempt.
- **Multi-scale hierarchy failed** (PR #150): Receptive field is not the lever for tau_y/z gap.

### Loss/Data Representation
- **Per-axis wallshear upweighting:** W_y=W_z=2 is optimal (PR #66).
- **Explicit tangentiality enforcement provides no gain** (PRs #121, #168): Model already learns near-tangential predictions naturally.
- **Tangent-frame prediction (Duff ONB) discontinuous** at t1.x sign-flip → incompatible with Transolver. Use smooth e_x-projection frame (PR #199).

---

## Key Constraints
- Training budget: ~270 min training + ~90 min val/test = 360 min total (~3-4 epochs at 6L/256d)
- VRAM: 96 GB per GPU; 6L/256d at bs=8 uses ~75 GB
- Gradient clipping: clip_grad_norm=1.0 standard; clip=0.5 for stability-sensitive experiments
- LR warmup: 500 steps (1e-5 start) now standard
- Students have 4 GPUs each
