# SENPAI Research State
- **Updated:** 2026-05-01 14:30 UTC
- **Branch:** `yi`
- **Baseline:** PR #99 (fern), `abupt_axis_mean_rel_l2_pct = 10.69`, W&B run `3hljb0mg`

---

## Most Recent Research Direction from Human Researcher Team

**Issue #18 (Morgan, 2026-04-30T20:29:19Z — overarching directive):**
> Stop incremental tuning. Rip out the model architecture and try completely new approaches. Students can handle radical departures from the reference train.py as long as logging/validation/checkpointing are maintained.

**Morgan's ordered priority list:**
1. Surface-tangent frame wall-shear prediction — 4× wall shear y/z error is a coordinate frame mismatch
2. **Perceiver-IO backbone replacement** — ~3× faster per epoch = more epochs within budget (**ASSIGNED #212 noam**)
3. asinh/log target normalization — tested, **NEGATIVE — tail suppression problem (PR #123)**
4. Physics-informed RANS constraint — soft divergence-free penalty (**ASSIGNED #201 nezuko**)
5. 1-cycle LR schedule with higher peak — tested, **NEGATIVE — budget starvation + stability ceiling (PRs #164, #191)**

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

## Active WIP PRs (as of 2026-05-01 14:30 UTC)

### Round-6 Assignments (Just Assigned)

| PR | Student | Hypothesis |
|---|---|---|
| **#207** | alphonse | Adaptive Gradient Clipping (AGC, NFNets) per-parameter stability |
| **#208** | askeladd | Sandwich-LN normalization to unlock 8L/256d depth |
| **#209** | frieren | Step-decay LR drop after ep1 (5e-4→1e-4, attacks ep1→ep2 divergence) |
| **#210** | kohaku | Gradient accumulation eff_bs=32 for smoother tau_y/z grads |
| **#211** | tanjiro | Relative magnitude-based grad-skip (EMA-adaptive) fleet infra |
| **#212** | noam | Perceiver-IO backbone replacement (2-4× speed → more epochs) |

### Ongoing From Earlier Rounds

| PR | Student | Hypothesis |
|---|---|---|
| **#201** | nezuko | Physics-informed RANS divergence-free penalty on volume velocity |
| **#200** | emma | Wall-shear magnitude/direction decomposition loss (τ y/z gap) |
| **#199** | stark | Smooth tangent-frame wall-shear prediction (continuous e_x-projection) |
| **#198** | senku | Stochastic Weight Averaging (SWA) free gain |
| **#197** | gilbert | k-NN local surface attention for tau_y/z gap |
| **#196** | edward | Lion optimizer sweep vs AdamW |
| **#193** | thorfinn | Curvature-biased surface point sampling |
| **#191** | haku | 1-cycle LR max=1e-3 (corrected epoch-limited schedule) |
| **#183** | fern | Omega-bank frequency sweep |
| **#171** | norman | Snapshot ensemble with cyclic LR |
| **#165** | chihiro | mlp_ratio=8 hardened (3-seed) |
| **#152** | violet | 14-dim analytic geometric moment conditioning |

---

## Round-6 Closed PRs (Negatives)

| PR | Student | Hypothesis | Best Val | Verdict |
|---|---|---|---:|---|
| **#168** | askeladd | Normal-consistency penalty λ∈{0.01,0.05,0.10} | 12.285 (ep2) | NEGATIVE — tangentiality enforcement provides no metric gain |
| **#164** | alphonse | 8L/256d + 1cycle LR | DNF (all diverged) | NEGATIVE — 8L/256d has LR ceiling < 5e-4 |
| **#123** | frieren | asinh/log wall-shear target normalization | 17.55 (ep1) | NEGATIVE — tail suppression kills the signal |

---

## Current Research Themes

### Theme 1: Coordinate-Frame Hypothesis for tau_y/z Gap (HIGHEST PRIORITY)
**Root question:** Is the 4× tau_y/z error a coordinate-frame problem? AB-UPT achieves equal error on tau_x/y/z (~3.6) while we have 10/14/15.

- **#199 stark:** Full tangent-frame prediction (smooth e_x-projection frame). Morgan's #1 directive.
- **#200 emma:** Magnitude/direction decomposition loss — decouple scale from alignment learning.
- **CLOSED NEGATIVE #168 askeladd:** Normal-consistency penalty — model already near-tangential naturally.
- **CLOSED NEGATIVE #121:** Hard Duff-ONB tangent-frame — discontinuous at t1.x sign-flip.

### Theme 2: Architecture Replacement (BOLD SWINGS — Morgan's directive)
- **#212 noam (NEW):** Perceiver-IO backbone — cross-attention bottleneck, 2-4× faster per epoch → 5+ epochs in budget. Morgan's #2 priority.
- **#208 askeladd (NEW):** Sandwich-LN to unlock 8L/256d depth — prior 8L attempts all diverged; sandwich-LN dampens gradient growth across depth.

### Theme 3: Optimizer Stability Infrastructure
**Four independent confirmations (PRs #123, #168, #165, #164):** Large-but-finite grad spikes bypass PR #169's NaN-skip.
- **#211 tanjiro (NEW):** Relative magnitude-based grad-skip (EMA-adaptive threshold).
- **#207 alphonse (NEW):** AGC per-parameter clipping (NFNets). Addresses root cause by making clip threshold proportional to weight norm.
- Together these form a complementary infra pair; combine if both show promise.

### Theme 4: LR Schedule Optimization
- **#209 frieren (NEW):** Step-decay LR drop after ep1 (5e-4→1e-4). Attacks the universal ep1→ep2 train/val divergence observed across 4 normalization variants in PR #123.
- **CLOSED NEGATIVE #164/#191:** OneCycleLR — 8L/256d and LR ceiling at ~6.5e-4 confirmed.

### Theme 5: Batch Statistics and Gradient Quality
- **#210 kohaku (NEW):** Gradient accumulation eff_bs=32. Zero VRAM cost; smoother gradient estimates for rare high-|τ| tail points.
- **#183 fern:** Omega-bank frequency sweep. Three arms healthy.

### Theme 6: Physics-Informed Constraints
- **#201 nezuko:** RANS divergence-free penalty on volume velocity. Morgan's #4 directive.
- **CLOSED NEGATIVE #168:** Normal-consistency (tangential) constraint.

### Theme 7: Ensemble and Budget Efficiency
- **#171 norman:** Snapshot ensemble with cyclic LR (V2).
- **#198 senku:** SWA — free post-train gain from weight averaging.
- **#196 edward:** Lion optimizer sweep.

### Theme 8: Data Representation and Sampling
- **#197 gilbert:** k-NN local surface attention for tau_y/z spatial locality.
- **#193 thorfinn:** Curvature-biased surface point sampling.
- **#152 violet:** 14-dim analytic geometric moment conditioning.

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
