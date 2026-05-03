# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-03 11:30Z — frieren EP8 PASS (7.60%), askeladd EP3 ahead of SOTA, nezuko EP5 PASS (9.13%); 8 students active, 0 idle; alphonse #510 critical surface_loss=0 incident in progress
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #489 (thorfinn vol-points curriculum 16k→65k), val_abupt **7.1792%** (EP11)

W&B run `r5rw40rn`, group `thorfinn-vol-curriculum`. All future PRs must beat val_abupt < **7.1792%**.

| Metric | PR #489 SOTA (val EP11) | AB-UPT |
|---|---:|---:|
| `abupt` (val_abupt EP11) | **7.1792%** | — |
| `surface_pressure` (val) | 4.783% | 3.82% |
| `wall_shear` (val) | 8.098% | 7.29% |
| `volume_pressure` (val) | **4.207%** | **6.08% (BEATEN)** |
| `tau_x` (val) | 7.019% | 5.35% |
| `tau_y` (val) | 9.187% | 3.65% |
| `tau_z` (val) | 10.701% | 3.63% |

Test metrics (best-val checkpoint): test_abupt=8.497%

---

## Latest research direction from human researcher team

No new directives since last cycle. All open human issues already responded to. Working off Issue #252 (Modded-NanoGPT-derived levers) plus organic vol_p / tau-axis attack programme.

---

## Currently in-flight (8 active WIP PRs on tay, ZERO idle students)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #523 | thorfinn | GradNorm dynamic loss balancing (Chen et al. NeurIPS 2018) | full GradNorm crashed (5× overhead); pivoted to EMA-loss-proxy; advisor confirmed `r_i^alpha` formula at 11:26Z; new run launching |
| #499 | fern | TTA mirror-y (post-hoc on frozen #387) | EP5 gate miss (10.02% vs 9.2%) but trajectory healthy; EP6=9.54%; post-hoc eval planned at timeout |
| #496 | tanjiro | Uncertainty-weighted multitask loss (dual-optimizer fix) | relaunched with AdamW for log_sigma, monitor sigma spread at EP2 |
| #501 | frieren | Anisotropic per-axis STRING freq (sigma_x/y/z, 3 arms) | **EP8 GATE PASS**: Arm A val_abupt=7.60% (run `kvywdebn`), pulling ahead of SOTA on tau_y/tau_z |
| #506 | nezuko | Surface point density 2× (65k→131k surface pts) | **EP5 GATE PASS**: val_abupt=9.13% (run `e4gz48nf`); ~22% per-epoch overhead |
| #510 | alphonse | Surface-loss-weight sweep (multi-arm) | **CRITICAL**: run `ojept9ov` had `surface_loss=0` and no val metrics; advisor flagged at 11:15Z; awaiting kill+diagnose+relaunch |
| #511 | edward | Extended training to EP13 (best-val extension) | run `5o7jc7wi` healthy at ~EP6/13, monotonic descent |
| #516 | askeladd | Per-channel tau_y/tau_z loss reweighting | Run A (tau_y×2.0, tau_z×2.5) tracking ahead of SOTA EP3-for-EP3; run `jeagf5zr` |

---

## Recent closeouts

- **PR #489 thorfinn (vol-points curriculum 16k→65k) — MERGED NEW SOTA.** val_abupt=7.1792% (EP11), test_abupt=8.497%. −0.1880pp (−2.55%) vs #488. Architecture: 4-stage ramp 16k→32k→49k→65k vol-points over first 9 epochs. W&B run `r5rw40rn`.
- **PR #471 askeladd (signed-log transform) — CLOSED-NEG.** A/B test conclusive: arm-b (signed-log) EP9=10.5449% vs arm-a (control) ~7.96%. Signed-log makes vol_p worse, not better. Dead end confirmed.
- **PR #458 nezuko (mlp_ratio=8) — CLOSED-NEG.** Non-monotonic capacity: wider FFN hurts. mlp4 optimal. Dead end confirmed.
- **PR #454 frieren (tau_yz w=1.5) — CLOSED-NEG.** Third NEG result in tau_yz loss-weight family. Spawned anisotropic STRING attack #501.
- **PR #483 edward (surf↔vol cross-attn bridge) — CLOSED.** Did not beat SOTA. Edward reassigned to extended training #511.
- **PR #482 thorfinn (TTA mirror-y in training) — CLOSED (EP5 gate fail).** TTA must be inference-only (revisited as fern #499).
- **PR #467 alphonse (per-axis output scaling) — CLOSED-NEG.** Confirms tau_y/tau_z gap is upstream in spectral representation. Spawned PR #488.

---

## Latest signals

### Alphonse #510 — surface loss-weight sweep (CRITICAL incident in progress)
- Run `ojept9ov` showed `train/surface_loss=0` and `train/surface_loss_weighted=0` throughout; no val metrics; vol_points stuck at 16384
- Multiple branch-state issues: pre-rebase train.py loaded in memory at launch; entrypoint hard-reset undid local rebase; fix commit `41df492` checkout-from-tay applied but lost again on subsequent reset
- Advisor flagged at 11:15Z; awaiting alphonse kill+diagnose+relaunch (pod healthy, on iteration 496)

### Edward #511 — extended training EP13
- Fell back from EP15 to EP13 (budget math: 25.2 min/epoch × 15 = 378 min > 360 min hard timeout)
- Run `5o7jc7wi` healthy at ~EP6/13; val curve monotonically descending

### Fern #499 — TTA mirror-y (inference-only on frozen SOTA #387)
- EP5 gate miss (10.02% > 9.2%); trajectory normal (EP6=9.54%); not concerning
- Post-hoc eval plan confirmed: after timeout, run `post_hoc_eval.py` with TTA on/off, compare tau_y delta

### Tanjiro #496 — uncertainty-weighted multitask loss (dual-optimizer fix)
- Lion lockstep diagnosed: sign-update is scale-invariant when losses share order of magnitude → all 5 log_sigma move identically (spread 5e-4 at EP2)
- Fix: separate AdamW(lr=1e-3, wd=0) for log_sigma; Lion for backbone
- Awaiting EP1 sigma divergence + EP5 gate result

### Frieren #501 — anisotropic STRING frequencies (3-arm sweep: sigma_x/y/z)
- **EP8 GATE PASS** — Arm A (sigma_x=1.0, sigma_y=2.0, sigma_z=2.0) val_abupt=7.60% < 8.0% gate
- Run `kvywdebn`, group `frieren-aniso-string`
- Already pulling ahead of SOTA on asymmetric tau_y/tau_z channels — strongest live signal in fleet

### Nezuko #506 — surface pts 131k (2× density)
- **EP5 GATE PASS** — val_abupt=9.1316% ≤ 9.5% gate
- Run `e4gz48nf`; tau_y/z still elevated (11.94%/13.25%) but converging
- ~22% per-epoch time penalty

### Askeladd #516 — per-channel tau_y/tau_z loss reweighting
- Run A (tau_y×2.0, tau_z×2.5) EP3 val_abupt=13.543% vs alphonse SOTA EP3 13.990% (−0.447pp ahead)
- Run `jeagf5zr`, group `askeladd-tau-channel-reweight`
- Note: PR #142, #454, #467 family marked EXHAUSTED — Run A's early lead vs PR #488 is interesting and bears watching

### Thorfinn #523 — GradNorm dynamic loss balancing → EMA-loss-proxy pivot
- Full GradNorm (Chen 2018) crashed at step 10823 (5× autograd overhead untenable in 6h budget)
- Pivoted to EMA-loss-proxy: `--gradnorm-mode ema_proxy --gradnorm-alpha 1.5 --gradnorm-ema-beta 0.9 --gradnorm-init-warmup-steps 50`
- Formula deviation: implemented `w_i ∝ r_i^alpha` (relative training rate, original GradNorm) rather than literal `1/ema_loss_i` (which would invert sign)
- Advisor confirmed at 11:26Z: thorfinn's interpretation matches the stated intent; gates set EP3≤14.0%, EP5≤9.5%, EP8≤8.0%

---

## Cross-cutting observations

### vol_p: MISSION ACCOMPLISHED broadly
- SOTA (PR #489) has vp=4.207%, well below AB-UPT ref 6.08%
- Multiple approaches achieved sub-AB-UPT vol_p: multi-sigma STRING-sep (spectral), vol-curriculum (data density)
- **DO NOT stack target-space transforms (signed-log/log1p)** — transforms degrade the already-excellent vol_p

### tau_y/tau_z: PRIMARY OPEN PROBLEM
- Current SOTA per-axis (PR #489 val EP11): tau_y=**9.187%**, tau_z=**10.701%**
- AB-UPT references: tau_y=3.65%, tau_z=3.63% — gap remains large (~2.5–3× relative)
- Loss reweighting confirmed inert (#142, #454, #467 — three NEGs, family exhausted)
- Active spectral attack: frieren #501 anisotropic per-axis STRING freq
- Active dynamic balancing attack: thorfinn #523 GradNorm
- Active uncertainty-weighting attack: tanjiro #496 (dual-optimizer fix)
- Active per-channel reweighting: askeladd #516
- Active data density attack: nezuko #506 131k surface pts
- Active TTA attack: fern #499 mirror-y inference

---

## Current research focus and themes

1. **Closing the tau_y/tau_z gap (×2.5–3 vs AB-UPT) — full-fleet upstream attack:**
   - **Dynamic loss balancing (GradNorm):** PR #523 thorfinn — data-driven adaptive task weighting, targets tau_y/z convergence speed
   - **Spectral representation, anisotropic per-axis:** PR #501 frieren (sigma_x/y/z 3-arm sweep) — directly targets anisotropic flow features
   - **Multitask uncertainty weighting:** PR #496 tanjiro (dual-optimizer fix, AdamW for log_sigma)
   - **Per-channel reweighting:** PR #516 askeladd (tau_y/z direct weight boost)
   - **Surface resolution:** PR #506 nezuko (131k surface pts)
   - **Inference TTA:** PR #499 fern (mirror-y on frozen SOTA)
   - **Loss reweighting (confirmed inert):** PRs #142, #454, #467 — THREE NEG results, family exhausted

2. **Consolidating vol_p gains and pushing further:**
   - SOTA vp=4.207% (PR #489) well below AB-UPT 6.08%
   - **DO NOT** stack target-space transforms (signed-log/log1p) on top — confirmed NEGATIVE

3. **Data curriculum and resolution:**
   - PR #489 thorfinn vol-curriculum MERGED as new SOTA
   - PR #506 nezuko (65k→131k surface pts) — orthogonal surface resolution lever
   - PR #511 edward — extended training to EP15 on best-val checkpoint

4. **Capacity scaling — CONFIRMED DEAD END:**
   - mlp_ratio=6 and mlp_ratio=8 both NEGATIVE; mlp4 is optimal for this stack

---

## Negative results catalog (do not retry on current stack)

| Lever | Outcome |
|---|---|
| Local tangent-frame INPUT features | NEGATIVE (#423) |
| Tangent-frame OUTPUT decomposition | EXHAUSTED |
| Channel-selective Huber on tau | NEGATIVE (#353) |
| Volume-loss-weight scalar rebalancing | NEGATIVE (#451) |
| Separate volume decoder (capacity) | NEGATIVE val→test overfitting (#452) |
| Muon optimizer | NEGATIVE (#299) +4.09pp |
| Sandwich-norm | NEGATIVE diverged |
| U-net skips | NEGATIVE (+0.555pp) |
| Lion β values >0.99 | All closed on older stacks |
| dropout > 0 | Default 0.0 best |
| 256d / 768d hidden | NEGATIVE on multiple stacks |
| 6L / 8L depth | NEGATIVE |
| Learnable per-axis output head scaling (#467) | NEGATIVE — tau_y/tau_z gap is upstream |
| TTA mirror-y in training loop (#482) | NEGATIVE — TTA must be inference-only |
| tau_yz loss-weight reweighting (#142, #454, #467) | EXHAUSTED — three NEG results, problem is upstream |
| mlp_ratio=6 FFN wider | NEGATIVE (#458 run 1) — non-monotonic; mlp4 is optimal |
| mlp_ratio=8 FFN wider | NEGATIVE (#458 run 2) — non-monotonic; mlp4 is optimal |
| Signed-log target transform for vol_p (#471 arm-b) | NEGATIVE — 10.5449% vs 7.96% control; SOTA multi-sigma init already handles vol_p better |
| log1p target transform for vol_p (#481 tanjiro) | NEGATIVE headline — vol_p OK (4.83%) but tau_y/z not helped |

---

## Composition candidates (when winners land)

- thorfinn vol-curriculum (#489 MERGED SOTA) + fern TTA (#499) = orthogonal data density + inference augmentation
- frieren anisotropic STRING (#501) + thorfinn GradNorm (#523) = orthogonal spectral + dynamic balancing tau attack
- frieren anisotropic STRING (#501) + tanjiro uncertainty weighting (#496) = two orthogonal multitask balance approaches
- **DO NOT compose:** any two target-space transforms (signed-log × log1p both touch vol_p distribution)
- **DO NOT compose:** additional loss reweighting on top of STRING-sep SOTA (three NEGs, confirmed inert)

---

## Potential next research directions (Round 28+)

1. **Compose thorfinn vol-curriculum (#489 SOTA) + frieren anisotropic STRING (#501 if wins)** — stack spectral + data density.
2. **Compose thorfinn vol-curriculum (#489 SOTA) + GradNorm (#523 if wins)** — data density + dynamic balancing.
3. **Wavelet/multi-resolution input encoding** — alternative tau_y/z spectral attack if #501 fails.
4. **Anisotropic positional encoding** — separate freq sets per spatial axis; extends #501 frieren logic.
5. **Surface point density 3× (196608 pts)** — if 2× (#506) wins, escalate.
6. **Learnable Fourier basis (NTK-style)** — learn a small Fourier basis matrix instead of STRING-sep fixed sinusoids; richer spectral coverage.
7. **Signed-log target for surface_pressure** — if SOTA surface_p shows multiplicative zero-region errors (separate from vol_p).
8. **EMA model-soup average** — port from yi-track if it wins there.
9. **Test-time augmentation: mirror-y + reflect-z + 90-deg rotations** — extend TTA family if fern #499 wins.
10. **Vol-points curriculum + anisotropic STRING composition** — thorfinn curriculum provides data density; frieren #501 provides spectral coverage.
11. **Slice-conditioned FFN width** — wider FFN only in middle (volumetric) slices; avoids global capacity penalty.
12. **Physics-informed boundary condition loss** — enforce no-slip wall condition explicitly via auxiliary loss on tau magnitude near wall.
