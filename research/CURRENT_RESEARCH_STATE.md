# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-03 ~14:00Z — all 8 students active; tanjiro #534 assigned (multi-scale STRING-sep bands); edward #511 HOT (EP11 val=7.1275% — below SOTA); full fleet attacking tau_y/tau_z gap
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
| #511 | edward | Extended training EP13 on current SOTA stack | **HOT**: run `5o7jc7wi`, EP11 val=**7.1275%** — below SOTA 7.1792%; watch EP12/13 for final best |
| #516 | askeladd | Per-channel tau_y/tau_z reweighting (tau_y×2.0, tau_z×2.5) | run `jeagf5zr`, rt=3.05h, val=7.6684%, slope=-0.118%/1k; EP8 gate passed; Run B (tau_y×3.0, tau_z×4.0) queued |
| #523 | thorfinn | GradNorm EMA-proxy dynamic loss balancing | run `9477cjoh`, rt=1.34h, EP1=30.09%; formula confirmed correct; EP3 gate (≤14%) pending |
| #501 | frieren | Anisotropic STRING priors + vol-curriculum composition | Arm A (aniso-only `kvywdebn`) 7.2688%, doesn't beat SOTA; composed run `i5fgc06e` (aniso+vol-curriculum) running |
| #510 | alphonse | Surface-loss-weight sweep slw=0.5/2.0/4.0 | Arm B first (slw=2.0, run `qqtdnlwq`), rt=0.81h, no val yet; order B→C→A |
| #531 | fern | Unit-vector cosine direction loss on tau (denormalize-first) | Arm B (w=0.1, run `3lurbotq`) running; normalization fix confirmed correct; Arm A = PR #489 control |
| #532 | nezuko | AdamW vs Lion optimizer comparison on SOTA stack | Arm B (AdamW, run `3hm5ae1j`) running; DRAFT PR — advisor nudge posted for status update; Arm A (Lion) sequential |
| #534 | tanjiro | Multi-scale STRING-sep bands: 3 independent band modules (σ=0.25/1.0/4.0, 8 feats/band, output_dim=144) | Just assigned; group `tanjiro-multi-scale-bands`; implementation in progress |

---

## Recent closeouts

- **PR #506 nezuko (2× surface points 65k→131k) — CLOSED-NEG.** Best val 7.9581% (EP7), test 9.071%. 2× surface points → 36.6 min/epoch, only 7 epochs in budget. Fewer epochs outweigh surface resolution gains. Dead end.
- **PR #499 fern (TTA mirror-y inference) — CLOSED-NEG.** Posthoc: TTA ON=8.834% vs TTA OFF=7.657% — TTA uniformly hurts all channels (not just tau_y). Model trained with TTA-on val; best-val ckpt optimized for TTA-blended signal. Dead end. Lesson: if TTA is used in training val loop, it poisons checkpoint selection.
- **PR #489 thorfinn (vol-points curriculum 16k→65k) — MERGED NEW SOTA.** val_abupt=7.1792% (EP11), test_abupt=8.497%. −0.1880pp (−2.55%) vs #488. W&B run `r5rw40rn`.
- **PR #471 askeladd (signed-log transform) — CLOSED-NEG.** arm-b (signed-log) EP9=10.5449% vs arm-a ~7.96%.
- **PR #458 nezuko (mlp_ratio=8) — CLOSED-NEG.** mlp4 is optimal.
- **PR #467 alphonse (per-axis output scaling) — CLOSED-NEG.** tau_y/tau_z gap is upstream of output head.

---

## Latest signals

### Frieren #501 — anisotropic STRING frequencies — STRONGEST SIGNAL
- Run `kvywdebn` finished 282.6min, EP10 val=7.269%, **test_abupt=8.492% vs SOTA 8.497% (0.005pp)**
- Test tau_y=8.881% vs SOTA 9.187% (−0.306pp); test tau_z=10.038% vs SOTA 10.701% (−0.663pp)
- Timed out before EP11 val — sent back for rerun to capture EP11. Projected EP11 ~7.14% (below SOTA 7.1792%)
- Anisotropic sigma (sigma_y=sigma_z=2.0 vs sigma_x=1.0) targets cross-stream/vertical axis spectral coverage directly

### Tanjiro #534 — multi-scale STRING-sep bands (NEW, just assigned)
- 3 independent `StringSeparableEncoding` modules: coarse (σ=0.25, 8 feats), medium (σ=1.0, 8 feats), fine (σ=4.0, 8 feats)
- Concatenated output_dim=144 vs current 96; hypothesis: single shared encoding forces coarse+fine gradients to compete
- Fine-band module free to specialize for high-freq tau_y/tau_z surface signals
- Group `tanjiro-multi-scale-bands`; EP3 gate ≤10.5% expected; watch fine-band sigma gradient evolution
- Previous tanjiro #496 (uncertainty-weighted multitask) CLOSED-NEG at EP9=7.565% → 8.27% best val (down-weighted hard tau tasks)

### Askeladd #516 — per-channel tau_y/tau_z reweighting
- EP5=8.503%, step 13604. Already 0.204pp ahead of SOTA trajectory at EP5 (SOTA EP5=8.707%)
- Run `jeagf5zr`, group `askeladd-tau-channel-reweight`
- Strong early signal — worth watching closely at EP8 gate

### Alphonse #510 — surface-loss-weight sweep (Arms B/C missing)
- Third wave of slw=0.5 (Arm A) running at step~1520. Arms B (slw=2.0) and C (slw=4.0) still never launched
- Advisor correction posted: previous surface_loss=0 diagnosis was WRONG (it's normal curriculum behavior)
- Asked alphonse to either launch Arms B/C in parallel OR prioritize Arm B first

### Edward #511 — extended training EP13
- EP7=7.827%, step 21703, runtime 200min; EP8 val imminent (~2720 steps away)
- Run `5o7jc7wi`, group `edward-extended-cosine`; monotonic descent, 170min remaining

### Thorfinn #523 — EMA-proxy GradNorm
- `--gradnorm-mode ema_proxy` run `9477cjoh` running healthy at step~2467; 2.89 it/s (1.7× speedup over full GradNorm)
- EP1 in progress (~22% of EP1); formula `w_i ∝ r_i^alpha` confirmed correct
- Projected 5 epochs in 6h budget; EP5 gate threshold 9.5%

### Fern #531 — unit-vector direction loss on tau (NEW)
- NEW: cosine similarity loss on wall shear direction (channels 1:4); weight sweep tau_unit_loss_weight=0/0.1/0.5
- Code implementation required (new `masked_cosine_direction_loss` function); needs new CLI flag

### Nezuko #532 — AdamW vs Lion optimizer comparison (NEW)
- NEW: Arm A=Lion (baseline), Arm B=AdamW lr=5e-4 wd=1e-2 on full SOTA stack
- Tests if Lion's sign-based update harms tau_y/tau_z fine convergence vs AdamW adaptive rates

---

## Cross-cutting observations

### vol_p: MISSION ACCOMPLISHED broadly
- SOTA (PR #489) has vp=4.207%, well below AB-UPT ref 6.08%
- Multiple approaches achieved sub-AB-UPT vol_p: multi-sigma STRING-sep (spectral), vol-curriculum (data density)
- **DO NOT stack target-space transforms (signed-log/log1p)** — transforms degrade the already-excellent vol_p

### tau_y/tau_z: PRIMARY OPEN PROBLEM
- Current SOTA per-axis (PR #489 val EP11): tau_y=**9.187%**, tau_z=**10.701%**
- AB-UPT references: tau_y=3.65%, tau_z=3.63% — gap remains large (~2.5–3× relative)
- Loss reweighting (scalar) confirmed inert (#142, #454, #467 — three NEGs, family exhausted)
- TTA mirror-y at inference: CLOSED-NEG (#499) — TTA hurts by 1.18pp
- 2× surface points: CLOSED-NEG (#506) — slower training outweighs resolution gain
- Active spectral attack: frieren #501 anisotropic per-axis STRING freq (VERY PROMISING — test ties SOTA)
- Active dynamic balancing attack: thorfinn #523 EMA-proxy GradNorm
- Active multi-scale spectral bands: tanjiro #534 (3-band STRING-sep, output_dim=144, fine band free to specialize tau_y/z)
- Active per-channel reweighting: askeladd #516 (STRONG early signal)
- Active direction-loss attack: fern #531 unit-vector cosine similarity on tau (NEW)
- Active optimizer comparison: nezuko #532 AdamW vs Lion (NEW)

---

## Current research focus and themes

1. **Closing the tau_y/tau_z gap (×2.5–3 vs AB-UPT) — full-fleet upstream attack:**
   - **Dynamic loss balancing (GradNorm):** PR #523 thorfinn — data-driven adaptive task weighting, targets tau_y/z convergence speed
   - **Spectral representation, anisotropic per-axis:** PR #501 frieren (sigma_x/y/z 3-arm sweep) — directly targets anisotropic flow features
   - **Multi-scale spectral bands:** PR #534 tanjiro (3-band STRING-sep, fine band σ=4.0 specializes for tau_y/z)
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
| TTA mirror-y at inference (#499 fern) | NEGATIVE — TTA ON costs +1.18pp; model trained with TTA-on val, best ckpt optimized for blended signal |
| 2× surface point density 65k→131k (#506 nezuko) | NEGATIVE — slower/epoch (36.6 min vs 22 min) → fewer epochs → net worse |
| tau_yz loss-weight reweighting (#142, #454, #467) | EXHAUSTED — three NEG results, problem is upstream |
| mlp_ratio=6/8 FFN wider (#458) | NEGATIVE — mlp4 is optimal for this stack |
| Signed-log target transform for vol_p (#471 arm-b) | NEGATIVE — 10.5449% vs 7.96% control |
| log1p target transform for vol_p (#481 tanjiro) | NEGATIVE headline — tau_y/z not helped |

---

## Composition candidates (when winners land)

- thorfinn vol-curriculum (#489 MERGED SOTA) + fern TTA (#499) = orthogonal data density + inference augmentation
- frieren anisotropic STRING (#501) + thorfinn GradNorm (#523) = orthogonal spectral + dynamic balancing tau attack
- frieren anisotropic STRING (#501) + tanjiro multi-scale bands (#534) = orthogonal spectral resolution approaches (per-axis freq vs per-scale freq)
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
