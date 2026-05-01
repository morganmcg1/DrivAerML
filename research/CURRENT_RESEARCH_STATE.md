# SENPAI Research State
- **Updated:** 2026-05-01 21:40 UTC
- **Branches:** `yi` (4L/512d Lion SOTA), `bengio` (4L/256d AdamW Wave 2/3)

---

## Most Recent Research Direction from Human Researcher Team

**Issue #252 (Morgan, 2026-05-01):** Get inspired by Modded-NanoGPT — review the world-record table and propose specific DrivAerML experiments. ADVISOR acknowledged at 19:48Z; 4 directions now assigned (Muon, linear-warmdown LR, U-net skips, tanh soft-cap).

**Issue #18 (Morgan, 2026-04-30, ongoing directive):** Stop incremental tuning. Take bigger architectural swings. Compounding small wins is also OK as long as they keep landing.

---

## Current Baselines

### yi branch (advisor branch, primary fleet)

**Bar to beat: val_abupt = 9.2910% (PR #222 fern, lr_warmup_epochs=1).** W&B run `ut1qmc3i`.

| Metric | yi best | AB-UPT | Ratio |
|---|---:|---:|---:|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **9.2910** | — | — |
| `val_primary/surface_pressure_rel_l2_pct` | **5.8707** | 3.82 | 1.54× |
| `val_primary/wall_shear_rel_l2_pct` | **10.3423** | 7.29 | 1.42× |
| `val_primary/volume_pressure_rel_l2_pct` | **5.8789** | 6.08 | **0.97×** ✓ |

Volume pressure has now beaten AB-UPT. Surface pressure and wall_shear remain the gap.

### bengio branch (Wave 2/3 long-schedule experiments)

**Bar to beat: val_abupt = 7.2091% at ep30 (alphonse `m9775k1v`, 4L/256d, AdamW, T_max=30).**

---

## Active WIP PRs (as of 2026-05-01 20:25 UTC)

### yi branch — 17 WIP PRs (zero idle)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #280 | frieren | **MLP activation ablation** (SwiGLU / ReLU² vs GELU) | NEW (just assigned) |
| ~~#279~~ | ~~frieren~~ | ~~no-slip BC penalty~~ | CLOSED 21:40Z — duplicates failed PR #201 (GT tau·n RMS=12% contradicts continuum BC) |
| #270 | violet | **tanh output soft-cap** (modded-NanoGPT logit-cap analog) | NEW (just assigned) |
| #262 | nezuko | **linear-warmdown LR schedule** (modded-NanoGPT WSD-style) | NEW (just assigned) |
| #261 | norman | **Muon optimizer** (Newton-Schulz orthogonalized momentum) | NEW (just assigned) |
| #249 | tanjiro | asinh wall-shear normalization | TempCollapse bug found; advisor authorized `T.clamp(min=1e-2)` + relaunch with `--lr-warmup-epochs 1`; sent back to wip 21:35Z |
| #247 | thorfinn | cosine T_max=14 (between 9 and 50) | Run live, mid-ep3 |
| #245 | gilbert | progressive EMA decay (0.99→0.9999 etc.) | Arm A surviving; B/C retries running |
| #244 | emma | surface-loss-weight {1.5, 2.0} | Arm A crashed lr=5e-4; Arm B healthy |
| #243 | chihiro | aux-rel-l2-weight {0.1, 0.5, 1.0} | A/B retries at lr=3e-4; C healthy |
| #239 | norman | Fourier PE num_freqs sweep (bengio-branch work) | NF=16 running, ep5 gate ~5h |
| #230 | senku | SWA tail-end weight averaging | All 4 arms running, no signal until ep25+ |
| #228 | edward | OHEM hard surface-point weighting | Arm B (f=1.0) running |
| #227 | stark | wall-shear in surface tangent frame | NO POD — RBAC blocked, Issue #248 |
| #225 | haku | mirror symmetry training augmentation | Running |
| #224 | fern | learned Fourier embeddings per-axis | K/L/N/O surviving; J finding: init=10 beats sincos by 6.7% at matched ep1 step |
| #221 | violet | per-channel adaptive loss reweighting | Run A `541ru1pv` ep5=11.12; gate-check stalled (cross-listed, primary on bengio) |
| #210 | kohaku | gradient accumulation eff_bs=32 | Running |
| ~~#209~~ | ~~frieren~~ | ~~step-decay LR drop after ep1~~ | CLOSED 2026-05-01 21:22 — hypothesis rejected; control 10.08 vs bar 9.291 on legacy stack |
| #208 | askeladd | sandwich-LN to unlock 8L/256d | Arm B running ~18866+ steps |
| #207 | alphonse | Adaptive Gradient Clipping (AGC) | lr=3e-4 arms only surviving |
| #193 | thorfinn | curvature-biased surface point sampling | 3 lr=3e-4 arms healthy past warmup |

### bengio branch — 16 WIP PRs (zero idle)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #266 | stark | **U-net long skip connections** (modded-NanoGPT-inspired) | NEW (just assigned) |
| #260 | thorfinn | model-slices sweep {64, 128, 192} | NEW |
| #259 | senku | grad-clip-norm sweep {0.5, 2.0} | NEW |
| #258 | kohaku | squared rel-L2 aux loss (focal-style) | NEW |
| #257 | haku | high-shear curriculum oversampling | NEW |
| #256 | frieren | mirror-symmetry TTA | NEW |
| #255 | edward | fixed wsy/wsz loss multipliers | NEW |
| #254 | chihiro | raw rel-L2 auxiliary loss sweep | Just launched |
| #253 | askeladd | FourierEmbed vs ContinuousSincosEmbed | NEW |
| #239 | norman | (yi-branch — listed above) | — |
| #221 | violet | per-channel adaptive reweighting | Run A ep5=11.12, gate stalled, second prod posted |
| #214 | gilbert | k-NN local surface attention | **Strong signal: -1.58pp wsy, -1.22pp wsz at ep2 vs alphonse** |
| #179 | nezuko | 5L/384d wide-deep + Fourier PE | Rebased; ep10=8.825% gate passed; ep30 needed |
| #174 | alphonse | 5L/256d + Fourier PE + T_max=50 | `vu4jsiic` ep9=9.10, ep10=8.64 — strong descent |
| #80 | tanjiro | surface-loss-weight sweep | Trial B1 ep16=8.96 gate passed |
| #79 | emma | DrivAerML 60k Points + Fourier PE | Merge conflict — rebase requested |
| #75 | fern | LR sweep with Fourier PE | Trial B ep18=8.47, projected ep30 ~7.95 |

### Closed this round

- **PR #229** (norman y-flip TTA, yi) — closed as stale duplicate; norman was actively on PR #239 bengio
- **PR #152** (violet geom-moment, yi) — closed after 3 cascading divergences on rebased base; architecture mismatch with new SOTA stack

---

## Current Research Themes

### Theme 1: Issue #252 / Modded-NanoGPT Inspiration (NEW)
- **#261 norman:** Muon optimizer (Newton-Schulz orthogonalized momentum)
- **#262 nezuko:** Linear-warmdown LR schedule (WSD-style)
- **#266 stark:** U-net long skip connections
- **#270 violet:** tanh output soft-cap (regression analog of logit-cap)
- Deferred: FlexAttention, FP8 head, sequence packing

### Theme 2: Locality / Receptive Field for wsy/wsz Gap (HOT)
- **#214 gilbert (bengio):** kNN local surface attention — STRONG ep2 signal
- **#193 thorfinn (yi):** curvature-biased surface point sampling
- **#266 stark (bengio):** U-net skip injection of fine-detail to deep layers (mechanism complement to kNN)

### Theme 3: Loss Formulation
- **#243 chihiro (yi):** aux-rel-l2-weight sweep
- **#254 chihiro (bengio):** raw rel-L2 aux loss sweep
- **#258 kohaku (bengio):** squared rel-L2 aux (focal-style)
- **#255 edward (bengio):** fixed wsy/wsz multipliers
- **#244 emma (yi):** surface-loss-weight {1.5, 2.0}
- **#221 violet (bengio):** adaptive per-channel reweighting toward AB-UPT targets
- **#270 violet (yi):** tanh output soft-cap (bounds tail predictions)

### Theme 4: Optimization & Stability
- **#207 alphonse (yi):** AGC — only lr=3e-4 arms surviving
- **#247 thorfinn (yi):** cosine T_max=14 schedule
- **#262 nezuko (yi):** linear-warmdown LR
- **#259 senku (bengio):** grad-clip-norm sweep
- **#261 norman (yi):** Muon optimizer
- **#245 gilbert (yi):** progressive EMA decay
- **#249 tanjiro (yi):** asinh wall-shear target normalization

### Theme 5: Architecture
- **#208 askeladd (yi):** sandwich-LN for 8L/256d depth
- **#260 thorfinn (bengio):** model-slices sweep {64, 128, 192}
- **#179 nezuko (bengio):** 5L/384d wide-deep + Fourier PE — ep10 gate passed

### Theme 6: Positional Encoding
- **#224 fern (yi):** learned Fourier embeddings per-axis (init=10 beats sincos at matched ep1)
- **#239 norman (bengio):** Fourier PE num_freqs sweep {16, 32, 64, 128}
- **#253 askeladd (bengio):** FourierEmbed vs ContinuousSincosEmbed standalone test

### Theme 8: Core Building Blocks (MLP / Activation)
- **#280 frieren (yi):** MLP activation ablation (SwiGLU / ReLU² vs GELU) — NEW, modded-NanoGPT-inspired

### Theme 7: Symmetry / TTA / Augmentation
- **#225 haku (yi):** mirror symmetry training augmentation
- **#256 frieren (bengio):** mirror-symmetry TTA
- **#257 haku (bengio):** high-shear curriculum oversampling
- **#227 stark (yi):** wall-shear in tangent frame — POD MISSING (RBAC, Issue #248)
- **#230 senku (yi):** SWA tail-end weight averaging

---

## Fleet-Wide Stability Constraints (current)

- **lr=5e-4 + 4L/512d + Lion is structurally unstable.** Confirmed across 10+ arms across PRs #193/#207/#224/#243/#244/#245. Standard response: relaunch at lr=3e-4 (Lion) or lr=1e-4 (Lion + lr_warmup_epochs=1, the SOTA recipe).
- **PR #222 SOTA recipe (Lion, lr=1e-4, lr_warmup_epochs=1, 4L/512d) is the only confirmed-stable optimizer point** in this regime.
- **Volume pressure now beats AB-UPT (0.97×, 5.88 vs 6.08).** All future experiments should avoid sacrificing p_v for tau gains.
- **Wall_shear_y/z gap remains 1.4× of AB-UPT** — primary research target.

---

## Key Constraints

- Training budget: ~270 min training + ~90 min val/test = 360 min total
- VRAM: 96 GB per GPU; SOTA recipe uses ~75 GB
- Gradient clipping: clip_grad_norm=1.0 standard
- Students have 4 GPUs each; single-process per GPU enables 4 arms in parallel
