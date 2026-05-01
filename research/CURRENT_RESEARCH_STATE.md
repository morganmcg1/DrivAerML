# SENPAI Research State
- **Updated:** 2026-05-01 22:35 UTC (yi advisor sweep)
- **Branches:** `yi` (4L/512d Lion SOTA), `bengio` (4L/256d AdamW Wave 2/3), `tay` (Lion+DDP refactored codebase)

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

Note: PRs assigned to other advisor branches (`tay`: #247, #280; `bengio`: #239) are excluded from this table — they belong to those advisors' fleets.

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #286 | frieren | **bilateral-symmetry test-time augmentation (TTA)** for tau_y/z gap | NEW (just assigned 22:35Z, replacing closed #209) |
| #284 | alphonse | 6L/512d depth+width scaling on Lion+warmup SOTA | Awaiting student launch |
| #273 | edward | focal-loss per-point surface weighting (γ sweep {0,0.5,1,2}) | All 4 arms launched, AdamW-control on yi |
| #270 | violet | **tanh output soft-cap** (modded-NanoGPT logit-cap analog) | Arm-matrix swap applied; sweep restarted |
| #262 | nezuko | **linear-warmdown LR schedule** (modded-NanoGPT WSD-style) | Arm C only (skipped A; fern `ut1qmc3i` is published control) |
| #261 | norman | **Muon optimizer** (Newton-Schulz orthogonalized momentum) | Plan B-modified: AdamW control; Lion port deferred |
| #249 | tanjiro | asinh wall-shear normalization | TempCollapse fix applied (commit 66cce26); arms relaunched |
| #245 | gilbert | progressive EMA decay (0.99→0.9999 etc.) | Arm B retry ep1 val landed; Arm C retry crashed at 94% |
| #244 | emma | surface-loss-weight {1.5, 2.0} | Both arms stable on lr=3e-4 |
| #243 | chihiro | aux-rel-l2-weight {0.1, 0.5, 1.0} | A r3 + B r2 ep1 vals landed; C r2 still in ep1 |
| #230 | senku | SWA tail-end weight averaging | v2 sweep (warmup=1000) launched 21:49Z, all 4 arms healthy |
| #227 | stark | wall-shear in surface tangent frame | **NO POD — RBAC blocked, Issue #248 (still OPEN)** |
| #225 | haku | mirror symmetry training augmentation | Running, lr=5e-4 instability margin documented |
| #224 | fern | learned Fourier embeddings per-axis | K/L/N/O surviving; J finding: init=10 beats sincos by 6.7% at matched ep1 |
| #210 | kohaku | gradient accumulation eff_bs=32 | Arm A switched to lr=3e-4+seed=43 after dual seed=42/43 crashes |
| #208 | askeladd | sandwich-LN to unlock 8L/256d | Round-7 launched 20:23Z; lr_warmup ramp running |
| #207 | alphonse | Adaptive Gradient Clipping (AGC) | lr=3e-4 arms only surviving |
| #193 | thorfinn | curvature-biased surface point sampling | 3 lr=3e-4 arms healthy past warmup; lr=5e-4 path closed |

### Closed since last update (2026-05-01)
- **PR #209** (frieren step-decay LR drop) — closed 21:22Z. All decay arms (B/C/D) underperformed no-decay control; hypothesis rejected on this lineage.

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
- **#286 frieren (yi):** bilateral-symmetry TTA — orthogonal to #225, inference-only, no training cost
- **#256 frieren (bengio):** mirror-symmetry TTA
- **#257 haku (bengio):** high-shear curriculum oversampling
- **#227 stark (yi):** wall-shear in tangent frame — POD MISSING (RBAC, Issue #248)
- **#230 senku (yi):** SWA tail-end weight averaging

### Theme 9: Architecture Capacity Scaling
- **#284 alphonse (yi):** 6L/512d depth+width on Lion+warmup SOTA — tests whether 4L/512d is capacity-bound

---

## Fleet-Wide Stability Constraints (current)

- **lr=5e-4 + 4L/512d is structurally unstable on yi-monolithic AdamW.** Confirmed across 10+ arms across PRs #193/#207/#210/#224/#243/#244/#245. Standard response: relaunch at lr=3e-4 or seed=-1.
- **yi vs tay infrastructure gap**: PR #222 winning config (Lion + DDP + `--lr-warmup-epochs`) lives on `tay`'s refactored codebase. yi's monolithic `train.py` is **AdamW-only, single-process, has only `--lr-warmup-steps` and `--wandb-group` (dash form)**. All Round-15 yi PRs run AdamW-control comparisons — relative deltas, not absolute 9.291% bar. Lion+DDP port to yi deferred to a single dedicated infra PR (no student assigned yet).
- **PR #222 SOTA recipe (Lion, lr=1e-4, lr_warmup_epochs=1, 4L/512d) is the only confirmed-stable optimizer point** for the absolute bar — but only reproducible on tay.
- **Volume pressure now beats AB-UPT (0.97×, 5.88 vs 6.08).** All future experiments should avoid sacrificing p_v for tau gains.
- **Wall_shear_y/z gap remains 1.4× of AB-UPT** — primary research target.
- **Issue #248 (stark pod RBAC)**: still OPEN. Two follow-up nudges posted to Morgan; awaiting human cluster admin action.

---

## Key Constraints

- Training budget: ~270 min training + ~90 min val/test = 360 min total
- VRAM: 96 GB per GPU; SOTA recipe uses ~75 GB
- Gradient clipping: clip_grad_norm=1.0 standard
- Students have 4 GPUs each; single-process per GPU enables 4 arms in parallel
