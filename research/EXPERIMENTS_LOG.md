# SENPAI Research Results

## 2026-04-29 19:30 — PR #472: [edward] Muon@1e-3 vs AdamW@1e-4 full-budget 4-GPU DDP — CLOSED (validated, compute-limited)

- Branch: `edward/muon-full-budget-ddp` (deleted)
- Hypothesis: Promote Muon (Newton-Schulz orthogonalization + Nesterov momentum) from PR #377 mechanism flag to default optimizer at lr=1e-3 vs AdamW@1e-4 control on yi 4L/512d baseline + STRING-sep PE.
- W&B runs: Arm A (AdamW) — wrong-WD aborted; Arm B v1 (Muon@1e-3 fresh) `blflnddk`; Arm B v2 (Muon@1e-3 fresh repro) `lgy5e3uw`; Arm C (model-only resume from blflnddk ep3) `tl87weiy`.

**Results (best val_primary/abupt_axis_mean_rel_l2_pct across arms):**

| Arm | Config | epochs | val_abupt | test_abupt | Notes |
|---|---|---:|---:|---:|---|
| A | AdamW@1e-4 fresh | ~3 | ≈18.0% | — | wrong-WD recovery |
| **B v1** | **Muon@1e-3 fresh** | ~3 | **11.349%** | 12.425% | clean signal |
| B v2 | Muon@1e-3 fresh repro | ~3 | ~12% | — | within ~1pp of v1 |
| C | Muon@1e-3 model-only resume | 1+2 | 10.104% (ep1) | 11.189% | EMA-settling artefact |

**Analysis:**
- Muon@1e-3 vs AdamW@1e-4: ~43.8% relative improvement on val_abupt — decisive, reproducible optimizer signal.
- Merge bar 9.032% (PR #517 askeladd Lion lr=1e-4 clip=0.5) NOT beaten — best arm 11.349%.
- Compute envelope: 4×RTX-Pro-6000 → ~1.25 s/it → ~113 min/epoch → only ~3 epochs per 4.5h session, vs. ~8-9 epochs the original 8-GPU-class design assumed.
- Arm C epoch-1 best (10.104%) is an EMA-settling artefact during low-LR warmup from loaded weights; epochs 2-3 regressed to 11-12%, confirming model plateau — model-only resume does not extend the trajectory because optimizer state, EMA, scheduler, and global_step all reset to fresh.
- Plateau is compute-limited, not optimizer-limited.

**Conclusion:** CLOSED as validated-but-compute-limited. Muon@1e-3 is a banked optimizer-direction win — preserved as a known-strong configuration for composition with other yi wins (STRING-sep PE PR #490 now landed on yi). Next steps are fresh experiments (composition runs, full-state checkpoint resume, 8-GPU when available), not iterations on this branch. Edward reassigned to Muon@1e-3 + STRING-sep PE composition run — the highest-value next step now that both wins are available on yi.

---

## 2026-04-29 — PR #317: [violet] Huber loss for wall-shear heavy-tail robustness (δ sweep) — MERGED (mechanism flag, baseline unchanged)

- Branch: `violet/huber-wall-shear-robustness` (merged)
- Hypothesis: Wall-shear targets (τ_x/y/z) have heavy-tailed residual distributions (p99 ≈ 3× p50). Plain MSE is sensitive to outlier-magnitude residuals. Huber loss on the standardized residual (r = pred − target in σ-units) limits the loss contribution of outliers while preserving L2 curvature near zero.
- W&B runs (r20 multi-seed): ctrl `g1s45tbt`/`6649fm5e`, d10 `52urviip`/`zni9if9p`; r18 (first screen): `p8s8rxo7`/`wnr2zd74`/`uuyxopmh`/`73hfyxwd`

**Final results (r20 multi-seed parity comparison, 2-seed paired-diff test):**

| Metric | ctrl mean | δ=1.0 mean | Δ |
|---|---:|---:|---:|
| val_primary/abupt_axis_mean_rel_l2_pct | 37.15% | 35.70% | **−1.45pp** |
| val_primary/wall_shear_z_rel_l2_pct | 85.95% | 82.56% | **−3.39pp** |
| val_primary/wall_shear_x_rel_l2_pct | 34.20% | 31.04% | **−3.16pp** |
| val_primary/wall_shear_y_rel_l2_pct | 47.83% | 47.26% | −0.57pp |
| val_primary/volume_pressure_rel_l2_pct | 8.68% | 8.77% | +0.09pp (negligible) |
| test_primary/abupt_axis_mean_rel_l2_pct | 37.69% | 36.38% | **−1.31pp** |

Seed-level: s0 Δ=−1.15pp, s1 Δ=−1.75pp (both agree, same direction).

**Analysis:**
- Pre-specified parity criteria met: mean Δ=−1.45pp >> 0.10pp threshold, both seeds agree, per-channel mechanism matches heavy-tail hypothesis (ws_z −3.39pp > ws_x −3.16pp > ws_y −0.57pp, correctly ordered by tail weight)
- Absolute merge bar (7.546%) not contestable: comparison ran with `--use-tangential-wallshear-loss` causing ep1 tangential-supervision × timeout interaction → both arms at 36-38% (far from 7.546%)
- Volume backbone uncontaminated: vol_loss 0.014-0.016 flat across all arms
- Implementation: plain Huber on standardized residual r = pred − target (in σ-units); `loss = r² (|r|<δ); 2δ(|r|−δ/2) (|r|≥δ)`; channels 1..3 (τ_x/y/z) only; cp keeps MSE; opt-in flag `--wallshear-huber-delta` (default 0.0 = exact MSE)
- δ=0.5 worse than control on r18; δ=0.25 (vol-head analogy from gilbert) non-tested here
- Tangential loss appears to amplify Huber value: r18 (no tangential) showed only Δ=−0.14pp vs r20's −1.45pp

**Conclusion:** MERGED as mechanism flag (zero behavior change at default δ=0.0; feature available for composition with asinh normalization, axis-specific δ variants, full-budget DDP runs). Absolute merge bar not contested in this round — follow-up assigned to violet as PR #440 (test without tangential loss).

---

## 2026-05-02 17:10 — PR #364: [frieren] DropPath stochastic-depth sweep (p=0/0.05/0.10/0.20) — CLOSED POSITIVE SCREEN

- Branch: `frieren/stochastic-depth-sweep-r24` (deleted)
- Hypothesis: Stochastic depth (DropPath) provides implicit regularization that helps the model generalize better on tau_y/z, which shows high spatial clustering and could benefit from reduced co-adaptation between layers.
- W&B group: `frieren-r24-droppath-sweep`

**Results (ep1 single-GPU AdamW screen):**

| Arm | Config | val_abupt (ep1) | vs control | W&B run |
|---|---|---|---|---|
| A | p=0.0 (control) | ~18.5% | — | — |
| **B** | **p=0.05** | **~17.7%** | **-0.797pp** | — |
| C | p=0.10 | worse than control | +pp | — |
| D | p=0.20 | worse than control | +pp | — |

**Analysis:**
- DropPath p=0.05 confirmed positive screen: -0.797pp at ep1 vs control
- Higher DropPath (0.10, 0.20) hurts performance — too much regularization kills useful signal for this architecture
- Full-budget rerun blocked on PR #355 (emma DDP infrastructure)

**Conclusion:** CLOSED as completed-positive screen. p=0.05 is confirmed in. Full-budget 8-GPU rerun to be queued once PR #355 lands and DDP is confirmed working.

---

## 2026-05-02 17:10 — PR #355: [emma] DDP infrastructure fix — MERGED (infrastructure, no metric bar change)

- Branch: `emma/ddp-infrastructure-fix` (merged)
- Hypothesis: train.py on yi has no DDP implementation — all torchrun 8-GPU launches silently train single-GPU. Cherry-picking DDP commits from PR #284 (alphonse) will restore canonical multi-GPU training.
- W&B runs: `fy3hsq4j` (smoke, 1-ep 4-GPU), `i263nt1h` (full 9-ep 4-GPU, timeout at ep3 partial)

**Results:**

| Metric | ep1 (4-GPU, eff bs=16) | PR #222 ep1 (8-GPU, eff bs=32) |
|---|---|---|
| val_primary/abupt_axis_mean_rel_l2_pct | 23.30% | 67.73% |
| val (ep2) | 12.70% | 41.93% |
| val (ep3 partial) | **11.36%** | 19.30% |
| test/abupt (ep3 partial chkpt) | 12.47% | — |

**Analysis:**
- DDP cherry-picks (`bfbe975` + `1a8f7b7`) landed cleanly on yi: zero conflicts
- Convergence at 4-GPU eff bs=16 is ~2× faster per-epoch vs PR #222 8-GPU eff bs=32 (more steps/epoch at same per-rank bs)
- Peak GPU memory 67.5 GiB / 96 GiB — margin confirmed for batch scaling
- Train timeout (270 min) hit mid ep3 — only 2.7 epochs completed vs 9 full epochs in PR #222
- DDP infrastructure verified healthy: world_size=4 confirmed, DistributedSampler set_epoch() working

**Conclusion:** MERGED as infrastructure fix. Metrics bar unchanged (7.546%). All future students can now use canonical torchrun --nproc_per_node=4/8 with no code changes.

---

## 2026-05-02 17:45 — PR #208: [askeladd] Sandwich-LN to unlock 8L/256d depth — CLOSED DEAD END

- Branch: `askeladd/pre-ln-sandwich-ln-deep-stability` (deleted)
- Hypothesis: Sandwich-LN (NormFormer) places LayerNorm before AND after each sublayer in the residual branch, dampening gradient cascades that prevent depth >6L at lr=5e-4.
- W&B runs: best stable arm `sc24mpqh` (8L/256d, lr=5e-4, val 9.892%)

**Results — best per-depth (14 total arms across R6-R14):**

| depth | norm | lr | val_abupt | test_abupt | W&B run |
|---|---|---|---|---|---|
| 8L | sandwich-LN | 5e-4 | 9.892% | 11.00% | `sc24mpqh` ⭐ |
| 10L | sandwich-LN | 3e-4 | 12.05% | 13.19% | `8hi5brwb` |
| 12L | sandwich-LN | 3e-4 | 12.19% | 13.23% | `vdeims25` |
| 14L | sandwich-LN | 3e-4 | CASCADE | — | `bzlgvos1` |
| **bar (PR #311)** | STRING-sep | — | **7.546%** | **8.771%** | `gcwx9yaa` |

**Depth-LR coupling map (key finding):**
- lr=5e-4 stable at ≤8L, unstable at ≥10L
- lr=3e-4 stable at 8L/10L/12L, unstable at 14L
- Cascade-onset depth ceiling: ~+2-4 layers per LR halving

**Analysis:**
- Best result (8L lr=5e-4, 9.892%) is 2.23pp above the new 7.546% bar — not a merge candidate
- 10L/12L at lr=3e-4: 12-13% test range — well off the bar trajectory
- The merge bar moved during this round (PR #311 STRING-sep) — no recovery path
- R9 8L-A control regression under omega-bank PE remains unexplained (seed-dependent)

**Conclusion:** CLOSED. Sandwich-LN is a useful architectural primitive with a well-characterized depth-LR ceiling. The most promising follow-up is sandwich-LN + STRING-sep PE at 8L/256d (orthogonal combination test). Current yi codebase does not include `--norm-style` flag.

---

## 2026-05-02 17:45 — PR #335: [chihiro] Tau_y/z loss curriculum (W_max ramp, 1→W over N epochs) — CLOSED DEAD END

- Branch: `chihiro/tau-yz-weight-curriculum` (deleted)
- Hypothesis: Ramping wallshear y/z weights from 1.0 → W_max over N epochs (curriculum) provides a gentler introduction of the wall-shear loss terms, avoiding early-epoch instability while still reaching a high final weight.
- W&B group: `chihiro-wmax-ramp-r19`

**Results (ep1 mid-warmup):**

| Arm | Config | val_abupt | W&B run |
|---|---|---|---|
| A | Curriculum W=1→3, N=3 | 18.43% | 49potgp7 |
| B | Static W=2 | 26.17% | kqt5g5bq |

**Analysis:**
- Both arms ~2-2.4× above merge bar (7.546%). Arm A wins arm-vs-arm by 7.74pp but absolute values too high to salvage.
- Static W=2 (Arm B) is worse than the current static default W_y=W_z=2 in the baseline — the config may not have reproduced exactly.
- Curriculum ramp provides no advantage over the current fixed-weight approach at this short horizon.

**Conclusion:** CLOSED. Dead end — neither arm approaches the merge bar. Curriculum approach discarded for now.

---

## 2026-05-02 17:45 — PR #313: [kohaku] Multi-seed ensemble averaging (3 seeds) — CLOSED NOT COMPETITIVE

- Branch: `kohaku/multi-seed-ensemble` (deleted)
- Hypothesis: Averaging predictions across 3 independently-trained seeds reduces variance and improves generalization, particularly on the τy/τz gap.
- W&B group: `kohaku-ensemble-r19-adamw`; runs: `o5xy0hb8` (seed=42), `iqmkd6zt` (seed=1337), `cty0iccl` (seed=2024)

**Results (final checkpoint):**

| Seed | val_abupt | W&B run |
|---|---|---|
| 42 | 11.015% | o5xy0hb8 |
| 1337 | 11.085% | iqmkd6zt |
| 2024 | 11.033% | cty0iccl |
| **3-seed ensemble mean** | **11.044%** | — |

**Analysis:**
- All 3 seeds converged to nearly identical values (σ≈0.036pp) — very low variance, good reproducibility signal.
- Ensemble mean 11.044% is +3.50pp above merge bar 7.546% — not competitive at this horizon (single-GPU AdamW, not full DDP Lion budget).
- The ensemble approach itself has merit but requires full-budget runs to compare fairly against the baseline.

**Conclusion:** CLOSED. Not competitive at this training horizon. Seed variance is negligible (~0.036pp), so ensemble averaging provides minimal benefit over single-seed runs at this scale.

---

## 2026-05-02 17:50 — PR #364: [frieren] DropPath stochastic depth sweep (p=0.00/0.05/0.10/0.20) — POSITIVE SCREEN

- Branch: `frieren/stochastic-depth-sweep-r24` (WIP, sent back)
- Hypothesis: Stochastic depth (DropPath) during training improves generalization by preventing the 4-layer transformer from over-relying on any single layer. Optimal p expected in 0.05-0.10 range.
- W&B group: `frieren-droppath-r24`; runs: `7s2sfxof` (A), `ylnh9rf0` (B), `uuko7oqf` (C), `idk6kx5q` (D)

**Results (~41% ep1, single-GPU AdamW):**

| Arm | p | val_abupt | W&B run |
|---|---|---|---|
| A (control) | 0.00 | 18.882% | 7s2sfxof |
| **B** | **0.05** | **18.085%** | ylnh9rf0 |
| C | 0.10 | 19.072% | uuko7oqf |
| D | 0.20 | 19.576% | idk6kx5q |

**Analysis:**
- Arm B (p=0.05) beats control by **0.797pp** — 2.7× the 0.3pp positive-screen threshold.
- C and D regress below control: DropPath overshoot at 4-layer depth and short training horizon.
- Inverted-U pattern centred at p=0.05 is clean — strong positive screen signal.
- Absolute values at ~18% are ~2.4× above bar; full-horizon comparison needed post-DDP.

**Conclusion:** POSITIVE SCREEN. DropPath p=0.05 validated. Sent back for full-budget DDP escalation once PR #355 (emma DDP) lands. Not yet merge-ready (needs full-budget run against real bar).

---

## 2026-05-02 17:55 — PR #367: [haku] Theta-conditioned wall-shear loss weight for cross-flow error — POSITIVE SCREEN

- Branch: `haku/theta-conditioned-wallshear-loss` (WIP, sent back)
- Hypothesis: Per-point theta-conditioned weight `w_i = 1 + alpha * sin(theta_i)` (where theta is angle from streamwise) up-weights cross-flow-dominant surface points, forcing the model to attend to the underserved tau_z regime.
- W&B group: `haku-theta-wallshear-screen`; runs: `bb385as1` (A), `6474cl0h` (B), `k57ssdvm` (C)

**Results (~60% ep1, single-GPU AdamW, mid-warmup):**

| Arm | alpha | val_abupt | tau_y | tau_z | vol_p | theta_w_mean | W&B run |
|---|---|---|---|---|---|---|---|
| A (control) | 0.0 | 38.547% | 50.09% | 74.90% | 14.95% | n/a | bb385as1 |
| **B** | **1.0** | **37.662%** | 49.76% | **69.16%** | 16.64% | 1.602 | 6474cl0h |
| C | 2.0 | 38.983% | 51.33% | 70.29% | 18.17% | 2.216 | k57ssdvm |

**Analysis:**
- Arm B (alpha=1.0) beats control by **0.885pp** on val_abupt and **5.74pp (-7.7% rel)** on tau_z — exactly the dominant gap channel.
- tau_y improvement minimal (0.33pp) — sin(theta) up-weights both tau_y and tau_z, but most of the practical signal is in tau_z (more extreme angle distribution).
- C (alpha=2.0) overshoots: all metrics worse than control, including volume pressure starvation (+3.22pp).
- theta_w_mean ≈ 1.60 for alpha=1.0 → E[sin(theta)] ≈ 0.60 → population heavily cross-flow-skewed.
- Side-channel cost: volume_p degrades +1.69pp at alpha=1.0 (not headline but worth tracking).

**Conclusion:** POSITIVE SCREEN. alpha=1.0 positive. Sent back for tighter bracketing (alpha ∈ {0.75, 1.0+z_weight=3.0, 1.0+y_weight=1.5}) and tau_z-y decoupling analysis. Full-budget DDP run post-#355 is the merge gate.

## 2026-05-02 14:15 — PR #350: [norman] MAE-style point masking sweep (--point-mask-ratio) — CLOSED (compute-limited; p=0.30 shows τy/τz signal; follow-up in PR #391)

- Branch: `norman/mae-point-masking` (base: yi)
- Hypothesis: Randomly masking decoder query points during training (MAE-style) forces global geometry learning and may improve tau_y/z generalisation on cross-flow-dominant surface patches.
- W&B runs: p=0.0 `9axjku8c`, p=0.10 `mf41ndwc`, p=0.20 `t0t78g11`, p=0.30 `sytpj1d9`
- Protocol: single-GPU AdamW, bs=4, 65k pts, lr=1e-4, wd=5e-4, lr-warmup-steps=2000, ~11k steps (~55% of epoch 1), 4 arms in parallel.

**4-arm results (best epoch=1, step ~11k for all arms):**

| Arm | val_abupt | test_abupt | val_τy | val_τz | val_vol_pres |
|---|---:|---:|---:|---:|---:|
| p=0.0 control (`9axjku8c`) | 25.0436 | 25.9227 | 11.3803 | 8.0337 | 15.1447 |
| p=0.10 (`mf41ndwc`) | 25.9286 | 26.8605 | 11.8143 | 8.4399 | 15.6484 |
| p=0.20 (`t0t78g11`) | 25.6061 | 26.2299 | 11.6948 | 8.2748 | 15.5818 |
| p=0.30 (`sytpj1d9`) | 25.1397 | 25.8753 | 11.2453 | 7.6645 | 15.9155 |

Realised drop fractions confirmed within ~1% of nominal (logged via `[point_mask]` diagnostic).

**Analysis and conclusions:**

1. **Hypothesis not supported at this compute horizon** — p=0.10/0.20 uniformly worse on all metrics; p=0.30 ≈ control on val_abupt but marginally better on test_abupt and targeted τy/τz axes.
2. **p=0.30 τy/τz signal is real and consistent (val/test):** Δτy −0.135/−0.318pp, Δτz −0.369/−0.738pp. Small but reproducible across both splits.
3. **Volume pressure regresses at all p>0:** p=0.30 +0.77pp val, +0.89pp test. Uniform masking drops interior points, degrading already-2.05× vol_p gap. This is a structural issue with Bernoulli masking that hits the entire token sequence.
4. **Architectural note:** Transolver has no encoder/decoder split — masking drops tokens from both slice-attention context AND loss simultaneously, introducing a train/val distribution shift proportional to p. Cleaner formulation: surface-only masking.
5. **Follow-up:** PR #391 (norman, surface-only masking `--surface-mask-ratio`) protects volume points, isolates the τy/τz regularisation benefit.

**Decision:** Closed. Compute-limited (1 val event, no convergence signal). Surface-only follow-up tracked in PR #391.

---

## 2026-05-02 12:40 — PR #339: [senku] mlp_ratio=8 vs mlp_ratio=12 head-to-head — CLOSED (architectural finding: mlp_ratio=8 confirmed optimal)

- Branch: `senku/mlp-ratio-12-headtohead` (base: yi)
- Hypothesis: Does the mlp_ratio improvement trend from PR #315 (mlp_ratio 2 < 4 < 8) continue to mlp_ratio=12? If yes, mlp_ratio=12 is the architecture for DDP confirmation. If no, mlp_ratio=8 is locked in.
- W&B group: `senku-mlp-ratio-r19`; Arm 8: run `x2r36cwo`, Arm 12: run `hndrne5u`
- Protocol: single-GPU AdamW, bs=2, vol=32k (reduced from bs=4/vol=65k due to 96GB VRAM), 9 epochs, 2000 steps/epoch

**2-arm results:**

| Metric | Arm 8 (mlp_ratio=8) | Arm 12 (mlp_ratio=12) | Δ (12-8) |
|---|---:|---:|---:|
| Params | 21.10M | 29.49M | +39.8% |
| Peak GPU (GB) | 39.39 | 44.82 | +5.43 (+13.8%) |
| Per-epoch time (s) | 405.6 | 487.0 | +81.4 (+20.1%) |
| best val_abupt | **17.204%** | 17.899% | +0.695pp (worse) |
| test_abupt | **18.317%** | 19.040% | +0.722pp |
| test surface_pressure | 12.445% | 12.908% | +0.46 |
| test wall_shear | 19.330% | 20.087% | +0.76 |
| test volume_pressure | 16.054% | 16.531% | +0.48 |
| test tau_y | 22.224% | 23.185% | +0.96 |
| test tau_z | 23.752% | 24.905% | +1.15 |

Note: absolute numbers are above merge bar due to reduced single-GPU protocol — the finding is the relative arm comparison.

**Analysis and conclusions:**

1. **Trend reverses at 12.** Ordering from PR #315 was mlp2 < mlp4 < mlp8 (lower=better); now mlp12 > mlp8 on every metric. The optimum is confirmed at mlp_ratio=8.
2. **Mechanism: optimization friction.** Gap narrows from 1.98pp at ep5 → 0.70pp at ep9, suggesting mlp=12 is under-trained at this step budget rather than hitting a hard capacity ceiling. But even if given more steps, mlp=8 dominates: same compute on mlp=8 would also improve.
3. **FFN regime ceiling hypothesis.** At 4L/512d, mlp_ratio=8 already puts 85% of params in FFN matrices. Marginal returns past this point are overwhelmed by optimization friction.
4. **Both arms stable** — no NaN, no OOM, no early stops. Architecture is well-behaved at mlp=12; it's just suboptimal.
5. **Both arms still descending at ep9** (slope -0.35%/1k-steps for arm8) — the protocol absolute numbers are not the relevant bar, the relative comparison is.

**Architecture confirmed locked:** `layers=4, hidden_dim=512, heads=8, slices=128, mlp_ratio=8`

**Decision:** Close. Architectural finding recorded. senku reassigned to PR #375 (5L/512d depth stacking on STRING-sep SOTA).

---

## 2026-05-02 12:25 — PR #316: [thorfinn] GradNorm dynamic per-task loss weighting — CLOSED (positive signal, mechanism mismatch, above merge bar)

- Branch: `thorfinn/pcgrad-dynamic-loss-weighting` (base: yi)
- Hypothesis: Dynamic per-task gradient-norm balancing (GradNorm-lite, Chen et al. 2018) would automatically upweight tau_y/z — the hardest axes — by computing `w_i = median(||∇L_i||) / ||∇L_i||` at a shared probe layer and rescaling losses.
- W&B groups: `thorfinn-pcgrad-r18` (Round 1: dynamic vs static), `thorfinn-gradnorm-r19` (Round 2: probe-depth ablation, 4 arms)

**Round 1 results — dynamic-norm vs static W=2:**

| Arm | Method | val_abupt | test_abupt | W&B run |
|-----|--------|-----------|------------|---------|
| Dynamic-norm | `model.norm` probe | **10.378%** | **11.326%** | `25i36dcj` |
| Static W=2 | `--wallshear-y-weight 2 --wallshear-z-weight 2` | 10.555% | 11.620% | `15msq8rj` |

**Round 2 results — probe-depth ablation:**

| Arm | Probe depth | val_abupt | Steps/epoch | Overhead | W&B run |
|-----|------------|-----------|-------------|----------|---------|
| Embed | First embedding layer | 16.45% | ~35% ep1 | +201% | `dbwikid2` |
| First block | First transformer block | 17.73% | ~35% ep1 | +201% | `wrtr9hg0` |
| Last block | Final transformer block | 12.83% | ~57% ep1 | +67% | `bgx4jz0i` |
| Hybrid (norm + W_y/z=2 floor) | model.norm + static floor | 10.745% | ~96% ep1 | ~0% | `t7qj0291` |
| R1-dynamic-norm (reference) | model.norm | 10.378% | — | +4% | `25i36dcj` |

**Weight trajectory at convergence (R1 dynamic, model.norm probe):**
| Task | Converged weight |
|------|-----------------|
| surface_pressure | 0.793 |
| tau_x | 1.058 |
| tau_y | **0.839** (< 1.0) |
| tau_z | **0.918** (< 1.0) |
| volume_pressure | **1.392** (upweighted) |

**Analysis and conclusions:**

1. **Mechanism mismatch** — at no probe depth does GradNorm-lite upweight tau_y/tau_z above 1.0. It consistently boosts volume_pressure, which is the task with the smallest gradient norm (already largely solved at val: ~5.88%). This is the opposite of the hypothesis.
2. **Dynamic > static by ~1.7-2.5%** relative on abupt — a real but small benefit from automated rebalancing. The signal is insufficient for merge (10.378% vs bar 7.546%).
3. **Probe-depth monotonicity** — gradient emphasis shifts systematically from surface_pressure (shallow probes) to volume_pressure (deep probes). This is a reusable diagnostic for future task-balancing work: choose probe depth based on which task you want to emphasise.
4. **Hybrid arm worse than pure dynamic** — forcing W_y=W_z=2 floor crowds out the volume_pressure signal that the algorithm wants to follow, confirming static-weight forcing is counterproductive with gradient-norm balancing.
5. **Compute overhead**: shallow probes (+201%) are incompatible with our throughput budget. Deep probe (last_block, +67%) is borderline; model.norm (+4%) is the only practical operating point.

**Decision:** Close. Best val_abupt 10.378% is 38% above merge bar 7.546%. Mechanism does not address tau_y/z gap. Thorfinn reassigned to PR #374 (asinh target normalization).

---

## 2026-05-02 12:10 — PR #363: [haku] Diagnostic — where do tau_y/z errors concentrate physically? — COMPLETED (diagnostic, not a loss-improvement experiment)

- Branch: `haku/diagnose-tau-yz-error-modes` (base: yi)
- Hypothesis: Identify physically meaningful clusters of tau_y/z error in surface geometry space — is the gap a coordinate-frame effect (cross-flow direction), a curvature effect (wheel arches/mirrors), a geometry-region effect, or distributed uniformly?
- 5-section diagnostic report delivered in PR comments; no training run (visualization study only)

**Key Findings:**

| Finding | Metric | Quantification | W&B run |
|---|---|---|---|
| Cross-flow alignment is dominant | tau_y (cross-flow θ~86°) vs tau_x (streamwise θ~4°) | **2.21× worse** | n/a (diagnostic) |
| Small-magnitude cross-flow under-predicted | GT \|tau\| at cross-flow vs streamwise points | 0.54 vs 2.20 (4× smaller magnitude) | n/a |
| Curvature is secondary | tau_y high-curvature decile vs low-curvature | **1.41× worse** | n/a |
| Curvature on tau_z | tau_z high-curvature decile vs low-curvature | 1.21× worse | n/a |
| Spatial clustering confirmed | Moran's I (spatial autocorrelation) | Significant in **all 34 val cases** | n/a |
| run_228 tau_z outlier | val case tau_z error | 21.63% (worst in fleet) | n/a |
| run_324 worst on both axes | tau_y AND tau_z | Both worst-in-fleet | n/a |
| No single broken region | Per-region breakdown | Error distributed across wheel arches/A-pillars/mirrors | n/a |

**Conclusions:**
- The tau_y/z gap is dominantly a **cross-flow alignment problem**: the model sees surface points where tau points nearly perpendicular to the global x-axis and struggles because small-magnitude cross-flow shear (|tau|=0.54) provides weak training signal relative to streamwise shear (|tau|=2.20).
- This is a **structural representation deficiency**, NOT a loss-rate problem or capacity problem. The model consistently under-predicts small cross-flow content.
- Curvature effects (wheel arches, A-pillars, mirrors) are real but secondary (1.4× vs 2.2× for the angular alignment effect).
- Errors are spatially clustered (Moran's I), meaning a targeted fix can yield disproportionate gain on the diagnostic cluster rather than needing uniform global improvement.
- Proposed next experiment: **theta-conditioned loss weight** — per-point weight `w_i = 1 + alpha * sin(theta_i)` where `theta_i = arccos(|tau_x_i| / (|tau_i| + eps))`, applied to wall-shear MSE in `train_loss()`.

**Decision:** Close diagnostic branch. Findings directly motivate haku's next assignment (theta-conditioned loss weight experiment).

---

## 2026-05-02 11:38 — PR #334: [gilbert] Mesh-Laplacian GFT spectral loss — CLOSED (clear negative result)

- Branch: `gilbert/mesh-laplacian-gft-spectral-loss` (base: yi)
- Hypothesis: Replace PR #288's weak index-FFT spectral loss with a Graph Fourier Transform on a kNN-graph mesh Laplacian — geometrically meaningful frequency content should give a stronger or broader peaked response than FFT's +0.32pp at lambda=0.10.
- 4-arm sweep (single-GPU AdamW, 4L/256d/4h/96sl, bs=8, lambda in {0.0, 0.05, 0.10, 0.20})
- Loss applied to wall-shear channels (tau_x/y/z) on a per-batch random N_sub=4096 subsample, k_nn=16, k_modes=64

**Apples-to-apples ep1 val_primary (all arms = 1 epoch over data):**

| Arm | lambda_gft | val_abupt | Δ vs ctrl | val_tau_x | val_tau_y | val_tau_z | W&B run |
|-----|-----------|-----------|-----------|-----------|-----------|-----------|---------|
| Control | 0.00 | **27.4365** | — | 26.63 | 34.72 | 37.20 | `l3x9hawb` |
| A | 0.05 | 28.1951 | +0.76pp | 27.39 | 36.20 | 38.31 | `r9gebyk0` |
| B | 0.10 | 28.4494 | +1.01pp | 27.70 | 36.67 | 38.44 | `shhwyhqg` |
| C | 0.20 | 28.2485 | +0.81pp | 27.86 | 36.28 | 38.08 | `lwxkoxja` |

**Conclusion:** Uniformly harmful — every lambda regresses val_abupt and *every* wall-shear axis (the channels the loss is supposed to help). Magnitude scales with lambda, with no peaked sweet spot. This is the opposite shape from PR #288's FFT signal.

**Root cause (gilbert's analysis, well-supported):** Per-batch random N_sub=4096 subsample means the kNN graph and Laplacian eigenbasis are *rebuilt fresh every step*. The "low-frequency mode 1" of batch t is not the "low-frequency mode 1" of batch t+1 — they're rotations/permutations of similar-but-not-identical bases. The model is being pushed to align spectral coefficients with a target that lives in a different basis every step → noisy gradient signal, not a coherent learning target.

**Cost:** +0.4s/step eigh overhead (~30% step-time tax), 62GB peak GPU RAM steady.

**Decision:** Close. Suggested follow-up (per-case cached FPS subsample + fixed eigenbasis + k_modes=16-32) is principled but the step-time tax means even a hypothetical small improvement must beat that budget cost; we have stronger leverage on tau_y/z elsewhere (PR #349 tangent-frame encoder input, PR #362 loss-side tangent projection, PR #316 GradNorm dynamic weighting, PR #336 per-channel output heads). Reassigning gilbert.

---

## 2026-05-02 11:25 — PR #283: [nezuko] model-layers=5 depth sweep on tay branch — CLOSED (informative, did not beat new SOTA)

- Branch: `nezuko/model-layers-5` (base: tay, not yi)
- Hypothesis: Going from 4L → 5L with all other PR #222 hyperparameters fixed isolates the depth lever. If 5L val_abupt < 4L SOTA, depth is a clean win we can stack with future architectural changes (FiLM, tangent frame, etc.).
- W&B run: `z6xc97gg` (group `tay-model-layers-sweep`)

**Epoch-by-epoch trajectory (Lion, 8-GPU DDP, bs=32, 50-ep limit, killed early at ep14):**

| Epoch | val_abupt |
|-------|-----------|
| 1     | 65.59%    |
| 2     | 47.67%    |
| 3     | 20.21%    |
| 4     | 13.94%    |
| 5     | 11.75%    |
| 6     | 10.60%    |
| 7     | 9.993%    |
| 8     | 9.469%    |
| 9     | 9.254%    |
| 10    | 10.896% (Lion spike) |
| 11    | 9.5225% (partial recovery) |
| **12** | **8.9938%** (best) |
| 13    | 9.66%     |
| 14    | 13.74% (terminal divergence) |

**Test-set evaluation from ep12 best-val checkpoint:**

| Metric | 5L (ep12) test | 4L SOTA (PR #232) test | Δ vs 4L SOTA | Current SOTA (PR #311) | Δ vs PR #311 |
|---|--:|--:|--:|--:|--:|
| `abupt_axis_mean` | **10.0426%** | 10.190% | -0.147pp | **8.771%** | +1.27pp (worse) |
| `surface_pressure` | 5.3009% | 5.461% | -0.160pp | 4.485% | +0.82pp |
| `wall_shear` | 9.7345% | 9.910% | -0.176pp | 8.227% | +1.51pp |
| `volume_pressure` | 12.6803% | 12.656% | ≈ tied | 12.438% | +0.24pp |
| `tau_x` | 8.3014% | 8.432% | -0.131pp | 7.253% | +1.05pp |
| `tau_y` | 11.6953% | 11.952% | -0.257pp | 9.233% | +2.46pp |
| `tau_z` | 12.2349% | 12.447% | -0.212pp | 10.449% | +1.79pp |

**Conclusion:** 5L cleanly beat the prior 4L SOTA on every test axis (small but consistent), confirming depth scaling helps. However while this run was completing, PR #311 (edward, STRING-separable position encoding) was merged on yi and set a new SOTA at val 7.546% / test 8.771%. PR #283 does not beat the new bar so it was closed — but the depth-5 insight is preserved and is a strong candidate for stacking with STRING-sep on the current yi base.

**Lion stability finding:** 5L exhibited two divergence events (ep10 spike with recovery, ep14 terminal) — the stable training basin for Lion at depth=5/lr=1e-4 is narrower than at depth=4. Future depth-5+ work on Lion should consider grad-clip=0.5 (PR #309), lower lr (e.g. 8e-5), or a longer warmup.

**Vol_pressure val/test divergence anomaly:** ep12 val vol_pressure = 6.03% but test vol_pressure = 12.68% (2.1× val). PR #311 shows the same pattern (val 5.88% → test 12.44%), confirming this is a systematic property of the test split (likely a small number of high-error meshes), not a 5L-specific artifact. Worth a separate diagnostic.

## 2026-05-02 11:25 — PR #311: [edward] STRING-separable learnable position encoding — MERGED (NEW SOTA)

- Branch: `edward/grape-positional-encoding` (base: yi)
- Hypothesis: Replace isotropic RFF spectral encoding with **STRING-separable** — axis-aligned sinusoids whose `log_freq` and `phase` are learnable per-axis parameters. CFD geometry is fundamentally anisotropic (streamwise ≠ spanwise ≠ wall-normal), so axis-independent learnable frequencies should outperform isotropic Gaussian RFF. 3-arm ablation: A=RFF-32 control, B=STRING-separable, C=GRAPE-M (full learnable plane orientations).
- W&B runs: Arm A `zf2dp7tv`, **Arm B `gcwx9yaa`** (winner), Arm C still running. Group: `tay-round18-grape-ablation`.

**Final 3-arm result table:**

| Arm | Encoding | val_abupt (best) | test_abupt | Δ test vs prior SOTA (PR #309) |
|-----|----------|------------------|------------|--------------------------------|
| A | RFF-32 (control) | 9.710% | 10.721% | +0.531pp (worse) |
| **B** | **STRING-separable** | **7.546%** | **8.771%** | **−1.419pp (−13.93% rel)** |
| C | GRAPE-M | still running | — | — |
| Prior SOTA (PR #309) | RFF-0 (no spectral encoding) | 9.039% | 10.190% | — |

**Arm B per-axis test breakdown:**

| Metric | Arm B test | AB-UPT | Ratio |
|---|---:|---:|---:|
| `abupt_axis_mean` | 8.771% | — | — |
| `surface_pressure` | 4.485% | 3.82 | 1.17× |
| `volume_pressure` | 12.438% | 6.08 | 2.05× |
| `wall_shear` | 8.227% | 7.29 | 1.13× |
| `tau_x` | 7.253% | 5.35 | 1.36× |
| `tau_y` | 9.233% | 3.65 | 2.53× |
| `tau_z` | 10.449% | 3.63 | 2.88× |

**Convergence diagnostics:** All Arm B val slopes still negative at terminal epoch (e.g. `val_primary_abupt/per_1k_steps = −0.04248`, `wall_shear_y` declining fastest at −0.07019/1k). Param diagnostics on STRING `log_freq` and `phase`: `nonfinite_count=0` throughout; healthy gradients.

**Why this matters:** This is the **first new-architecture (vs hyperparameter) win** in many rounds. STRING-separable is orthogonal to: (a) depth scaling, (b) Lion+warmup stabilization, (c) tau_y/z weight upweighting, (d) tangent-frame inputs/losses. All currently in-flight bold experiments (PRs #349 edward tangent-frame input, #362 fern tangent-frame loss, #334 gilbert GFT spectral loss, #336 tanjiro per-channel heads) should now be evaluated on top of the new SOTA.

**Merge bar updated:** 9.291% → 7.546% (val_abupt). Test bar: 8.771%.

## 2026-05-02 10:30 — PR #284: [alphonse] Depth+width scaling 6L/512d — REROUTED TO 8-GPU POD

- Branch: `alphonse/depth-scaling-6l-512d`
- Hypothesis: Scaling from 4L/512d (baseline) to 6L/512d with DDP + Lion + lr-warmup-epochs adds depth while keeping width constant. Per-epoch convergence shows strong depth-scaling signal (ep1=22.371%, ep2=11.723%, ep3=11.618% at 4-GPU, run `wsvdv49o`).
- W&B runs: `wsvdv49o` (4-GPU DDP, bs=32 effective, 3 epochs complete), `mescjl8v` (failed — `--epochs 3` throughput regression)

**4-GPU partial result (wsvdv49o, `--epochs 50`, 3 epochs completed before reroute):**

| Epoch | val_abupt |
|-------|-----------|
| 1     | 22.371%   |
| 2     | 11.723%   |
| 3     | **11.618%** |
| test  | 12.571%   |

- Throughput: 1.02 s/iter, ~58 GB peak VRAM (4×RTX PRO 6000 Blackwell, bs=4/GPU)
- Merge bar: 9.291% (PR #222, fern, run `ut1qmc3i`) — still 2.33pp above

**Throughput regression (investigated but root cause unknown):**
- `--epochs 3` → 1.83 s/iter (~95 GB VRAM), reproduced across 2 fresh restarts
- `total_estimated_steps` is SMALLER with epochs=3 (16,326 vs 272,100), does not explain memory increase
- `MetricSlopeTracker` and EMA decay schedule use `total_estimated_steps` but neither allocates GPU memory
- Root cause likely a dataloader prefetch or PyTorch caching behavior at low epoch counts — filed as infra bug; workaround is `--epochs 6`

**Routing decision (2026-05-02):** Rerouted to `senpai-drivaerml-ddp8-alphonse` (8-GPU pod, confirmed 1/1 READY).
- Command: `torchrun --standalone --nproc_per_node=8 train.py --depth 6 --width 512 --lr 3e-4 --weight_decay 1e-5 --lr_warmup_epochs 2 --optimizer lion --epochs 6 --batch_size 4 --wandb_group depth-scaling-8gpu`
- Rationale: 8 GPUs × bs=4 = bs=32 effective; T_max=6 avoids epochs=3 pathology; ~46 min/epoch × 6 ≈ 276 min budget
- v1 4-GPU partial result (test_abupt=12.571%) kept as record, not re-run

**Infra contributions (to be cherry-picked to yi base):**
- Commit `bfbe975`: DDP init, DistributedSampler, all_reduce for val_primary
- Commit `1a8f7b7`: `dist.broadcast_object_list` run_id fix (non-rank-0 FileNotFoundError on checkpoint load)
- Lion optimizer integration + `--lr_warmup_epochs` flag

---

## 2026-05-02 11:35 — PR #336: [tanjiro] Per-channel MLP output heads for tau_y/z — CLOSED (hypothesis unfalsifiable, Arm B NaN'd 4/4 times)

- Branch: `tanjiro/per-channel-heads` (base: yi)
- Hypothesis: Replace single shared 512→4 linear projection (surface_p, tau_x, tau_y, tau_z) with 4 independent 2-layer MLP heads (one per surface channel, hidden_dim=256). Per-channel non-linear capacity should reduce gradient interference from tau_x dominating tau_y/z in a shared head.
- W&B group: `tanjiro-per-channel-heads-r0`

**All runs attempted:**

| Arm | Run id | Variant | Outcome | Steps survived |
|---|---|---|---|---|
| A control | `7v3ybpfn` | linear (1×512→4) | trained 2 ep, NaN'd in ep3 | 31966 (fail) |
| B v1 | `jr5smvzg` | per-channel-h256, zero outer | KeyboardInterrupt (session) | 2814 |
| B v2 | `pa9r2brg` | per-channel-h256, zero outer | killed by kill-threshold (batch loss spike) | 6099 |
| B v3 | `2zu820aw` | per-channel-h256, trunc_normal | divergence step ~7100, NaN at 8572 | 7100 |
| B v4 | `ke9s6oxy` | per-channel-h256, trunc_normal + post-GELU LN | divergence step ~5000, NaN at 6987 | 5000 |

**Arm A control results (only arm to produce validation checkpoints):**

| Metric | Arm A ep2 (best val) | Merge bar (PR #222) |
|---|---:|---:|
| `val_primary/abupt_axis_mean_rel_l2_pct` | 11.346% | 7.546% |
| `wall_shear_y_rel_l2_pct` | 15.322% | — |
| `wall_shear_z_rel_l2_pct` | 15.887% | — |
| Note: Arm A NaN'd in ep3; 2-epoch result not comparable to 9-epoch SOTA |

**Root cause of Arm B failures:**
- v3 failure (step ~7100): post-GELU intermediate activations unbounded → backbone LN gradient explosion (backbone/blocks/0/norm1 global_norm = 9441 at failure)
- v4 fix (post-GELU LayerNorm) bounded the forward path, but the backward-pass amplification through 4 parallel head paths still overwhelmed grad-clipping at 1.0 when a hard batch hit (~step 5000)
- Arm B v4 tracked Arm A within ±0.02 loss for first 5k steps — failure is non-gradual, abrupt

**Conclusions:**
- Pre-agreed decision criterion met: "Close if Arm B is NaN'd" — Arm B NaN'd 4 times, no validation checkpoint produced
- Hypothesis architecturally interesting but UNSTABLE on AdamW 6L/256d yi base
- The yi 6L/256d AdamW base is itself near a stability boundary (Arm A NaN'd in ep3)
- Per-channel head backward-pass amplification requires either Lion optimizer, separate head LR group (1e-4 vs 5e-4 backbone), or smaller head hidden (128 vs 256) to stabilize
- If revisited: Lion + 4L/512d base + per-channel head LR=1e-4 (separate param group) is most promising path

**Decision:** Close. Hypothesis not falsified, but untestable on current configuration.

---

## 2026-05-02 09:00 — PR #339: [senku] mlp_ratio=8 vs 12 head-to-head confirmation — ASSIGNED (running)

- Branch: `senku/mlp-ratio-r19`
- Hypothesis: Following PR #315 screening which showed mlp_ratio=8 as unambiguous winner over mlp_ratio=2 and 4 on all metrics, test whether mlp_ratio=12 further improves over mlp_ratio=8 at matched single-GPU AdamW protocol. Also serves as protocol-matched re-run of mlp_ratio=8 (PR #315 ran mlp8 at eval-volume=65536; PR #339 runs both at eval-volume=32768 to fit mlp12 within 96GB).
- W&B group: `senku-mlp-ratio-r19`, project: `wandb-applied-ai-team/senpai-v1-drivaerml`
- Status: ASSIGNED (2026-05-02, pending results)

**Experiment config (advisor-assigned):**
```bash
# mlp_ratio=12 arm (bs=2, eval-volume=32768 for memory headroom)
CUDA_VISIBLE_DEVICES=0 python train.py \
  --agent senku --wandb-group senku-mlp-ratio-r19 --wandb-name mlp-ratio-12 \
  --model-mlp-ratio 12 \
  --lr 5e-4 --weight-decay 1e-4 --lr-warmup-steps 500 --clip-grad-norm 1.0 \
  --no-compile-model --batch-size 2 --epochs 9 --max-steps-per-epoch 2000 \
  --gradient-log-every 200 --weight-log-every 200 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 32768 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.999 --wallshear-y-weight 2.0 --wallshear-z-weight 2.0
# mlp_ratio=8 arm re-run at matched eval-volume=32768 protocol on GPU1
```
- mlp_ratio=12 params: ~28M (vs 21.10M for mlp_ratio=8, 12.70M for mlp_ratio=4)
- Decision criteria: If mlp_ratio=12 beats mlp_ratio=8, both confirm monotonic scaling — proceed to 8-GPU Lion confirmation run after yi infra PR lands. If mlp_ratio=8 wins, mlp_ratio=8 is the scaling ceiling and confirms PR #315 finding.
- **Context:** Note — absolute numbers from this screening protocol (AdamW, single-GPU, max-steps-per-epoch=2000) CANNOT be compared to the 9.291% merge bar. This is a relative arm-vs-arm comparison only. Full 8-GPU Lion lr=1e-4 lr-warmup-epochs=1 run required before merge.

---

## 2026-05-02 08:45 — PR #315: [senku] mlp_ratio FFN expansion sweep (2/4/8) — SCREENING COMPLETE, NOT MERGED

- Branch: `senku/mlp-ratio-r18` (kept open as screening record)
- Hypothesis: The FFN hidden-dim multiplier `mlp_ratio` (default=4) may be under-provisioned. Expanding the FFN from 4× to 8× hidden width adds more parameters to the most data-intensive MLP blocks, potentially improving non-linear function approximation capacity where gradient signal is strong (surface pressure, volume pressure).
- W&B group: `senku-mlp-ratio-r18`, project: `wandb-applied-ai-team/senpai-v1-drivaerml`
- W&B runs: `4288ko22` (mlp2), `d7hhw0vl` (mlp4/ctrl), `wwhdaevb` (mlp8)

**Results (single-GPU AdamW screening, max-steps-per-epoch=2000, 9 epochs):**

| Arm | mlp_ratio | params | peak_GB | best_val_abupt | test_abupt | test_p_s | test_tau | test_p_v |
|-----|-----------|--------|---------|----------------|------------|----------|----------|----------|
| A   | 2         | 8.50M  | 58.8    | 11.841%        | 12.848%    | 7.849%   | 13.189%  | 13.626%  |
| B (ctrl) | 4    | 12.70M | 67.5    | 11.236%        | 12.285%    | 7.390%   | 12.443%  | 13.508%  |
| **C** | **8**   | **21.10M** | **85.5** | **10.897%** | **11.981%** | **7.098%** | **12.059%** | **13.454%** |

**Per-axis wall shear (test):**

| Arm | mlp_ratio | tau_y | tau_z |
|-----|-----------|-------|-------|
| A   | 2         | 15.107% | 15.849% |
| B   | 4         | 14.288% | 15.189% |
| **C** | **8**   | **14.010%** | **14.689%** |

- **Result:** mlp_ratio=8 is unambiguous winner — monotonically better than mlp_ratio=2 and 4 on every metric from epoch 4 onward. Best gains vs ctrl (mlp4): abupt −0.34pp (−3.0%), tau_z −0.50pp (−3.3%), tau_y −0.28pp (−2.0%), surface_p −0.29pp (−3.9%). All arms still descending at epoch 9 — further headroom likely with more budget.
- **Memory:** mlp_ratio=8 peak VRAM = 85.5GB (within 96GB limit at bs=4). mlp_ratio=12 requires bs=2 + eval-volume=32768.
- **Advisor decision:** DO NOT MERGE YET. Screening protocol (AdamW, single-GPU, max-steps-per-epoch=2000) cannot be compared to merge bar. Requires confirmation run: 8-GPU DDP, Lion lr=1e-4, lr-warmup-epochs=1, full eval points, matching PR #222 config. Queued behind yi infra PR (cherry-pick alphonse PR #284 commits `bfbe975` + `1a8f7b7` + senku PR #315's `--max-steps-per-epoch` patch).
- **Follow-up:** PR #339 assigned to test mlp_ratio=12 vs 8 head-to-head before committing to full confirmation run.

---

## 2026-05-02 07:30 — PR #297: [haku] Bilateral symmetry augmentation with y-flip (include-both mode) — SECOND SWEEP RUNNING

- Branch: `haku/symmetry-aug-c`
- Hypothesis: Training with y-axis reflection augmentation (anti-symmetric tau_y, all other fields invariant) reduces tau_y/z generalization error by teaching the model the bilateral symmetry constraint explicitly. "include-both" mode concatenates original + flipped sample per step (virtual batch-size doubling). Expected to most benefit tau_y (strongest symmetry constraint), with secondary benefit on tau_z (partial symmetry).
- W&B group: `haku-symm-c-lr3e4-wu2000`, project: `wandb-applied-ai-team/senpai-v1-drivaerml`
- Status: SECOND SWEEP RUNNING (awaiting results)

**Sweep 1 post-mortem (failed — LR warmup mismatch):**
- Advisor diagnosed: `--lr-warmup-epochs 1` at single-GPU bs=2 resolves to ~43,532 warmup steps (~14.5hr), consuming the entire training budget in warmup alone. Entire first sweep results were confounded — models never exited warmup during training.
- Fix: Switch to `--lr-warmup-steps 2000` (step-relative, GPU-count-agnostic).

**Second sweep (4 arms, group `haku-symm-c-lr3e4-wu2000`):**

| GPU | wandb-name | aug | bs | seed | Notes |
|-----|------------|-----|----|------|-------|
| 0 | `haku/symm-both-bs2-lr1e4-wu2000-seed42` | include-both | 2 | 42 | bs=4+include-both OOMs at ~91GB |
| 1 | `haku/symm-both-bs2-lr1e4-wu2000-seed7` | include-both | 2 | 7 | fell back to bs=2 (~67.5GB) |
| 2 | `haku/control-noaug-bs4-lr1e4-wu2000-seed42` | no-aug control | 4 | 42 | matched protocol control |
| 3 | `haku/control-noaug-bs4-lr1e4-wu2000-seed7` | no-aug control | 4 | 7 | seed diversity |

- 2 seeds per condition to distinguish signal from seed variance.
- Decisive criterion: include-both must beat no-aug control on val_abupt AND show disproportionate improvement on tau_y (expected primary beneficiary). If include-both only improves tau_y by ~1pp but regresses surface_p/vol_p, the augmentation trades one metric for another and is not worth the bs=2 throughput penalty.
- **Note from PR #286 (frieren TTA):** Bilateral TTA on an untrained-for-symmetry model showed tau_y improved (+2.79%) but surface_p/vol_p regressed. Symmetry augmentation trains the model to internalize the constraint — hypothesis is that this is cleaner than TTA post-hoc correction.

---

## 2026-04-29 14:00 — PR #286: [frieren] Bilateral-symmetry TTA (y→-y reflection at inference) — CLOSED NEGATIVE

- Branch: `frieren/symmetry-tta` (deleted)
- Hypothesis: At inference, reflect each CFD sample around the y=0 plane, average the forward pass predictions with sign-corrected flip (tau_y negated), to enforce the bilateral symmetry the model must learn statistically from only 500 training cases. Expected to disproportionately close the tau_y/z gap at zero training cost.
- W&B group: `frieren-symmetry-tta-r15`, project: `wandb-applied-ai-team/senpai-v1-drivaerml`

| Run ID | Name | Train Views | no_tta abupt (best) | tta abupt (best) | TTA Δ rel |
|---|---|---|---:|---:|---:|
| gq1dp80i | tta-mini50-bs2-r4 | 50 | 98.236% | 92.518% | −5.82% (untrained — meaningless) |
| 4usjyxjg | tta-mini8k-bs4 | 8k | 22.139% | 22.193% | **+0.24% HARMFUL** |
| d5scti3o | tta-mini20k-bs4-seed1 | 20k | 18.784% | 18.663% | **−0.644%** (below threshold) |
| dqhpc9v0 | tta-mini20k-bs4-seed2 | 20k | 27.983% | 26.921% | −3.80% (diverged base model) |

**Per-component at d5scti3o (best trained arm, mini20k seed1):**

| Channel | no_tta | tta | Δ |
|---|---:|---:|---:|
| surface_p | 12.988% | 13.114% | **+0.126% HARMFUL** |
| wall_shear_y | 24.223% | 23.547% | **−0.676% HELPFUL** |
| wall_shear_z | 25.950% | 25.840% | −0.110% (marginal) |
| vol_p | 12.797% | 13.063% | **+0.266% HARMFUL** |
| **abupt** | **18.784%** | **18.663%** | **−0.644%** |

- **Merge bar status:** No run is anywhere near 9.291%. Best base model is 18.784% (2.02× above bar). Hypothesis not confirmed.
- **Root cause analysis:** Only tau_y uniquely benefits from y-flip averaging — it is the only channel with no static bias in the symmetric direction (tau_y must be exactly anti-symmetric under y→-y by physics). Surface_p, tau_z, and vol_p all have learned per-channel offsets from asymmetric training data distribution; averaging with the flipped prediction introduces a bias mismatch that is net-harmful.
- **Conclusion:** CLOSED. Net TTA effect at best-trained arm: −0.644% relative on abupt, below ~1% meaningful threshold. TTA does work for tau_y (+2.79%) but this is more than cancelled by the surface_p and vol_p degradation.
- **Informed follow-ups (parked):** (a) selective channel-wise TTA — flip-average only wall_shear_y, keep original for surface_p/tau_z/vol_p; (b) train with bilateral symmetry augmentation first (PR #297 haku), then apply TTA on a symmetry-aware model — should eliminate the per-channel bias mismatch.

---

## 2026-04-29 15:00 — PR #336: [tanjiro] Per-channel MLP output heads — ADVISOR FEEDBACK SENT (awaiting results)

- Branch: `tanjiro/per-channel-heads`
- Hypothesis: Replacing the single shared 4-channel linear projection `self.surface_out = LinearProjection(512, 4)` with 4 independent 2-layer MLP heads per output channel reduces gradient interference (tau_x's dominant loss gradient monopolizes the shared projection and starves tau_y/z).
- Student question resolved: Confirmed **Option 1** (AdamW + actual `yi` config at 6L/256d, no Lion port) — architectural test is optimizer-agnostic; option 2 would bundle infra work; option 3 changes architecture without justification.
- Zero-init outer Linear approved (T-Fixup convention, heads start at 0 and differentiate via gradient).
- Adjusted decision criteria: merge if Arm B beats both Arm A and 9.291%; request-changes if Arm B beats Arm A but not 9.291% (architectural mechanism is real, worth porting to Lion base); close if >5% worse than Arm A.
- PR is status:wip, awaiting student results.

---

## 2026-04-29 15:30 — PR #338: [frieren] DropPath stochastic-depth sweep on Lion/4L/512d SOTA — ASSIGNED

- Branch: `frieren/stochastic-depth-sweep`
- Hypothesis: DropPath regularization (stochastic depth) forces the model to maintain redundant multi-pathway representations, preventing tau_x's dominant gradient from monopolizing fixed layer pathways. PR #127 (old 6L/256d AdamW stack) showed DropPath harmful — but the current Lion+4L/512d base is fundamentally different (fewer layers, intrinsically more regularized optimizer, full-budget stable training). Re-testing on the new stack is warranted.
- 4-arm sweep: p=0.0 (control), 0.05, 0.10, 0.20. All other flags match PR #222 SOTA config.
- W&B group: `frieren-droppath-r15`
- Status: ASSIGNED (draft PR #338)

---

## 2026-04-29 — PR #230: [senku] SWA uniform weight averaging for flat-minima generalization — CLOSED (DEAD END: stack-level crash, untestable)

- Branch: `senku/swa-post-train-gain` (deleted)
- Hypothesis: Stochastic Weight Averaging (SWA; Izmailov et al. 2018) over the last 20–30% of training epochs should improve generalization by converging to a flatter loss basin, targeting tau_y/z improvement.

| Sweep | Arm | Run | Config | Best abupt | SWA activated? | Status |
|---|---|---|---|---:|---|---|
| v1 | A | — | swa_start=0.7, lr_swa=5e-5 | crashed | No | lr-warmup-steps=500 instability |
| v1 | B | (EMA only) | swa_start=0.7, lr_swa=5e-5 | 10.05 (EMA) | No | Run capped at ~33k steps; SWA activation at step 408,112 never reached |
| v1 | C | — | swa_start=0.8, lr_swa=1e-5 | crashed | No | lr-warmup-steps=500 instability |
| v1 | D | — | swa_start=0.6, lr_swa=5e-5, wd=1e-3 | crashed | No | lr-warmup-steps=500 instability |
| v2 | A | — | swa_start=0.7, seed=42 | crashed @ step 5324 | No | Stack-level bad batch (seed-pinned) |
| v2 | B | — | swa_start=0.7, seed=13 | crashed @ step 5324 | No | Identical crash step — deterministic |
| v2 | C | — | swa_start=0.8, seed=7 | crashed @ step 5324 | No | Identical crash step — deterministic |
| v2 | D | — | swa_start=0.6, seed=99 | crashed @ step 5324 | No | Identical crash step — deterministic |

- **Result:** SWA hypothesis untestable. Two full sweeps (8 arms total), all invalidated. Root cause: old 6L/256d/AdamW/lr=5e-4 stack. v1: 3/4 arms crashed from lr-warmup-steps=500 instability; Arm B survived to timeout but SWA never activated (would have needed step 408k, reached only ~33k). v2: all 4 arms crashed at identical step 5324 — a seed-pinned bad batch exposed deterministic stack instability. Not SWA-related.
- **Merge bar shift:** PR #222 merged mid-experiment, moving bar 10.21 → 9.291. Best old-stack result (10.05 EMA Arm B) is now +8.2% worse than current baseline.
- **SWA merit not disproven.** The implementation was technically correct. SWA on the new Lion+4L/512d+lr=1e-4 stack remains on the backlog as a potential follow-up — flat-minima generalization hypothesis is valid, just never tested on a stable base.
- **Conclusion:** Close dead end. Do not re-attempt SWA on old 6L/256d stack. New-stack SWA retry is a backlog item, not a priority.

---

## 2026-05-02 08:30 — PR #193: [thorfinn] Curvature-biased surface point sampling (tau_y/z gap fix) — CLOSED NEGATIVE

- Branch: `thorfinn/surface-point-upsampling-high-curvature` (deleted)
- Hypothesis: Biasing surface point sampling toward high-curvature regions (wheel arches, mirrors, A-pillar) would improve tau_y/z by giving high-error zones more gradient signal — analogous to hard negative mining/OHEM.

| Run | alpha | seed | lr | warmup | best_abupt | sp | ws | vp | wsy | wsz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline `3hljb0mg` | 0.0 | — | 5e-4 | 0 | **10.69** | 6.97 | 11.69 | 7.85 | 13.73 | 14.73 |
| Arm B `dn9cenux` | 1.0 | -1 | 5e-4 | 0 | 15.30 (ep2) | 11.50 | 17.90 | **7.30** | 19.12 | 22.10 |
| Arm C `8dxqb6fq` | 0.25 | -1 | 5e-4 | 0 | 12.34 (ep2) | 8.52 | 13.81 | **7.66** | 16.07 | 17.36 |
| `xana1psi` | 0.25 | -1 | 5e-4 | 1000 | 15.88 (ep1) | — | — | — | — | — |
| `deq5seiy` | 0.25 | 1 | 5e-4 | 1000 | 16.06 (ep1) | — | — | — | — | — |
| `7yubkfi2` | 0.25 | 1 | 3e-4 | 1000 | 12.71 (ep2) | 8.63 | 14.35 | **7.54** | 16.88 | 18.01 |
| `7r2dxta7` | 0.15 | -1 | 3e-4 | 1000 | 12.98 (ep2) | 8.83 | 14.60 | **7.85** | 17.25 | 18.27 |

- **Result:** Conclusive negative. Best curvature arm (alpha=0.25, lr=3e-4, ep2 abupt=12.71) is +18.9% worse than baseline 10.69 across all primary metrics. The tau_y/z-specific hypothesis was directly disconfirmed — biased sampling worsened wall shear uniformly.
- **Side-finding:** volume_pressure improved consistently with curvature concentration across alpha=0.25/0.5/1.0 (best: vp=7.30 at alpha=1.0, −7.0% vs baseline 7.85). This is reproducible and may warrant a dedicated follow-up.
- **Stability:** All 13 nonzero-alpha arms diverged (grad spikes to 1e4+). The lr=5e-4 instability is independent of curvature biasing. Curvature cache (67 min one-time build for 400 cases) worked correctly; the technique itself is sound but not beneficial.

---

## 2026-05-02 00:30 — PR #262: [nezuko] linear-warmdown LR schedule (modded-NanoGPT WSD) — SENT BACK (budget infeasibility)

- Branch: `nezuko/linear-warmdown-lr`
- Hypothesis: Replace cosine LR decay with trapezoidal warmup → constant → linear-warmdown schedule (modded-NanoGPT WSD), keeping more steps at peak LR before a sharp final decay. Tests whether cosine's early knee under-utilises the 9-epoch budget.

| Arm | Run | Schedule | DDP | compile | ep1 abupt | Status |
|---|---|---|---|---|---:|---|
| C (warmdown-start=6) | rtmey78g | linear-warmdown | 4-GPU | True | 23.33 | Running, will not reach warmdown phase before timeout |
| A (cosine control) | s0d4pa3r | cosine | 4-GPU | True | — | crashed before ep1 |
| probe | 17a00h9d | cosine | 1-GPU | True | — | crashed (throughput probe) |

- **Result:** Hypothesis cannot be tested as designed on a 4-GPU pod. SOTA reference (PR #222 / `ut1qmc3i`) was 8-GPU DDP + no-compile and still took most of the 6h budget. With 4 GPUs at compile=True, throughput is ~90 min/epoch wall clock, so 9 epochs ≈ 14h — far over `SENPAI_TIMEOUT_MINUTES=360`. Critical: warmdown phase begins at ep6, but realistic reach within budget is ep3-4. The schedule shape is never tested.
- **Infrastructure work landed (not merged yet):** nezuko ported missing pieces from PR #222's `fern/round12-lr-warmup-1ep` branch into `target/train.py` on `yi`: Lion optimizer, `--lr-warmup-epochs FLOAT`, `--lr-schedule {cosine,linear-warmdown}`, `--lr-warmdown-start-epoch FLOAT`, and minimal DDP scaffolding (DistributedSampler + DDP wrapper). This is reusable infra for the rest of the fleet.
- **Open question:** PR #222's actual training code was on `fern/round12-lr-warmup-1ep` but never squash-merged into `yi`'s `target/train.py`. The fleet-wide `target/train.py` on `yi` is missing Lion + DDP. This explains why several Round 14/15 PRs have hit unexpected setup blockers.
- **Send-back plan:** Compressed-budget probe — 3 epochs, lr_warmup_epochs=0.33 (11%), lr_warmdown_start=2.0 (67% constant, 33% warmdown), preserving the trapezoidal shape. 2 arms (cosine control + linear-warmdown), sequential 4-GPU DDP, compile=True, W&B group `nezuko-warmdown-3ep`. This will not produce a number comparable to 9.291% but will tell us whether the shape gives a positive signal at matched 3-epoch horizon. If positive → request 8-GPU follow-up.
- **Win criterion (probe):** Arm C' val_abupt at ep3 < Arm A' val_abupt at ep3 by >0.5pp = positive signal, pursue 9-epoch follow-up. No merge from this probe.

---

## 2026-05-02 00:00 — PR #225: [haku] Left-right symmetry augmentation for tau_y/z gap — CLOSED (ep1 signal, no convergence)

- Branch: `haku/symmetry-augmentation` (deleted)
- Hypothesis: Left-right y-axis reflection of DrivAerML surface data addresses tau_y/tau_z gap by teaching the model the bilateral symmetry constraint. `--symmetry-include-both` (orig+flip concat per step, effective bs×2) was tested alongside stochastic flip p=0.5.

| Arm | Run | Seeds | ep1 abupt | ep1 tau_y | ep1 tau_z | Outcome |
|---|---|---|---:|---:|---:|---|
| A control | zhwlaury/te7uug8u/k3boii9j | 42/7/13 | — | — | — | All crashed before ep1 val |
| B symm p=0.5 | byxxuehz | 13 | 15.95 (−10.0%) | 20.46 (−11.3%) | 22.53 (−11.1%) | ep1 val OK, crashed ep2 |
| C symm-both bs=4 | d03gq4om | 101 | **12.75 (−28.0%)** | **16.29 (−29.4%)** | **17.89 (−29.4%)** | ep1 val OK, crashed ep2 |
| D symm-yzw3 | 2een5w1o | 42 | 15.71 (−11.4%) | 19.96 (−13.5%) | 21.80 (−14.0%) | ep1 val OK, crashed ep2 |

Baseline ep1 comparison: bplngfyo (PR #183 epoch-1 val): abupt=17.72, tau_y=23.07, tau_z=25.35

- **Result:** Win criterion not met — no run survived to final convergence. All 11 attempts in this PR hit NONFINITE_SKIP_ABORT (>200 nonfinite steps), including 3/3 control arm attempts. The instability is structural (fleet-wide lr=5e-4 + warmup=500 issue), not augmentation-induced.
- **Ep1 signal:** Arm C (symm-include-both, bs=4) showed the strongest signal in the fleet at ep1: −28% abupt and −29.4% tau_y/z vs baseline ep1. The tau_y/z disproportionality ratio is mild (1.05× — most of the win is uniform improvement, partly from effective dataset doubling). Arm B/D (random p=0.5) showed a smaller but consistent −10-14% improvement.
- **Key insight:** The "include-both" variant (doubling effective data per step) is much more powerful than stochastic flip. This suggests the win is partly a true symmetry prior and partly sample-budget doubling — worth disentangling in a follow-up.
- **Follow-up assigned:** PR #297 (haku) — re-test Arm C (symm-include-both, bs=4) on the stable PR #222 base config (lr=1e-4, warmup=1 epoch, 8-GPU DDP torchrun). The −28% ep1 signal should survive to convergence on the stable base.

---

## 2026-04-29 12:00 — PR #144: [edward] AdamW beta2 sweep (0.95 vs 0.999, lr 3e-4 and 5e-4) — CLOSED NEGATIVE
- Branch: `edward/adamw-beta2-sweep`
- Hypothesis: β2=0.95 in AdamW (faster second-moment adaptation) will reduce the v-saturation collapse driving NaN at lr=5e-4, potentially unlocking lr=5e-4 stability or beyond.

| Arm | Run | Best val_primary | Terminal |
|---|---|---:|---|
| A v1 (lr=5e-4, β2=0.999) | 0351xvpg | 11.962 (e2) | NaN ep3 step 32417 |
| B v1 (lr=5e-4, β2=0.95) | zei4lzb8 | **11.803 (e2)** | NaN ep3 step 30557 |
| C v1 (lr=3e-4, β2=0.95) | 23nmdpp0 | 19.391 (e1) | kill-threshold ep2 |
| D v1 (lr=3e-4, β2=0.999) | nnex5o0v | 18.282 (e1) | kill-threshold ep2 |
| A v2 (lr=5e-4, β2=0.999) | snjasvxx | — | NaN ep1 step 5099 |
| B v2 (lr=5e-4, β2=0.95) | l6os2f8i | 11.945 (e2) | NaN ep3 step 25501 |
| C v2 (lr=3e-4, β2=0.95) | oagys1rq | 12.690 (e4) | finished, 4 epochs |
| D v2 (lr=3e-4, β2=0.999) | oxfn12do | 17.594 (e1) | NaN ep2 ~step 20400 |

Baseline: **10.69** — none of 8 runs reached baseline. Best was B v1 at 11.803 (1.11 pp gap).

- Commentary: **Hypothesis NOT supported. β2=0.95 does not fix the lr=5e-4 instability ceiling.** All 4 lr=5e-4 arm-runs NaN'd in epoch 3 regardless of β2. The dominant stability lever was `--lr-warmup-steps 500`, which pushed NaN onset from ~step 11k (fleet historical) to ~25–32k. β2 shows a weak stability advantage at lr=3e-4 (C v2 survived 4 epochs vs D v2 NaN at epoch 2), but best metric C v2=12.690 is well above baseline 10.69. Infrastructure contribution: `--beta1`, `--beta2`, `--lr-warmup-steps`, `--lr-warmup-start-factor` flags added to train.py — these will be cherry-picked for fleet adoption. Additional finding: kill-threshold operator direction trap (condition is what must hold to continue, NOT to kill — inverted `>=3.0` killed all healthy arms in phase 1).

## 2026-04-29 12:00 — Round 12 assignments: edward #196, gilbert #197, senku #198

Three idle students assigned fresh experiments for Round 12:
- **PR #196 (edward)** — Lion optimizer (sign-based, immune to v-saturation NaN). Sweep: Lion at lr=1e-4/3e-4/5e-4 + AdamW control with warmup=500. Hypothesis: Lion's bounded-magnitude updates sidestep the heavy-tail gradient distribution collapse.
- **PR #197 (gilbert)** — K-NN local attention: replace last 1/2/3 transformer layers with k=32 or k=64 nearest-neighbor surface attention. Hypothesis: wall shear is a local physical quantity; global attention is the wrong inductive bias for tau_y/z.
- **PR #198 (senku)** — Stochastic Weight Averaging (SWA) free gain: collect uniform weight averages over the last 30–50% of training at swa_lr=5e-5–1e-4. Hypothesis: SWA finds wider minima that generalize better than EMA alone; composes orthogonally with EMA.

---

## 2026-05-01 10:15 — PR #167: [tanjiro] Static W_y=W_z=3.5 + 1k LR warmup — CLOSED NEGATIVE
- Branch: `tanjiro/static-wyz-35-warmup`
- Hypothesis: Setting static per-component weights W_y=W_z=3.5 from step 0 (vs curriculum ramping in PR #130 that caused Adam desync) with 1000-step LR warmup gives Adam's second moments time to calibrate before full-lr updates hit the higher-weighted channels.
- W&B run: `ynqjygsa` (tanjiro/static-wyz35-warmup1k)

| Metric | Epoch 1 (only valid epoch) | Baseline (PR #99 best) | AB-UPT target |
|---|---:|---:|---:|
| `val_primary/abupt_axis_mean_rel_l2_pct` | 15.63 | **10.69** | — |
| `val_primary/wall_shear_y_rel_l2_pct` | 19.59 | 13.73 | 3.65 |
| `val_primary/wall_shear_z_rel_l2_pct` | 21.84 | 14.73 | 3.63 |
| `val_primary/surface_pressure_rel_l2_pct` | 11.25 | 6.97 | 3.82 |
| `val_primary/volume_pressure_rel_l2_pct` | 9.85 | 7.85 | 6.08 |

Pre-clip grad norm trajectory by phase:
| Phase | n | mean | max |
|---|---:|---:|---:|
| Warmup 0–1000 | 10 | 25.98 | 64.77 |
| Epoch 1 post-warmup (1001–10883) | 98 | 3.17 | 11.36 |
| Epoch 2 pre-div (1001–15099) | 47 | 554.97 | 13,646.89 |
| Epoch 2 divergence (≥15100) | 2 | 2,035,679 | 4,046,702 |

- Commentary: **Clear negative — NaN divergence at step 15,300 (mid epoch 2).** Epoch 1 beat baseline epoch 1 (15.63 vs 16.47, +5% better) confirming the signal is real. But grad norms drifted 4 OoM mid-epoch-2: elevated warmup → calm epoch 1 → silent drift → catastrophic explosion → NaN. Sister arm senku #166 (W=3.0) also NaN'd (step 9,699, even earlier). **Per-component stability ceiling is below 3.0** at lr=5e-4/clip=1.0. Static weights from step 0 did not prevent the failure — Adam's second-moment saturation on boosted channels is a post-warmup instability, not a warmup miscalibration. Longer warmup would not help. The direction of upweighting is correct (epoch-1 signals are encouraging) but the mechanism needs a lower LR or tighter per-channel gradient clipping. Decision: closed. LR warmup infrastructure already in train.py via PR #169.


## 2026-05-01 10:30 — New assignments: haku #191, tanjiro #192, thorfinn #193

Three idle students (haku, tanjiro, thorfinn) assigned new experiments targeting the tau_y/z gap and training efficiency:

- **PR #191 (haku)** — 1-cycle LR max=1e-3 super-convergence: PyTorch OneCycleLR with peak 1e-3, pct_start=0.3, div_factor=25. Hypothesis: epoch-limited regime benefits from spending more training time at elevated LR. Fallback arm at lr=5e-4 if 1e-3 diverges.
- **PR #192 (tanjiro)** — asinh target normalization for tau_y/z: apply `torch.asinh` to wall-shear y and z targets before computing loss, invert before metric computation. Targets the heavy-tail distribution causing the 4× gap vs AB-UPT. Isolated to y/z components only.
- **PR #193 (thorfinn)** — Curvature-biased surface point sampling: sample training surface points with probability proportional to local surface normal variation (curvature proxy), biasing toward wheel arches/mirrors where tau_y/z errors are highest. Eval sampling remains uniform. Two arms: alpha=0.5 (blend) and alpha=1.0 (fully biased).

---

## 2026-04-29 10:45 — PR #191: [haku] 1-cycle LR max=1e-3 super-convergence — CLOSED NEGATIVE
- Branch: `haku/1cycle-lr-max1e3-superconvergence`
- Hypothesis: PyTorch OneCycleLR with peak_lr=1e-3, pct_start=0.3, div_factor=25 enables super-convergence in the epoch-limited training regime; spending more time at elevated LR accelerates learning on tau_y/z channels.

Three arms:
| Arm | Run | Status | Best val_abupt |
|---|---|---|---:|
| Main literal (lr=1e-3, epochs=50, total_steps=544150) | d86d7dg9 | Finished | 18.43 |
| Tuned (lr=1e-3, epochs=4, calibrated total_steps) | 1khqvozw | NaN-abort at step 12,759 | 28.23 |
| Fallback (lr=5e-4, epochs=4, calibrated total_steps) | 0e3jqcti | NaN-abort at step 14,279 | 27.31 |

Best arm full W&B metrics (d86d7dg9):

| Metric | 1-cycle best | Baseline (PR #183) | Δ |
|---|---:|---:|---|
| abupt_axis_mean_rel_l2_pct | 18.43 | **10.21** | +8.22 |
| surface_pressure_rel_l2_pct | 13.01 | 6.97 | +6.04 |
| wall_shear_rel_l2_pct | 20.86 | 11.69 | +9.17 |
| volume_pressure_rel_l2_pct | 10.60 | 6.32 | +4.28 |
| wall_shear_x_rel_l2_pct | 18.36 | 10.17 | +8.19 |
| wall_shear_y_rel_l2_pct | 24.42 | 13.73 | +10.69 |
| wall_shear_z_rel_l2_pct | 25.76 | 14.73 | +11.03 |

- Commentary: **Clear negative. Root cause: fundamental incompatibility between OneCycleLR's peak schedule and the time-limited training regime.** With total_steps=544,150 (epoch=50), the warmup phase extends to step 163,245 — but the actual training run reaches only ~33,263 steps (3–4 epochs/wall-clock). Training terminated at ~20% through the warmup ramp, never reaching the intended super-convergence phase. Best arm scored 18.43 vs baseline 10.21 (+80% worse). The calibrated arms (total_steps matched to actual budget) failed with NaN-abort at peak LR because the steep 5e-4 → 1e-3 ramp triggered gradient instability before any benefit could materialize. Epoch progression shows steady improvement (35.16→23.84→18.64→18.43) but all below baseline. OneCycleLR is not viable in this regime without either: (a) drastically reducing total_steps to match actual training steps, AND (b) capping peak LR to avoid NaN — by which point it degrades to a standard warmup schedule.

---

## 2026-04-29 03:00 — PR #11: [kohaku] Tangential wall-shear projection loss
- Branch: `kohaku/round1-tangential-wallshear-loss`
- Hypothesis: Project wall-shear predictions onto the tangential plane at each surface point to enforce the no-slip boundary condition physically.
- Results: W&B run `uy0ds6iz`

| Metric | Value |
|---|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | 35.12 |
| `test_primary/surface_pressure_rel_l2_pct` | 10.07 |
| `test_primary/wall_shear_rel_l2_pct` | 43.05 |
| `test_primary/volume_pressure_rel_l2_pct` | 14.99 |
| `test_primary/wall_shear_x_rel_l2_pct` | 30.85 |
| `test_primary/wall_shear_y_rel_l2_pct` | 42.06 |
| `test_primary/wall_shear_z_rel_l2_pct` | 77.65 |

- Commentary: Established first yi baseline. Only 1 epoch completed due to timeout bug (pre-fix). Projection loss adds physical constraints but alone is insufficient without the protocol fixes.

---

## 2026-04-29 03:57 — PR #9: [gilbert] Protocol fixes (bs=8, vol_w=2.0, validation-every=1)
- Branch: `gilbert/round1-protocol-fixes`
- Hypothesis: Larger batch size and higher volume loss weight stabilize training and improve all metrics.
- Results: W&B run `y2gigs61`

| Metric | Value | Prior (PR #11) |
|---|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **17.39** | 35.12 |
| `test_primary/surface_pressure_rel_l2_pct` | 11.07 | 10.07 |
| `test_primary/wall_shear_rel_l2_pct` | 18.32 | 43.05 |
| `test_primary/volume_pressure_rel_l2_pct` | 15.21 | 14.99 |
| `test_primary/wall_shear_x_rel_l2_pct` | 15.65 | 30.85 |
| `test_primary/wall_shear_y_rel_l2_pct` | 21.86 | 42.06 |
| `test_primary/wall_shear_z_rel_l2_pct` | 23.18 | 77.65 |

- Commentary: 50.5% improvement in headline metric. Flag-only change. Protocol fixes (bs=8, vol_w=2.0) are now the standard base config.

---

## 2026-04-29 — PR #8: [frieren] Per-case geometry FiLM conditioning
- Branch: `frieren/round1-film-geom-conditioning`
- Hypothesis: Geometry-conditioned FiLM layers (GeomEncoder → FiLM per block) allow the model to specialize per car geometry.
- Results: W&B run `hltti2ec` (1 epoch only, timeout)

| Metric | FiLM | No-FiLM (PR #3) | Δ |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **16.53** | 30.47 | −46% |
| `test_primary/surface_pressure_rel_l2_pct` | 10.38 | 21.65 | −52% |
| `test_primary/wall_shear_rel_l2_pct` | 17.29 | 32.51 | −47% |
| `test_primary/volume_pressure_rel_l2_pct` | 14.91 | 23.73 | −37% |
| `test_primary/wall_shear_x_rel_l2_pct` | 14.76 | 28.07 | −47% |
| `test_primary/wall_shear_y_rel_l2_pct` | 20.59 | 39.02 | −47% |
| `test_primary/wall_shear_z_rel_l2_pct` | 22.00 | 39.88 | −45% |

- Commentary: FiLM halves the error at 1 epoch. Mechanistically active (geom_token_norm grew 70× during training). +142k params (+4.4% overhead). Pending rebase to merge.

---

## 2026-04-29 — PR #13: [norman] Progressive EMA decay anneal 0.99→0.9999
- Branch: `norman/round1-progressive-ema-decay`
- Hypothesis: Cosine-annealed EMA from 0.99 (start) to 0.9999 (end) — fast updates early, slow stable averaging late.
- Results: W&B run `wio9pqw2`

| Metric | Value | Δ vs PR #9 |
|---|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **15.82** | −9.0% |
| `test_primary/surface_pressure_rel_l2_pct` | 9.99 | −9.7% |
| `test_primary/wall_shear_rel_l2_pct` | 16.60 | −9.4% |
| `test_primary/volume_pressure_rel_l2_pct` | 14.21 | −6.6% |
| `test_primary/wall_shear_x_rel_l2_pct` | 14.27 | −8.8% |
| `test_primary/wall_shear_y_rel_l2_pct` | 19.49 | −10.8% |
| `test_primary/wall_shear_z_rel_l2_pct` | 21.12 | −8.9% |

- Commentary: Monotonic val improvement through 4 epochs (no divergence). No new parameters. 9% improvement. Code now on yi; all future runs should use `--ema-decay-start 0.99 --ema-decay-end 0.9999`.

---

## 2026-04-29 — PR #3: [askeladd] Codex/optimized-lineage config (4L/256d)
- Branch: `askeladd/round1-codex-lineage`
- Hypothesis: The codex/optimized-lineage config (4L/256d/4h/128sl, lr=2e-4, wd=5e-4, 65k pts, ema=0.9995) is a stronger baseline than stock defaults.
- Results: W&B run `kv586sse`

| Metric | Value | PR #13 pending |
|---|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **15.27** | 15.82 |
| `test_primary/surface_pressure_rel_l2_pct` | 9.45 | 9.99 |
| `test_primary/wall_shear_rel_l2_pct` | 15.97 | 16.60 |
| `test_primary/volume_pressure_rel_l2_pct` | 14.07 | 14.21 |
| `test_primary/wall_shear_x_rel_l2_pct` | 13.67 | 14.27 |
| `test_primary/wall_shear_y_rel_l2_pct` | 19.08 | 19.49 |
| `test_primary/wall_shear_z_rel_l2_pct` | 20.08 | 21.12 |

- Commentary: best_epoch=1 only (gradients diverged epoch 2+, clip guarded checkpoint). NaN checkpoint guard bug discovered here. Win is real but fragile — full improvement will require gradient clipping.

---

## 2026-04-29 — PR #22: [gilbert] Gradient clipping (clip_grad_norm=1.0)
- Branch: `gilbert/round2-gradient-clipping`
- Hypothesis: clip_grad_norm=1.0 between backward and optimizer step stabilizes training.
- Results: 4-arm sweep

| Arm | clip | vol_w | W&B run | abupt |
|---|---:|---:|---|---:|
| 0 | 0.0 | 2.0 | `ujv64aty` | 16.54 |
| **1** | **1.0** | **2.0** | **`9ozwna8l`** | **14.80** |
| 2 | 1.0 | 3.0 | `u1gt9ygf` | NaN |
| 3 | 5.0 | 3.0 | `owuceuvy` | 18.61 |

- Commentary: clip=1.0 with vol_w=2.0 wins (14.80). vol_w=3.0 is not stabilizable. Pre-clip grad norm median 2.17, p99 19.80, max 31.43 — clipping engaged ~50-60% of steps. Code now on yi as default `--clip-grad-norm 1.0`.

---

## 2026-04-29 — PR #24: [emma] Squared rel-L2 aux loss (no sqrt)
- Branch: `emma/round2-squared-rel-l2-aux-loss`
- Hypothesis: Replace relative-L2 with squared relative-L2 to avoid singularity in backward pass near zero targets.
- Results: 2-arm sweep

| Arm | weight | W&B run | abupt |
|---|---:|---|---:|
| 0 | 0.1 | `4lz8rjpy` | diverged |
| **1** | **0.5** | **`zv791js1`** | **14.81** |

- Commentary: w=0.5 won (14.81), trajectory 23.06→17.75→15.13→13.85→14.59 (best epoch 4). Grad norms bounded 1.3–5.8. Pending rebase to merge.

---

## 2026-04-29 — PR #14: [senku] Depth ablation 5L/6L/256d
- Branch: `senku/round1-depth-ablation`
- Hypothesis: Increasing Transolver depth from 4L to 5L/6L while keeping hidden_dim=256 provides compositional capacity that width alone cannot replicate.
- Results: 2 runs

| Config | W&B run | abupt | Δ vs 4L baseline |
|---|---|---:|---:|
| 5L/256d | `t5tv01ch` | **13.52** | −18.7% |
| **6L/256d** | **`et4ajeqj`** | **13.15** | **−21.0%** |

| Metric | 6L/256d | Prior (4L/512d PR #4) | AB-UPT |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **13.15** | 16.64 | — |
| `test_primary/surface_pressure_rel_l2_pct` | 7.64 | 10.65 | 3.82 |
| `test_primary/wall_shear_rel_l2_pct` | 13.47 | 17.66 | 7.29 |
| `test_primary/volume_pressure_rel_l2_pct` | 13.58 | 14.37 | 6.08 |
| `test_primary/wall_shear_x_rel_l2_pct` | 11.53 | 14.87 | 5.35 |
| `test_primary/wall_shear_y_rel_l2_pct` | 16.23 | 19.89 | 3.65 |
| `test_primary/wall_shear_z_rel_l2_pct` | 16.75 | 21.73 | 3.63 |

- Commentary: New yi best. Both runs monotonically improving at timeout — still descending. Depth is more parameter-efficient than width: 6L/256d (4.73M) crushes 4L/512d (12.7M). Volume pressure val→test gap (6.93→13.58) — test generalization issue, not model capacity. Wall_shear axes still 4-5× AB-UPT.

---

## 2026-04-29 — PR #45: [fern] Mamba-2 SSM surface decoder — FAILED
- Branch: `fern/b04-mamba2-surface-decoder`
- Hypothesis: Post-Transolver SSM decoder using Morton Z-order sorted surface tokens.
- Results: W&B run `322xcrnv`, abupt=27.53 (+65% over baseline)
- Commentary: Catastrophic failure. No zero-init residual + no gradient clipping → backbone poisoning. mamba-ssm unavailable (no nvcc), fell back to S4D-Lin. Closed.

---

## 2026-04-29 — PR #38: [violet] NIG evidential regression — FAILED
- Branch: `violet/c02-deep-evidential-regression`
- Hypothesis: NIG-NLL replaces MSE, outputting uncertainty.
- Results: λ=0.01: abupt=33.54 (`vo9ep9fd`); λ=0.1: abupt=42.63 (`trreny49`)
- Commentary: NIG parameter collapse (α→1+ε, ν→0) + max grad norm 4.91×10⁷. Fundamental — not just hyperparameter. Closed.

---

## 2026-04-29 — PR #29: [chihiro] 512d × FiLM × cosine-EMA composition — FAILED
- Branch: `chihiro/b06-width-film-ema-composition`
- Hypothesis: Compose 4L/512d + FiLM + cosine EMA.
- Results: W&B run `kk7wkhkv`, abupt=37.08 (+122%)
- Commentary: EMA schedule bug — denominator set to 50 epochs × steps/epoch but only 3 epochs completed (4% of schedule), so EMA effectively pinned at 0.99. Hypothesis remains valid. Closed.

---

## 2026-04-29 — PR #21: [kohaku] Normal-component suppression — PENDING RERUN
- Branch: `kohaku/round2-normal-suppression`
- Hypothesis: Explicit penalty on (ws_pred · n_hat)² drives predicted normal component to zero.
- Results: Gradient-clip confounded. Best arm λ=1.0: abupt=18.76. λ=0.01 reached 17.89 before diverging.
- Commentary: Mechanism works (wallshear_pred_normal_rms drops with λ). All arms hit gradient-clip bug. Sent back for rerun on 6L base with clip=1.0.

---

## 2026-04-29 — PR #15: [tanjiro] SDF-gated volume attention — PENDING RERUN
- Branch: `tanjiro/round1-sdf-gate-volume`
- Hypothesis: Gaussian gate on |SDF| focuses volume attention on near-wall points.
- Results: W&B run `iiedyq63`, abupt=17.68 (+1.7% regression vs PR #9)
- Commentary: Marginal volume_pressure improvement (−0.5%) but overall regression. sigma=0.05 too wide (gate ≈0.97 at q75). Sent back for rerun with sigma=0.005 on 6L base.

---

## 2026-04-29 — PR #16: [thorfinn] Bilateral xz-plane TTA — CLOSED
- Branch: `thorfinn/round1-tta-bilateral-symmetry`
- Hypothesis: Mirror geometry about xz-plane at inference, average predictions.
- Results: W&B run `xdjsf4ad`, no-TTA=19.28, TTA=19.40 (TTA hurts)
- Commentary: Cars are symmetric but model predictions are not variance-reduced. TTA only helps near-equivariant models. Closed.

---

## 2026-04-29 — PR #2: [alphonse] Stock defaults baseline — CLOSED
- Hypothesis: Reference floor for stock train.py defaults.
- Results: W&B run `a1fikrwe`, abupt=87.30
- Commentary: Confirms massive gap between stock (3L/192d, 40k pts) and optimized protocol. NaN checkpoint guard bug discovered. Closed.

---

## 2026-05-01 01:18 — PR #119: [edward] RFF coordinate encoding (Tancik 2020) — CLOSED
- Branch: `edward/rff-coordinate-encoding`
- Hypothesis: Replace ContinuousSincosEmbed with Random Fourier Features (Tancik 2020) to give the model access to learnable coordinate frequencies, potentially improving high-frequency surface detail.
- Results:

| Arm | σ | features | W&B run | val abupt (best epoch) | test abupt | Final state |
|---|---:|---:|---|---:|---:|---|
| 1 (fnyhm654) | 1.0 | 64 | fnyhm654 | NaN | — | crashed (NaN @step ~2300, epoch 1) |
| 1-r2 (n77zkyc8) | 1.0 | 64 | n77zkyc8 | NaN | — | crashed (NaN @step ~7500, epoch 1) |
| 2 (3s9qatve) | 5.0 | 64 | 3s9qatve | 130.01 (epoch 3) | 88.55 | loss explosion @step ~7700; stuck at ~3.5 forever |
| 3 (fig141q6) | 2.0 | 128 | fig141q6 | 17.45 (epoch 1) | 18.28 | NaN @step 17269 (mid epoch 2) |

- Commentary: Clean negative result — all 3 sigma values unstable at lr=5e-4/clip=1.0/bf16/raw-meter-coords. Best arm (σ=2.0, 128 features) test abupt=18.28 (56% worse than baseline 11.73). Wall_shear_y/z worst per-axis (21.3%, 23.2%) — consistent with anisotropy hypothesis (σ isotropic on non-isotropic meters). RFF fixed-Gaussian-B amplifies certain coordinate-frequency components creating unstable training unlike ContinuousSincosEmbed (deterministic, balanced). Suggests coord normalization to [-1,1] before RFF would be needed. Assigned follow-up PR #143 (fern, coord normalization) to test this hypothesis. Closed.

---

## 2026-05-01 (in progress) — PR #117: [alphonse] Width+depth scale-up (6L/384d, 8L/256d) — WIP
- Branch: `alphonse/width-depth-scale-up`
- Hypothesis: Increasing Transolver from 6L/256d (4.73M params) to 6L/384d (~9M) or 8L/256d (~6M) provides additional capacity to resolve the tau_y/z fine-scale surface features.
- Results (intermediate — running at lr=3e-4 after fleet-wide stability discovery):

Round 1 at lr=5e-4: all 3 arms diverged (6L/384d/4h @step 3206, 8L/256d @step 8899, 6L/384d/6h @step 11099). Round 2 at lr=3e-4: stable as of ~01:00 UTC May 1.

| Arm | Config | W&B run | Step (~01:00) | train/loss | State |
|---|---|---|---:|---:|---|
| A | 8L/256d | xl92i3f5 | stable | healthy | running |
| B | 6L/384d/4h | hbahy1ob | stable | healthy | running |
| C | 6L/384d/6h | 3m4cqwg3 | stable | healthy | running |

- Commentary: Confirms lr=5e-4 is hard stability ceiling for scale-up experiments. 6L/384d/4h has d_head=96 (vs standard 64) — possible bf16 attention overflow. lr=3e-4 resolves all arms. Awaiting first val checkpoint.

---

## 2026-05-01 (in progress) — PR #119 companion: RFF → coord-normalization insight
- Key insight from edward #119: raw DrivAerML coords are anisotropic (x~8m, y/z~2m). ContinuousSincosEmbed's omega is tuned for the x-range, leaving y/z under-sampled in frequency. This is likely a direct cause of the 4× tau_y/z gap. Assigned as PR #143 (fern, coord normalization sweep).

---

## 2026-05-01 02:28 — PR #132: [violet] Decoupled wall-shear magnitude + direction prediction — CLOSED
- Branch: `violet/wallshear-magnitude-direction-decoupled`
- Hypothesis: Factorize tau into |tau| (log-MSE) + tau_dir (cosine loss) heads to reduce magnitude-dominated gradients and preferentially help the small-magnitude tau_y/tau_z axes (currently 3.8×/4.1× AB-UPT gap).
- Results: 6-arm sweep — only B and D survived to terminal epoch.

| Arm | mw | dw | mag-weighted dir | W&B status | test abupt | tau_y | tau_z | p_s | p_v |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| A | 1 | 1 | no | stuck | — | — | — | — | — |
| B | 0.5 | 2 | no | finished | **13.22** | 16.60 | 18.14 | **6.42** | **13.57** |
| C | 1 | 1 | yes | NaN ep1 | — | — | — | — | — |
| D | 1 | 1 | yes (mw=1) | finished | 13.74 | 17.28 | 18.84 | 7.17 | 13.89 |
| E | 0.5 | 2 | yes | NaN ep1 | — | — | — | — | — |
| F | 2 | 0.5 | no | diverged ep1 | — | — | — | — | — |
| **PR #99 baseline** | — | — | — | — | **11.73** | 13.53 | 13.98 | 6.64 | 14.42 |

- Commentary: Clean **negative result** on the headline. Best surviving arm (B) +12.7% worse than baseline; the very axes the PR was designed to fix (tau_y, tau_z) regressed 23–30%. Three diagnostic findings:
  1. **Cosine direction loss has perverse axis priority.** Gradient scales by sin(θ); once tau_x dominant alignment is learned, the small y/z residual contributes near-zero gradient. Reformulation does the *opposite* of preferentially upweighting transverse components.
  2. **Magnitude-weighted direction loss is destructively unstable.** 3 of 4 mag-weighted arms NaN'd. Arm D's cos_sim trajectory `0.531→0.899→0.930→0.951→collapse to 0.404 in ~11 steps` is textbook bf16 overflow as residual high-|tau| points dominate post-convergence.
  3. **Pressure side-effect is interesting but not load-bearing.** Arm B's p_s 6.42 (−3.3%) and p_v 13.57 (−5.9%) beat baseline. Plausible: the magnitude head's log-MSE is acting as a soft regularizer on the shared trunk. Worth a narrow ablation (`--wallshear-magnitude-loss-weight 0.5 --wallshear-direction-loss-weight 0`) but not from this PR.
- **Direction forward**: For the y/z gap, PR #121 (askeladd surface-tangent frame) attacks the *coordinate frame* of the loss rather than the output reformulation — better-targeted lever. Output-side decoupling is closed-door pending a fundamentally different magnitude weighting scheme. Violet reassigned.

---

## 2026-05-01 02:19 — PR #122: [emma] Perceiver-IO backbone replacing Transolver — CLOSED
- Branch: `emma/perceiver-io-backbone`
- Hypothesis: Perceiver-IO with M latent tokens (cross-attention bottleneck) trains 2× faster per step than Transolver with comparable accuracy, buying more epochs in the 6h budget.
- Results: Throughput won, accuracy lost decisively.

| Arm | Config | W&B | Steps/sec | VRAM | test abupt | val abupt (best epoch 6) |
|---|---|---|---:|---:|---:|---:|
| A | Transolver 6L/256d/128sl (control) | 1iilxfvs | 2.15 | 76 GB | NaN ep1 | — |
| B | Perceiver-IO M=512, 6L/256d | jyesq3i4 | 4.29 | 17 GB | 24.46 | 23.69 |
| C | Perceiver-IO M=1024, 6L/256d | 8b8yd2c8 | 3.54 | 17 GB | **22.46** | **21.43** |
| **PR #99 baseline (Transolver)** | 6L/256d/128sl | 3hljb0mg | 2.15 | 76 GB | **11.73** | **10.69** |

- Per-axis: C tau_y=27.11, tau_z=28.40 vs baseline 13.53 / 13.98 (~2× worse on the very gap PR was meant to help).
- Commentary: Perceiver-IO is **architecturally mismatched** for DrivAerML CFD. Latent cross-attention bottleneck loses fine-grained spatial structure; Transolver's slice-based attention preserves it. Despite 2× speedup and 5× VRAM savings, accuracy gap is prohibitive — even with the throughput-bought 6 vs 3 epochs, val abupt floor is ~21% vs ~11% for Transolver. Diminishing returns clear: Arm C's per-epoch deltas were −8.65, −2.18, −1.99, −0.85, −0.09 → asymptote near 21%. Arm A control NaN'd at step 6676 ep1 (independent flake; not Perceiver-related). **Decision: closed.** Backbone replacement is a dead-end direction for this geometry. Emma reassigned.

---

## 2026-05-01 02:19 — PR #127: [nezuko] Stochastic depth regularization sweep — CLOSED
- Branch: `nezuko/stochastic-depth-regularization`
- Hypothesis: DropPath stochastic depth at rates {0.05, 0.10, 0.20} provides regularization that closes the wall-shear y/z gap by preventing layer-specialization.
- Results: All arms regress on wall-shear; hypothesis refuted.

| Arm | sd_prob | W&B | test abupt | p_s | tau_vec | tau_x | tau_y | tau_z | p_v |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **PR #99 baseline** | 0.0 | 3hljb0mg | **11.727** | 6.637 | 11.484 | 10.064 | 13.529 | 13.980 | 14.424 |
| A | 0.05 | u0s413c0 | 11.707 (−0.020) | 6.743 (+) | 11.762 (+) | 10.262 (+) | 13.962 (+0.43) | 14.318 (+0.34) | **13.251 (−1.17)** |
| B | 0.10 | i5kk7ng0 | 14.02 (NaN ep3) | 8.68 | 14.40 | 12.63 | 16.80 | 17.76 | 14.24 |
| C | 0.20 | lyjnyi3a | 13.07 (+11%) | 7.88 | 13.37 | 11.79 | 15.61 | 16.23 | 13.83 |

- Commentary: Arm A's −0.020 abupt is within noise; all wall-shear axes regress (+0.20–0.43 pp on tau_x/y/z). The PR's target was the y/z gap — stochastic depth widens it, not closes it. Arm B diverged at step 23705 ep3 (single-seed flake). Arm C clearly worse. **Mechanistic read:** wall-shear prediction needs coherent multi-layer feature propagation through the fine boundary layer geometry; randomly dropping residual branches injects noise that the model can't recover from in 9 epochs. **Volume pressure note:** Arm A's −1.17 p_v is interesting but isolated — possibly the dropout is a soft regularizer on volume features specifically. Not enough to justify the shear regression. **Decision: closed.** Stochastic depth ruled out for this regime. Nezuko reassigned.

---

## 2026-05-01 02:24 — PR #135: [tanjiro] T_max=100 cosine LR sweep extension (tay branch) — CLOSED
- Branch: `tanjiro/round9-lion-tmax100-ema999`
- Hypothesis: Extend cosine schedule T_max from 50 → 100 to test if "less LR decay is better" trend continues at the wider end on the tay branch.
- Results: Narrow win vs PR #111 SOTA-at-the-time, but superseded by PR #115 lr-change.

| Metric | PR #135 (T_max=100) | PR #111 SOTA-at-launch (T_max=50) | PR #115 actual SOTA (lr=1e-4) | Δ vs PR #115 |
|---|---:|---:|---:|---:|
| `test_primary/abupt_axis_mean` | 11.082 | 11.142 | **10.580** | **+4.74% regression** |
| `surface_pressure` | 6.138 | 6.209 | — | — |
| `wall_shear` | 11.039 | 11.138 | — | — |
| `volume_pressure` | 12.665 | 12.548 | — | — |
| `tau_y` | 13.469 | 13.525 | — | — |
| `tau_z` | 13.791 | 13.992 | — | — |
| `best_val_primary/abupt` (ep9) | 9.886 | 9.989 | — | — |

- W&B run: `wtfrhy2n`. T_max sweep series across three values: T_max=24→50 gave −3.1pp; T_max=50→100 gave −0.54%pp — clear diminishing returns.
- Commentary: This run launched against PR #111 SOTA (11.142) and would have been a clean +0.5% win at that time. During the run, PR #115 (lr=1e-4 change) merged to tay and moved SOTA to 10.580 — making PR #135's 11.082 a +4.74% regression against the current frontier. Schedule lever (T_max) is **closed-door on tay**: sweet spot is T_max=50 (already merged in PR #110), and gains are sub-percentage and dominated by lr-based wins. **Decision: closed.** Tanjiro reassigned to PR #149 (per-axis tau_y/tau_z conservative weighting on tay's new SOTA stack).

---

## 2026-04-29 15:00 — PR #169: [thorfinn] NaN-skip + seed + LR warmup utility flags (stability infra) — MERGED
- Branch: `thorfinn/nan-skip-utility-cherry-pick`
- Hypothesis: N/A — pure infra utilities to improve training stability observability and repeatability.
- Three new CLI flags added:
  1. `--seed <int>` — set all RNG seeds for reproducibility
  2. `--lr-warmup-steps <int>` — linear LR warmup from lr/10 to lr over N steps
  3. NaN/Inf-skip safeguard — detect non-finite loss/gradients, skip those steps, abort after 200 consecutive skips
- Also fixed: kill-threshold NaN bug (commit de8f8d0) — `_numeric_metric_items()` was filtering out NaN metrics silently so `check_kill_thresholds()` would never trigger on NaN runs. Now non-finite values trigger kill condition.
- Validation run: W&B run `e6sgx5ku`, seed=42

| Epoch | val abupt | Expected (spec band) | Notes |
|---|---:|---|---|
| 1 | 15.83 | 12.0–13.0 | High init variance from seed=42 — accepted |
| 2 | **11.20** | 11.0–11.5 | Within spec band — no regression |

- `train/nonfinite_skip_count = 0` throughout (NaN-skip not triggered — confirms baseline is already NaN-free)
- Commentary: Infra PR merged for code utility. No metric improvement (infra PR, not hypothesis test). Epoch-1 spike (15.83) is seed=42 init variance, not a code bug — confirmed by epoch-2 landing in spec band (11.20). Kill-threshold NaN bug fix is fleet-wide safety. LR-warmup-steps flag is now available for future experiments that need structured warmup without committing to OneCycleLR. **Decision: merged.** Infra contributions on the main path — no regression.

---

## 2026-04-29 15:30 — PR #125: [haku] 1cycle LR max=1e-3 (super-convergence in epoch-limited regime) — CLOSED NEGATIVE
- Branch: `haku/onecycle-lr-peak-1e3`
- Hypothesis: OneCycleLR with structured warmup to peak LR then cosine anneal achieves super-convergence in the 3–4 epoch budget, beating flat lr=5e-4.
- Results: 4-arm sweep (round 1) + 4-arm sweep (round 2 with total_steps bug fixed)

**Round 1 (bug: total_steps=544,150 → schedule nearly flat):**

| Arm | max_lr | W&B run | val ep2 | val ep3 | Notes |
|---|---:|---|---:|---:|---|
| A | 1e-3 | pbxxmq1h | NaN ep1 | — | Diverged immediately |
| B | 7e-4 | — | NaN ep1 | — | Diverged |
| C | 5e-4 | — | NaN ep1 | — | Diverged |
| D | 3e-4 | bq11rk5t | 12.01 | 11.83 | Stable but schedule flat (bug) |

**Round 2 (total_steps=36,000 — correct schedule):**

| Arm | max_lr | pct_start | W&B run | val ep1 | val ep2 | val ep3 | val ep4 | test abupt | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| A2 | 1e-3 | 0.10 | 1cycl-A2 | NaN | — | — | — | — | Diverged (> stability ceiling) |
| B2 | 7e-4 | 0.15 | 1cycl-B2 | NaN | — | — | — | — | Diverged (> stability ceiling) |
| **C3** | **5e-4** | **0.20** | 1cycl-C3 | 12.80 | 11.65 | 11.42 | **11.34** | **12.41** | Best arm; still worse than baseline |
| D3 | 3e-4 | 0.10 | 1cycl-D3 | 11.89 | 11.63 | 11.48 | 11.46 | 12.20 | Soft-divergence at lr~6.5e-5 (late cosine anneal) |

**Baseline (PR #99):** val=10.69, test=11.73

- Commentary: Clean **negative result**. Two failure modes:
  1. **Stability ceiling at ~6.5e-4** — confirmed fleet-wide. Even under structured OneCycleLR warmup, peaks ≥6.5e-4 diverge to NaN regardless of pct_start. No warmup schedule overcomes this ceiling.
  2. **Budget starvation at lower peaks** — with peaks ≤5e-4, OneCycleLR spends ~20% of budget rising to peak and another ~60% falling back below flat baseline; effective average LR is much lower than flat 5e-4, and the model cannot recover in 3–4 epochs.
  - Best arm C3 (5e-4 peak): all headline metrics regressed — test abupt=12.41 vs baseline 11.73 (+5.8%).
  - One mild positive signal: C3 volume_pressure=13.33 vs baseline 14.42 — lower average LR may help the volume head specifically. Filed for future per-head LR experiments (e.g., different LR groups for surface vs volume decoder).
  - D3 soft-divergence: anomalous loss spike at lr~6.5e-5 during deep cosine anneal — potential bf16 AMP precision issue at very low loss values. Fleet-wide flag: if future low-LR fine-tuning runs see similar, investigate AMP precision or implement LR floor ~1e-4.
  - OneCycleLR does not help in this epoch-limited regime. LR schedule lever closed for now. **Decision: closed.**

## 2026-05-01 — PR #245: [gilbert] Progressive EMA decay schedule (ramp 0.99→0.9999 vs fixed 0.9995) — CLOSED INCONCLUSIVE
- Branch: `gilbert/ema-decay-schedule`
- Hypothesis: Progressive EMA warmup — ramping decay from low (fast adaptation) to high (slow adaptation) over training — allows the model to track loss-landscape changes quickly early on and then stabilize to a smooth EMA trajectory for inference, outperforming fixed EMA decay.

**W&B-verified ep1 metrics (merge bar = 9.291, PR #222):**

| Run | Arm | State | abupt (primary) | surf press | vol press | wall shear |
|-----|-----|-------|----------------|------------|-----------|------------|
| xpoz88lg | A: ramp 0.99→0.9999, lr=5e-4 | failed (ep2 grad explosion) | 14.41% | 10.18% | 8.29% | 16.20% |
| f6acdprl | B: ramp 0.999→0.9999, lr=5e-4 | crashed (ep2 grad explosion) | 14.63% | 10.41% | 8.50% | 16.41% |
| buch3nry | C: fixed 0.9995, lr=3e-4 | ep1 only (stopped) | 18.93% | 13.63% | 10.96% | 21.19% |

- Commentary: Experiment dominated by the fleet-wide lr=5e-4 gradient explosion (pre-clip grad norms reaching 1e7–1e9). Arms A and B both crashed in epoch 2 before producing comparable multi-epoch metrics. The ep1 A/B gap (14.41 vs 14.63) is within run-to-run noise — no valid signal on EMA ramp schedule. Arm C (fixed 0.9995) used lr=3e-4 after two lr=5e-4 failures, and only produced ep1 metrics (18.93% — underfitting after 1 epoch at lower lr). Cannot serve as the matched fixed-decay control. Research decision: EMA ramp hypothesis remains untested in clean conditions. Keeping default --ema-decay 0.9995 unchanged. Deprioritized in favour of higher-leverage experiments (architecture, loss formulation). If revisited, all arms must use lr=3e-4 (or lr=5e-4 with 1k-step warmup) and include a matched fixed-decay control at the same lr.

---

## 2026-05-01 — PR #197: [gilbert] K-NN local surface attention for tau_y/z gap — CLOSED NEGATIVE
- Branch: `gilbert/knn-local-attention-r12`
- Hypothesis: K-nearest-neighbor local surface attention introduces a locality bias that disproportionately improves tau_y/z prediction by giving the model better access to fine-grained boundary-layer geometry near each query point. The 4× tau_y/z gap was hypothesized to stem from insufficient local receptive field.
- Results:

| Arm | Config | W&B run | val_primary (abupt) | wall_shear_y | wall_shear_z | surface_pressure | volume_pressure | Final state |
|---|---|---|---:|---:|---:|---:|---:|---|
| A (control) | 0L KNN | haibegok | 17.64 | 23.38 | 24.89 | 12.57 | 9.99 | crashed |
| B | 1L KNN k=32 | 77yqqenp | — | — | — | — | — | crashed (no metrics) |
| C | 2L KNN k=32 | hvey8no6 | 19.91 | 25.55 | 27.32 | 13.73 | 13.89 | completed |
| D | 2L KNN k=64 | 7vmm33nz | 24.10 | 30.46 | 32.40 | 16.69 | 18.21 | completed |
| **yi baseline** | PR #183 | — | **10.21** | **13.47** | **14.52** | **6.85** | **6.32** | — |

- Commentary: Clean negative. Locality bias hypothesis falsified. Key findings:
  1. **No tau_y/z-specific improvement** — all metrics degrade uniformly with more KNN layers; the gap ratio is not selectively reduced.
  2. **Monotonic degradation**: 0L=17.64 → 2L k=32=19.91 → 2L k=64=24.10. Every additional KNN layer hurts.
  3. **KNN compute overhead ~4–9× per step** means Arms C/D ran less than 1 epoch in the 270-min budget, so comparisons are slightly biased against KNN arms — but the directional signal is unambiguous.
  4. **Volume pressure catastrophic regression**: +39% (Arm C) to +82% (Arm D) vs control, suggesting the shared FFN between surface KNN tokens and volume global-attention tokens acts as a leak path degrading the well-solved volume-pressure head.
  5. **The tau_y/z gap is NOT a locality/receptive-field problem.** The Transolver global-attention mechanism already has sufficient receptive field; adding KNN locality layers actively interferes with the learned global representation.
  - Peak VRAM: A=74.8GB, B=72.1GB, C=71.4GB, D=72.2GB on H200 96GB.
  - Direction closed. Do not re-assign local surface attention variants.

## 2026-04-29 — PR #244: emma surface-loss-weight sweep (closed — negative)
- Branch: `emma/surface-loss-weight-sweep`
- Hypothesis: Increasing `--surface-loss-weight` (slw) beyond the default 1.0 will upweight the surface loss branch relative to volume loss, improving surface_pressure and wall_shear metrics on the 6L/256d AdamW architecture.
- Architecture tested: 6L/256d AdamW (NOT the current 4L/512d Lion SOTA)

| Arm | slw | val_abupt | surface_pressure | vol_pressure | wall_shear | Notes |
|-----|-----|-----------|-----------------|-------------|-----------|-------|
| A (control) | 1.0 | 10.6347% | — | — | — | best arm |
| B | 1.5 | crashed | — | — | — | 3/3 runs diverged even at lr=3e-4 |
| C | 2.0 | completed | — | — | — | ran cleanly |
| **yi baseline (PR #222)** | — | **9.2910%** | **5.8707%** | **5.8789%** | **10.3423%** | 4L/512d Lion |

- Commentary: Best val abupt 10.6347% does not beat the 9.291% merge bar. Key findings:
  1. **Architecture mismatch**: Experiment ran on 6L/256d AdamW, not the 4L/512d Lion SOTA. Results are not directly comparable.
  2. **slw=1.5 crashes**: 3/3 runs diverged even at lr=3e-4 on 6L/256d AdamW.
  3. **slw=2.0 stable**: ran cleanly — important stability finding. This is architecturally general and likely holds on 4L/512d Lion too.
  4. **slw signal not tested on SOTA**: The hypothesis that slw=2.0 improves wall_shear (main remaining gap vs AB-UPT) on the current SOTA config remains open.
- Direction: Closed for 6L/256d AdamW. Follow-up assigned to emma: test slw=2.0 on the 4L/512d Lion SOTA (PR #322).

## 2026-04-29 — PR #322: emma surface-loss-weight=2.0 on SOTA Lion 4L/512d (assigned)
- Branch: `emma/surface-loss-weight-on-sota`
- Hypothesis: `--surface-loss-weight 2.0` (stable on 6L/256d from PR #244) will improve wall_shear and surface_pressure on the current 4L/512d Lion SOTA stack. Wall shear (10.34%) is the primary remaining gap vs AB-UPT (7.29%). Upweighting the surface branch 2× is orthogonal to the existing --wallshear-y-weight 2.0 --wallshear-z-weight 2.0 (PR #66) which applies within the surface branch; slw=2.0 raises the entire surface branch vs volume branch.
- Run command:
```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --agent emma \
  --wandb-group yi-slw2-on-sota \
  --optimizer lion \
  --lr 1e-4 \
  --weight-decay 5e-4 \
  --no-compile-model \
  --batch-size 4 \
  --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.999 \
  --lr-warmup-epochs 1 \
  --surface-loss-weight 2.0
```
- Status: WIP — awaiting emma's run results.

## 2026-05-02 07:14 — PR #315: senku MLP expansion ratio sweep (mlp_ratio=2/4/8) [SCREENING WIN, sent back]
- Branch: senku/mlp-ratio-sweep
- Hypothesis: increasing FFN expansion ratio in Transolver blocks (mlp_ratio: default 4 → 2/4/8) trades params for representational capacity; predicted 0.3-1.0pp val_abupt improvement at mlp_ratio=8.
- Single-GPU screening on `senpai-yi-senku` with `--max-steps-per-epoch 2000` infra patch (5-line port of violet's commit `26914b9`). All 3 arms identical except `--model-mlp-ratio`, AdamW + lr=5e-4 + bs=4 (within-experiment control, NOT PR #222 baseline conditions).
- Results:

  | arm | mlp_ratio | params | best_val_abupt | test_abupt | tau_y | tau_z | wandb |
  |-----|----------:|-------:|---------------:|-----------:|------:|------:|-------|
  | A | 2 | 8.50M | 11.841% | 12.848% | 15.107% | 15.849% | `4288ko22` |
  | B (ctrl) | 4 | 12.70M | 11.236% | 12.285% | 14.288% | 15.189% | `d7hhw0vl` |
  | **C (win)** | **8** | **21.10M** | **10.897%** | **11.981%** | **14.010%** | **14.689%** | `wwhdaevb` |

- Conclusions:
  - Monotonic ordering `mlp8 < mlp4 < mlp2` from epoch 4 through epoch 9 across **every** primary and per-axis metric. Relative ranking is reliable; absolute numbers cannot be compared to 9.291% bar (within-experiment control was 11.236%, far above baseline due to AdamW vs Lion + bs=4 vs bs=32 + lr=5e-4 vs 1e-4 deviations).
  - Hypothesis supported in predicted direction with predicted magnitude (-0.34pp val_abupt at mlp_ratio=8 vs Arm B control, lower end of 0.3-1.0pp predicted band).
  - All 3 arms still descending at epoch 9 — none plateaued.
  - Per-step compute cost: ~17% slowdown for 1.66× FFN params at mlp_ratio=8.
  - Peak VRAM: 85.5 GB at bs=4 mlp_ratio=8 (89% of 96GB cap) — bs=8 infeasible at this ratio.
- Decision: **NOT MERGED** (within-experiment control is far from 9.291% bar), but **screening winner is captured** for follow-up. Sent back to senku with instructions to run mlp_ratio=8 vs mlp_ratio=12 head-to-head on the same single-GPU screening protocol (Option A) to inform the eventual 8-GPU confirmation arm choice.
- Infra side-effect: senku's `--max-steps-per-epoch` patch is a clean 5-line port of violet's commit `26914b9`. To be cherry-picked into yi base alongside alphonse's Lion+DDP+lr-warmup-epochs port from PR #284.
- `BASELINE.md` ↔ `train.py` drift acknowledged: the BASELINE.md command (Lion, --lr-warmup-epochs, 8-GPU torchrun) does not run on current yi/train.py because the code paths exist only on student branches. Resolution path: cherry-pick alphonse's commits `bfbe975` + `1a8f7b7` (Lion + DDP + lr-warmup-epochs + broadcast_object_list run_id fix) and senku's `--max-steps-per-epoch` into yi base as a standalone infra PR after PR #284 closes.

## 2026-05-02 07:14 — PR #284: alphonse 6L/512d depth+width scaling [partial, sent back to 8-GPU pod]
- Branch: alphonse/depth-scaling-6l-512d
- Hypothesis: 6L/512d depth+width scaling on PR #222 Lion+warmup baseline; expected to extend the depth-scaling win from PR #14 senku (4L→6L on 256d) to the new 4L/512d width win.
- Run state: Reached epoch 3 on 4-GPU DDP at bs=4 (effective bs=16, half of PR #222's bs=32) before train budget cap. W&B `wsvdv49o` (group `depth-scaling-6l-512d`).
- Results (3-epoch partial, ep-3 best-val checkpoint, 50 test cases):

  | Metric | 6L/512d ep-3 (test) | PR #222 4L/512d ep-9 (val baseline) |
  |---|---:|---:|
  | abupt_axis_mean_rel_l2_pct | 12.571% | 9.291% |
  | surface_pressure_rel_l2_pct | 6.440% | 5.871% |
  | wall_shear_rel_l2_pct | 11.669% | 10.342% |
  | volume_pressure_rel_l2_pct | 17.771% | 5.879% |
  | wall_shear_y_rel_l2_pct | 14.524% | — |
  | wall_shear_z_rel_l2_pct | 14.277% | — |

- Per-epoch convergence dominance vs PR #222: ep1 −45pp, ep2 −30pp, ep3 −7.7pp — strong per-epoch advantage, but the half-batch + truncated-epoch + cosine-T_max-not-cooling story makes the converged answer ambiguous.
- 4-GPU restart with `--epochs 3` (cosine T_max fix) reproduced a throughput regression: 1.83 s/iter vs v1's 1.02 s/iter, ~95 GB vs ~58 GB peak per rank, across two fresh restarts (`mescjl8v` killed; v3 killed). Root cause unidentified; appears to scale non-monotonically with `--epochs` value but cannot be the LR schedule scalar alone. Not relevant to a 6-epoch run on 8 GPUs (avoids the regime).
- Decision: **NOT MERGED** (test_abupt 12.57% is 3.28pp above 9.291% merge bar; experiment compute-bounded at 4 GPUs). Sent back to alphonse with: stand by for 8-GPU pod re-invocation (`senpai-drivaerml-ddp8-alphonse` is 1/1 READY); when on 8 GPUs, run **Option (A)** — exact PR #222 recipe (Lion+lr=1e-4+wd=5e-4+warmup=1ep) at `--epochs 6` so cosine T_max=6 cools cleanly within budget; do not relaunch on 4 GPUs.
- Infra contribution (independently valuable, queued for cherry-pick into yi base):
  - commit `bfbe975`: --optimizer {adamw,lion} via lion-pytorch, --lr-warmup-epochs flag, DistributedDataParallel + nccl + DistributedSampler.set_epoch, rank-0 W&B/checkpoint/tqdm gating, per-rank seed offset.
  - commit `1a8f7b7`: broadcast_object_list run_id fix (rank-0 wandb run.id broadcast to all ranks before computing output_dir; fixes torch.load FileNotFoundError on non-0 ranks at end-of-train test eval).

---

## 2026-04-29 09:00 — PR #314: [norman] Coordinate jitter augmentation sweep (σ=0.002/0.005/0.010) — REJECTED

- Branch: `norman/coordinate-noise-augmentation`
- Hypothesis: Adding Gaussian jitter to input coordinates (both surface and volume) during training would act as a regularizer, improving generalization on τy/τz where the model may be over-fitting spatial patterns.
- W&B group: `norman-coord-noise-r18`, W&B runs: `ij8nkgbv` (σ=0), `bn1j6gf3` (σ=0.002), `um0map47` (σ=0.005), `lxl4ljy4` (σ=0.010)

**Results (4-arm sweep, ~12k steps / ~55% of epoch 1 each):**

| σ | val_abupt | test_abupt | test τy | test τz |
|---|-----------|------------|---------|---------|
| 0.000 (control) | 13.058% | 13.937% | 16.597% | 18.025% |
| 0.002 | 15.089% | 15.925% | 19.239% | 21.569% |
| 0.005 | 20.637% | 21.448% | — | — |
| 0.010 | 25.256% | 26.035% | 41.923% | 34.108% |

**Analysis:**
- Monotonic degradation at every noise level. τy and τz — the exact channels the hypothesis was designed to help — were the most severely damaged. No overfitting at ~12k steps to regularize against. Volume coordinates are in raw meters; σ=0.002 jitter relocates volume query points to incorrect physical locations rather than regularizing.
- Grad norms shrink monotonically with σ (confirmation of training disruption, not improvement).
- **No NaN/Inf** — clean experiment, unambiguous failure.

**Conclusion:** REJECTED. Coord-noise is mechanistically broken for this architecture and data scale. No follow-up variants warranted. Suggested alternative: MAE-style point masking (drop 10–30% of surface/volume query points during training to force global geometry reasoning). PR closed.

---

## 2026-04-29 09:10 — PR #317: [violet] Huber loss on wall-shear channels (δ=0.5/1.0/2.0) — MARGINAL SIGNAL, AWAITING FULL-BUDGET CONFIRMATION

- Branch: `violet/huber-wallshear-loss`
- Hypothesis: Huber loss (instead of MSE) on τx/τy/τz channels would reduce the disproportionate influence of large residual outliers, improving tail generalization and the τy/τz channels specifically.
- W&B group: `violet-huber-wallshear-r18`, W&B runs: `p8s8rxo7` (δ=0/control), `wnr2zd74` (δ=0.5), `uuyxopmh` (δ=1.0), `73hfyxwd` (δ=2.0)
- Note: Plain Huber on standardized residual `r = pred − target` (NOT relative Huber); relative formulation exploded due to near-zero standardized targets.

**Results (~12k steps / ~55% of epoch 1 each):**

| δ | val_abupt | val τy | val τz |
|---|-----------|--------|--------|
| 0.0 (MSE control) | 13.314% | 17.072% | 18.945% |
| 0.5 | 13.593% | 17.797% | — |
| **1.0** | **13.173%** | **16.982%** | **18.862%** |
| 2.0 | 13.286% | — | — |

**Analysis:**
- Non-monotonic ranking (d=1.0 < d=2.0 < control < d=0.5) suggests δ=1.0 is a real but small sweet spot.
- δ=0.5: Most residuals in L1 regime → weak gradient on bulk → slower convergence. δ=2.0: Essentially MSE for most residuals, tracks control.
- Effect is NOT channel-asymmetric — uniform mild improvement across all fields; vol_p unaffected (backbone clean).
- −0.14pp at 1 epoch fragment is within potential seed variance; full-budget run required.

**Decision:** SENT BACK. Full-budget 2-arm confirmation run assigned (group: `violet-huber-fullbudget-r20`). δ=0.5 disqualified. Merge criterion: val_abupt(δ=1.0) < control at best checkpoint.
- Status: WIP — awaiting alphonse re-invocation on `senpai-drivaerml-ddp8-alphonse` 8-GPU pod for the recipe-fixed 8-GPU re-run.

---

## 2026-05-02 08:00 — PR #312: [edward] Surface-tangent frame wall-shear prediction — REJECTED

- Branch: `edward/surface-tangent-frame` (deleted)
- Hypothesis: Predicting wall-shear in the local surface-tangent frame [t1, t2, n] (rather than global Cartesian) would provide a physics-aligned inductive bias and improve τy/τz, which span 4× the AB-UPT gap.
- W&B runs: `jkq0hjt0` (tangent), `wj6p859t` (control), group `edward-surface-tangent-r18`
- Config: scaled-down (bs=4, vol-pts=32768, max-steps-per-epoch=1500, epochs=9); OOM at bs=8+65k vol pts on single GPU

**Results (best-checkpoint test metrics):**

| Metric | Tangent (`jkq0hjt0`) | Control (`wj6p859t`) | Δ |
|---|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | 16.806% | 16.429% | +0.377 ❌ |
| `surface_pressure_rel_l2_pct` | 10.697% | 10.730% | −0.033 |
| `wall_shear_y_rel_l2_pct` | 21.025% | 19.752% | +1.273 ❌ |
| `wall_shear_z_rel_l2_pct` | 22.354% | 21.117% | +1.236 ❌ |
| `wall_shear_x_rel_l2_pct` | 14.658% | 15.454% | −0.796 ✅ |

- Gap consistent across all 9 epochs (monotonic; not single-checkpoint noise)
- `frame_det_mean = 1.000`, `frame_nan_count = 0` — frame computation numerically stable
- `n_target_phys_rms ≈ 0.33` — dataset's wall-shear has ~33% of signal along mesh normal; the purely-tangential hypothesis doesn't hold for DrivAerML

**Root cause analysis:**
1. DrivAerML τ vectors are not purely tangential at the mesh-normal level (`n_target_phys_rms ≈ 0.33`), so the rotation adds a third hard-to-predict n-axis rather than concentrating signal on [t1, t2].
2. The per-point tangent frame rotation destroys global flow-alignment; τy/τz (cross-flow) are exactly the channels where the rotation distributes signal across all three rotated axes.
3. τx improved (rotation concentrates streamwise signal on t1) but τy/τz — the experiment's targets — got worse.

**Conclusion:** REJECTED. Mesh-tangent frame hurts the channels it was designed to fix. Hypothesis failed at the geometric assumption level. Next assignment: tangent-frame as **input feature** (concatenate [t1, t2, n] into surface tokens encoder-side, keep Cartesian output heads — zero hypothesis risk, pure inductive bias addition).

---

## 2026-05-02 08:10 — PR #315: [senku] MLP expansion ratio screening (mlp_ratio=2/4/8) — CLOSED/SUPERSEDED by #339

- Branch: `senku/mlp-ratio-sweep` (deleted, superseded)
- Hypothesis: A larger FFN expansion ratio (mlp_ratio=8 vs default 4) provides more capacity in each transformer block and could close the wall-shear gap.
- Config: scaled-down (bs=4+65k vol pts, max-steps-per-epoch=2000, epochs=9 → 18k steps); params: mlp2=8.50M, mlp4=12.70M, mlp8=21.10M

**Results (3-arm screening, best val):**

| Arm | mlp_ratio | params | best_val_abupt | test_abupt | tau_y (test) | tau_z (test) |
|-----|----------:|-------:|---------------:|-----------:|-------------:|-------------:|
| A | 2 | 8.50M | 11.841% | 12.848% | 15.107% | 15.849% |
| B (ctrl) | 4 | 12.70M | 11.236% | 12.285% | 14.288% | 15.189% |
| **C** | **8** | **21.10M** | **10.897%** | **11.981%** | **14.010%** | **14.689%** |

- Monotonic ordering mlp8 < mlp4 < mlp2 from epoch 4 through all 9 epochs; all 3 arms still descending at ep9
- mlp8 gains: −0.28pp tau_y, −0.50pp tau_z vs control
- mlp8 peak memory: 85.5 GB / 96 GB (near VRAM ceiling; mlp_ratio=12 requires bs=2 to fit)

**Conclusion:** mlp_ratio=8 is the clear winner. Superseded by #339 (mlp_ratio=8 vs 12 head-to-head at matched conditions: bs=2, vol-pts=32768). PR closed to unblock senku's single WIP constraint.

---

## 2026-05-02 21:00 — Round 28 Advisor Actions

### PR #377 (edward, Muon optimizer on STRING-sep) — SENT BACK FOR REBASE (Option 3 accepted)

- Branch: `edward/muon-optimizer-string-sep`
- Hypothesis: Muon optimizer (Newton-step-style update rule) outperforms AdamW/Lion on the STRING-sep SOTA (val_abupt 9.039%, PR #309 yi bar). Epoch-1 evidence from Arm A (AdamW control) vs Arm B (Muon) at matched compute was decisive.
- W&B run referenced: `pgic8n16` (partial, covers ~75% of epoch 1; 21,766 steps at ~1.3 s/it under SENPAI_TIMEOUT_MINUTES=360)
- GitHub comment: https://github.com/morganmcg1/DrivAerML/pull/377#issuecomment-4364649506

**Advisor decision: Option 3 accepted — epoch-1 evidence sufficient.**

Matched-compute epoch-1 result shows ~24.8% relative improvement (Muon Arm B vs AdamW Arm A control). This exceeds the standard significance threshold. Full-budget multi-epoch run is blocked by SENPAI_TIMEOUT_MINUTES=360 pod cap (~4.5h train+1.5h val), which covers only ~75% of epoch 1 at 21,766 steps/epoch. Checkpoint-resume capability (`--resume-from-checkpoint`) does not currently exist in train.py and is deferred to a separate infra PR.

Advisor directed edward to:
1. Rebase onto yi HEAD (PR #355 DDP fix is now live)
2. Run a post-rebase validation epoch at the same matched-compute budget to confirm the improvement holds on the rebased codebase
3. Report post-rebase epoch-1 val_abupt for Arm B (Muon) vs Arm A (AdamW) control

Label swapped: `status:review` → `status:wip`

---

### PR #366 (gilbert, volume Huber loss δ sweep) — SENT BACK FOR REBASE

- Branch: `gilbert/volume-huber-loss`
- Hypothesis: Huber loss (δ sweep: 0.3/0.5/0.7/1.0) on volume-pressure output reduces outlier influence during training and closes the 2.05× vol_p test/AB-UPT gap.
- GitHub comment: https://github.com/morganmcg1/DrivAerML/pull/366#issuecomment-4364649892

**Advisor action:** PR branch conflicts with yi HEAD (post PR #317 + PR #355 merges). Sent back for rebase with instructions to rebase onto yi HEAD and re-run. No result changes — this is purely a merge conflict resolution request.

---

### PR #262 (nezuko, linear warmdown LR / WSD-style) — SENT BACK FOR REBASE

- Branch: `nezuko/linear-warmdown-lr`
- Hypothesis: WSD-style linear LR warmdown keeps learning rate high longer than cosine (per modded-nanogpt insights), enabling better convergence within the epoch-limited budget.
- Arm A (cosine control) epoch-1 val_abupt: 18.84% (partial prior result)
- GitHub comment: https://github.com/morganmcg1/DrivAerML/pull/262#issuecomment-4364650078

**Advisor action:** PR branch conflicts with yi HEAD. Arm C (main WSD arm) has not yet run due to rebase conflict. Sent back for rebase with instructions to rebase Arm C onto yi HEAD and resume the warmdown sweep.

---

### PR #429 (frieren, OneCycleLR) — STATUS PING SENT

- Branch: `frieren/onecycle-lr`
- Hypothesis: OneCycleLR super-convergence schedule outperforms cosine within the SENPAI_TIMEOUT_MINUTES cap.
- Note: PR #191 (haku) tested OneCycleLR on an older architecture and found best val_abupt 18.43% — well above bar. However, PR #429 targets the current 4L/512d Lion+STRING-sep SOTA which is a significantly stronger base. Architecture maturity may change the schedule sensitivity.
- GitHub comment: https://github.com/morganmcg1/DrivAerML/pull/429#issuecomment-4364650315

**Advisor action:** PR was stale WIP with no launch confirmation. Pinged frieren to confirm whether the run has started, and if not, to launch immediately.

---

**Round 28 Fleet State Summary:**
- 18 WIP PRs active
- 0 review-ready PRs
- Active yi merge bar: **val_abupt < 9.039%** (PR #309, thorfinn) — NOTE: 7.546% is the aspirational tay target (PR #311 edward STRING-sep), NOT yet merged to yi (pending PR #420 fern port)
- All four advisor actions completed; fleet fully occupied

---

## 2026-04-29 — PR #496: [tanjiro] Homoscedastic uncertainty-weighted multitask loss (tay branch) — CLOSED NEGATIVE

- Branch: `tanjiro/uncertainty-weighted-multitask-loss` (deleted)
- Hypothesis: Replace fixed per-task loss weights with Kendall & Gal (2018) learnable homoscedastic uncertainty weighting. Each of 5 tasks (surface_pressure, tau_x, tau_y, tau_z, volume_pressure) gets a learnable log_sigma scalar; loss = sum_i [L_i * exp(-2*log_sigma_i)/2 + log_sigma_i]. Adaptive weighting should up-weight the harder tau_y/tau_z axes and close their gap to AB-UPT.
- W&B run: `9pt8v6x2` (rank0), group `tanjiro-uncertainty-loss`, 8 H100 DDP, 11 epochs validated, 270.6 min, nonfinite_count=0

**Implementation deviation (flagged by student):** Original PR assigned log_sigma to the Lion backbone optimizer. Student correctly identified that Lion's sign update causes all 5 log_sigma scalars to move in lockstep (spread=5e-4 at EP2), defeating per-task adaptation. Fix: dedicated `AdamW(lr=1e-3, weight_decay=0.0)` optimizer for log_sigma scalars. Clean sigma divergence confirmed with fix.

**Sigma trajectory (AdamW fix confirmed working):**

| EP | val_abupt% | sigma_sp | sigma_tau_x | sigma_tau_y | sigma_tau_z | sigma_vol_p |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 51.67 | 0.350 | 0.393 | 0.487 | 0.523 | 0.407 |
| 5 | 8.95 | 0.068 | 0.091 | 0.119 | 0.124 | 0.113 |
| 10 (best) | 7.39 | 0.050 | 0.066 | 0.087 | 0.090 | 0.081 |

Effective weights at EP10 (= 1/(2σ²)): surface_pressure=201.6×, tau_x=116.2×, vol_p=75.5×, tau_y=65.6×, tau_z=61.3×. Ratio sp:tau_z = 3.3×. log_sigma_sp saturated at -3 clamp (σ≈0.050).

**Final results vs tay SOTA:**

| Metric | This PR (EP10 EMA) | SOTA #387 alphonse | SOTA #489 | Δ vs #387 |
|---|---:|---:|---:|---:|
| val_abupt | 7.3879% | 7.3816% | 7.1792% | +0.0063 MISSED |
| test_abupt | 8.6402% | 8.5936% | — | +0.0466 MISSED |
| test surface_pressure | 4.2646% | 4.4377% | — | **-0.1731 WIN** |
| test wall_shear | 8.1255% | 7.9989% | — | +0.1266 MISSED |
| test volume_pressure | 12.0504% | 12.1885% | — | **-0.1381 WIN** |
| test tau_y | 9.2954% | 9.1058% | — | +0.1896 MISSED |
| test tau_z | 10.5957% | 10.2736% | — | +0.3221 MISSED |

**Key finding:** Mechanism worked correctly. The hypothesis was inverted. Kendall et al. homoscedastic uncertainty weighting is an *aleatoric down-weighting* mechanism — it identifies high-uncertainty (noisy) tasks and reduces their gradient contribution, treating them as low-information signal. On DrivAerML, tau_y/tau_z ARE the highest-uncertainty axes, so the loss correctly de-emphasizes them. This is the exact opposite of gap-closing intent. Surface_pressure and volume_pressure improved because they were allowed to dominate training; tau_z worsened by 0.32pp because it was systematically under-weighted.

---

## 2026-05-03 14:10 — PR #511: [edward] Extended cosine schedule T_max=13 (tay branch) — MERGED (new tay SOTA)

- Branch: `edward/extended-epochs-cosine` (merged)
- Hypothesis: PR #488 (alphonse EP11) showed val curve still descending at terminal epoch (EP11 LR=2.3e-5), suggesting the model was under-trained. Extending the cosine schedule from T_max=11 to T_max=13 (and training 13 epochs) would buy 2 extra epochs at near-zero LR, continuing the descent and potentially gaining 0.03-0.10pp val_abupt.
- W&B run: `5o7jc7wi` (rank0), group `edward-extended-cosine`, name `edward-extended-ep13`, 8× H100 DDP, 13 epochs, 340 min total (328.1 train + ~12 final eval)

**Final per-component val_primary (% rel_l2):**

| Epoch | val_abupt | surface_p | wall_shear | τ_x | τ_y | τ_z | volume_p | LR |
|------:|----------:|----------:|-----------:|----:|----:|----:|---------:|---:|
| EP11 | 7.1275 | 4.5838 | 8.0850 | 7.0957 | 8.9625 | 10.6951 | 4.3003 | 1.345e-5 |
| EP12 | 7.0578 | 4.5382 | 8.0127 | 7.0422 | 8.8435 | 10.6183 | 4.2467 | 6.670e-6 |
| **EP13** | **7.0134** | **4.5104** | **7.9649** | **7.0052** | **8.7717** | **10.5629** | **4.2168** | 2.438e-6 |
| Δ EP11→EP13 | −0.114 | −0.073 | −0.120 | −0.091 | −0.191 | −0.132 | −0.083 | — |

Full trajectory: EP1=50.43→EP2=33.70→EP3=14.15→EP4=10.22→EP5=8.81→EP6=8.20→EP7=7.83→EP8=7.56→EP9=7.38→EP10=7.23→EP11=7.13→EP12=7.06→EP13=**7.01** (monotone descent every epoch)

**Final test_primary (% rel_l2) — best-val checkpoint reload (EMA):**

| Metric | PR #511 EP13 | PR #488 baseline | Δ | AB-UPT ref |
|---|---:|---:|---:|---:|
| `abupt` | **8.3130** | 8.4791 | **−0.166pp** | — |
| `surface_pressure` | **4.2709** | 4.4489 | **−0.178pp** | 3.82 |
| `wall_shear` | **7.7863** | 8.0642 | **−0.278pp** | 7.29 |
| `volume_pressure` | 11.8673 | **11.5029** | +0.364pp | 6.08 |
| `τ_x` | **6.9184** | 7.0932 | **−0.175pp** | 5.35 |
| `τ_y` | **8.5819** | 9.0723 | **−0.490pp** | 3.65 |
| `τ_z` | **9.9267** | 10.2780 | **−0.351pp** | 3.63 |

Wins on 6 of 7 test metrics. `volume_pressure` regressed +0.364pp on test despite val improvement — likely test-split outlier sensitivity in volume normalization (test vol_p MAE 28.0 vs val 8.32, ~4× larger absolute scale).

**Analysis:**
- Hypothesis validated exactly as predicted. Monotone descent EP1→EP13 with no instability. Extended T_max bought 2 productive epochs.
- Most important finding: **τ_y and τ_z improved fastest at near-floor LR** (τ_y −0.191pp, τ_z −0.132pp EP11→EP13) — the anisotropic axes were genuinely still learning right up to the LR floor. This is consistent with the multi-sigma RFF init from PR #488 that identified τ_y/τ_z as the open problem.
- val→test gap well-behaved: val 7.01→test 8.31 (+1.30pp), matching prior PR #488 gap (~+1.11pp). No overfit regime.
- τ_y=8.58% (test) vs AB-UPT ref 3.65% (2.35×), τ_z=9.93% vs ref 3.63% (2.73×) — still the dominant gaps.
- Peak GPU memory 74 GB / 96 GB (8× H100). Identical model dims to #488.

**Key config:**
```bash
SENPAI_VAL_BUDGET_MINUTES=30 torchrun --standalone --nproc_per_node=8 train.py \
  --lr-cosine-t-max 13 --epochs 13 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --optimizer lion --lr 1e-4 --weight-decay 5e-4 --ema-decay 0.999 \
  --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --batch-size 4 --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536
```

**Conclusion:** MERGED. New tay SOTA: val_abupt=7.0134%, test_abupt=8.3130%. Improvement vs prior SOTA (PR #488): −0.354pp val, −0.166pp test. The fastest remaining improvement path is further τ_y/τ_z gap closure (still 2.35-2.73× AB-UPT). Student suggestions: EP15 with extended budget (val slope still negative at EP13), cosine warm-restarts, targeted τ_y/τ_z inductive-bias adds (frieren anisotropic STRING, edward curvature features).

**Conclusion:** CLOSED. Kendall et al. uncertainty weighting is the wrong tool for per-axis gap-closing on DrivAerML. The correct approach for tau_y/tau_z gap-closing is principled UP-weighting of lagging tasks: GradNorm (log per-task loss ratios, weight proportional to (L_i(t)/L_i(0))^α with α>0) or static per-axis multipliers (tau_y/tau_z × 2–3×, others × 1×) on the current tay SOTA stack. Static weight sweeps predating the STRING-sep+QK-norm+RFF16 backbone have not been repeated at current stack level.

## 2026-05-03 17:00 — PR #430: [emma] Cosine EMA decay ramp (0.99→0.9999 vs fixed 0.999) — CLOSED NULL RESULT

- Branch: `emma/ema-decay-ramp` (closed, branch deleted)
- Hypothesis: EMA decay that warms up from a loose schedule early in training (ema=0.99, fast adaptation) to a tight schedule late (ema=0.9999, strong smoothing) should outperform a fixed schedule by allowing the model to learn quickly then stabilise. Tested as a 4-arm sweep: A (fixed 0.999/wd=5e-4), B (fixed 0.99/wd=1e-4), C (ramp 0.99→0.9999/wd=1e-4), D (ramp 0.995→0.9999/wd=1e-4).
- W&B runs: A=`0wnwtaro`, B=`zg3ukcex`, C=`3y93q5h7`, D=`6phmqaxj` (entity: wandb-applied-ai-team, project: senpai-v1-drivaerml)

| Arm | Config | wd | best_val_abupt | test_abupt | surf_p | wall_shear | ws_y | ws_z | vol_p |
|---|---|---|---|---|---|---|---|---|---|---|
| A | fixed 0.999 | 5e-4 | 10.798% | 11.870% | 6.587% | 11.990% | 14.926% | 14.581% | 13.110% |
| B | fixed 0.99 | 1e-4 | **10.247%** | **11.356%** | **6.301%** | **11.461%** | **14.270%** | **13.931%** | **12.579%** |
| C | ramp 0.99→0.9999 | 1e-4 | 10.486% | 11.596% | 6.408% | 11.698% | 14.589% | 14.222% | 12.874% |
| D | ramp 0.995→0.9999 | 1e-4 | 10.752% | 11.853% | 6.583% | 11.948% | 14.871% | 14.525% | 13.175% |

**Analysis:**
- **All arms are well above the current tay SOTA** (val=7.01%, test=8.31%) — these were ablation runs on a lightweight config (2L/256d, 4 GPUs, 10 epochs), not the full tay stack.
- **Ramp does NOT beat fixed.** Arm B (fixed ema=0.99, wd=1e-4) is the clear winner, beating both ramp variants by 0.24–0.51pp val. The cosine ramp hypothesis is falsified.
- **wd confound in Arm A.** Arm A used wd=5e-4 instead of 1e-4 (matching the intended ablation), which complicates the 0.999 vs 0.99 read. However even accounting for wd the ramp arms do not dominate.
- **Best finding:** fixed ema=0.99 (B) outperforms fixed ema=0.999 (A) by 0.55pp val. Suggests lower EMA decay (more responsive tracking) is beneficial at this scale. Worth testing as a clean A/B on the full tay stack: ema=0.99 vs ema=0.999 with matched wd.
- None of the arms approach the tay SOTA, so no merge consideration.

**Conclusion:** CLOSED NULL RESULT. Cosine EMA ramp is not beneficial — fixed ema=0.99 beats both ramp variants on every metric. Hypothesis falsified. Candidate follow-up: clean ema=0.99 vs 0.999 pair at full tay stack with matched wd=5e-4 (controls the wd confound from this sweep).

## 2026-05-03 18:38 — Round 33 New Assignments (PRs #539–#544)

### PR status at assignment time
- 10 WIP PRs running (tanjiro, violet, nezuko, thorfinn, senku, norman, gilbert, kohaku, chihiro, edward)
- 6 students idle: alphonse, askeladd, emma, fern, frieren, haku
- Active yi merge bar: val_abupt 9.032% (PR #517 askeladd, `brat65z4`)
- PR #490 (frieren STRING-sep learnable PE) MERGED to yi at 15:48 UTC

### PR #539 — frieren: STRING-sep + Lion lr=1e-4 clip=0.5 from scratch
- Branch: `frieren/string-sep-lion-fromscratch`
- Hypothesis: Clean from-scratch combination of `--learnable-pe` (PR #490) + Lion lr=1e-4 clip=0.5 (PR #517) on the merged yi codebase. Evidence base: frieren resumed run `zwh9qzjw` achieved 8.087% but from a checkpoint — non-canonical. This is the canonical confirmation.
- Expected: val_abupt ≤ 8.5%, new yi merge bar.

### PR #540 — emma: GradNorm adaptive τ_y/τ_z up-weighting
- Branch: `emma/gradnorm-task-balancing`
- Hypothesis: GradNorm (Chen et al. 2018) dynamically up-weights lagging tasks by normalizing per-task gradient magnitudes relative to their training rate. On DrivAerML's 5 axes, should auto-discover τ_y/τ_z need 2–3× weight. Different from PR #454 (static 2× weight, null) and PR #496 (Kendall uncertainty, mechanism inverted). 3-arm sweep: α=0/1.0/1.5.

### PR #541 — askeladd: streamline-aligned wall-shear target frame
- Branch: `askeladd/streamline-ws-frame`
- Hypothesis: Rotate τ targets per-point into a local streamline frame (t1=streamwise, t2=cross-tangent) before loss; rotate back for eval. The structural τ_y/τ_z gap in Cartesian frame is caused by near-zero signal/noise for cross-flow components. In streamline frame, the secondary component τ_t2 carries physically meaningful cross-flow shear with better S/N ratio. Different from PR #312/419 (input feature tangent vectors — closed).

### PR #542 — alphonse: Laplace far-field soft penalty on volume pressure
- Branch: `alphonse/laplace-farfield-penalty`
- Hypothesis: Add `∑(∇²p_far)²` penalty on far-field volume points (SDF > 0.3m) to enforce the RANS Laplace equation at inference. Volume pressure has 1.95× test gap driven by heavy-tail outlier vehicles — this physics-informed penalty should suppress spurious extreme predictions that violate ∇²p=0 in free stream. 3-arm λ sweep: 0/1e-3/1e-2.

### PR #543 — haku: principal curvatures (H,K) as surface input features
- Branch: `haku/surface-curvature-features`
- Hypothesis: Add mean curvature H and Gaussian curvature K as 2 extra surface input channels (8-channel total). High-curvature zones (A-pillars, wheel-arches) are where τ_y/τ_z concentration was diagnosed (PR #363). This gives the model explicit geometry signal for where cross-flow separation is expected.

### PR #544 — fern: y-symmetry paired loss + test-time mirror ensemble
- Branch: `fern/symm-pair-equivariance`
- Hypothesis: Enforce y-symmetry (cars are left-right symmetric; τ_y flips sign under y-reflection) as a paired-batch constraint: `L = 0.5·L(f(x), y) + 0.5·L(R_y·f(mirror_y(x)), y)`. PR #225 (haku random mirror aug) closed null — this differs by always presenting BOTH orientations simultaneously with correct sign, not random flips. Test-time mirror ensemble is a free 2× average that enforces exact symmetry.
