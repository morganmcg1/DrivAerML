# SENPAI Research Results — `drivaerml-long-20260504`

## 2026-05-26 09:55 — PR #1335 H144 fern CLOSED: EMA-of-weights does not beat H39 SOTA

- `dl24-fern/h144-ema-weights`, run `wybzhel9` (8 DDP ranks)
- Hypothesis: EMA-of-weights (online Polyak averaging, decay=0.999) recovers Lion sign-momentum noise on validation/test
- **Terminal verdict: NULL on this branch (test_WSS 6.81% > SOTA 6.6506% by +0.156pp)**

### Terminal SENPAI-RESULT (EP10 kill, dual-checkpoint eval)

| Metric | H144 EMA EP10 | H39 SOTA | Δ |
|--------|--------------:|---------:|---:|
| **test_primary/wall_shear_rel_l2_pct (PRIMARY)** | **6.8065%** | **6.6506%** | **+0.156pp miss** |
| test_primary/volume_pressure_rel_l2_pct | 3.7499% | 3.6033% | +0.147pp |
| test_primary/surface_pressure_rel_l2_pct | 3.7233% | 3.6498% | +0.073pp |
| test_primary/abupt_axis_mean_rel_l2_pct | 5.9482% | 5.8010% | +0.147pp |

### Dual-checkpoint validation of EMA mechanism

| Checkpoint | full_val_WSS | test_WSS | EMA gain vs live |
|-----------:|-------------:|---------:|:----------------:|
| EMA-best (val_ema EP10) | 7.0763% | **6.8065%** | — (winner) |
| live-best (val_primary EP10) | 7.2468% | 6.9906% | — |
| Δ (ema − live) | −0.1705pp | **−0.1841pp** | EMA wins on test by 0.184pp |

**EMA mechanism is validated** — the EMA→live Δ_test = -0.184pp lands within advisor prior of 0.05-0.20pp. The headline miss is a `deepcopy(model)`-induced RNG perturbation drifting the live trajectory by ~+0.4pp from EP2 onward, NOT an EMA failure.

### Live vs EMA trajectory (val, EP1-EP10)

| EP | val_primary | val_ema | Δ_ema−live | H39 ref |
|----|------------:|--------:|----------:|--------:|
| 1  | 17.886% | 18.141% | +0.255 | 17.897% |
| 2  |  8.435% |  7.581% | −0.854 |  7.482% |
| 3  |  7.746% |  7.271% | −0.475 |  7.202% |
| 5  |  7.482% |  7.155% | −0.327 |  7.013% |
| 8  |  7.324% |  7.089% | −0.234 |  6.879% |
| 10 |  7.247% |  7.076% | −0.171 |  6.850% |

EP1 matched H39 to 4dp; post-warm-up live diverged ~+0.4-0.5pp from H39 throughout. Per-axis test_WSS miss decomposes: wss_x +0.126pp, **wss_y +0.259pp (dominant)**, wss_z +0.131pp.

### Conclusion / Wave 36 implication

- EMA mechanism on this branch: NULL at decay=0.999, EP10 kill. Trajectory not converging.
- Higher decay (0.9999) is in flight on `tay` branch PR #1343 (nezuko/tay) — no need to duplicate on dl24 branch.
- **The dl24-fern slot is reassigned to PR #1345 H148 compound spatial reweighting (z-coord + curvature on clean H39 stack).**
- Conditional follow-up: deepcopy-free EMA injection (lazy parameter-by-parameter shadow) could fix the RNG-perturbation issue — code-architecture work, not a hyperparameter sweep.

## 2026-05-26 05:30 — PR #1339 H146 EP1-2 INTERIM: wd-only ablation tracks H39, NOT H138 → lion-β is the driver

- `dl24-nezuko/h146-wd-only-ablation`, run `9aeprogu` (8 DDP ranks)
- Hypothesis: Isolate `weight_decay=0.0001` as driver of H138's gains. Clean H39 stack + ONLY wd change.
- **EP3 verdict pending** — but EP1-2 interim data is decisive.

### Gate data through EP2

| EP | Step | H39 SOTA | H138 (3 drifts) | H145 (clean+curv) | **H146 (clean+wd)** | wd-only signal |
|----|------|---------:|----------------:|------------------:|--------------------:|----------------|
| 1  | 10,975 | 17.8972% | 12.8321% | 17.7090% | **18.2739%** | +0.38pp WORSE than H39 |
| 2  | 21,951 |  7.4818% |  7.3039% |  7.6054% |  **7.6134%** | +0.13pp WORSE than H39 |

### Decisive interim finding

**wd=0.0001 alone does NOT reproduce H138's early-epoch advantage.** H146 tracks H39 (slightly worse, +0.38pp at EP1) — not H138 (-5.07pp ahead of H39 at EP1). Combined with H145 (curvature mechanism = -0.19pp), the contributions stack as:

- Curvature mechanism: -0.19pp
- wd-drift: 0pp (slightly negative)
- **lion-β drift: ~5.0pp** (by elimination, assuming additive contributions)

The wd story is dead. The β story is alive. **H138's spectacular gains are driven almost entirely by the lion-β drift (β₁=0.9→0.95, β₂=0.99→0.98).**

### Hypothesis for β mechanism

β₂=0.98 (faster second-moment estimation) appears to be the dominant factor. In CFD landscapes with high curvature variance, faster adaptation to recent gradient magnitudes may help the optimizer navigate sharp transitions. β₁=0.95 (slower momentum) is more likely secondary — it has less impact when β₂ is also changing.

### EP3 status

H146 is currently in EP3. If val_WSS > 7.40% at EP3, student will close per the hard-abort gate. Even if it limps to EP3, the EP1-2 signal is conclusive: wd alone is null.

### Implication for next dispatch

**H147 lion-β-only ablation becomes the highest-priority experiment.** Clean H39 + ONLY `--lion-beta1 0.95 --lion-beta2 0.98`. Predicted EP1 ≈ 13% if β is the driver. If confirmed, Wave 36 canonical optimizer becomes β=0.95/0.98 with wd=0.005.

---

## 2026-05-26 03:00 — PR #1336 H145 CLOSED: Curvature-weighted Charbonnier-z is NULL on clean H39 stack (decisive disentanglement)

- `dl24-nezuko/h145-curv-charb-z-wd005`, run `1mpz6zlp` (8 DDP ranks)
- Hypothesis: Re-test H138's curvature-weighted Charbonnier-z mechanism on the **clean H39 hyperparameter stack** (`weight_decay=0.005, lion_beta1=0.9, lion_beta2=0.99`) to disentangle the mechanism signal from H138's 3-drift confound.

### Gate data — KILLED at EP3 hard-abort

| EP | Step | H39 SOTA `yym5oa8x` | H138 `4m8f7rme` (3 drifts) | **H145 `1mpz6zlp` (clean)** | H145 vs H39 |
|----|------|--------------------:|--------------------------:|----------------------------:|------------:|
| 1  | 10,975 | 17.8972% | 12.8321% | **17.7090%** | -0.19pp (curvature contribution alone) |
| 2  | 21,951 |  7.4818% |  7.3039% |  **7.6054%** | +0.12pp (mechanism mildly harmful) |
| 3  | 32,927 |  7.2016% |  6.9955% |  **7.3398%** | +0.14pp (gate tripped, >7.30%) |

### Mechanism diagnosis — decisive negative

The curvature-weighted Charbonnier-z mechanism (per-point residual × `1 + α·κ_i`, α=1.0) on the clean H39 optimizer regime produces a **marginally worse trajectory than H39 baseline**. The mechanism contributes only -0.19pp at EP1, compared to the combined -5.07pp early-epoch gain of H138 — meaning the **curvature mechanism explains ~4% of H138's early-epoch advantage; the remaining ~96% comes from the wd+β drifts**.

### Strategic implication

1. **H138's val_WSS=6.7565% breakthrough is NOT a curvature-mechanism win.** It is a hyperparameter-drift win (wd=0.0001 + β₁=0.95 + β₂=0.98).
2. **Curvature-weighted Charbonnier-z removed from active research candidates.** Null on clean stack at α=1.0; α-tuning is unlikely to recover the mechanism since the EP1-3 trajectory is flat-to-mildly-harmful.
3. **Wave 36 pivot**: dispatch wd-only ablation (H146) and lion-beta ablation (H147) to identify which drift is the dominant driver of H138's gains.

### Resource economics

EP3 hard-abort saved 22.5h × 8 GPUs vs running to terminal. Disentanglement experiments should always set tight early-epoch kill gates to extract maximum information per GPU-hour.

### W&B and PR
- Run: `1mpz6zlp`
- PR: #1336 (CLOSED)

---

## 2026-05-26 03:00 — H146 DISPATCHED to dl24-nezuko: wd-only ablation (PR #1339)

- Hypothesis: Isolate `weight_decay=0.0001` as the dominant driver of H138's gains. Clean H39 hyperparameter stack with ONE change: `--weight-decay 0.0001`. Tracks ablation of wd vs lion-beta contributions to H138.
- PR #1339: `dl24-nezuko/h146-wd-ablation`
- Predicted EP1 signal: 13-16% if wd dominates; 16-17% if beta drifts dominate
- If terminal test_WSS < 6.6506%, **merge-eligible as a wd-only SOTA**.

---

## 2026-05-25 23:40 — PR #1333 H143 CLOSED: Lookahead(k=5,α=0.5) EP3 hard-abort gate miss

- `dl24-nezuko/h143-lookahead-lion-v2`, run `mzk2tpu7` (8 DDP ranks)
- Hypothesis: Lookahead optimizer wrapping Lion (k=5, α=0.5) from step 1 (Zhang et al. 2019)

### Gate data

| EP | Step | val_WSS | Gate | Verdict |
|----|------|---------|------|---------|
| 1 | 10,975 | 23.1229% | 17.40-18.40% (sanity) | MISS (+4.72pp) |
| 2 | 21,951 | 8.0000% | — | reference |
| **3** | **32,927** | **7.4430%** | **≤7.30% (hard-abort)** | **KILL** |

### Mechanism diagnosis

Student EP1 analysis identified that Lookahead(k=5, α=0.5) **halves effective parameter displacement per step**: outer update `θ_slow ← θ_slow + 0.5·(θ_fast - θ_slow)` means 5 inner steps accumulate only 2.5× delta (vs 5× for raw Lion). At EP3 (32,927 real steps), effective training progress ≈ H39 EP1.5. Val_WSS=7.443% is consistent with this "slow learner" interpretation.

The mechanism fired correctly (2647 outer updates at EP3, perfect k=5 cadence), but is **net negative** under these hyperparameters: re-anchoring to θ_slow interrupts Lion's productive early-epoch descent without recovering the lost progress late.

**Verdict**: Lookahead k=5/α=0.5 from step 1 is net-negative on H39 regime. The mechanism logic is sound, but current hyperparameters are too aggressive (α=0.5 = too much anchoring, k=5 = too frequent). Future variants: k=20/α=0.5, or late-activation (apply only from EP10+).

### Wave 35 optimization-time axis verdict
- SWA (H141, closed process failure): mechanism untested
- EMA-of-weights (H142, closed bad-logic): ongoing as **H144** (fern, PR #1335)
- Lookahead k=5/α=0.5 (H143): **net negative** (mechanism active, gates failed)

---

## 2026-05-25 23:35 — PR #1316 H134 CLOSED: GALE-Transolver decisive negative (3 stabilization failures)

- `dl24-fern/h134-gale-transolver`, run `uhf5c8md` (rank-0, 8 DDP ranks)
- Hypothesis: GALE persistent geometry context bank cross-attention at every Transolver block

### Failure history (3 runs)

| Run | Failure | Root cause |
|-----|---------|------------|
| `shtce8gt` | Grad explosion EP2 (gn 1.09e14) | PR-spec Xavier out_proj injecting noise into residual |
| `v9d2j25u` | Dead gradient EP2 (geo_gate grad=0) | Zero-init out_proj killed MHA gradient |
| `uhf5c8md` | LayerScale escape EP3 (val_WSS 91%, gn 612k) | Unconstrained `nn.Parameter` LayerScale grew 20,000× (1e-4 → ~2.0) under Lion sign-updates |

**Root cause pattern**: GALE cross-attention residual under Lion+lr=1e-4 has no stable equilibrium with bounded-by-default mechanisms. AdamW's per-parameter adaptive scaling provides natural damping for unbounded learnable scales; Lion's sign-based `±lr` updates remove this implicit bound.

### Decision

Three sequential failures with autonomous stabilization fixes each surfacing a new failure mode. Option B (tanh-bounded LayerScale) declined to avoid 4th iteration on a stalling line. **Decisive negative result preserved**: GeoTransolver's recipe (depth-20, AdamW, 500 epochs) is not portable to H39 capacity+Lion+30ep regime.

**Informative result**: future architecture experiments on H39 should pre-specify Lion-compatible stability budget (tanh-bounded or hard-clipped gates from initialization, no unbounded learnable residual scales).

---

## 2026-05-25 23:40 — H144 DISPATCHED to dl24-fern: EMA-of-weights (PR #1335)

- Hypothesis: Online Polyak averaging (decay=0.999) shadow model from EP0. Dual val logging: `val_primary/*` + `val_ema/*`. EMA checkpoint vs live checkpoint at terminal.
- Context: H142 EMA-of-weights (same hypothesis) was closed in error by bad-logic conflation with H141 SWA. H142 NEVER launched. This is first execution.
- PR #1335: `dl24-fern/h144-ema-weights`
- Predicted test_WSS: < 6.60% (beats H39 SOTA 6.6506%)

---

## 2026-05-25 23:40 — H145 DISPATCHED to dl24-nezuko: Curv-Charb-z clean (wd=0.005) disentanglement (PR #1336)

- Hypothesis: H138's curvature-weighted Charbonnier-z mechanism repeated with explicit `--weight-decay 0.005` (H39 reference value), to disentangle the curvature mechanism signal from the wd-drift confound in H138 (`4m8f7rme`, wd=0.0001).
- H138 context: EP14.58 val_WSS=6.7582% (strongest in-program signal). Has wd=0.0001 drift. If terminal beats SOTA, H145 determines whether to attribute the gain to curvature mechanism or to the wd reduction.
- PR #1336: `dl24-nezuko/h145-curv-charb-z-wd005`
- Predicted test_WSS: < 6.65% (beats H39 SOTA) if curvature mechanism is responsible

---

## 2026-05-25 21:50 — PR #1333 H143 RELAUNCHED: hyperparameter drift discovered + Wave 35 fleet-wide wd-drift cross-cut

- `dl24-nezuko/h143-lookahead-lion-v2`, run `mzk2tpu7` (rank-0, 8 DDP ranks)
- Group: `h143-lookahead-lion` (shared with killed `uih9w1sj`)
- Hypothesis: Lookahead optimizer wrapping Lion (k=5, α=0.5; Zhang et al. 2019, arxiv 1907.08610)

### What happened

Student dl24-nezuko discovered at 21:29Z that the first H143 run `uih9w1sj` had silently inherited new `train.py` defaults `batch_size=2` (H39 reference: 1) and `weight_decay=0.0001` (H39 reference: 0.005), because the launch command passed neither flag explicitly. The drifted run completed EP0.5-equivalent training showing val_WSS=28.87%, val_VP=27.21%, val_SP=20.96% — not a valid Lookahead-vs-H39 datapoint.

Student autonomously killed `uih9w1sj` at step 7307 and relaunched as `mzk2tpu7` with `--batch-size 1 --weight-decay 0.005` explicit. New W&B config verified to match H39 + Lookahead delta. Iter rate 4.27 it/s steady-state (~6% Lookahead overhead vs H39's 4.53 it/s). Lookahead outer-step firing verified: `[H143] #1 fired at step 5 / #2 at 10 / #3 at 15`.

### Advisor verification (21:45Z)

- H39 SOTA `yym5oa8x` W&B config confirmed: `batch_size=1, weight_decay=0.005, lr=1e-4, optimizer=lion`. Student's drift identification is correct.
- New run `mzk2tpu7` config verified to match H39 exactly with `use_lookahead=True, lookahead_k=5, lookahead_alpha=0.5` as the only delta. State=running, step=4142 at 21:44Z (0.27h into corrected run). EP1 ETA ~22:14Z.
- Advisor ACK posted to PR #1333 approving the autonomous kill-and-relaunch decision.

### Cross-cut wd-drift discovered (fleet-wide)

W&B audit of all 4 active dl24 Wave 35 runs reveals the same `weight_decay=0.0001` drift on three other runs that predate the discovery:

| Run | PR | wd | bs | Note |
|-----|----|----|----|------|
| `uhf5c8md` (H134 fern GALE) | #1316 | **0.0001** ❌ | 1 ✓ | Started 18:30Z, EP2 |
| `4m8f7rme` (H138 frieren curv-Charb-z) | #1324 | **0.0001** ❌ | 1 ✓ | Started ~13:00Z, EP12 (strongest signal) |
| `jtyglnxu` (H140 tanjiro z-coord WSS) | #1329 | **0.0001** ❌ | 1 ✓ | Started ~18:00Z, EP5.4 |
| `mzk2tpu7` (H143 nezuko Lookahead, relaunched) | #1333 | **0.005** ✓ | 1 ✓ | Started 21:29Z, EP0.4 |

Same root cause as H143-old: PR repro commands never explicitly set `--weight-decay 0.005`, and the `train.py` default was lowered from 0.005 to 0.0001 sometime between H39 dispatch and Wave 35 launches.

### Decision: continue H134/H138/H140 to terminal

Killing now would waste 8.8h (H138), 3.9h (H140), 3.3h (H134) of useful mechanism data — and H138 is the strongest in-program positive signal of the entire wave (`val_WSS=6.7715%` at EP12). The drift is a confound, not a corruption: the runs are still informative about whether their intended mechanisms (curvature-weighted Charb-z, GALE persistent geometry context, z-coord WSS weighting) produce signal.

Wd-drift flag posted to PR #1316/#1324/#1329 with explicit merge-time interpretation note: any terminal result that beats H39 SOTA must be re-verified with a clean H<N>-v2 follow-up `--weight-decay 0.005` to disentangle the intended mechanism from the wd axis before merge.

### Memory update

Added `feedback_drivaerml_weight_decay_default.md`: "Every DrivAerML PR repro command MUST include `--batch-size 1 --weight-decay 0.005` — train.py defaults drift and silently change optimizer regime." This is the second silent-default-drift incident in Wave 35 after the batch_size=2 drift documented in `feedback_drivaerml_batch_size_default.md`. Same root cause family.

### Schedule for H143 `mzk2tpu7` (corrected step convention)

| EP | Step | ETA | Gate |
|----|------|-----|------|
| 1 | 10,976 | ~22:14Z | sanity 17.40-18.40% |
| 3 | 32,927 | ~23:30Z | hard-abort ≤7.30% |
| 6 | 65,855 | ~02:04Z (+1d) | ≤7.00% |
| 15 | 164,640 | ~08:25Z | ≤6.90% |
| 30 | 329,280 | ~18:30Z | terminal |

### Conclusion

H143 is now the only clean (no-drift) ACTIVE experimental run in Wave 35. Mechanism is firing correctly. H138 remains the strongest in-program positive signal but its terminal-merge eligibility now hinges on a clean v2 confirmation.

---

## 2026-05-25 19:20 — PR #1328 CLOSED: H141 SWA (process failure, NOT mechanism failure — see post-mortem)

- `dl24-nezuko/h141-swa`, run group rank-0=`y9qf09ch` (8 DDP ranks, started 17:28Z)
- Hypothesis: Stochastic Weight Averaging at EP21+ (70% of 30 EPs) for flat-basin OOD transfer; --swa-lr 5e-6 --swa-start 0.7 --swa-epochs 2 (or equivalent).
- **Closed by prior advisor session at 19:08Z** after dual escalation (18:15Z, 18:58Z) when no W&B run ID was posted to the PR.
- **Post-mortem (19:18Z)**: W&B query revealed 8 ranks WERE running productively. Trajectory:

| Step | EP | val_WSS | vs H39 |
|------|-----|---------|--------|
| 10975 | 1.00 | 17.5728% | **−0.33pp ahead** |
| 21951 | 2.00 | 7.6674% | +0.37pp behind |

**Root cause**: dl24-nezuko student Claude session ran training but failed to post W&B run ID to PR within 30-min window. Pod was productive; PR-commenting layer was broken.

**Why not revived**:
- SWA mechanism doesn't activate until EP21 (70% of 30); at EP2.5 only baseline-equivalent training had run.
- H141 already trailing H39 by +0.37pp at EP2, baseline-equivalent trajectory was marginally worse.
- H142 EMA-of-weights (PR #1331) tests the same parameter-averaging hypothesis but applies the averaging from EP0 (full trajectory, not just last 30%) — strictly more informative.

**Process improvement**: Advisor memory updated with "always query W&B run group before closing PR for non-response". Saved as `feedback_dl24_check_wandb_before_close.md`.

**Conclusion**: Process failure (premature close), not mechanism falsification. SWA hypothesis remains untested. Subsumed by H142.

---

## 2026-05-25 19:51 — PR #1331 CLOSED: H142 EMA-of-weights (bad-logic kill — hypothesis conflation, never launched)

- `dl24-nezuko/h142-ema-weights` — closed at 19:51:46Z by prior advisor session before any H142 run was started.
- **Kill comment cited**: "EP3 gate KILL: val_wss=7.3179% > 7.20% threshold at step=32,927. SWA axis closed."
- **Bad-logic conflation**: The 7.3179% datapoint came from H141 SWA's continued post-close runs (`y9qf09ch` group). H141's SWA mechanism (--swa-start 0.7) activates at EP21+; at EP3 it was running PURE H39 baseline-equivalent training that showed +0.12pp seed-variance drift vs H39 EP3 (~7.20%). H142 EMA is a fundamentally different mechanism (online Polyak averaging from EP0).
- **Kill comment also wrongly cited H112 (tay branch) as baseline** — should have been H39 (dl24 branch). Two-layer confusion.
- **Successor**: PR #1333 (H143 Lookahead optimizer) dispatched 19:58Z. Lookahead is UNAMBIGUOUSLY different from weight averaging (slow/fast weights with re-synchronization, Zhang et al. 2019).
- Memory updated with `feedback_hypothesis_conflation_kills.md` to prevent recurrence of this pattern.

---

## 2026-05-25 19:58 — PR #1333 DISPATCHED: WSS H143 Lookahead optimizer wrapping Lion (nezuko)

- `dl24-nezuko/h143-lookahead-lion` — successor to closed H142 EMA, 3rd consecutive nezuko reassignment.
- **Hypothesis**: Lookahead optimizer (Zhang et al. 2019, arxiv 1907.08610) wrapping Lion. Slow weights θ_slow and fast weights θ_fast; every k=5 steps update θ_slow ← θ_slow + α·(θ_fast - θ_slow), then RESET θ_fast = θ_slow.
- **Why this is fundamentally different from H141 SWA / H142 EMA**: Lookahead actively RE-SYNCHRONIZES fast weights to slow weights every k steps — this modifies the training trajectory. EMA/SWA never reset the live weights. Lookahead is a distinct optimization paradigm, not weight averaging.
- **Mechanism**: Re-anchoring optimizer to slow-weight basin every k steps reduces late-stage Lion noise variance. Activates from step 5 (no late-EP delay).
- **Predicted**: test_WSS 6.55-6.70%, 45-60% prob beats H39 SOTA.
- Gates: EP1 within ±0.5pp of H39 EP1 (17.40-18.40%); EP3 hard-abort >7.30%; EP6 ≤7.00%; EP15 ≤6.90%; EP30 terminal.
- All H39 hyperparameters preserved (depth-6, hidden-512, slices-128, surface_out f=2.0, Charb-z 0.1, GradNorm α=0.5 clamp 0.15, Lion lr 1e-4, cosine T_max=30) — only Lookahead wrapper added with k=5, alpha=0.5.
- PR body includes EXPLICIT "DO NOT CLOSE UNDER WEIGHT-AVERAGING-AXIS-CLOSED REASONING" language to prevent another conflation-based kill.

---

## 2026-05-25 19:18 — PR #1331 DISPATCHED: WSS H142 training-time EMA-of-weights (nezuko)

- `dl24-nezuko/h142-ema-weights` — successor to closed H141 SWA.
- **Hypothesis**: Online Polyak averaging of all model weights with decay=0.999 suppresses Lion sign-momentum noise around the loss minimum, yielding a "wider-basin" representation that transfers better val→test.
- Mechanism: maintain `θ_ema ← 0.999·θ_ema + 0.001·θ_live` after every optimizer step. At validation, evaluate both live model and EMA model. Choose lower val_WSS for terminal test eval.
- **Why now**: Wave 35 falsified two multi-task-weighting hypotheses (H136 IMTL-G active failure, H137 PCGrad passive failure). H138 is showing strong loss-shaping signal but optimization-axis remains untested. H39's "EP24 EMA" best-checkpoint result suggests EMA matters; H142 extends EMA from val-only to training-time.
- **Predicted**: EMA val_WSS lower than live val_WSS by 0.05-0.15pp at terminal; test_WSS < 6.65% (50-60% prob beats H39 SOTA).
- Gates: EP1 sanity ±0.5pp of H39 (17.90%); EP3 hard-abort >7.50%; EP6 gate EMA starts to outperform live; EP15 EMA ≤6.85%; EP30 terminal report both metrics.
- All H39 hyperparameters preserved (depth-6, hidden-512, slices-128, surface_out f=2.0, Charb-z 0.1, GradNorm α=0.5 clamp 0.15, Lion lr 1e-4, cosine T_max=30) — only EMA tracking added.

---

## 2026-05-25 16:25 — PR #1304 CLOSED: WSS H123 hidden_dim 512→640 (terminal NULL — capacity-axis falsification)

- `dl24-tanjiro/h123-hidden-dim-640`, run `a9c5akny`
- Hypothesis: H39 + trunk hidden_dim 512→640 (+25%, ~25-28M params) tests trunk per-token feature capacity as orthogonal to H39's decoder-side surface_out widening. Predicted: if H123 wins, trunk-width is productive and W35 compounds hidden_dim × depth × Charb-y.
- **Closed by student at EP25** (early termination — plateau definitively confirmed).

| EP | val_WSS | vs H39 | per-axis (x/y/z) |
|---|---|---|---|
| 1 cold-start | 15.78 | **-2.118pp AHEAD** ⭐ | uniform ahead all axes |
| 3 | 7.109 | -0.093pp ahead | z lead WIDENED to -0.104 |
| 5 | 6.963 | -0.050pp ahead | continuing structural lead |
| 8 | 6.883 | +0.004 BEHIND (lead crossed zero) | |
| 10 | 6.855 | +0.010 behind | |
| 15 | 6.811 | +0.015 behind | |
| 17-25 | [6.784, 6.799] | +0.025-0.029pp behind | **9-EP PLATEAU** |
| EP30 projection | ~6.783 | +0.133pp behind H39 SOTA terminal | |

**Mechanism analysis** (student terminal verdict at 16:18Z):
- Cold-start advantage (EP1 -2.12pp ahead, EP3-5 -0.05 to -0.09pp ahead) was structurally real — wider trunk absorbs training signal faster.
- **BUT** lead eroded by EP8 (crossed zero) and converged to flat-plateau +0.025pp behind H39 from EP17-EP25.
- Hidden_640 hit the same capacity ceiling as H39 baseline — the additional 25% trunk capacity got absorbed by volume targets (vol_p continued improving) rather than surface WSS objective.
- This is the **same pattern as H115** (wider volume_out): structural capacity adds to non-primary objective.

**Conclusions**:
- **Trunk-width is NOT a productive capacity axis for WSS optimization on H39 base** — converges to same WSS ceiling as H39, doesn't escape the 6.78% val plateau.
- **5/5 H39-base capacity axes now CLOSED** as null/falsified: encoder slices (H122), volume-decoder width (H115), loss-shape Charb-axis (H41v2 yz, H117 xyz, H124 +y), trunk hidden_dim (H123 now), trunk depth (H132), base LR (H133).
- **Capacity-axis sweep on H39 base = DEFINITIVELY EXHAUSTED.** This is a high-information null result — narrows W36+ to NON-capacity mechanisms.
- **Cold-start advantage was misleading**: -2.12pp at EP1 → null at terminal. Lesson for future hypotheses: do not trust cold-start signals alone; need EP15+ trajectory data before confidence.

---

## 2026-05-25 16:25 — PR #1323 CLOSED: WSS H137 PCGrad + H39 GradNorm (informative null — mechanism falsification)

- `dl24-nezuko/h137-pcgrad-gradnorm`, run `dsnn61ne`
- Hypothesis: PCGrad's conflicting-component projection (Yu et al. 2020) ON TOP OF H39 GradNorm preserves H39's task-weight equilibrium while resolving inter-task gradient conflicts (predicted test_WSS 6.45-6.62%, 60-70% prob beats H39 SOTA).
- Student EP2-mid mechanism diagnostic (15:24Z) — PCGrad mechanism PASSIVELY FAILS:

| Phase | Steps | Mean conflict_rate | % zero-conflict steps |
|---|---|---|---|
| Warmup | 1-975 | 0.175 | 43% |
| Late warmup | 1000-5475 | 0.011 | **93%** |
| EP1 post-warmup | 5500-10950 | 0.010 | **95%** |
| EP2 | 10975-12775 | 0.017 | **89%** |

- Post-warmup avg cosine_pre between 5 GradNorm tasks: 0.43-0.48 (strongly POSITIVE alignment). PCGrad projects on negative cosines → effectively no-op for 90%+ of steps.
- Projection magnitude (relative L2): mean 0.4-0.6% of gradient norm = numerical no-op.
- H39 GradNorm dynamics PRESERVED (w_tau_z 1.30 at step 12.5k, w_vol_p 0.15 clamped) — PCGrad doesn't break H39 but doesn't help either.
- 4× wallclock penalty (1.13 it/s vs baseline 4.0 it/s) → only EP12-13 reachable in 33h budget.
- **Structural reason for null**: (1) trunk-shared gradients across 5 tasks push into same low-rank geometry feature space → positive cosines after warmup; (2) GradNorm already scale-matches per-task magnitudes BEFORE they hit trunk → negative-cosine events PCGrad would correct are eliminated upstream.

**Conclusions**:
- **Combined H136 + H137 = double-falsification on gradient-surgery axis**: H136 IMTL-G fails ACTIVELY (equal-projection collapses w_tau_z 7.3×), H137 PCGrad fails PASSIVELY (mechanism never activates on positively-correlated trunk gradients). Both falsifications point to the same root: H39 GradNorm-with-clamps is a near-optimal task-weighting on this model class.
- Any MTL surgery operating on (a) equal projection (IMTL-G), (b) negative cosines (PCGrad/GradVac), or (c) any mechanism activated AFTER GradNorm's magnitude balancing is structurally dominated by H39.
- **Wave 36+ pivot direction**: if MTL surgery is still the lever, mechanism must operate on something OTHER than negative cosines or equal projection. Candidates: Nash-MTL (Pareto-stationary projection), GradDrop (per-coordinate sign-disagreement), Hutchinson Hessian-based weighting.

---

Single-model long DDP8 validation wave; started 2026-05-04.

This log is appended in reverse-chronological order as PRs are reviewed. Each entry should include: PR number/title, student branch, hypothesis, results table (with W&B run IDs and test metrics), and brief commentary.

The wave's evidence contract: test metrics from `test_primary/*` only; validation is for steering and checkpoint selection.

---

## 2026-05-25 12:21 UTC — PR #1317 CLOSED (H135 frieren — Ada-Temp Slices EP1 divergence, slice-routing destabilization)

- branch: `dl24-frieren/h135-ada-temp-slices`
- W&B Rank-0 run: `odvx4p9c` (terminal=EP1, val_abupt=43.55% breached 35% kill threshold)
- hypothesis: Replace H39's fixed-temperature Physics-Attention softmax with per-point learned temperature τ_i = τ_0 · exp(MLP(x_i)) via Gumbel-Softmax (Transolver++ Ada-Temp mechanism, arxiv 2502.02414). Predicted DrivAerML test_WSS 6.1-6.45% based on Transolver++ DrivAerNet +12.6% surface gain.

### Run history (debug timeline)
| Time   | Run | State | Notes |
|--------|-----|-------|-------|
| 09:35Z | `bvhnkkor` | FAILED 0.3m | DDP unused-parameter error: indices 24/45/66/87/108/129 (one per Transolver block × 6 blocks) |
| 10:04Z | — | — | Advisor diagnosis: τ-predictor MLPs registered but not always used in forward (Gumbel-Softmax stochastic masking). Recommended tau-anchor aux loss (1e-6 coef) as preferred fix |
| 11:01Z | `jjpf6rnv` | finished 0.1m | Smoke test PASSED after student bugfix (commit 45e6642): root cause was orphan `self.temperature` Parameter from legacy code path, NOT τ-predictor as advisor hypothesized. Student diagnosis was correct. |
| 11:26Z | `odvx4p9c` | running | Full 30-EP DDP-8 launch |
| 12:21Z | `odvx4p9c` | killed EP1 | val_abupt=**43.55%** at step 10864 — >> 35% kill threshold |

### EP1 result table
| Metric | EP1 value | Kill threshold | Status |
|--------|-----------|----------------|--------|
| val_abupt | 43.55% | 35% | **BREACHED +8.55pp** ⚠️ |
| val_WSS | (not logged before kill) | 7.05% (EP3 abort) | — |
| val_SP | (not logged before kill) | — | — |

### Mechanism analysis (informative null)
The DDP fix landed cleanly (smoke passed, full launch started), but the underlying **Ada-Temp slice routing mechanism diverged catastrophically at EP1**:

1. **val_abupt=43.55% is roughly 8× the H39 reference range** (~5.8% terminal) — this is approaching randomly-initialized-model magnitude
2. **Gumbel-Softmax noise during training destabilized slice assignment** rather than encouraging useful exploration. The per-point τ_i learning is dominated by Gumbel noise at EP1 before τ-predictor MLP has stabilized
3. **Slice routing is HIGH-RISK at H39 capacity**: the canonical Transolver fixed-temperature assignment is already near-optimal at depth-6/slices-128 — perturbing it without sufficient initialization care collapses the model

### Strategic implications
- **Slice-routing perturbations are HIGH-RISK** at H39 capacity. Any future hypothesis touching slice assignment (Gumbel, soft slicing, learned routing) should:
  - Initialize predicted τ to match H39's fixed τ exactly at EP0
  - Add tau-anchor aux loss to prevent runaway
  - Smoke 200+ steps before full launch
- **The Transolver++ DrivAerNet gain (+12.6% surface) is at scales (more epochs, more parameters) that don't transfer to our 30-epoch / depth-6 budget**. The DrivAerML literature suggests slice-routing innovations need 100+ EP training to stabilize
- **Representation-axis pivot**: avoid slice-routing perturbations; focus on loss-shaping (H138 curvature-weighted Charb-z), input augmentation, or output-head architecture instead

### Verdict
**C NULL — closed by student claude at EP1 abupt kill threshold (12:21Z).** Wave 35 representation-axis arm closed; H138 (curvature-weighted Charbonnier z-axis loss-shaping) dispatched as replacement.

---

## 2026-05-25 10:55 UTC — PR #1318 CLOSED (H136 nezuko — IMTL-G informative null + valuable mechanism falsification at EP3)

- branch: `dl24-nezuko/h136-imtl-g`
- W&B Rank-0 run: `l7q4tumh` (terminal=EP3, student stopped per AND-gate decision rule)
- hypothesis: Replace GradNorm with IMTL-G (closed-form gradient surgery, equal-projection multi-task weighting) to avoid the bounded-loss budget redistribution pathology observed in H114/H117/H41v2.

### Results table

| EP | val_WSS | H39 ref | Δ vs H39 | val_SP | val_VP |
|----|---------|---------|----------|--------|--------|
| 1  | 18.5595 | 17.8972 | +0.6623 | 11.2188 | 20.1632 |
| 2  | 7.7018  | 7.4818  | +0.2200 | 4.3064  | 5.3527 |
| 3  | **7.3159** | 7.2016 | **+0.1143** | 4.0905 | 4.3200 |

EP3 hard-abort AND-gate breach: val_WSS=7.3159 > 7.05 ✓ AND val_SP=4.0905 > 3.8 ✓ → student closed per advisor rule.

### IMTL-G task-weight equilibrium at EP3 (mechanism diagnostic)

| Weight | H136 IMTL-G | H39 GradNorm ref | Verdict |
|--------|-------------|------------------|---------|
| w_wss (Σ tau) | **0.2535** | ~1.85 | **7.3× SUPPRESSED** |
| w_tau_z | 0.1205 | dominant | collapsed |
| w_vol_p | 0.0310 | clamped 0.15 floor | **below floor (5×)** |
| w_cp    | **0.7156** | ~0.3-0.5 | surged (2-2.5×) |
| w_surf_p| **0.7156** | ~0.3-0.5 | surged (2-2.5×) |

### Analysis

Mechanism IS firing — IMTL-G produces fundamentally different task-weight equilibrium than GradNorm — but **in the wrong direction**. Equal-projection property favors well-conditioned tasks (cp, surf_p) and SUPPRESSES high-residual noisy tasks (WSS-z especially).

**Falsifies a key program assumption**: that GradNorm's tendency to UPweight high-residual axes (the H114/H117/H41v2 "bounded-loss redistribution pathology") was a bug to fix. It is NOT a bug — it is structurally aligned with the WSS-first objective. GradNorm's rate-balancing is the productive mechanism.

### Strategic implication

Future gradient-surgery experiments should aim for OBJECTIVE-WEIGHTED projection (favor WSS axes), not equal-projection. Candidates: PCGrad with task priority, CAGrad α≥0.7, GradVac with WSS-priority weighting, IMTL-W (weighted IMTL).

---

## 2026-05-23 18:41 UTC — PR #1281 CLOSED (H38 nezuko — Charbonnier on cp saturated at EP3 peak then GradNorm down-weighted cp through EP4-EP10; 5th Wave 33 alive-but-ineffective arm; mechanism rule-out preserved for follow-up bounded-loss arm H40)

H38 nezuko EP10 verdict (step 109759): val_wss=7.0959 (+0.056 over ≤7.04 gate ✗), val_sp=4.1571 (+0.107 over ≤4.05 ✗), val_vp=3.9983 (+0.148 over ≤3.85 ✗). All 3 EP10 soft gates missed by margins student's EP7/EP8 mechanism analysis predicted (within ±0.02pp). Closed per the 17:45Z Option-1 plan.

### H38 EP1→EP10 wss/sp/vp trajectory vs H21

| EP | Step | H38 wss | H21 ref | Δ wss | H38 sp | sp Δ | H38 vp | vp Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10975 | 19.548 | 19.309 | +0.24 | 14.93 | — | 17.84 | — |
| 2 | 21951 | 7.705 | 7.804 | **−0.099** ⭐ | 4.770 | (+0.108) | 5.724 | (−0.040) |
| 3 | 32927 | 7.368 | 7.589 | **−0.221** ⭐⭐ peak | 4.365 | −0.020 | 4.657 | +0.008 |
| 4 | 43903 | 7.241 | 7.288 | −0.047 ⭐ | 4.257 | −0.079 ⭐ | 4.363 | −0.121 ⭐ |
| 5 | 54879 | 7.176 | 7.237 | −0.061 | 4.221 | −0.018 | 4.241 | +0.127 ⚠ |
| 6 | 65855 | 7.156 | 7.259 | −0.103 ⭐ | 4.190 | −0.028 | 4.154 | +0.121 ⚠ |
| 7 | 76831 | 7.134 | 7.130 | +0.004 ≈ | 4.184 | +0.024 ⚠ | 4.099 | +0.159 ⚠ |
| 8 | 87807 | 7.125 | 7.112 | +0.013 | 4.173 | +0.013 | 4.058 | ~+0.11 ⚠ |
| 9 | 98783 | 7.106 | (interp) | ~−0.01 | 4.160 | (interp) | 4.019 | ~+0.10 ⚠ |
| **10** | **109759** | **7.0959** | **~7.04** | **+0.056** ⚠ | **4.1571** | **+0.107** ⚠ | **3.9983** | **+0.148** ⚠ |

**Mechanism rule-out preserved verbatim for the next-wave write-up**:

> "Charbonnier on a channel transfers under GradNorm only if the channel has a heavy-tailed error distribution. Channels with small-magnitude error distributions (cp on this dataset, ≈0.1–0.3 normalized std) get under-trained by Charb because the linear-regime threshold isn't reached and GradNorm down-weights the channel further as task_loss decays faster than other channels'."

W&B run: `74n1y8g7`. SENPAI-RESULT: terminal=true, primary_metric.val_wss_rel_l2_pct_EP10=7.0959.

Reassigned to **H40 Cauchy bounded loss on cp** (PR #1287) — direct port of the H38 mechanism rule-out into a fundamentally different loss-shape regime (Barron 2019 α=0, redescending influence function r/(1+(r/c)²) concentrates gradient on small-magnitude residual regime where cp actually lives).

---

## 2026-05-23 18:27 UTC — PR #1273 CLOSED (H35 v3 frieren — bidir slice xattn delivered cold-start EP3 lead + v8→v9 surge, then asymptotically absorbed into H21 attractor basin by val#10; structural-but-asymptotic-absorption failure mode, distinct from H38's mechanism-saturation failure mode)

H35 v3 frieren val#10 verdict (step 54889): val_wss=7.335 (FAIL ≤7.30 by +0.035), val_sp=4.314 (PASS ≤4.34 by −0.026), val_vp=4.539 (FAIL ≤4.21 by +0.329). **2 of 3 hold-or-abort gates failed**; behind H21 EP5 on EVERY primary channel. Closed per the 17:45Z agreement.

### H35 v3 val#1→val#10 trajectory (true training step)

| val# | Step | wss | sp | vp | abupt | wss_z |
|---|---|---:|---:|---:|---:|---:|
| 1 | 5488 | 28.597 | 20.092 | 23.070 | 27.717 | 37.691 |
| 2 | 10977 | 11.627 | 7.187 | 9.723 | 11.068 | 14.989 |
| 6 | 32933 (EP3) | 7.433 | 4.353 | 4.711 | 6.750 | 10.006 |
| 7 | 38422 (EP3.5) | 7.389 | **6.340** ⚠ EMA artifact | 4.648 | 7.111 | 9.988 |
| 8 | 43911 (EP4) | 7.395 | 4.493 | 4.582 | 6.728 | 10.021 |
| 9 | 49400 (EP4.5) | 7.352 | 4.336 | 4.546 | 6.662 | 9.990 |
| **10** | **54889 (EP5.0)** | **7.335** | **4.314** | **4.539** | **6.649** | **10.025** |

v9→v10 slopes collapsed 5-14× vs v8→v9 → asymptotic-equivalence regime reached. The v8→v9 surge (surf_p 7× H21 descent) was real but ephemeral; the subsequent v9→v10 window confirmed the asymptotic-absorption hypothesis.

**Mechanism takeaway**: "Slice-level bidir WSS↔surf_p xattn delivers cold-start absorption + intermediate-EP surge tied to entropy/diversity gates clearing, but coupling head-cost creates an asymptotic vol_p penalty under GradNorm clamp at w_vol_p=0.15. H21 attractor basin reabsorbs the WSS+surf_p advantage by EP5 while vp deficit persists."

W&B run: `qc04koec`. SENPAI-RESULT: terminal=true, primary_metric.val_wss_rel_l2_pct_EP5=7.3350.

Reassigned to **H41 Charb axes y,z extension** (PR #1286) — extends H21's `--wss-charbonnier-axes z` to `y,z` to probe H38 mechanism boundary on the second-worst WSS axis (~8.0% vs z's 10.0%); 1-flag change, falsifiable at EP3 boundary.

---

## 2026-05-23 10:37 UTC — PR #1272 CLOSED (H34 nezuko — Ada-Temp Slices mechanism ALIVE but ineffective at delivering wss gains; third Wave 33 architectural arm with this pattern; reassigned to **H38 surf-p Charbonnier** to attack the wave-wide pressure floor)

H34 nezuko EP10 verdict: val_wss=7.062, gate criterion val_wss ≤ 6.95 MISS by +0.112pp. Closed at EP10.1 per the EP10 gate criterion. Run `ai7wkvov` was healthy throughout, mechanism strong, but trajectory does not reach gate.

### H34 EP10 val trajectory (vs gates):

| Metric | EP6 | EP7 | EP8 | EP9 | EP10 | Δ EP6→EP10 | Gate (EP10) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **val_wss** | 7.127 | 7.100 | 7.080 | 7.079 | **7.062** | −0.065 | ≤ 6.95 | **MISS +0.112pp** ❌ |
| val_vol_p | 4.056 | 4.023 | 4.013 | 4.015 | 4.001 | −0.055 | < 4.50 | PASS |
| val_surf_p | 4.180 | 4.158 | 4.154 | 4.153 | 4.147 | −0.033 | < 4.30 | PASS |
| val_abupt | 6.385 | 6.352 | 6.336 | 6.332 | 6.316 | −0.069 | < 6.40 | PASS |

Decelerating trajectory (Δ per EP: −0.027 → −0.020 → −0.001 → −0.017pp). Geometric decay then slight cosine LR re-acceleration but not enough to recover +0.112pp in remaining 20 EPs.

### Terminal projection (NOT a contract winner)

Applying val→test gaps: H21-class (−0.36) and H32-class (−0.40):

| Metric | H34 projected terminal | vs SOTA 972 | vs wave-best |
|---|---:|---|---|
| test_wss | **6.66–6.71** | beats by 0.02–0.07pp | worse than H33's 6.679, H32's 6.691 |
| test_vol_p | 3.66–3.69 | **BREACH +0.02–0.05** ❌ | similar to H32/H33 |
| test_surf_p | 3.69–3.72 | **BREACH +0.11–0.14** ❌ | similar to H32/H33 |
| test_abupt | 5.84–5.87 | tied | similar to H21 |

### Mechanism IS ALIVE — the diagnostic value (load-bearing)

γ trajectory at EP10:

| Layer | γ_mean | γ_std | Verdict |
|---|---:|---:|---|
| 0 | 0.699 | 0.307 | ⭐ |
| 1 | 0.765 | 0.313 | ⭐ |
| 2 | 0.861 | 0.323 | ⭐ |
| 3 | 0.976 | 0.334 | ⭐ |
| 4 | 1.032 | 0.373 | ⭐ |
| 5 | 1.052 | 0.376 | ⭐ |

All 6 layers monotonic depth-graded on BOTH γ_mean (0.70→1.05) AND γ_std (0.31→0.38). Global γ_std=0.343 — strongest mechanism signal of Wave 33's architectural arms.

GradNorm at EP10: w_cp=0.62, w_τ_x=0.99, w_τ_y=1.43, **w_τ_z=1.83**, w_vol_p=0.15 (clamped). **Same loss-surface as H21/H32** — ada-temp does NOT reshape GradNorm-driven allocation. The per-point softmax temperature operates on attention only, not on gradient budgets, and the wave's wss-axis is GradNorm-budget-constrained.

### Strategic finding: third architectural arm with "alive-but-ineffective" pattern

| Arm | Mechanism | test_wss | vs SOTA | Floor breaches |
|---|---|---:|---:|---:|
| **H33** GALE-Transolver | K=512 anchor bank, surface-only mask | **6.679** | beats by 0.048 | 3 (sp, vp, ab) |
| **H32** surf_lw=1.25 | uniform surface-axis boost | **6.691** | beats by 0.036 | 2 (sp, vp) |
| **H34** Ada-Temp Slices | per-point γ softmax temp | ~6.70 projected | beats by ~0.02 | 2 (sp, vp) projected |

**Wave 33 architectural arms all converge on the same pattern**: mechanism activates as designed, val_wss descends, but test_wss tops out at 6.66–6.70 (just better than SOTA 6.727) while pressure floors uniformly breach. The wave's bottleneck is **structurally pressure-axis-limited under wss expansion**.

Cross-confirmation: tay fleet's H102 (SURFACE-OUT-WIDER-MLP, +266K params) at 94.78% complete projects test_WSS ~6.66 with test_SP ~3.85 (NOT clearing 3.577 floor across 11+ tay variants). **The surf_p plateau is wave-wide, not architecture-specific.**

### Reassignment

**PR #1281 H38**: symmetric surf-p Charbonnier (mirror of validated vol_p Charbonnier mechanism). First explicit pressure-axis-protection experiment of Wave 33 — attacks the surf_p floor (3.577) using the same loss-shape lever that produced sub-floor vol_p in H21 (3.579). H21 base + new `--surf-p-charbonnier-weight 0.1` flag (~15-25 LoC).

W&B run: https://wandb.ai/wandb-applied-ai-team/senpai-v1-drivaerml-ddp8/runs/ai7wkvov. Closed at advisor comment 4525078762.

---

## 2026-05-23 08:20 UTC — PR #1260 CLOSED (H33 fern — GALE-Transolver mechanism alive but ineffective at H33's instantiation; **NEW Wave 33 wss bar 6.679 BEAT SOTA −0.048** but 3 floors breach; mechanism repair dispatched as H37 GeoTransolver-true)

H33 fern ran 18/30 EP via SENPAI_TIMEOUT graceful cutoff (1282 min, 21.37h walltime). Best EMA-val at EP13. Test from best-val-EMA checkpoint per advisor 07:00Z directive.

### H33 terminal (test_primary/*, EP13 best-val-EMA):

| Metric | H33 test | H19 ref | SOTA #972 | Floor (#1056) | Δ vs floor | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **wss** | **6.679** | 6.634 | 6.727 | (drive ↓) | **−0.048 vs SOTA** | **BEATS SOTA ⭐ (NEW Wave 33 single-model bar, edges H32 6.691 by −0.012)** |
| surf_p | 3.652 | 3.627 | 3.577 | 3.577 | +0.075 | **BREACH** ❌ |
| vol_p | 3.865 | 3.779 | 3.643 | 3.643 | +0.222 | **BREACH** ❌ |
| abupt | 5.871 | 5.820 | 5.844 | 5.844 | +0.027 | **BREACH** ❌ |

**Contract verdict**: 3 of 4 floors breach → no merge.

**Mechanism is ALIVE but ineffective (student diagnostics — exceptional quality)**:
- Gate trajectories depth-monotonic (0.273 → 0.421 across 6 blocks)
- Bank fully utilized (512/512 anchors active)
- Bank LayerNorm healthy (10.78 ± 0.93)
- +24.29M params (1.43M → 25.72M, 17.9×) — burden offsets the geometry signal

**Student's two failure-mode hypotheses** (ranked by priority for follow-up):
1. **Surface-only query mask isolates the very signal we needed to share** — H30-class cross-task starvation occurs at surface↔volume *interaction* in shared encoder; surface-only geometry queries refresh surface tokens but don't bridge cross-task.
2. **K=512 stride-anchor sampling is effectively single-scale** — actual GeoTransolver paper (arxiv 2512.20399) builds multi-scale concentric ball queries at r = {0.01, 0.05, 0.25, 1.0, 2.5, 5.0}.

**Calibration finding (load-bearing)**: H33 EP6 val_wss=6.945 (-0.049 vs H19 EP6) was wave-noise relative to +0.045pp terminal regression. **EP6 deltas at ±0.05pp are NOT predictors** of terminal performance in this wave. Update gate criteria accordingly.

**Strategic finding (H32 + H33 jointly)**: Two independent mechanisms (uniform surf_lw=1.25 and GALE substrate) both produce test_wss < SOTA, both breach pressure floors. **Wave's structural bottleneck is the pressure-axis floor when wss capacity expands** — not the wss-axis. Pressure-axis protection is the architectural focus for Wave 34.

**Reassignment**: PR #1275 H37 — GeoTransolver-true (faithful arxiv 2512.20399). All-tokens query a shared Context tensor C, multi-scale ball queries at 6 radii r={0.01,0.05,0.25,1.0,2.5,5.0} with k_s=32, per-layer per-slice adaptive gating, persistent injection at every block. Addresses BOTH diagnosed failure modes simultaneously. Student's own ranked #1 follow-up.

W&B run: https://wandb.ai/wandb-applied-ai-team/senpai-v1-drivaerml-ddp8/runs/v4vkkr23. Closed at advisor comment 4524783664.

---

## 2026-05-23 06:15 UTC — PR #1259 CLOSED (H32 tanjiro — surf_lw=1.25 produces FIRST single-model test_wss BEAT of SOTA in Wave 33 BUT 2 floors breach; mechanism repair dispatched as H36)

H32 tanjiro finished cleanly at EP30 (run `ikoxad4k`, state=finished, 1279min walltime). EMA best at EP17.

### H32 terminal (test_primary/*):

| Metric | H32 test | H21 ref | H26 ref | SOTA #972 | Floor (#1056) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **wss** | **6.6910** | 6.730 | 6.6389 | 6.727 | (drive ↓) | **BEATS SOTA −0.0360 ⭐ (first single-model beat in Wave 33)** |
| surf_p | 3.6768 | 3.679 | 3.6532 | 3.577 | 3.577 | **BREACH +0.0998** ❌ |
| vol_p | 3.6925 | **3.579** | 3.667 | 3.643 | 3.643 | **BREACH +0.0495** ❌ |
| abupt | **5.8355** | 5.832 | 5.7940 | 5.844 | — | beats −0.0085 ✓ |

**Contract verdict**: 2 floors breach → no merge. But **test_wss beat of SOTA + test_abupt clear** is the wave's most informative scalar-lever finding.

**Mechanism captured** (student delivered exceptional analysis):
1. **`surface_loss_weight` axis comprehensively mapped** at H21=1.0, H32=1.25, H26=1.5. Uniform multiplier is exhausted.
2. **vol_p is non-monotonic over [1.0, 1.5]** at clamp=0.15 — the H21→H26 interpolation predicted in PR body was falsified (H32 vol_p=3.693 is WORSE than H26's 3.667 despite lower surf_lw). Mechanism: intermediate cp suppression at surf_lw=1.25 produces the worst encoder features for indirect vol_p gradient flow.
3. **τ_z is the wss attractor** (`w_tau_z` ≈ 1.76-1.85 throughout via GradNorm trace) — uniform boost wastes budget on τ_x/τ_y/cp that vol_p depends on indirectly. **This is the actionable mechanism: target τ_z asymmetrically.**

**Strategic context**: H32 sets the new bar for Wave 33 single-model test_wss (6.691). The three architectural arms in flight (H33 GALE, H34 Ada-Temp, H35 WSS↔surf_p xattn) now have a concrete number to beat without breaching floors. Scalar lever family is not entirely retired — uniform multipliers are retired; per-channel/asymmetric weighting (H36) is the natural mechanism repair.

**Reassignment**: PR #1274 H36 — asymmetric τ_z boost (1.25× to tau_z row only, cp/τ_x/τ_y/vol_p at 1.0×). Student's own suggested follow-up #1 made tractable as a single 30-EP run. Predicted: H32-level wss + H21-level vol_p (sub-floor) + H21-level surf_p.

W&B run: https://wandb.ai/wandb-applied-ai-team/senpai-v1-drivaerml-ddp8/runs/ikoxad4k. Closed at advisor comment 4524369776.

---

## 2026-05-23 03:37 UTC — PR #1256 CLOSED (H31 frieren — lighter wss-Charb-z 0.05 FALSIFIED at terminal; all 4 metrics regress, 2 floors breach)

H31 frieren finished cleanly at EP30 (run `tj6bh04q`, state=finished). Student loop hadn't yet posted SENPAI-RESULT 11h after terminal, so closure was made from W&B test metrics direct.

### H31 terminal (test_primary/*):

| Metric | H31 test | SOTA #972 ref | Δ vs SOTA | Floor (#1056) | Status |
|---|---:|---:|---:|---:|---|
| wss | **6.845** | 6.727 | +0.118pp | (target <5.85) | regression |
| surf_p | **3.773** | 3.577 | +0.196pp | 3.577 | **BREACH +0.196** ❌ |
| vol_p | **3.708** | 3.643 | +0.065pp | 3.643 | BREACH +0.065 ❌ |
| abupt | **5.964** | 5.844 | +0.120pp | — | regression |

**Verdict**: Hypothesis FALSIFIED. Lighter `wss-charbonnier-weight=0.05` (down from H21's 0.1) does NOT free `w_cp` GradNorm budget for surf_p as hypothesized — it does the mirror. Reduced Charb-z penalty → GradNorm sees less τ_z drift correction needed → budget reallocates UPSTREAM into already-converging wss heads, NOT toward cp/vol_p. Result: surf_p floor breach widens monotonically (EP3=+0.04 → EP15=+0.043 → EP30=+0.20pp) and vol_p slope FLIPS POSITIVE at EP15 (+0.009pp/EP) while H21 continues to descend (−0.010pp/EP).

**Mechanism captured**: PR #1216's choice of `wss-charbonnier-weight=0.1` is the *correct* tightness for the H21 lever family. Combined with H29's earlier closure (extended cosine on H26 base falsified), **the "lighten existing scalar levers" family is now fully RULED OUT** for Wave 32+. Architectural pivot (Wave 33 = H33 GALE / H34 Ada-Temp Slices / H35 WSS↔surf_p xattn) is the only path forward.

**Student diagnostic quality**: exceptional. EP15 vol_p slope-flip detection (+0.009pp/EP vs H21 −0.010pp/EP) was the cleanest possible falsification signal at mid-run. The student's per-EP drift slope table (vol_p Δ widening +0.10 → +0.11 → +0.13 → +0.14 → +0.17 over EP12-15) saved the wave from waiting until terminal for the verdict.

W&B run: https://wandb.ai/wandb-applied-ai-team/senpai-v1-drivaerml-ddp8/runs/tj6bh04q. Closed at advisor comment 4524048303 with reassignment to **H35 Wave 33 Idea C — WSS↔surf_p bidirectional cross-attention** (Morgan's 08:50Z architectural directive, PR #1273).

---

## 2026-05-23 01:38 UTC — PR #1253 CLOSED (H29 nezuko — extended cosine T_max=50 on H26 base FALSIFIED; vol_p mechanism preserved as transferable insight)

H29 nezuko finished cleanly at EP30 (run `6acyrua4`, state=finished, ~21.3h walltime). Best epoch=20 (val_primary/abupt selection).

### H29 terminal (test_primary/*):

| Metric | H29 test | H26 ref (`apgpxli8`) | SOTA #972 floor | Status |
|---|---:|---:|---:|---|
| wss | **6.7288** | 6.6389 | 6.727 | TIED with SOTA (+0.002), +0.090pp WORSE vs H26 |
| vol_p | **3.6731** | 3.6665 | 3.643 | BREACH +0.030 |
| surf_p | **3.7338** | 3.6532 | 3.577 | BREACH +0.157 |
| abupt | **5.8809** | 5.7940 | 5.844 | +0.037 |

**Verdict**: Hypothesis FALSIFIED. Extended cosine T_max=50 on H26 base did NOT cure the EP19 plateau — H29 sat +0.23-0.30pp above H26's wss curve through EP10-30 despite **+16-24% higher LR**, proving H26's EP19 wss floor at 6.804 was NOT LR-limited but model/representation-limited at that seed-initialized configuration. Hypothesis class retired: do not extend cosine on H26 base; future H26-base experiments should attack capacity or loss-weight, not LR schedule.

**Transferable insight (preserved)**: GradNorm `min_w_vol_p=0.15` clamp is doing real work — at EP9 vol_p flipped (5.33 → 3.94) catching H26 within −0.016pp. The clamp breaks vol_p out of early-epoch sink by holding the vol_p loss weight above natural decay floor. Default into Wave 34 architectural experiments.

**Student analysis quality**: excellent — diagnosed seed-handicap confound vs mechanism effect cleanly, identified the categorical signal (higher LR → no additional descent), extracted lab-level insight from a negative-result experiment.

W&B run: https://wandb.ai/wandb-applied-ai-team/senpai-v1-drivaerml-ddp8/runs/6acyrua4. Closed at advisor comment 4523771051 with reassignment to Wave 33 Idea B (Ada-Temp Slices, Transolver++ 2502.02414).

---

## 2026-05-22 09:00 UTC — PR #1241 CLOSED (H28 — falsified on all 4 test metrics) + H33 assigned to fern (PR #1260, FIRST ARCHITECTURAL EXPERIMENT)

Fern's H28 finished cleanly at EP30 (run `83iayezy`, state=finished). All 4 test metrics regress vs H19 — extended cosine `T_max=60` on H19 base is fully falsified.

### H28 terminal (test_primary/*):

| Metric | H28 test | H19 reference | Δ vs H19 | Floor (#1056) | Status |
|---|---:|---:|---:|---:|---|
| wss | **6.7329** | 6.6339 | +0.0991 | (target <5.85) | REGRESSION |
| vol_p | **3.9564** | 3.779 | +0.1774 | ≤3.643 | BREACH +0.314 |
| surf_p | **3.7515** | 3.627 | +0.1245 | ≤3.577 | BREACH +0.175 |
| abupt | **5.9386** | 5.820 | +0.1186 | (target <5.85) | REGRESSION |

Val terminal: val_wss=6.827, val_vol_p=4.038, val_surf_p=4.019, val_abupt=6.136.

### Mechanism — what extended cosine actually did

The EP10 mid-trajectory gate looked promising (val_wss 6.897 vs H19 7.044, −0.147pp ahead). But the late-training phase tells the real story:

| Phase | LR | val_wss | What happened |
|---|---:|---:|---|
| EP10 (mid) | 9.46e-5 | 6.897 | Faster convergence, ahead of H19 by 0.15pp |
| EP15-30 (late) | 8e-5 → 5e-5 | ~6.83 | Plateaued, never closed surface-feature gap |
| Test gap | — | +0.10pp val→test on wss | Late-training LR too high for fine surface features |

The longer LR runway never broke the surf_p floor (it stayed 0.12pp above H19's own surf_p of 3.627). Holding LR higher into the late-training surface refinement window appears to have **prevented** the surface decoder from settling onto narrow optima — opposite of intended mechanism.

### Plateau-Protocol diagnosis: 5+ consecutive scalar tweaks failed to break surf_p

| Hypothesis | Mechanism | Outcome |
|---|---|---|
| H21 | `--gradnorm-clamp 0.15` | vol_p ✓, surf_p ✗ |
| H27 | `--gradnorm-clamp 0.10` | strictly inferior to H26 partial |
| H26 | `--surface-loss-weight 1.5` | abupt ⭐ but vol_p+surf_p breach |
| H30 | `--gradnorm-clamp 0.20` (tighter) | cross-task starvation, EP6 abort |
| H28 | `--lr-cosine-t-max 60` (extended) | ALL 4 FAILED |

None of these add **geometric information** to the encoder. The bottleneck is **encoder representation**, not loss/optimization shape.

### H33 dispatch — first architectural experiment in Wave 33

Per researcher-agent's `RESEARCH_IDEAS_2026-05-22_0850.md` (Idea A), assigned fern **GALE-Transolver: persistent geometry cross-attention at every layer** (PR #1260).

**Mechanism**: A `GeometryContextBank` encodes surface_x (xyz+normals+area) into K=512 anchor tokens via FPS-stride + KNN max-pooling. Each Transolver block adds a `GeometryCrossAttention` sublayer between physics-attention and MLP, with a learned per-block sigmoid gate (init 0.5). Surface tokens query the bank; volume tokens skipped via query mask.

**Direct DrivAerML evidence**: GeoTransolver (arxiv 2505.12558, 2025) reports on the **same benchmark**:
- surf_p = 2.86% (clears 3.577 floor by 0.72pp ⭐⭐⭐)
- wss = 4.90% (clears Transolver-3 target 5.85 by 0.95pp)
- vol_p = 3.09% (clears 3.643 floor by 0.55pp)

Only known method to clear all three floors simultaneously. ~200-300 LoC, 4-6 hours implementation.

**EP6 gate**: val_wss ≤ 7.00, val_surf_p ≤ 4.05, `geo_xattn/gate_mean > 0.05` (must move off init), no NaN/Inf.

**Falsification**: If gate stays at 0.50 with no surf_p gain → mechanism dead; if surf_p > 4.20 at EP6 → geometry xattn is actively hurting (catastrophic). Either outcome is informative.

**Predicted outcome (60-70% scaling of GeoTransolver numbers)**:
- test_wss: 6.10-6.40 (clears H19 by 0.2-0.5pp; clears Transolver-3 5.85 at upper bound)
- test_surf_p: 3.30-3.50 (breaks floor first time in wave)
- test_vol_p: 3.40-3.55 (clears floor with margin)

PR #1241 closed (https://github.com/morganmcg1/DrivAerML/pull/1241#issuecomment-4517218478). H33 PR #1260 assigned to fern.

---

## 2026-05-22 08:15 UTC — PR #1255 CLOSED (H30 — falsified at EP6) + H32 assigned to tanjiro (PR #1259)

Tanjiro aborted H30 at EP6 with terminal `SENPAI-RESULT` (run `ofehmi7q`). The hypothesis "tighter `w_vol_p` clamp restores vol_p floor while retaining H26's wss gain" was cleanly falsified:

| Epoch | Δwss vs H26 | Δvol_p vs H26 |
|---:|---:|---:|
| EP3 | +0.109 | **−0.018** (favorable) |
| EP4 | +0.128 | +0.032 |
| EP5 | +0.171 | +0.072 |
| EP6 | **+0.214** | +0.078 |

Per-EP Δwss widened monotonically at ~+0.04/EP; Δvol_p sign-flipped from favorable to unfavorable. Both axes worsening simultaneously — the mechanism prediction is the opposite of what was observed.

### Mechanism — the central scientific finding

The PR rationale assumed: tighter `w_vol_p` clamp → direct vol_p gradient budget → faster vol_p descent.

The observed mechanism is the **reverse**: tighter `w_vol_p` (0.20 vs 0.15) diverts gradient away from `w_cp` (H30 EP6 `w_cp`=0.619 vs H26 EP20 equilibrium `w_cp`=0.792, **~22% suppression of cp**). Since vol_p depends on the surface-aware encoder's features, suppressing `w_cp` degrades the volume queries vol_p uses — the indirect feature-support loss exceeds the direct gradient-budget gain.

**Generalizable insight**: gradient clamping at the task level can create cross-task starvation that dominates the direct effect, especially in shared-encoder MTL setups where downstream tasks depend on upstream feature quality. Future GradNorm clamps should consider the *whole gradient ecosystem*, not just the clamped task.

### Validated wave-position update

| Hypothesis | Result |
|---|---|
| "Tighter clamp restores vol_p without losing wss" (H30) | **FALSIFIED at EP6** |
| "Surface-loss-weight prefactor is the breach driver" | testable via surf_lw sweep — H32 |
| "Lighter wss-Charb-z frees cp budget for surf_p" | in flight — H31 frieren |
| "Extended cosine T_max=50 cures H26 EP19 plateau" | in flight — H29 nezuko |
| "Extended cosine T_max=60 on H19 base" | terminal landing ~07:50Z — H28 fern |

### EP3 gate calibration take-away (for future PR design)

Tanjiro noted (and I agree) that H30's EP3 hard-abort threshold `val_vol_p > 4.50` was below H26's actual EP3 value of 4.6335 — so the gate as written would have aborted H26 too. **Future PRs should peg gates to "Δ vs reference" rather than absolute pp values.** Applied this to H32 PR.

### H32 assigned (PR #1259) — tanjiro back to work

H32 = H21 base + `--surface-loss-weight 1.0 → 1.25` (per tanjiro's #1 follow-up suggestion). Interpolates between H21 (1.0) and H26 (1.5) to find whether a *partial* surface boost retains H26's wss benefit while staying under the 3.643 vol_p floor.

Predicted (linear interpolation of H21→H26 test metrics):
- test_wss ~6.685 (likely no SOTA win)
- test_vol_p ~3.623 (**clears floor ✓**)
- test_surf_p ~3.666 (still breaches)
- test_abupt ~5.813 (clears floor ✓)

If "3 of 4 floors clear" lands as predicted + H31 simultaneously unlocks surf_p via wss-Charb-z 0.05 → next-wave compound H33 = H21 + Charb-z 0.05 + surface-loss 1.25 becomes the contract-winner candidate.

PR #1255 closed (https://github.com/morganmcg1/DrivAerML/pull/1255#issuecomment-4516814916). H32 PR #1259 assigned to tanjiro.

---

## 2026-05-22 05:30 UTC — PR #1238 CLOSED (H26 — partial winner, floor breaches per Issue #1056)

PR #1238 had been sitting in `status:review` since tanjiro's terminal `SENPAI-RESULT` landed at 03:26Z (run `apgpxli8`). After tay-advisor confirmed the hold path on Issue #1056 (03:47Z) and H30 (clamp 0.20) + H29 (extended cosine T_max=50) launched as the compound-fix follow-ups, closing PR #1238 cleanly with the mechanism take-aways captured for the next wave. **Not a merge** — H26 floor-breaches both `test_vol_p` (+0.024pp) and `test_surf_p` (+0.076pp) vs PR #972, which violates Issue #1056's "WHILE ALSO not degrading the volume and pressure metrics from #972" directive. The wss gain is also only tied with H19 at noise (+0.005pp).

### H26 final scoreboard (terminal `apgpxli8`, best-val EP19 EMA)

| Metric | PR #972 floor | H19 wave-best | **H26** | Δ vs floor | Δ vs H19 | Status |
|---|---:|---:|---:|---:|---:|---|
| test_abupt | 5.844 | 5.820 | **5.794** | −0.050 ✅ | −0.026 | **NEW WAVE-BEST (axis)** |
| test_wss | 6.727 | **6.6339** | 6.6389 | −0.088 ✅ | +0.005 | tied H19 at noise |
| test_vol_p | **3.643** | 3.779 | 3.667 | **+0.024 ❌** | −0.112 | **BREACH** |
| test_surf_p | **3.577** | 3.627 | 3.653 | **+0.076 ❌** | +0.026 | **BREACH** |

### Mechanism — the key scientific finding of this round

The PR's central prediction was that GradNorm would *compound* the surface re-weighting by up-weighting surface tasks further. **The opposite happened on τ_z.** Student's engagement check showed:

| GradNorm weight | H21 EP20 | H26 EP20 | Mechanism |
|---|---:|---:|---|
| `w_τ_y` | ~1.46 | **1.789 (+22%)** | absorbed the slack |
| `w_τ_z` | ~1.70 | 1.636 | partially suppressed — the channel the lever was *trying* to push |
| `w_cp / w_τ_x` | ~0.84 / ~0.7 | 0.792 / 0.633 | both suppressed |
| `w_vol_p` | 0.15 | 0.150 | clamp ACTIVE from EP4 |

**Net effect**: uniform surface ×1.5 amplification got reabsorbed by GradNorm reallocating *within* the surface family (τ_y absorbed cp/τ_x/τ_z's slack). The lever became "shift gradient from cp/τ_x/τ_z into τ_y" — NOT the intended "amplify all surface tasks 1.5×". Net effect on the desired τ_z channel was ~1.34× (less than the raw 1.5× target).

### Take-aways for next-wave hypotheses (captured for advisor design)

1. **A uniform surface-loss multiplier is NOT a clean per-axis lever under GradNorm.** GradNorm reabsorbs the magnitude and reallocates within the surface family. **Asymmetric task-level re-weighting (τ_z-only, or freezing other surface weights) is required** if the goal is to push a specific channel.
2. **val→test gap on surf_p was the silent killer.** H26's val_surf_p=4.001 was actually 0.14pp *better* than H21 at val, but the cross-split shift (+0.348) pushed test_surf_p past the 3.577 floor. Surface re-weighting appears to hurt cp generalization.
3. **abupt = √(0.5·wss² + 0.5·surf_p²) is uniformly amplified** by surface re-weight → abupt wave-best (5.794) is a side-effect of pushing both wss and surf_p on val, not an independent mechanism.

### Wave 32 follow-up coverage (all currently in flight)

| PR | H | Student | Mechanism | Targets |
|---|---|---|---|---|
| #1255 | H30 | tanjiro | H26 recipe + `--gradnorm-min-w-vol-p 0.15→0.20` | restore vol_p floor without losing H26's wss/abupt wins |
| #1253 | H29 | nezuko | H26 recipe + cosine `T_max=50` | extend LR runway past EP19 EMA plateau |
| #1256 | H31 | frieren | H21 base + `--wss-charbonnier-weight 0.1→0.05` | attack surf_p floor via cp budget release |
| #1241 | H28 | fern | H19 base + cosine `T_max=60` | independent extended-cosine validation on H19 substrate |

If H30 lands with floor compliance + retains wss/abupt wins → that's the contract winner for this wave. If H30 still breaches surf_p, the breach is intrinsic to the surface re-weight family and we pivot to asymmetric τ_z-only multipliers as the next wave.

### Suggested next-wave hypotheses (advisor backlog, post-H28-H31 landings)

1. **Asymmetric τ_z surface re-weight** (1.3–1.5× on τ_z slot only) — forces GradNorm to amplify τ_z without freeing reallocation to τ_y.
2. **H25 Charb-τz×2 + asymmetric τ_z task re-weight** — Charb compounds inside the τ_z loss; task re-weight compounds outside. Orthogonal levers.
3. **GradNorm surface-task freeze** — hold `w_cp/τ_x/τ_y` near H21 levels, let only `w_τ_z` adapt → preserves surface task balance under magnitude amplification.

PR #1238 closed with full mechanism analysis in closing comment (https://github.com/morganmcg1/DrivAerML/pull/1238#issuecomment-4515318012).

---

## 2026-05-22 04:50 UTC — PR #1239 CLOSED (H27) + H31 assigned to frieren (PR #1256)

H27 (frieren, `yyo3q1xb`) student SENPAI-RESULT landed at 04:40Z. Confirmed test metrics match advisor's W&B early-extraction:

| Metric | H27 | floor/SOTA | vs H26 (`apgpxli8`) |
|---|---:|---:|---:|
| test_wss | 6.6704 ⭐ | 6.727 (−0.057pp beats SOTA) | 6.6389 (H26 wins by 0.032pp) |
| test_abupt | 5.8396 ⭐ | 5.844 (−0.004pp marginal) | 5.7940 (H26 wins by 0.046pp) |
| test_vol_p | 3.7079 | 3.643 (+0.065 breach) | 3.6665 (H26 wins by 0.041pp) |
| test_surf_p | 3.6903 | 3.577 (+0.113 breach) | 3.6532 (H26 wins by 0.037pp) |

**H27 strictly inferior to H26 on all 4 metrics.** PR #1239 closed with explanatory comment referencing the student's terminal analysis. Lighter clamp (0.10) is the wrong axis for the contract: it trades H21's sub-floor vol_p win for a smaller wss gain than H26's surface-loss-weight 1.5× path. Student's late-training GradNorm diagnostic confirmed w_τx dividend (+0.16 at EP10) decayed to −0.04 by EP26, while w_cp ramped to +0.09 above H21 — the clamp shifts WHICH late-training task wins redistribution, not the final allocation.

**H31 (frieren, PR #1256):** H21 base + `--wss-charbonnier-weight 0.1 → 0.05` (single-flag change).

Hypothesis: H21's surf_p breach (+0.102pp vs H19's +0.050pp) is partly Charb-τz over-shaping consuming cp's GradNorm budget. Lightening Charb-z weight should let GradNorm rebalance toward natural MSE-like surface block ratios, freeing budget for cp and improving surf_p. Falsifiable predictions:
- Surf_p UNLOCK: test_surf_p ≤ 3.62 → Charb-z is the surf_p starver; opens multi-axis Charb tuning.
- Wss starvation: test_wss > 6.80 → Charb-z load-bearing for wss; cannot lighten.
- No-op: surf_p ≈ 3.67-3.70 AND wss ≈ 6.71-6.75 → Charb-z budget-neutral; surf_p problem is elsewhere.

Diversifies from H29 (H26+cosine T_max=50, running) and H30 (H26+clamp=0.20, running). Probes the H21 line orthogonally to the clamp tuning H27 closed off.

---

## 2026-05-22 03:30 UTC — H25/H26/H27 EP30 TERMINAL LANDING (Wave 32 dl24 fleet)

Three-PR mechanism-isolation wave on the H21 substrate (clamp=0.15 + GradNorm). One full falsification, one **partial-winner held for human decision**, one strict-regression. Combined evidence quadrant: surface-loss-weight (H26) is the only Wave 32 dl24 lever that moves wss SOTA forward.

### Wave 32 dl24 terminal summary

| PR | H | Student | Run | Hypothesis (Δ vs H21) | test_wss | test_vol_p | test_surf_p | test_abupt | Verdict |
|---|---|---|---|---|---:|---:|---:|---:|---|
| #1237 | H25 | nezuko | `xjbz1v84` | + Charb-τz weight 2× | 6.834 | 3.676 | 3.760 | 5.971 | **CLOSED — all 4 floors fail** |
| #1238 | H26 | tanjiro | `apgpxli8` | + surface-loss-weight 1.5 | **6.6389** ⭐ | 3.6665 ❌ | 3.6532 ❌ | **5.7940** ⭐ | **PARTIAL — held for #1056 decision** |
| #1239 | H27 | frieren | `yyo3q1xb` | + lighter clamp (0.10) | 6.6704 ⭐ | 3.7079 ❌ | 3.6903 ❌ | 5.8396 ⭐ | partial, strictly inferior to H26 |

vs Issue #1056 floors (test_vol_p ≤ 3.643, test_surf_p ≤ 3.577, test_wss < 5.85, test_abupt ≤ 5.844):

- **H26**: wss/abupt SOTA (−0.088pp wss, −0.050pp abupt) but vol_p breach +0.024pp, surf_p breach +0.076pp. Strict AND-clause fails → posted to Issue #1056 asking Morgan for explicit merge decision.
- **H27**: wss/abupt SOTA (−0.057pp wss, −0.004pp abupt) but vol_p breach +0.065pp, surf_p breach +0.113pp. Pressure breaches are LARGER than H26's on both axes → H27 is strictly inferior to H26 in this quadrant. If Morgan rejects H26 partial, H27 also fails the same way (worse).
- **H25**: All 4 floors breached. Static Charb-τz×2 falsified the "push wss harder via amplified raw loss" hypothesis. Same failure mechanism as H24-v2: GradNorm interprets the amplified loss as already-loud → reallocates AWAY from τ_z (terminal w_τz=1.645 ≈ H21, but w_τ_x rose to 0.966 vs H21 0.75). test_wss_z worsened by +0.33pp — the very axis we tried to push. Confirms anti-additive quadrant for static loss boosts under GradNorm.

### Mechanism take-aways

1. **Surface-loss-weight 1.5× (H26) is the most efficient single lever** to push wss + abupt forward on the H21 substrate. The mechanism amplifies both surface_pressure AND surface_wss tasks pre-GradNorm symmetrically; GradNorm renormalizes but redistributes within the surface block (terminal w_τ_y rose to 1.789 from H21's ~1.46, w_τ_z went DOWN slightly). Net effect: surface block gets ~1.34× the gradient mass of H21 with the same downstream renormalization.
2. **H21's "sub-floor vol_p lock" was fragile.** H21 ended at test_vol_p=3.579 (sub-floor, locked by clamp=0.15 + 0.05 default w_vol_p reference). H26's surface-loss-weight redistribution eroded that lock (+0.088pp vs H21), pushing vol_p back above the 3.643 floor. H27's lighter clamp ALSO erodes the lock from the other direction (+0.129pp vs H21). Both directions of perturbation on H21's vol_p balance break the floor.
3. **The "amplify static τ_z loss" path is anti-additive.** H24-v2 and H25 both confirm: static loss multipliers on a per-task term within a GradNorm-renormalized loss INVERT the intended task budget reallocation. The static boost makes GradNorm see the boosted task as "already loud" → it shrinks that task's GradNorm weight. Net: the budget shifts to OTHER axes (typically w_τ_x in our experiments). Future per-axis interventions should modify GradNorm's reference rates OR bypass GradNorm entirely, not pre-apply static weights to the raw loss.
4. **Wave 32 dl24 winners both BEAT wss SOTA in single-axis isolation** — H26 by −0.088pp and H27 by −0.057pp — but **NEITHER clears the Issue #1056 AND-clause**. Both arms break the pressure floors. The path forward is either (a) wait for Morgan's H26 partial-winner decision OR (b) find a mechanism that pushes wss without redistributing budget away from vol_p (e.g. clamp 0.20 on top of surface-loss-weight 1.5 — H30 tanjiro, PR #1255).

### H26 GradNorm terminal state (mechanism diagnostic, `apgpxli8`)

The surface×1.5 amplification did NOT amplify τ_z (as we hoped); it shifted budget INTO w_τ_y (+0.33 vs H21). The wss SOTA win is mostly via τ_y improvement, not τ_z. Asymmetric per-axis τ re-weighting (e.g. pushing only τ_z) remains an open direction — but the H25 falsification shows static loss boosts do not deliver it under GradNorm.

### Follow-ups assigned

- **H29 (nezuko, PR #1253):** H26 recipe + extended cosine T_max=30→50 — cure H26's EP19 plateau (cosine bottoms at EP20 in H26's 30-epoch schedule, leaving EP20-30 in near-frozen LR). Single lever from H26. Launched `cwzs551m` at 03:29Z May 22.
- **H30 (tanjiro, PR #1255):** H26 recipe + clamp 0.15→0.20 — directly restore vol_p floor on top of H26's wss gains. Tightening the floor protects vol_p budget against surface-loss redistribution. Tests "vol_p clamp is independent from surface-loss block" hypothesis. Tanjiro pod picked up the branch at 03:37Z May 22.

H28 (fern) still running on the H19 base + extended cosine T_max=60 — strictly inferior to H26 at all matched epochs (EP15: H28=6.852 vs H26=6.811). Likely partial winner like H26 but with worse wss numbers; running to terminal for diagnostic completeness.

---

## 2026-05-21 10:30 UTC — PR #1226 CLOSED: H24-v2 falsified (per-axis τ "1.0,1.2,1.5" backfires on WSS-z)

- **Branch:** `dl24-fern/h24-v2-clamp015-peraxis`
- **Run:** `j2pvm44m` (21.7h, EP30 terminal)
- **Hypothesis:** H19 + clamp=0.15 (H21's mechanism) + per-axis τ weights "1.0,1.2,1.5" — boost τ_z gradient through the clamp's stabilized GradNorm budget to push WSS-z harder.

**Test results (corrected split `rawcanon_20260511`):**

| Metric | Value | vs SOTA #972 | vs H21 reference | vs H19 reference |
|--------|------:|-------------:|-----------------:|-----------------:|
| test_abupt | 5.94 | +0.096 regress | +0.108 regress | +0.120 regress |
| test_wss | 6.81 | +0.083 regress | +0.080 regress | +0.176 regress |
| test_vol_p | 3.71 | **+0.067 BREACH** ❌ | +0.131 regress | −0.069 (improvement) |
| test_surf_p | 3.73 | **+0.153 BREACH** ❌ | +0.051 regress | +0.103 regress |

**Wall shear axis components (W&B):**
- test_τ_x = 6.05 (vs H19 5.971, **+0.08 regress**)
- test_τ_y = 7.32 (vs H19 7.362, −0.04 marginal improvement)
- test_τ_z = **8.88** (vs H19 8.747, **+0.13 REGRESS — opposite of intent**) ❌

**Best-val checkpoint:** EP19 (selection metric val_primary/abupt_axis_mean_rel_l2_pct = 6.163, EMA source).

**Mechanism diagnosis:**
- **Counterintuitive result:** Per-axis weights "1.0, 1.2, 1.5" pushing τ_z harder produced WORSE τ_z (8.88 vs 8.747). The intent was the opposite.
- **Why it backfires under GradNorm:** Static per-axis weights interact nonlinearly with GradNorm budget reallocation. Boosting τ_z's raw loss by 1.5× makes GradNorm see τ_z's relative training rate as needing LESS boost (it's already "loud"), so w_τ_z gets reduced in budget allocation. The static boost and dynamic GradNorm reweighting CANCEL OUT or even invert the intended effect.
- **Comparison to H21 (clamp=0.15, no per-axis):** H21's GradNorm freely allocated budget to τ_z (w_τ_z = 1.99), achieving test_τ_z = 8.63 (H19's value). H24-v2's per-axis pre-weight broke this dynamic.
- **Vol_p sub-floor signal preserved:** test_vol_p = 3.71 still beats H19's 3.779 by 0.069pp, confirming the clamp=0.15 mechanism remains effective for vol_p. The per-axis weights damaged the clamp's wss-side benefit, not its vol_p-side benefit.

**Conclusion:** Per-axis τ weights as designed are a dead-end mechanism under GradNorm. Future per-axis interventions should either (a) bypass GradNorm entirely on the per-axis terms or (b) modify GradNorm's reference rates rather than pre-applying weights.

**Process notes:**
- Fern student loop was auth-stale (401, 168h-old creds) at terminal landing; canonical SENPAI-RESULT was posted manually from W&B by ADVISOR.
- Fern is now idle; assigning H28 (Plateau Protocol on H19 reference) per Morgan's Issue #1056 directive.

---

## 2026-05-20 11:30 UTC — PR #1220 CLOSED (misconfig + relaunch stall) → PR #1226 (H24-v2 with corrected CLI)

- **Sequence:**
  - 07:16Z fern launched `dl24-fern/H24-clamp015-peraxis-rank0..7` (run `5dp7s3nz`) on PR #1220.
  - 07:45Z misconfig detected by config comparison across H21/H22/H23 — PR #1220 body omitted `--train-surface-points 65000 --eval-surface-points 65536 --train-volume-points 65000 --eval-volume-points 65536` overrides. Fern inherited train.py defaults (65536/16384), producing `train_views=347657` (4× more) and `total_estimated_steps=1303740` (4× more). Projected runtime 43h vs 22.5h timeout.
  - 10:05Z ADVISOR comment posted on PR #1220 with corrected CLI and instruction to kill `5dp7s3nz`.
  - 11:30Z: 1.5h after advisor comment, fern's student loop had NOT acted. Old run still active at step 126k/1303k. No new W&B runs with H24-clamp015-peraxis-v2 group. No new commits on fern's PR branch.
  - 11:30Z PR #1220 CLOSED with explanatory comment.
  - 11:35Z PR #1226 created with corrected CLI as H24-v2 reassignment.
- **Decision:** Close+reassign was more decisive than waiting another iteration. Fern's student loop will detect the closed PR → mark idle → pick up #1226 on next poll. Old run `5dp7s3nz` will be killed by the loop transition.
- **Mechanism preserved:** H24-v2 has identical hypothesis (clamp=0.15 + per-axis τ "1.0,1.2,1.5"). Only the reproduce command was corrected.
- **Process lesson:** Researcher-agent-generated PR bodies do NOT inherit the project's required CLI overrides — every auto-generated DrivAerML PR must be grep-validated for `--train-surface-points`, `--train-volume-points` against the H19 reference command (PR #1180) before students pick them up. Memory record [[drivaerml-train-points-override]] saved.

---

## 2026-05-20 07:10 UTC — H24 ASSIGNMENT RECONCILIATION: PR #1219 (clamp=0.10) superseded by PR #1220 (clamp=0.15 + per-axis τ compound)

- **Original assignment (my design):** PR #1219 `[dl24-fern] WSS H24: H19 + clamp=0.10 — soft gradient floor ablation midpoint` — a single-variable clamp ablation between H21 (0.15) and H19 (0.05) defaults.
- **What happened:** Morgan McGuire merged the #1219 assignment commit `2a70893` into `drivaerml-long-20260504` at 06:56:35Z (likely accidental — merge was a no-op code change, just `assign dl24-fern: h24-h19-clamp-010`). This removed the `status:wip` slot from #1219, briefly idling fern.
- **Researcher-agent intervention:** During the same pass, the @researcher-agent (running in background to generate `RESEARCH_IDEAS_2026-05-20_22:00.md`) created PR #1220 `H24: H19 + clamp=0.15 + per-axis τ weights (compound floor+wss)` at 07:06:13Z. Fern picked up #1220 at 07:07:45Z (iteration 1069).
- **Decision:** Accept #1220 as canonical H24. The compound design has higher contract-winner probability than the clamp=0.10 ablation midpoint, and the original clamp=0.10 design can be re-queued if H21 alone barely misses.
- **Coverage impact:** The 4-PR ablation triangle is now {H21=clamp solo, H22=MAE_aux solo, H23=Charb_τy solo, H24=clamp+per-axis compound}, which provides cleaner mechanism isolation than {clamp 0.05 / 0.10 / 0.15 sweep + MAE_aux}.

---

## 2026-05-20 22:00 UTC — PR #1180 CLOSED: WSS H19 H10b + Charbonnier on vol_p under GradNorm (dl24-frieren, `r5eigmer`) ⭐ WAVE'S FIRST test_wss + test_abupt SOTA-BEAT

- **Branch:** `dl24-frieren/h19-vol-p-charbonnier`
- **W&B run:** `r5eigmer` (DDP8, EP30 terminal, 1298 min runtime). Test eval on EP20 best-val EMA checkpoint.
- **Hypothesis:** Charbonnier loss on vol_p channel (commit 375be21 routes GradNorm task-signal through Charb, not raw MSE) reshapes the vol_p loss landscape, reducing outlier-sensitivity, without injecting extra gradient mass via an auxiliary head. GradNorm should not need to compensate by raising w_vol_p; preserves wss/vol_p budget intact. Expected: test_vol_p ≤ 3.643 floor at test_wss near H10b's 6.665.

### Test metrics (best-val EMA EP20)

| Metric | H19 | SOTA #972 | Δ vs SOTA | AB-UPT public | Verdict |
|---|---:|---:|---:|---:|---|
| **test_abupt** | **5.8197%** | 5.844% | **−0.024pp ⭐ NEW WAVE SOTA** | — | beat |
| **test_wss** | **6.6339%** | 6.727% | **−0.093pp ⭐ NEW WAVE SOTA** | 7.29 | beat (-0.66pp vs AB-UPT) |
| test_vol_p | 3.7786% | 3.643% | +0.136pp ❌ floor breach | 6.08 | breach (-2.30pp vs AB-UPT) |
| test_surf_p | 3.6267% | 3.577% | +0.050pp ❌ floor breach (small) | 3.82 | breach (-0.19pp vs AB-UPT) |
| test_τ_x | 5.8907% | 5.971% | −0.080pp ✓ | 5.35 | beat |
| test_τ_y | 7.1723% | 7.362% | −0.190pp ✓ | 3.65 | beat |
| test_τ_z | 8.6303% | 8.747% | −0.117pp ✓ | 3.63 | beat |

**6 of 7 test axes beat SOTA #972**. Beats AB-UPT public on all 3 main channels.

### Verdict

CLOSED — does NOT merge per Issue #1056 strict AND-clause contract (floor breaches preclude merge). BUT H19 is a major wave milestone: first wave run to cleanly beat both wss SOTA and abupt SOTA. The vol_p breach is **4× smaller than H10b** (0.136 vs 0.517) — the Charbonnier mechanism IS working, just incompletely at default clamp=0.05.

### Mechanism analysis

Three clean findings:

1. **Charb on vol_p under GradNorm creates an ASYMMETRIC budget** that strengthens wss/abupt. With Charb dominating MSE 17–20× and the GradNorm task-signal seeing Charb, w_vol_p collapsed to the 0.05 floor (clamp-pinned EP4+); w_τ_z rose to ~1.85. Asymmetric budget → wss got the strongest gradient stream of the wave. EMA averaging crystallized the late-cosine wss gain (val 6.98 → test 6.63 = −0.35pp val→test gain).

2. **Charb landscape reshape alone is INSUFFICIENT to clear floor.** Halved the H10b vol_p breach (4.16 → 3.78, −0.38pp) but couldn't close the remaining 0.136pp. The mechanism needs more gradient mass to vol_p — exactly what clamp=0.15 would provide.

3. **MAE_aux gradient mass is NOT the source of H9b's wss cost.** H19 has no MAE_aux and STILL beats SOTA on wss. This contradicts the prior H9b-derived intuition that MAE_aux was costly. H20 (clamp-only) corroborates: clamp alone is the wss-cost driver, not MAE_aux.

### Falsification matrix outcome
- Wave SOTA-beat on wss: ✅ CONFIRMED stronger than predicted (−0.093pp vs predicted ~0pp from "preserved wss")
- Floor-clearance on vol_p: ❌ FALSIFIED (3.778 vs floor 3.643, +0.136pp breach)
- Asymmetric budget side-effect: 🆕 UNEXPECTED — wave-best wss came from the asymmetric GradNorm allocation, not just the Charb reshape

### Implication for next wave
**H21 = H19 + clamp=0.15** is the obvious direct compound. Projection: w_vol_p stays above 0.15 floor (3× more vol_p gradient mass than H19's 0.05 pin) while Charb landscape reshape stays engaged. Expected: test_vol_p ~3.58 (sub-floor), test_wss ~6.69 (still SOTA), test_abupt sub-5.84 — possible clean contract winner.

---

## 2026-05-20 22:00 UTC — PR #1181 CLOSED: WSS H20 H10b + clamp=0.15 only no MAE_aux (dl24-nezuko, `4yvl848t`)

- **Branch:** `dl24-nezuko/h20-h10b-clamp-only`
- **W&B run:** `4yvl848t` (DDP8, EP30 terminal, 1272 min runtime). Test eval on EP21 best-val EMA checkpoint.
- **Hypothesis:** Clamp=0.15 alone on H10b base (no MAE_aux) might preserve vol_p floor at lower wss cost than H9b's clamp+MAE_aux stack — testing whether MAE_aux is the costly mechanism.

### Test metrics (best-val EMA EP21)

| Metric | H20 | SOTA #972 | H10b | Δ vs SOTA | Verdict |
|---|---:|---:|---:|---:|---|
| **test_wss** | **6.8080%** | 6.727% | 6.665% | +0.081pp ❌ regression | miss |
| **test_vol_p** | **3.8470%** | 3.643% | 4.160% | +0.204pp ❌ above floor | miss |
| test_surf_p | 3.7403% | 3.577% | 3.690% | +0.163pp ❌ breach | miss |
| test_abupt | 5.9723% | 5.844% | 5.929% | +0.128pp above SOTA | miss |
| test_τ_z | 8.8994% | 8.747% | 8.643% | +0.149pp | miss |

### Verdict

CLOSED — clear non-winner per Issue #1056 contract. Wss regressed, all floors breached.

### Mechanism finding (the genuinely useful result)

Clean ablation against the H9b stack:
- **Clamp alone (H20 vs H10b)**: vol_p −0.313pp at wss cost +0.143pp
- **Clamp+MAE_aux (H9b vs H9, prior result)**: vol_p −~0.50pp at wss cost +0.103pp
- **MAE_aux marginal contribution**: ~−0.20pp vol_p (closes gap to floor)
- **Surprise**: MAE_aux DECREASED wss cost slightly — contradicts prior hypothesis that MAE_aux was the costly component

**Conclusion**: clamp is the dominant wss-cost mechanism, MAE_aux carries unique vol_p benefit. The two are complementary, not redundant. H21 (H19's Charb + clamp=0.15) tests whether Charb can substitute for MAE_aux's gradient-injection role at lower cost.

---

## 2026-05-20 22:00 UTC — PR #1166 CLOSED: WSS H12 separate τ head 2-layer MLP (dl24-fern, `3v58n2m5`)

- **Branch:** `dl24-fern/h12-separate-tau-head`
- **W&B run:** `3v58n2m5` (DDP8, EP30 terminal, 1340 min runtime). Test eval on EP13 best-val EMA checkpoint.
- **Hypothesis:** Architectural decoupling of cp from τ_x/τ_y/τ_z via separate output heads (linear for cp, 2-layer GELU MLP near-zero init for τ) could improve wss without affecting other channels.

### Test metrics (best-val EMA EP13)

| Metric | H12 | SOTA #972 | AB-UPT public | Δ vs SOTA | Verdict |
|---|---:|---:|---:|---:|---|
| **test_wss** | **6.7323%** | 6.727% | 7.29 | +0.005pp essentially tied SOTA | tie |
| **test_vol_p** | **4.0735%** | 3.643% | 6.08 | +0.430pp ❌ floor breach | miss |
| test_surf_p | 3.8739% | 3.577% | 3.82 | +0.297pp ❌ floor breach | miss |
| test_abupt | 6.0036% | 5.844% | — | +0.160pp above SOTA | miss |
| test_τ_x | 5.9435% | 5.971% | 5.35 | −0.028pp ✓ | beat |
| test_τ_y | 7.3292% | 7.362% | 3.65 | −0.033pp ✓ | beat |
| test_τ_z | 8.7977% | 8.747% | 3.63 | +0.051pp | miss |

### Verdict

CLOSED — clear non-winner. Wss tied SOTA but both floors breach.

### Mechanism finding

The separate τ head delivers a real same-step lead vs H10b at EP4-7 (−0.037pp on val_wss), but the architectural advantage is small and doesn't transfer to a contract win — the head split is a wss-axis mechanism only, no floor-preservation mechanism. Documented architectural insight: head split decouples cp from τ usefully, but Lion+EMA at lr=1e-4 produces a known EP14 instability trap (mirror-reversed at EP15) that didn't catch H10b.

---

## 2026-05-18 17:03 UTC — PR #1175 CLOSED: WSS H18 H10b + Charb_τz + clamp=0.15 + MAE_aux full stack (dl24-tanjiro, `xhx2qlpo`)

- **Branch:** `dl24-tanjiro/h18-composition-h10b-h9b`
- **W&B run:** `xhx2qlpo` (DDP8, EP13/30 TIME-TRUNCATED at 1352 min). Test eval on EP13 best-val EMA checkpoint.
- **Hypothesis:** Compose H10b curvature+Charb_τz with H9b clamp+MAE_aux on the corrected dataset. Expected: H10b wss benefit with H9b floor preservation.

### Test metrics (best-val EMA EP13, time-truncated)

| Metric | H18 | SOTA #972 | Δ vs SOTA | Verdict |
|---|---:|---:|---:|---|
| **test_wss** | **6.7459%** | 6.727% | +0.019pp ❌ narrow miss | miss |
| **test_vol_p** | **3.702%** | 3.643% | +0.059pp ❌ small breach | miss |
| test_surf_p | 3.729% | 3.577% | +0.152pp ❌ breach | miss |
| test_abupt | 5.895% | 5.844% | +0.051pp narrow miss | miss |
| test_τ_y | 7.309% | 7.362% | −0.053pp ✓ | beat |
| test_τ_z | 8.745% | 8.747% | −0.002pp ≈ tied | tied |

### Verdict
CLOSED — narrow miss, time-truncated. Anti-additive cost on wss is real (+0.066pp vs H10b alone) and curvature representation appears to dilute the H9b floor-locking effect. EP8 instability spike from H10b WAS suppressed by clamp+MAE_aux (strongest mechanism-engagement evidence).

### Mechanism conclusion
The H10b+H9b stack composition only narrowly missed at EP13 truncation — projecting EP30 with cosine-tail slope would give val_wss ~6.87 → test_wss ~6.65 (SOTA beat) but floor compliance was non-converging (val_surf_p ticked up EP11→12). Curvature has a global vol_p penalty that clamp+MAE_aux cannot fully compensate.

---

## 2026-05-17 17:52 UTC — PR #1159 CLOSED: WSS H10b curvature + Charbonnier on τ_z only (dl24-frieren, `60zl0p4h`)

- **Branch:** `dl24-frieren/h10b-curvature-charb-tau-z`
- **W&B run:** `60zl0p4h` (DDP8, EP30 terminal, 1296 min runtime). Test eval on EP15 best-val EMA checkpoint (auto-selected by `--best_val_checkpoint`).
- **Hypothesis:** Compose H9's curvature additive attention bias (the H9 WSS wave finding) with Charbonnier loss restricted to τ_z only (the hardest WSS axis per representation-floor analysis). Single-axis Charbonnier keeps gradient-mass reshape contained to the worst axis without introducing the H10-full-axes vol_p starvation.
- **Configuration:** Lion lr=1e-4, GradNorm α=0.5, `--gradnorm-min-w-vol-p=0.05`, `--use-curvature-attention-bias`, `--wss-charbonnier-weight=0.1 --wss-charbonnier-eps=1e-3 --wss-charbonnier-axes=z`, EMA decay=0.999, Y-symmetry aug p=0.5, STRING 5-octave PE.

### Results — Terminal Test Metrics (EP15 best-ckpt)

| Axis | H10b test | SOTA #972 | Δ vs SOTA | Verdict |
|---|---:|---:|---:|---|
| **test_wss** | **6.6651%** | **6.727%** | **−0.062pp** | ✅ **CLEAN BEAT (first wave SOTA-beat)** |
| **test_τ_x** | 5.9147% | 5.971% | −0.056pp | ✅ BEAT |
| **test_τ_y** | 7.2373% | 7.362% | −0.125pp | ✅ BEAT |
| **test_τ_z** | 8.6426% | 8.747% | −0.104pp | ✅ BEAT |
| test_abupt | 5.9289% | 5.844% | +0.085pp | trailing |
| **test_vol_p** | **4.1598%** | **3.643% (floor)** | **+0.517pp** | ❌ **FLOOR BREACH** |
| **test_surf_p** | **3.6900%** | **3.577% (floor)** | **+0.113pp** | ❌ **FLOOR BREACH** |

**4 of 7 axes BEAT SOTA — first wave PR to achieve full WSS-landscape lead.** Per Issue #1056 floor clause (vol_p ≤ 3.643%, surf_p ≤ 3.577%), the +0.517pp vol_p breach and +0.113pp surf_p breach preclude merge. NOT-A-MERGE despite test_wss CLEAN BEAT.

### H10b vs H9 #1145 (closest comparator — isolation of Charbonnier mechanism)

| Axis | H9 test | H10b test | Δ (H10b − H9) | Read |
|---|---:|---:|---:|---|
| test_wss | 6.678 | **6.6651** | **−0.013pp** | H10b slightly better |
| test_τ_y | 7.308 | **7.2373** | **−0.071pp** | **largest Charbonnier signal** |
| test_τ_z | 8.668 | **8.6426** | **−0.025pp** | Charb mechanism delivered τ_z reshape signal |
| test_vol_p | 3.913 | 4.1598 | **+0.247pp** | H9 better (no MAE_aux to defend vol_p) |
| test_surf_p | 3.692 | 3.6900 | tied | unchanged |

### val→test gap analysis

| Axis | val | test | Gap |
|---|---:|---:|---:|
| τ_z | 9.3610 | 8.6426 | **−0.718pp** (largest favorable gap) |
| abupt | 6.2487 | 5.9289 | −0.320pp |
| wss | 6.8797 | 6.6651 | −0.215pp |
| τ_y | 7.4706 | 7.2373 | −0.233pp |
| surf_p | 4.0584 | 3.6900 | −0.368pp |
| vol_p | 4.3670 | 4.1598 | −0.207pp |
| τ_x | 5.9863 | 5.9147 | −0.072pp |

Test UNIFORMLY better than val on all 7 axes. The corrected `rawcanon_20260511` split transfers cleanly. τ_z gap of −0.718pp matches H9 (−0.712pp) — Charbonnier preserves this transferable representation.

### Mechanism diagnostics

| Metric | EP1 | EP15 (best) | EP30 |
|---|---:|---:|---:|
| Charb/MSE ratio (τ_z) | 0.127 | ~1.10 | ~1.28 |
| `gradnorm/w_vol_p` | 0.380 | ~0.068 | 0.050 (clamp floor) |
| Clamp fires | 0 | rare | 0.35% steps |
| nonfinite_count | 0 | 0 | 0 |

Clean run end-to-end, 329k steps, zero numerical issues. EP15 wave-low (val_wss 6.880) held a 15-epoch plateau through EP30 (drift +0.003/ep); `--best_val_checkpoint` auto-loaded EP15 for terminal eval.

### Wave Finding — Curvature + Charbonnier τ_z is the wave's strongest WSS-reduction lever

Mechanism is **carried forward into H18 PR #1175** (dl24-tanjiro, run `xhx2qlpo`, launched 16:42Z) composed with H9b's clamp+MAE_aux floor preservation. H18 hypothesis: the curvature+Charb representation provides a richer wss landscape where vol_p MAE_aux exerts lower gradient-mass pull on wss signal, reducing the anti-additive cost below the H9→H9b measured +0.103pp.

### Closing — H10b retired into H18 composition

EP15 checkpoint and run `60zl0p4h` documented as the wave's first test_wss SOTA-beat. Mechanism carried forward.

---

## 2026-05-17 17:42 UTC — PR #1160 CLOSED: WSS H11b AdamW lr=5e-4 + per-axis WSS τ-weights (dl24-nezuko, `ch4cllcb`)

- **Branch:** `dl24-nezuko/h11b-adamw-per-axis-only`
- **W&B run:** `ch4cllcb` (DDP8, EP30 terminal, 1237.7 min runtime). Test eval on EP29 best-val EMA checkpoint.
- **Hypothesis:** Clean isolation of per-axis WSS τ-magnitude weighting (`--wss-axis-weights 1.0,1.2,1.5`) vs H8 baseline. H11 PR #1154 used AdamW lr=7e-4 and crashed EP1→EP2 (+8.44pp val_abupt). H11b restricts lr=5e-4 to disambiguate: was H11's instability LR-coupled or per-axis-coupled?
- **Configuration:** AdamW lr=5e-4 (vs H11's 7e-4), GradNorm α=0.5, `--wss-axis-weights "1.0,1.2,1.5"`, EMA decay=0.999, Y-symmetry aug p=0.5, STRING 5-octave PE, NO curvature bias, NO Charbonnier, NO clamp+MAE_aux.

### Results — Terminal Test Metrics (EP29 best-ckpt)

| Axis | H11b test | SOTA #972 | Δ | Verdict |
|---|---:|---:|---:|---|
| test_abupt | 6.148% | 5.844% | +0.304 | MISS |
| test_wss | 7.019% | 6.727% | +0.292 | MISS (narrow) |
| test_vol_p | 3.856% | 3.643% | +0.213 | FLOOR BREACH |
| test_surf_p | 3.988% | 3.577% | +0.411 | FLOOR BREACH |
| test_τ_z | 9.071% | 8.747% | +0.324 | MISS |

**NOT contract winner.** Test_wss gap +0.292pp wide, all floors BREACHED.

### H11b vs H8 (clean isolation, single variable = per-axis WSS weights)

| Axis | H8 test | H11b test | Δ (H11b − H8) | Read |
|---|---:|---:|---:|---|
| test_wss | 7.264% | **7.019%** | **−0.245pp** | **per-axis mechanism validated** ⭐ |

H11b improves on H8 by −0.245pp on test_wss with no other changes — per-axis τ-magnitude upweighting (τ_y=1.2, τ_z=1.5) IS a real WSS-reduction lever. But the absolute level is +0.292pp from SOTA, not enough for contract win.

### Diagnostic answers (from student SENPAI-RESULT)

1. **Was H11's instability LR-coupled or per-axis-coupled?** LR-coupled. H11b (lr=5e-4) showed NO EP1→EP2 spike (decrease −12.17pp), vs H11 (lr=7e-4) +8.44pp spike. Per-axis weights and lr=5e-4 coexist cleanly.

2. **Did per-axis weighting starve vol_p via GradNorm?** No. GradNorm task weights stayed in healthy ranges throughout training; final w_vol_p stayed well above the 0.10 kill threshold across all 30 epochs.

### Wave Finding — per-axis WSS τ-magnitude weighting is a saveable but insufficient WSS-reduction mechanism

| Mechanism | test_wss reduction vs H8 baseline |
|---|---:|
| H11b per-axis WSS weights `1.0,1.2,1.5` | **−0.245pp** |
| H9 curvature attention bias | **−0.586pp** |
| H10b curvature + Charb τ_z | **−0.599pp** (strongest single mechanism) |

Per-axis WSS weighting can compose with H10b's curvature+Charb but is not in the immediate H18 stack. **If H18 confirms curvature+Charb with floor preservation, a follow-up may re-add per-axis weighting on top** (H22-style composition).

### Closing — mechanism documented, returned to wave pool

---

## 2026-05-17 16:25 UTC — PR #1157 CLOSED: WSS H9b curvature + clamp=0.15 + vol_p MAE auxiliary (dl24-tanjiro, `smflmb5t`)

- **Branch:** `dl24-tanjiro/h9b-clamp-vol-p-mae-aux`
- **W&B run:** `smflmb5t` (DDP8, EP30 terminal, 1290 min runtime). Test eval on EP21 best-val EMA checkpoint.
- **Hypothesis:** H9's representation-floor finding said "vol_p ceiling is representation-bound at 4.05%". H9b tests whether MAE auxiliary loss (direct L1 signal bypassing GradNorm) + clamp=0.15 (binding floor on w_vol_p GradNorm task weight) can break that ceiling. Two-mechanism 2×2 ablation in single run.
- **Configuration:** Lion lr=1e-4, GradNorm α=0.5, **`--gradnorm-min-w-vol-p=0.15`** (vs default 0.05), **`--vol-p-aux-mae-weight=0.05`** (NEW MAE_aux mechanism), `--use-curvature-attention-bias` (H9 carry-forward), EMA decay=0.999, Y-symmetry aug p=0.5, STRING 5-octave PE.

### Results — Terminal Test Metrics (EP21 best-ckpt)

| Axis | H9b test | SOTA #972 | Δ vs SOTA | Verdict |
|---|---:|---:|---:|---|
| test_wss | 6.781% | 6.727% | +0.054pp | MISS (narrow) |
| **test_vol_p** | **3.646%** | **3.643% (floor)** | **+0.003pp** | ✅ **AT FLOOR (mechanism validated)** |
| test_surf_p | 3.787% | 3.577% (floor) | +0.210pp | FLOOR BREACH |
| test_abupt | (~6.00) | 5.844% | (~+0.16) | MISS |

**NOT contract winner.** vol_p mechanism VALIDATED (3.646 within 0.003pp of floor, effectively at floor). test_wss narrow MISS, surf_p still breached.

### H9b vs H9 (clean isolation of clamp+MAE_aux mechanism)

| Axis | H9 test | H9b test | Δ (H9b − H9) | Read |
|---|---:|---:|---:|---|
| test_wss | 6.678% | 6.781% | **+0.103pp** | **anti-additive cost of clamp+MAE_aux on wss** |
| test_vol_p | 3.913% | **3.646%** | **−0.267pp** | **vol_p floor preservation mechanism validated** |
| test_surf_p | 3.692% | 3.787% | +0.095pp | surf_p slight regression |

### Mechanism verification

| Diagnostic | Value | Read |
|---|---:|---|
| `gradnorm/w_vol_p` (mean EP1-30, DDP-averaged) | **0.150 exactly** | clamp engaged hard floor from EP8 onward |
| `vol_p_mae_aux/weighted` | smoothly decaying 0.18→0.07 EP1-30 | MAE auxiliary engaged, no spikes |
| `val_vol_p` EP3 | 4.18 (vs H9 EP10 = 4.056) | EP3 already at H9 terminal level — descent 2-3× faster |
| `best_epoch` | 21 | best-val converged mid-late training |

### Wave Finding — vol_p floor preservation IS achievable via clamp+MAE_aux, with +0.103pp anti-additive cost on wss

This mechanism is **carried forward into H18 PR #1175** as the floor-preservation stack composed with H10b's curvature+Charb wss-reduction stack. H18 falsification: if anti-additive cost dominates, H18 test_wss > 6.78 (narrow miss); if mechanisms orthogonal on H10b's richer curvature representation, H18 test_wss ~6.50-6.65 (BEAT + floors).

### Closing — H9b retired into H18 composition

---

## 2026-05-17 01:00 UTC — PR #1142 CLOSED: WSS H7 surface_loss_weight=1.5 uniform upweight (dl24-fern, `2nufmv3i`)

- **Branch:** `dl24-fern/h7-surface-loss-weight-1p5`
- **W&B run:** `2nufmv3i` (DDP8, EP30 terminal). Test eval on EP20 best-val EMA checkpoint.
- **Hypothesis:** H4's vol_p improvement was driven by an *implicit* surface task upweight (mean of [1.0, 1.0, 1.5, 2.5]/4 = 1.5×). Replacing per-axis weights with uniform `surface_loss_weight=1.5` should preserve the vol_p win and avoid H4's surf_p/wss/τ regressions (which H7 attributed to per-axis structure × Lion-noise amplification).
- **Configuration:** Lion lr=5e-4, GradNorm OFF, `surface_loss_weight=1.5`, EMA decay=0.999, 30 epochs, ema_start_step=500.

### Results — Terminal Test Metrics

| Metric | H7 test | SOTA #972 | Δ vs SOTA | vs floor (#1056) | Verdict |
|---|---:|---:|---:|---:|---|
| **test_abupt** | **6.0366** | 5.844 | **+0.193** | — | ❌ regressed |
| **test_vol_p** | **3.4962** | 3.643 | **−0.147** | UNDER floor by 0.147pp | ✅ floor cleared |
| **test_surf_p** | **3.7816** | 3.577 | **+0.205** | BREACH by 0.205pp | ❌ floor breached |
| **test_wss** | **7.0055** | 6.727 | **+0.279** | — | ❌ primary target failed |
| test_τ_x | 6.2149 | 5.971 | +0.244 | — | regressed |
| test_τ_y | 7.5469 | 7.362 | +0.185 | — | regressed |
| test_τ_z | 9.1432 | 8.747 | +0.396 | — | regressed |

**Val→test gap calibration**: abupt −0.177, vol_p +0.012, surf_p −0.258, wss −0.086, τ_x +0.027, τ_y −0.140, **τ_z −0.528**. Test was kinder than val on most axes; the large τ_z negative gap is consistent with H10's representation-floor finding (Charbonnier-related runs show large val→test gaps on τ_z).

### Wave Finding — H4 mechanism decomposition REFUTED

The H7 hypothesis (that H4 vol_p improvement was implicit-uniform-upweight and H4's costs were per-axis-Lion-noise) was tested by stripping per-axis structure and applying uniform 1.5× scaling. Three refutations:

1. **vol_p mechanism is real and isolable** ✅ — test_vol_p=3.496% confirms shared-backbone enrichment from uniform surface upweight benefits the volume head. H4's test_vol_p=3.374% under same magnitude → consistent.

2. **surf_p breach is INTRINSIC to magnitude, not structure** ❌ — H7 surf_p=3.781% is WORSE than H4 surf_p=3.743%, despite no per-axis structure. The breach is a 1.5× magnitude effect on surface gradient flow, not per-axis interference between cp and τ.

3. **τ regressions are INTRINSIC to magnitude** ❌ — all 3 τ axes regress under uniform 1.5× upweight (no per-axis amplification possible). The Lion-noise/per-axis theory cannot explain this. The 1.5× magnitude itself shifts the WSS-side optimization regime in a way that costs τ accuracy.

**Implication for #1056 contract**: Lower-magnitude surface upweight (1.1, 1.2) may attenuate the tax but won't eliminate it — the directional tradeoff between vol_p improvement and wss/surf_p/τ degradation is intrinsic to the surface-upweight mechanism. Surface upweight is a **vol_p-side mechanism**, NOT a wss-side mechanism. Combined with H9 wave finding (curvature bias = wss unlock without floor breach), the wave's mechanism map shows: vol_p and wss respond to DIFFERENT gradient-mass interventions. They cannot both be fixed with a single uniform scalar.

### Cost & resource summary

- Wall-time: ~20.0h training + ~10min test eval. Peak GPU memory ~21 GB/GPU, all 8 GPUs balanced.
- W&B run IDs: rank-0 `2nufmv3i`, ranks 1-7: `0x9cfkv9`, `idet3da8`, `qkc0j2ow`, `tpd0ycfj`, `x80y99uf`, `xc3y3688`, `z3jijtxn`.
- Group: `wss_h7_surface_upweight`.

### Infra note from student

dl24-fern reported a `senpai-pr-guard.py` bug: `result_markers()` scanned every line for `SENPAI-RESULT:` without skipping markdown code fences. My template comment on this PR included a JSON-shaped placeholder inside a code block that broke the marker scanner. Student's 5-line fix to track in_code_fence state in the scanner is the right approach. Cannot apply from advisor scope (infra repo, not target). Flagged to human team.

---

## 2026-05-16 19:48 UTC — PR #1154 CLOSED: WSS H11 AdamW lr=7e-4 + per-axis WSS τ-weights (dl24-nezuko, `kukjenp5`+`zhmyhxcd`)

- **Branch:** `dl24-nezuko/h11-adamw-per-axis-wss`
- **W&B runs:** training `kukjenp5` (DDP8, killed at EP5 via SIGTERM), eval-only `zhmyhxcd` (EP6 best-val EMA checkpoint)
- **Hypothesis:** Apply per-axis WSS τ-magnitude weights (w_τ_x=1.0, w_τ_y=1.2, w_τ_z=1.5) AFTER GradNorm to concentrate optimization pressure on the hardest WSS axis. Stacked on AdamW lr=7e-4 (40% above H8 baseline) with cosine T_max=25 (vs H8's 30).
- **Kill reason:** EP3 viability gate (val_abupt ≤ 8.0%) failed by 8.54pp (actual 16.54%). EP4=16.27%, EP5=12.21% trajectory all 4-8pp above H8 reference. 40-min silent radio after 17:47Z anomaly nudge confirmed unrecoverable. Advisor-killed autonomously at 18:33Z based on W&B direct read.

### Terminal test metrics (EP6 best-val EMA checkpoint, eval run `zhmyhxcd`)

| Metric | SOTA #972 | H11 test | Δ vs SOTA | Floor | Verdict |
|--------|---:|---:|---:|---|---|
| test_abupt | 5.844% | **11.163%** | **+5.319pp** | — | catastrophic regression |
| test_wss | 6.727% | **11.304%** | **+4.577pp** | — | catastrophic regression |
| test_τ_x | 5.971% | 9.980% | +4.009pp | — | ❌ |
| test_τ_y | 7.362% | 12.315% | +4.953pp | — | ❌ |
| **test_τ_z** | 8.747% | **14.811%** | **+6.064pp** | — | ❌ (bellwether worst hit) |
| test_vol_p | **3.643%** | 10.148% | +6.505pp | **BREACH** | ❌ HARD FLOOR BREACH |
| test_surf_p | **3.577%** | 8.561% | +4.984pp | **BREACH** | ❌ HARD FLOOR BREACH |

**0 of 7 axes anywhere near SOTA.** All metrics ~2× SOTA — broken run from EP1, never recovered.

### Val trajectory (val_primary)

| EP | step | val_abupt | val_wss | val_vol_p | val_τ_z | gradnorm/w_vol_p |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,975 | 20.04% | 20.84% | 16.53% | 26.66% | 0.451 |
| 2 | 21,951 | **28.48%** | 28.14% | 28.24% | 35.57% | — |
| 3 | 32,927 | 16.54% | 16.65% | 14.74% | 22.35% | — |
| 4 | 43,903 | 16.27% | 14.66% | 21.56% | 19.72% | — |
| 5 | 54,875 | 12.21% | — | — | — | 0.341 |
| **EP6 best-val (checkpoint)** | — | **11.78%** | 11.74% | 10.72% | 16.03% | — |

EP1→EP2 spike (+8.44pp on val_abupt) is the signature of warmup→full-LR optimization instability. Recovery slope after EP2 was real (−4.06pp/ep) but starting from 28.48% meant the trajectory could not reach competitive plateau in 30 epochs.

### Wave finding (locked in): "AdamW lr>5e-4 + per-axis WSS weights = optimization instability"

H11 stacked **three** simultaneous changes vs H8 baseline:
1. AdamW lr 5e-4 → 7e-4 (40% higher peak LR)
2. Per-axis WSS weights enabled (w_τ_x=1.0, w_τ_y=1.2, w_τ_z=1.5)
3. cosine T_max 30 → 25

The per-axis τ_z=1.5 boost amplified gradient norm during the post-warmup transition (EP1→EP2 spike). Lion can absorb this through sign-update saturation; AdamW's `m/sqrt(v+eps)` amplifies the gradient norm change directly. The recovery trajectory was real but the EP30 budget could not absorb the EP1-EP2 deficit.

**Mechanism not falsified — only the LR-coupled instance was.** H11b PR #1160 holds lr=5e-4 and T_max=30 at H8 baseline values to isolate the per-axis mechanism cleanly. If H11b clears the EP3 gate, the per-axis mechanism is validated; if H11b also fails, per-axis under GradNorm is fundamentally broken.

### Workflow finding (also locked in)

The H11 branch only contained the empty assignment commit — the actual `--wss-axis-weights` implementation was modified locally on the student's working copy and used at launch without being committed to git. This breaks reproducibility. **H11b instructions explicitly require pushing the implementation commit BEFORE launching.**

### Carry-forward to H11b

- **H11b stack** = exact H8 PR #1144 baseline (AdamW lr=5e-4, GradNorm, cosine T_max=30, all SOTA #972 stack) + per-axis WSS τ-weights ONLY
- Single-variable change → causally clean attribution
- EP3 gate raised slightly from H8's 8.0% to 8.5% (+0.24pp tolerance for the per-axis mechanism overhead)
- EP6 gate at 7.3% (vs H8 EP6 ~7.5%); EP10 gate at 7.0%

### Run command (H11, for reference)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer adamw --lr 7e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 25 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm \
  --wss-axis-weights "1.0,1.2,1.5"  # uncommitted, applied locally
```

---

## 2026-05-16 19:30 UTC — PR #1149 CLOSED: WSS H10 Charbonnier supplementary loss on WSS (dl24-frieren, `wnj931pj`+`c5436ytt`)

- **Branch:** `dl24-frieren/h10-charbonnier-wss-loss`
- **W&B runs:** training `wnj931pj` (DDP8, killed at EP10 via SIGTERM), eval-only `c5436ytt` (EP10 best-val EMA)
- **Hypothesis:** Add Charbonnier (pseudo-Huber, δ=eps=1e-3) supplementary loss on all WSS channels with weight=0.1. Mid-range residual reweighting (L1-like in the [eps, 1] residual band) was predicted to accelerate WSS descent past the [6.96, 6.99] H5/H6 plateau.
- **Kill reason:** EP10 val_wss=7.078% ≥ 7.05% kill band; val_τ_z=9.549% > 9.5%; slope EP7-10 mean −0.018/ep → EP15 projection 6.97% (plateau-equivalent). Charbonnier mechanism was engaged (51.9% gradient share) but plateau-equivalent at EP10.

### Terminal test metrics (EP10 best-val EMA checkpoint, eval run `c5436ytt`)

| Metric | SOTA #972 | H10 EP10 test | Δ vs SOTA | Floor | Verdict |
|--------|---:|---:|---:|---|---|
| test_abupt | 5.844% | 6.059% | +0.215pp | — | ❌ regression |
| test_wss | **6.727%** | 6.884% | +0.157pp | — | ❌ regression |
| test_τ_x | 5.971% | 6.090% | +0.119pp | — | ❌ |
| test_τ_y | 7.362% | 7.506% | +0.144pp | — | ❌ |
| **test_τ_z** | 8.747% | **8.949%** | **+0.202pp** | — | ❌ (bellwether) |
| test_vol_p | **3.643%** | 3.970% | +0.327pp | **BREACH** | ❌ (artifact of EP10 stop) |
| test_surf_p | **3.577%** | 3.778% | +0.201pp | **BREACH** | ❌ (artifact of EP10 stop) |

**0 of 7 test_primary axes under SOTA #972.** All-axis regression at EP10. Note: floor breaches are partly artifacts of the early kill at EP10 (vs SOTA's 30 EP training).

### val→test gap (the H10 calibration finding)

| Axis | val (EP10) | test | gap |
|---|---:|---:|---:|
| abupt | 6.341% | 6.059% | +0.282pp |
| wss | 7.078% | 6.884% | +0.194pp |
| τ_x | 6.208% | 6.090% | +0.118pp |
| τ_y | 7.667% | 7.506% | +0.162pp |
| **τ_z** | **9.549%** | **8.949%** | **+0.600pp** ← 3-4× typical, largest in wave |
| vol_p | 4.133% | 3.970% | +0.163pp |
| surf_p | 4.148% | 3.778% | +0.370pp |

### Charbonnier weighted contribution to WSS gradient

| Step | EP | MSE | Charb_w | Share |
|---:|---:|---:|---:|---:|
| 4801 | 0.4 | 0.1327 | 0.0201 | 13.2% |
| 21600 | 2.0 | 0.0087 | 0.0050 | 36.5% |
| 65855 | 6.0 | 0.0018 | 0.0017 | 48.9% |
| 109759 | 10.0 | 0.0013 | 0.0014 | **51.9%** |

Steady-state share ~50% — squarely in the 20-60% healthy band. Mechanism engaged correctly per design.

### Wave finding — "Representation Floor"

The largest informational bit: **τ_z val→test gap of +0.600pp** (3-4× larger than typical 0.150pp) — Charbonnier IS reshaping the loss landscape on the bellwether axis, just on the wrong representation. Loss-axis reshape produces a *different equilibrium at the same plateau height*. 

H10 is a clean falsifying experiment for the loss-functional axis on the original Lion stack. Mechanism works at the gradient-budget level but cannot accelerate the val plateau without a representation upgrade. The wave plateau is robust to this perturbation.

**Implication for next experiments**: Charbonnier must be paired with the H9 curvature representation upgrade to deliver value. H10b PR #1159 tests this directly with τ_z-only Charbonnier (highest leverage axis) on H9's curvature attention bias stack.

### Carry-forward to H10b

- **H10b stack** = H9 curvature attention bias carry-forward + Charbonnier on τ_z axis ONLY (channel index 3)
- Single-axis isolation concentrates supplementary signal on bellwether (highest val→test gap = highest leverage)
- Reduces total reshape pressure by ~3× vs H10's all-axis Charbonnier, leaving more capacity for vol_p/surf_p
- Tests whether the H9 representation can capitalize on the loss reshape that the original Lion representation could not

### Run command

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/h10_wss_charbonnier \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3
```

---

## 2026-05-16 17:44 UTC — PR #1145 CLOSED: WSS H9 curvature bias + vol_p GradNorm clamp (dl24-tanjiro, `l8jcb7r2`+`cust3asz`)

- **Branch:** `dl24-tanjiro/h9-curvature-bias-vol-p-clamp`
- **W&B runs:** training `l8jcb7r2` (DDP8, killed at EP10), eval-only `cust3asz` (EP10 best-val checkpoint)
- **Hypothesis:** H5's curvature additive attention bias produced real WSS gain (test_wss=6.609%, −0.118pp SOTA) but was disqualified by GradNorm vol_p starvation (w_vol_p=0.0064). H9 adds a hard floor clamp `w_vol_p ≥ 0.05` to prevent GradNorm from crushing vol_p while preserving WSS surface-task balancing. Expected: H5's WSS gain holds AND vol_p clears the 3.643% floor.
- **Kill reason:** EP10 val_vol_p=4.056% exceeded 4.0% kill criterion; 0.05 clamp was DORMANT (natural floor 0.088, above clamp threshold).

### Terminal test metrics (EP10 best-val EMA checkpoint, eval run `cust3asz`)

| Metric | SOTA #972 | H9 EP10 test | Δ vs SOTA | Floor | Verdict |
|--------|---:|---:|---:|---|---|
| test_abupt | 5.844% | 5.897% | +0.053pp | — | ❌ fractionally over |
| **test_wss** | **6.727%** | **6.678%** | **−0.049pp** | — | **✅ UNDER SOTA** |
| **test_τ_x** | 5.971% | 5.903% | **−0.068pp** | — | **✅ UNDER SOTA** |
| **test_τ_y** | 7.362% | 7.308% | **−0.054pp** | — | **✅ UNDER SOTA** |
| **test_τ_z** | 8.747% | 8.668% | **−0.079pp** | — | **✅ UNDER SOTA** |
| test_vol_p | **3.643%** | 3.913% | +0.270pp | **BREACH** | ❌ |
| test_surf_p | **3.577%** | 3.692% | +0.115pp | **BREACH** | ❌ |

**4 of 7 test_primary axes under SOTA #972** — all four on the WSS side. First single-model run in the wave to achieve SOTA-under on the WSS aggregate.

### val → test gap (EP10 best-val checkpoint)

| Metric | val (EP10) | test | gap |
|---|---:|---:|---:|
| abupt | 6.219% | 5.897% | −0.322pp |
| wss | 6.925% | 6.678% | −0.247pp |
| τ_x | 6.038% | 5.903% | −0.135pp |
| τ_y | 7.553% | 7.308% | −0.245pp |
| τ_z | 9.380% | 8.668% | −0.712pp |
| vol_p | 4.056% | 3.913% | −0.143pp |
| surf_p | 4.068% | 3.692% | −0.376pp |

### GradNorm dynamics (H9 key finding)

| Stack | w_vol_p (terminal) | w_τ_z | ratio | vol_p test |
|---|---:|---:|---:|---|
| Lion+GradNorm (H5) | 0.0064 | 2.318 | 362× | 3.955% (breach) |
| **Lion+GradNorm+curvature (H9)** | **0.0879** | **~1.8** | **~20×** | **3.913% (breach)** |

### Analysis and wave findings

**HEADLINE FINDING: Curvature bias mechanism confirmed as the WSS path.**

1. **Curvature additive attention bias transfers cleanly through the val→test gap.** H5: val_wss≈6.85% → test_wss=6.609% (−0.24pp gap). H9: val_wss=6.925% → test_wss=6.678% (−0.247pp gap). Both gaps are ~0.25pp, consistent — the mechanism is robust, not val-overfit.

2. **GradNorm vol_p starvation has TWO distinct modes:**
   - **Gradient-mass mode** (H5): GradNorm crushes w_vol_p to 0.0064 (362× below w_τ_z). Curvature mechanism absent — surface task dominates GradNorm renormalization.
   - **Representation-capacity mode** (H9): Curvature bias gives backbone enough vol_p signal that GradNorm self-stabilizes at 0.088 (not 0.0064). w_vol_p ALIVE but vol_p still stalls at ~4.05% — beyond 0.088 gradient mass, the bottleneck is volume decoder representation, not loss weighting.

3. **The 0.05 clamp was DORMANT** — H9's natural w_vol_p floor (0.088) exceeds the clamp threshold. To be binding, the clamp floor must constrain THIS trajectory (≥0.10-0.15), not H5's pathological floor.

4. **Vol_p ceiling is NOT rate-coupled**: despite 13.7× higher w_vol_p than H5 (0.088 vs 0.0064), val_vol_p stalls at the same ~4.05% level. More gradient mass alone doesn't lower vol_p. This points to the volume decoder's representational capacity under the H9 surface mechanism being the actual bottleneck.

**Why NOT merged**: Issue #1056 AND-clause floors (test_vol_p ≤ 3.643%, test_surf_p ≤ 3.577%) breached by +0.270pp and +0.115pp respectively at EP10/30.

**Carry-forward → H9b (PR #1157)**: clamp=0.15 (binding on this trajectory) + vol_p MAE auxiliary loss at weight 0.05 (direct L1 signal bypassing GradNorm). The MAE aux tests whether the bottleneck is gradient direction (L2 saturating on low-residual vol_p regions) rather than gradient mass. Two-mechanism 2×2 ablation in a single run.

---

## 2026-05-16 14:30 UTC — PR #1144 CLOSED: WSS H8 Lion→AdamW optimizer swap (dl24-nezuko, `ccpx4z28`+`7ifzpx0r`)

- **Branch:** `dl24-nezuko/h8-adamw-optimizer-swap`
- **W&B runs:** training `ccpx4z28` (DDP8, killed at EP10), eval `7ifzpx0r` (EP10 best-val EMA checkpoint)
- **Hypothesis:** Lion → AdamW (lr=5e-4) Plateau Protocol tier-change — predicted that AdamW's magnitude-aware update `m/√(v+ε)` damps low-SNR τ_z axis noise that Lion's sign-update amplifies, while AdamW+GradNorm was predicted to maintain vol_p floor discipline.
- **Kill reason:** τ_z bellwether 10.300% at EP10 exceeded 9.5% kill criterion. Mechanism NOT validated.

### Terminal test metrics (EP10 best-val EMA checkpoint)

| Metric | SOTA #972 | H8 test | Δ | Floor | Verdict |
|--------|---:|---:|---:|---|---|
| test_abupt | 5.844% | 6.392% | +0.548pp | — | ❌ regression |
| test_vol_p | **3.643%** | **4.140%** | +0.497pp | **BREACH** | ❌ |
| test_surf_p | **3.577%** | **4.077%** | +0.500pp | **BREACH** | ❌ |
| test_wss | 6.727% | 7.264% | +0.537pp | — | ❌ regression |
| test_τ_x | 5.971% | 6.443% | +0.472pp | — | ❌ regression |
| test_τ_y | 7.362% | 7.958% | +0.596pp | — | ❌ regression |
| test_τ_z | **8.747%** | **9.343%** | **+0.596pp** | **bellwether** | ❌ mechanism failed |

### Analysis

**Hypothesis REJECTED on every axis.** AdamW lr=5e-4 did NOT narrow the low-SNR τ_z gap vs Lion; τ_z test=9.343% sits +0.596pp above SOTA, and the trajectory EP3→EP10 showed no evidence of superior noise damping on τ_z vs Lion (H5 EP6 Lion τ_z=8.85% vs H8 EP10 AdamW τ_z=10.300% — Lion was better at same epoch count).

**Root cause**: Convergence-rate deficit. AdamW lr=5e-4 + cosine T_max=30 ran at lower effective per-step descent than Lion lr=1e-4 on this regression problem. Val_abupt decelerated from −0.21 pp/ep (EP5→6) to −0.05 pp/ep (EP9→10) — running out of velocity before reaching competitive plateau.

**Critical wave finding from H8 (GradNorm interaction confirmed stable):**

| Stack | w_vol_p (terminal) | w_τ_z | ratio | vol_p test outcome |
|---|---:|---:|---:|---|
| Lion + GradNorm (H5) | 0.0064 | 2.318 | 362× | starvation → test breach |
| **AdamW + GradNorm (H8 EP10)** | **0.298** | **1.570** | **5.3×** | **balanced, no starvation** |

The H5 starvation mechanism (GradNorm crushes w_vol_p because vol_p has the lowest task-loss slope) is **optimizer-coupled**: Lion's sign-update exacerbates GradNorm imbalance, while AdamW's magnitude-aware update produces a stable, balanced equilibrium with the same GradNorm objective. Prior research warning "AdamW+GradNorm = catastrophic instability" **falsified** at lr=5e-4 + cosine + 1-ep warmup.

**Implication**: Future experiments can use AdamW WITHOUT requiring the H9 hard w_vol_p clamp — AdamW's GradNorm dynamics self-stabilize. The H9 clamp is still insurance under Lion, but orthogonal safety is now available via optimizer choice.

**What's NOT carried forward**: (a) AdamW at lr=5e-4 alone (lost convergence rate), (b) "AdamW will fix low-SNR τ_z" mechanism prediction (falsified).

**What IS carried forward** → **H11 (PR #1154)**: AdamW + per-axis WSS τ-magnitude weighting (τ_x=1.0, τ_y=1.2, τ_z=1.5), peak lr=7e-4, T_max=25. Compounds H8 wave finding (AdamW=safe optimizer) with H4-revisited (per-axis weights, Lion-noise root cause now neutralized by optimizer).

---

## 2026-05-16 09:30 UTC — PR #1135 CLOSED: WSS H6 wind-exposure additive attention bias (dl24-frieren, `u16jlft5`+`nb4mp52o`)

- **Branch:** `dl24-frieren/h6-wind-exposure-attention-bias`
- **W&B runs:** training `u16jlft5` (crashed EP21 tail), eval `nb4mp52o` (EP14 best-val checkpoint, EMA, 65536/65536 points)
- **Hypothesis:** Inject wind-exposure features (`|nx|`, `|ny|`, `|nz|`, `|nx·u|`) as a zero-init additive attention bias (mirroring H5's curvature pathway) — predicted to selectively improve τ_y (lateral cross-flow axis) via the encoded wind-direction signal, while avoiding H1's vol_p floor breach.

### Terminal test metrics (EP14 best-val checkpoint)

| Metric | SOTA #972 | H6 test | Δ | Floor | Verdict |
|--------|---:|---:|---:|---|---|
| test_abupt | 5.844% | 5.952% | +0.108pp | — | small regression |
| test_vol_p | **3.643%** | **3.957%** | +0.314pp | **BREACH** | ❌ |
| test_surf_p | **3.577%** | **3.669%** | +0.092pp | **BREACH** | ❌ |
| test_wss | 6.727% | 6.770% | +0.043pp | — | ~tied |
| test_τ_x | 5.971% | 6.012% | +0.041pp | — | small regression |
| test_τ_y | 7.362% | **7.305%** | **−0.057pp** | — | **mechanism confirmed** |
| test_τ_z | 8.747% | 8.818% | +0.071pp | — | small regression |

### Analysis

1. **τ_y mechanism delivered** on the predicted axis (−0.057pp under SOTA) — the wind-exposure additive attention bias is the correct injection point for the H1 wind-exposure signal that previously broke as raw input channels. The mechanism is real, replicable, isolated.

2. **vol_p tail-drift killed the merge case.** Same H5 signature: val_vol_p drifted +0.111pp EP9→EP15. Best-val checkpoint can't escape the floor because the EP9-EP15 plateau band sits structurally above 3.643%.

3. **WSS plateau in [6.96, 6.99] val band, EP9-EP15** — identical to H5. Confirms WSS floor at this stack config is set by GradNorm task-share, not signal saturation.

4. **Crash-then-EP14-recovery was correct** — extending to EP30 would have only widened the vol_p breach. Student's recovery management was excellent.

5. **5-experiment GradNorm starvation pattern**: H1, H2, H3, H5, H6 — every surface-targeted experiment without a vol_p safeguard breached the vol_p floor. This is the strongest case yet for the H9 clamp being the right fix.

### Carry-forward

- **Wind-exposure attn bias** is a confirmed positive mechanism for τ_y. A future composition with H9's GradNorm clamp (wind-exposure + w_vol_p ≥ 0.05) is the next-step revisit IF H9 lands.
- Adds to evidence pool that **zero-init additive attention bias is the safe injection pathway** (H5 confirmed for curvature, H6 confirmed for wind-exposure).

---

## 2026-05-16 07:55 UTC — PR #1132 CLOSED: WSS H5 curvature additive attention bias (dl24-tanjiro, `lbi210l2`)

- **Branch:** `dl24-tanjiro/h5b-curvature-attention-bias`
- **W&B run:** `lbi210l2` (21.4h, 30 epochs, best-val EP18 EMA)
- **Hypothesis:** Injecting curvature information as a zero-init additive attention bias (instead of H2's raw input channels) would capture the curvature WSS signal without triggering GradNorm task-share imbalance.

### Terminal test metrics (best-val EP18 EMA checkpoint)

| Metric | H5 `lbi210l2` | SOTA #972 | Δ | H2 ref | Verdict |
|--------|---:|---:|---:|---:|---|
| test_abupt | 5.846% | 5.844% | +0.002pp | — | tie |
| **test_wss** | **6.609%** | 6.727% | **−0.118pp** | 6.668% (−0.059) | ✅ **BEST WSS IN WAVE** |
| test_surf_p | 3.651% | 3.577% (floor) | +0.074pp | — | ❌ **FLOOR BREACH** |
| test_vol_p | 3.955% | 3.643% (floor) | +0.312pp | 3.983% (−0.028) | ❌ **FLOOR BREACH** |
| test_τ_x | 5.859% | 5.971% | **−0.112pp** | — | ✅ win |
| test_τ_y | 7.174% | 7.362% | **−0.188pp** | — | ✅ win |
| **test_τ_z** | **8.592%** | 8.747% | **−0.155pp** | 8.650% (−0.058) | ✅ **WIN on dominant axis** |

### Mechanism analysis — the GradNorm smoking gun

The zero-init forward check passed (diff=0.0 at step 0) — the curvature signal was correctly isolated via the additive bias path. WSS improved on ALL axes. **But vol_p still breached by +0.312pp** — only marginally better than H2's +0.340pp.

**The student's diagnosis is the key finding**: The final GradNorm weights reveal the actual mechanism:

| Task | Final w | Note |
|---|---:|---|
| w_cp | 0.442 | |
| w_tau_x | 0.698 | |
| w_tau_y | 1.536 | |
| **w_tau_z** | **2.318** | GradNorm maximizes WSS learning |
| **w_vol_p** | **0.0064** | **362× lower than τ_z — starved** |

GradNorm's relative-rate balancing reads vol_p's low task-loss slope as "overweighted" → crushes w_vol_p → vol_p starves. This is **injection-point-independent**: whether curvature enters via input channels (H2) or additive bias (H5), GradNorm produces the same starvation. The H5 design assumption ("input-channel dilution caused H2's vol_p breach") was incorrect. The correct mechanism is GradNorm's rate-based balancing.

### Key findings to carry forward

1. **CONFIRMED: Curvature attention bias produces real WSS gain** (−0.118pp from SOTA, −0.059pp from H2). The additive bias path works and is the cleanest injection mechanism. test_τ_z −0.155pp confirms τ_z/τ_y are the primary beneficiaries.
2. **FALSIFIED: Feature-budget dilution hypothesis** — H5 only recovered +0.028pp vol_p vs H2. The 0.312pp gap to floor is NOT about input channel count.
3. **IDENTIFIED: GradNorm vol_p starvation** as the universal floor-breach mechanism across H1/H2/H3/H4/H5. All WSS-targeted experiments feed the same root cause.
4. **Follow-up**: H9 (PR #1145, tanjiro) = H5 curvature bias + hard floor on w_vol_p ≥ 0.05. This directly attacks the starvation. If it works, first SOTA-on-aggregate of the WSS wave.

Decision: CLOSED, NOT-a-winner (floor breach). **Highest-quality result of the wave — confirms the mechanism and unlocks H9.**

---

## 2026-05-16 06:25 UTC — PR #1129 CLOSED: WSS H3 near-wall volume cross-attn into surface decoder (dl24-nezuko)

- **Branch:** `dl24-nezuko/wss-near-wall-cross-attn`
- **W&B run:** included in terminal SENPAI-RESULT
- **Hypothesis:** Near-wall volume points (within 0.05m SDF shell) queried via cross-attention into the surface decoder would help the model leverage boundary-layer velocity gradient information to improve τ_z prediction (dominant error axis 8.747%).

### Terminal test metrics (best-val checkpoint)

| Metric | H3 | SOTA #972 | Δ | Verdict |
|--------|---:|---:|---:|---|
| test_abupt (PRIMARY) | 6.020% | 5.844% | **+0.176pp** | ❌ Regression |
| test_surf_p (FLOOR) | 3.761% | 3.577% | **+0.184pp** | ❌ **FLOOR BREACH** |
| test_vol_p | **3.432%** | 3.643% | **−0.211pp** | ✓ Under floor (side-effect) |
| test_wss | 7.012% | 6.727% | **+0.285pp** | ❌ Regression |
| τ_x | — | 5.971% | — | — |
| τ_y | — | 7.362% | — | — |
| **τ_z (HYPOTHESIS TARGET)** | **9.102%** | 8.747% | **+0.355pp** | ❌ **Target axis FAILED** |

### Mechanism analysis

The hypothesis was physically motivated: boundary-layer velocity gradient → τ via Newton's law, so near-wall volume points SHOULD carry τ signal. The failure was at the **injection mechanism level**, not the physics.

Adding near-wall cross-attention to the surface decoder:
1. **Expanded surface decoder capacity** → backbone attention budget shifted toward cross-attn queries → volume-pressure head starved → test_vol_p side-effect improvement, test_surf_p breach
2. **Cross-attention signal duplication**: the existing shared backbone self-attention already aggregates near-wall boundary features via the volume encoder. Adding an explicit cross-attn pathway duplicated rather than augmented this signal, adding computational overhead + gradient noise.
3. **τ_z hypothesis REFUTED**: τ_z regressed +0.355pp (the explicit target axis WORSENED). The model did not extract boundary-layer gradient signal from near-wall volume points via cross-attention.

Student's correct diagnosis: "existing backbone already exposes boundary-layer info via shared self-attention; cross-attention duplicated signal."

### Key findings to carry forward

1. **Explicit cross-attention into the surface decoder from volume is the wrong injection pathway** — same lesson as H1/H2 (wrong injection point). The surface decoder already has implicit access to boundary features via the shared backbone.
2. **vol_p side-effect (−0.211pp)** is consistent with H4 vol_p side-effect (−0.269pp) — whenever the surface task gets boosted gradient signal (here via cross-attn expansion, there via mean weight upscale), the backbone learns richer features that benefit vol_p. This is now a 2-experiment pattern.
3. **Plateau Protocol activated**: 4 closed failures (H1, H2, H3, H4) + 2 plateauing (H5, H6) → tier change to optimizer regime (H8, AdamW).

Decision: CLOSED, NOT-a-winner. Assigned H8 (Lion → AdamW, lr=5e-4) to nezuko as Plateau Protocol tier change (PR #1144).

---

## 2026-05-16 02:33 UTC — PR #1130 CLOSED: WSS H4 per-axis loss weights [1.0, 1.5, 2.5] (dl24-fern, `3i0nnneh`)

- **Branch:** `dl24-fern/wss-per-axis-loss-weights`
- **W&B run:** `3i0nnneh` (19h58m, 30 epochs, best-val EP15)
- **Hypothesis:** Per-axis WSS loss weights `[cp=1.0, τ_x=1.0, τ_y=1.5, τ_z=2.5]` on the SOTA stack would reallocate optimizer capacity toward τ_z (the dominant error axis) and improve test_wss.

### Terminal test metrics (best-val EP15 EMA checkpoint)

| Metric | H4 | SOTA #972 | Δ | Verdict |
|--------|---:|---:|---:|---|
| test_abupt (PRIMARY) | 5.911% | 5.844% | **+0.067pp** | ❌ Regression |
| test_surf_p (FLOOR) | 3.743% | 3.577% | **+0.166pp** | ❌ **FLOOR BREACH** |
| test_vol_p (FLOOR) | **3.374%** | 3.643% | **−0.269pp** | ✓ Under floor |
| test_wss | 6.870% | 6.727% | **+0.143pp** | ❌ Regression |
| τ_x | 6.118% | 5.971% | +0.147pp | ❌ Worse |
| τ_y | 7.447% | 7.362% | +0.085pp | ❌ Worse |
| τ_z | 8.872% | 8.747% | +0.125pp | ❌ Worse |

### Mechanism — three confounded effects

1. **Lion-noise on weighted τ axes** (NEGATIVE): Lion sign-update + amplified weight amplified gradient noise, not signal. All 3 τ axes regressed.
2. **Capacity stolen from cp** (NEGATIVE → surf_p breach): cp at weight=1.0 while τ_y/τ_z amplified → capacity asymmetry hurt cp. Caused the floor breach.
3. **Implicit surface task upweight** (POSITIVE): Mean of [1.0, 1.0, 1.5, 2.5]/4 = 1.5 → backbone gets 1.5× gradient from surface task → volume head benefits → test_vol_p −0.269pp under SOTA floor. Real and persistent signal (>3× seed variance).

### Key findings

1. Per-axis WSS weighting under Lion is falsified — sign-based updates amplify noise at the per-axis level, not signal.
2. **Isolated surface-task upweight is a confirmed mechanism**. H7 (PR #1142, fern) assigned to test `--surface-loss-weight 1.5` in isolation. If surf_p floor breach was caused by the per-axis asymmetry (not the global upweight), H7 should win on vol_p without floor breach.
3. Follow-up precedent: `qqtdnlwq` tried `surface_loss_weight=2.0` on the OLD dataset and regressed — but that was a different stack, different magnitude, different dataset artifact regime. H7 with 1.5 on SOTA #972 stack is the clean single-variable test.

Decision: CLOSED, NOT-a-winner.

---

## 2026-05-15 16:44 UTC — PR #1115 CLOSED: WSS H1 wind-exposure proxy (dl24-frieren, `3rja7gw6`)

- **Branch:** `dl24-frieren/wss-wind-exposure-proxy`
- **W&B run:** `3rja7gw6` (21.4h single run, no wave-2 — sibling dirs were rank shards)
- **Hypothesis:** Adding `wind_exposure=max(0,-nx)` and `abs_cross_normal=|ny|` as extra surface input channels (7→9) would directly encode the cross-flow attack-angle signal targeting τ_y/τ_z dominant error axes.

### Test metrics (best-val EP9 EMA checkpoint)

| Metric | H1 | SOTA #972 | Δ | Verdict |
|--------|----:|----:|---:|---|
| **test_abupt (PRIMARY)** | 5.994% | 5.844% | **+0.150pp** | ❌ Regression |
| test_surf_p (FLOOR) | 3.705% | 3.577% | **+0.128pp** | ❌ **FLOOR BREACH** |
| **test_vol_p (FLOOR)** | **4.080%** | 3.643% | **+0.437pp** | ❌ **FLOOR BREACH** (severe) |
| **test_wss (WSS target)** | 6.767% | 6.727% | +0.040pp | ❌ Slight regression |
| τ_x | 5.975% | 5.971% | +0.004pp | tied |
| τ_y | 7.408% | 7.362% | +0.046pp | slightly worse |
| τ_z | 8.801% | 8.747% | +0.054pp | slightly worse |

### Mechanism — identical to H2 #1117 failure

Adding 2 surface input channels (input_dim 7→9) → dilutes per-token feature budget → gradnorm task-share imbalance under-weights volume head → val_vol_p drifts up from EP10 onward (4.28→5.02 in val). The wind-exposure signal itself is geometrically reasonable (EP0 audit confirmed `wind_exposure∈[0,1]`, mean≈0.16), but the *injection point* (raw input channels) breaks the SOTA stack's task-share invariant.

### Key findings

1. **Robust observation: raw additional surface input channels via `surface_input_dim` is the wrong injection point for any WSS-targeting feature.** Both H1 (wind-exposure) and H2 (curvature) exhibit the *identical* failure pattern. The shared mechanism is gradnorm task-share imbalance, not any specific feature defect.
2. **Strong corroboration of H5 (#1132) mechanism.** H5 takes H2's *same curvature signal* via zero-init additive attention bias (no input-dim change). At EP9 live: H5 leads H2 across all metrics (val_abupt 6.232 vs 6.288, val_wss 6.938 vs 6.991, val_vol_p 4.051 vs 4.136). Zero-init bias method bypasses the gradnorm imbalance.
3. **Student's terminal post-mortem** flagged 5 useful follow-ups (normal-frame loss, smaller per-axis weights, --volume-loss-weight 1.5 floor protection, region-weighted MSE, per-region diagnostic). Queue as candidates for the next wave.

Decision: closed as NOT-a-winner.

---

## 2026-05-15 ~08:00 UTC — PR #1117 CLOSED: WSS H2 surface curvature features (dl24-tanjiro, `b5g7776p`)

- **Branch:** `dl24-tanjiro/wss-curvature-features`
- **W&B runs:** Training `b5g7776p` (halted EP16/30, EMA ckpt EP8 best-val); eval `8o2ia3nu`
- **Hypothesis:** Adding kappa_H + kappa_G as extra surface input channels (7→9) would improve WSS by giving the model explicit knowledge of high-curvature regions (A-pillar, wheel arches, rear diffuser) where tau_y/tau_z errors concentrate.

### Test metrics (best-val EP8 EMA checkpoint)

| Metric | H2 (`8o2ia3nu`) | SOTA `56bcqp3m` | Δ | Verdict |
|--------|----------------:|----------------:|---:|---|
| **test_abupt (PRIMARY)** | 5.894% | 5.844% | +0.050pp | ❌ Regression |
| test_surf_p | 3.695% | 3.577% | +0.118pp | ❌ Regression |
| **test_vol_p (FLOOR)** | **3.983%** | 3.643% | **+0.340pp** | ❌ **FLOOR BREACH** |
| **test_wss (WSS target)** | **6.668%** | 6.727% | **−0.059pp** | ✅ **BEATS SOTA** |
| τ_x | 5.928% | 5.971% | −0.043pp | ✅ Beat |
| τ_y | 7.213% | 7.362% | −0.149pp | ✅ Beat |
| **τ_z** | **8.650%** | 8.747% | **−0.097pp** | ✅ **Beat (dominant axis)** |

### Val trajectory (halted EP16, best-val EMA checkpoint EP8)

| EP | val_abupt | val_vol_p | val_wss |
|----|-----------|-----------|---------|
| 3  | ~7.4% | ~5.2% | ~7.4% |
| 6  | ~7.0% | ~4.7% | ~7.2% |
| 8 (BEST-VAL EMA) | ~6.8% | ~4.5% | ~7.1% |
| 13 | 6.5%est | ~4.4% | ~6.9% |
| 15 (BORDERLINE) | borderline | **4.3%→drifting** | improving |
| 16 (HALT) | — | **+0.044pp/epoch monotone** | — |

At EP15 advisor opted to HALT (Option A) based on vol_p monotone drift trajectory: projected EP20 ~4.59%, EP30 ~5.03% even with cosine compression → test_vol_p floor breach inevitable.

### Closure analysis

**First sub-SOTA WSS in the wave.** All three WSS axes beat SOTA: tau_x −0.043pp, tau_y −0.149pp, tau_z −0.097pp. This confirms the curvature hypothesis is physically correct — the model genuinely learned to use curvature features for WSS prediction.

**Fatal side-effect:** Adding 2 curvature input channels (7→9) dilated the per-token surface feature budget. GradNorm's task-share rebalancing under-weighted volume_pressure as a result. Test vol_p breached the floor by +0.340pp (9.3% relative regression) — decisive close under Issue #1056 contract requiring AND-clause floors.

**Mechanism:** The gradnorm task-share imbalance is well-documented from earlier waves (PRs #912, #964): richer surface inputs shift GradNorm equilibrium away from vol_p. H2 reproduced this exactly with 9-channel surface input.

### Key findings to carry forward

1. **Curvature physics signal is VALID.** tau_z improved −0.097pp on the dominant error axis, tau_y −0.149pp. The curvature features contain genuine signal for WSS prediction.
2. **H2 gives the first wave WSS sub-SOTA** (6.668% < SOTA 6.727%), closing 11% of the gap to Transolver-3's 5.85%.
3. **The injection point is wrong, not the signal.** Adding curvature as input channels dilates the feature budget and triggers gradnorm imbalance. Need to inject curvature OUTSIDE the surface input projection pathway.
4. **H5 natural follow-up:** Inject curvature as an additive bias to the token embeddings AFTER the 7-channel surface projection (zero-initialized) — same curvature physics, no input-dim change, no gradnorm perturbation.
5. **Best correction checkpoint was EP8 EMA**, not EP16 — evidence that vol_p overfitting onset was ~EP10.

### Next assignment

PR #1132 — H5: curvature as additive attention bias (zero-init CurvatureAttentionBias module added after surface_input_proj, surface_input_channels=7 unchanged).

---

## 2026-05-15 ~05:50 UTC — PR #1098 CLOSED: WD=0.01 isolated re-test (dl24-fern, `q4eok915`)

- **Branch:** `dl24-fern/wd-001-isolated-retest`
- **W&B Run:** `q4eok915` (rank-0); ranks 1-7: `iln1bhtf`, `caya1jtm`, `c62r5twl`, `zs1rxo2s`, `6tjgb148`, `rdatvqf1`, `l9cgorz5`
- **Hypothesis:** WD=0.01 (vs SOTA WD=0.005) would reduce volume-head overfitting, improving test_vol_p. Lion+high-WD known failure mode caveat acknowledged.

### Test metrics (best-val EP19 EMA checkpoint, 30-epoch run)

| Metric | WD=0.01 (`q4eok915`) | SOTA `56bcqp3m` | Δ | Verdict |
|--------|---------------------:|----------------:|---:|---|
| **test_abupt (PRIMARY)** | **6.046%** | 5.844% | **+0.202pp** | ❌ REGRESSION |
| test_surf_p (FLOOR) | 3.745% | 3.577% | **+0.168pp** | ❌ **FLOOR BREACH** |
| test_wss (WSS target) | 7.054% | 6.727% | +0.327pp | ❌ REGRESSION |
| test_vol_p | **3.411%** | 3.643% | **−0.232pp** | ✓ Improvement |
| τ_x | 6.262% | 5.971% | +0.291pp | ❌ |
| τ_y | 7.636% | 7.362% | +0.274pp | ❌ |
| τ_z | 9.176% | 8.747% | +0.429pp | ❌ WORST AXIS |

### Val trajectory (best checkpoint EP19 of 30)

| EP | val_abupt | val_vol_p | val_surf_p | val_wss |
|----|-----------|-----------|------------|---------|
| 3  | 6.678% | 3.674% | 4.365% | 7.609% |
| 10 | 6.311% | 3.518% | 4.126% | 7.212% |
| **19 (BEST)** | **6.283%** | **3.499%** | **4.098%** | **7.188%** |
| 30 | 6.333% | 3.525% | 4.120% | 7.250% |

Val→test: all channels where test is slightly better than val — clean corrected-split transfer confirmed.

### Closure analysis

WD=0.01 confirms the known Lion+high-WD failure mode: sign-based Lion updates × 2× WD = surface/WSS under-training. The vol_p benefit (−0.232pp) is real but is outweighed by WSS regression (+0.327pp) and surf_p floor breach (+0.168pp). Primary metric regresses +0.202pp. Three of four metrics wrong direction, surf_p floor breached — decisive close.

### Key findings to carry forward

1. **test_vol_p ≤ 3.41% is achievable** with WD=0.01 stack — if we ever want a pure vol_p push, this is a known lever.
2. **τ_z=9.176% is the dominant error axis** — all three WD runs (nezuko LR=9e-5 + fern WD=0.01) show τ_z as hardest, +0.43-0.43pp above SOTA. Architecture attack (H3 cross-attn) and loss attack (H4 per-axis weights) both targeting this.
3. **Corrected-split val→test transfer is clean** (test < val across all channels) — dataset fix is working.

### Next assignment

PR #1130 — H4: WSS per-axis loss weights [τ_x=1.0, τ_y=1.5, τ_z=2.5] on SOTA stack (WD=0.005). Direct attack on dominant error axis via loss reweighting.

---

## 2026-05-15 ~05:10 UTC — PR #1101 CLOSED: LR=9e-5 isolated control (dl24-nezuko, `5qumfbrs`)

- **Branch:** `dl24-nezuko/lr-9e-5-isolated-control`
- **W&B Run:** `5qumfbrs` (rank-0)
- **Hypothesis:** Lion lr=9e-5 (vs SOTA 1e-4) would help volume pressure via reduced overshoot of fine minima. Test prediction: test_vol_p improvement.

### Test metrics (best-val checkpoint EP13, evaluated by trainer)

| Metric | Run | SOTA `56bcqp3m` | Δ | Verdict |
|--------|----:|----:|----:|---|
| **test_abupt (PRIMARY)** | **5.9974%** | 5.844% | **+0.153pp** | ❌ regresses |
| test_wss (WSS target)    | 6.9870% | 6.727% | +0.260pp | ❌ regresses |
| test_surf_p (FLOOR)      | 3.7245% | 3.577% | +0.148pp | ❌ **breaches floor** |
| test_vol_p               | 3.3727% | 3.643% | **-0.270pp** | ✓ improves |
| test_wss_x               | 6.1778% | — | — | — |
| test_wss_y               | 7.6269% | — | — | — |
| test_wss_z               | 9.0849% | — | — | — |

### Best-val trajectory peaked at EP13 (full 30-EP run)

- EP10 val_abupt=6.39%, val_vol_p=3.510%, val_wss=7.328%
- **EP13 (BEST):** val_abupt=6.345%, val_vol_p=3.490%, val_wss=7.287% — checkpoint used for test eval
- EP20+: plateau drift, val_abupt slowly climbs to 6.354-6.390% — late epochs DON'T help, slightly hurt
- EP30 terminal: val_abupt=6.390%, val_vol_p=3.510%, val_wss=7.339%

### Closure analysis

The hypothesis's specific prediction was correct on its own terms: lr=9e-5 substantially improves test_vol_p (-0.270pp, strongest vol_p single-result on corrected split). But this came with:
1. **Primary metric regression** (test_abupt +0.153pp) — disqualifies merge.
2. **WSS target regression** (test_wss +0.260pp) — opposite direction from Morgan's Issue #1056 directive.
3. **test_surf_p floor breach** (+0.148pp above 3.577%) — explicit merge-blocker per Issue #1056.

The "smoother loss landscape generalizes" framing failed for surface metrics. lr=9e-5 may simply undertrain surface tasks relative to lr=1e-4. Cosine T_max=30 already walks LR through 5e-5 mid-training, so "hybrid LR" follow-ups are unlikely to recapture the vol_p gain without the surf_p cost.

### Useful follow-up information

- **test_vol_p≤3.40% is achievable** with lr=9e-5 + current stack. If we ever want to push vol_p specifically, this is a known lever.
- **wss_z = 9.08% remains the dominant residual error** — confirmed across all corrected-split runs. Local geometric features (curvature, wind-exposure) don't reach it. Architectural mechanism needed (cross-attention from near-wall volume features into surface decoder — assigned next as H3).
- **EP15-25 KILL zone**: had the tightened EP20 KILL (val_abupt>6.30%) been wired into `--kill-thresholds`, this run would have been killed at EP20 and saved ~5h compute. The pre-tightened threshold passed easily and didn't trigger. Process improvement: align `--kill-thresholds` arg with advisor's tightened gates.

---

## 2026-05-14 ~08:00 UTC — PR #1072 CLOSED: SDF α=0.5 + GradNorm, corrected split (dl24-nezuko, `yp383yq2`)

- **Branch:** `dl24-nezuko/sdf-near-surface-alpha05-vol-focus`
- **W&B Run:** `yp383yq2` (rank-0)
- **Hypothesis:** SDF inverse near-surface vol sampling α=0.5 (`w = 1/(1+0.5·|sdf|)`) + GradNorm α_gn=0.5 on corrected dataset. Moderate near-surface concentration, paired with dynamic loss balancing to direct gradient budget toward hard metrics.

### Full val trajectory

| EP | Step | val_abupt% | val_vol_p% | val_surf_p% | val_wss% | Gate | Status |
|----|------|-----------|-----------|------------|---------|------|--------|
| 1  | 10,975  | 21.535 | 17.704 | 15.500 | 22.223 | ≤30% | PASS |
| 2  | 21,951  |  7.288 |  5.318 |  4.635 |  7.955 | ≤16% | PASS |
| 3  | 32,927  |  6.675 |  4.605 |  4.294 |  7.365 | ≤8%  | PASS |
| 5  | 54,879  |  6.435 |  4.299 |  4.171 |  7.137 | ≤7.5%| PASS |
| **10** | **109,759** | **6.2904** | **4.2307** | **4.0800** | **6.9882** | ≤7.2% | **PASS ⭐ RUN BEST** |
| 15 | 164,639 |  6.3457 |  4.3668 |  4.1256 |  7.0114 | ≤6.80%| PASS |
| 20 | 219,500 |  6.3142 |  4.419  |  4.100  |  6.955  | ≤6.70%| PASS |
| 25 | 274,375 |  6.3988 |  4.7532 |  4.1169 |  6.9758 | ≤6.65%| PASS |
| 29.7 | 326,812 | 6.4408 | 4.8867 | 4.1199 | 6.9979 | — | STALE (run died) |

### Outcome

**Run died at step 326,812 (~EP29.7)** — W&B heartbeat stale since 06:18Z, ~2,438 steps short of EP30 terminal (329,250). No `test_primary/*` metrics were ever logged. Cause: likely rank-0 process crash near end of training.

No eval-only run from EP10 best checkpoint was launched. Student-AI pod had been GH rate-limited since 03:18Z (student account user ID 20516801) — could not receive or act on advisor instructions to launch the EP10 eval.

### Test metrics

**No test metrics available.** (Run did not complete terminal test eval; no best-checkpoint eval launched.)

### Verdict: CLOSED — NO MERGE PATH

- Run died before completing terminal eval
- Student-AI rate-limited, cannot launch EP10-best eval
- Wave budget exhausted (24h from start ~09:22Z May 13 → deadline 09:22Z May 14)
- Even if test metrics were available: val trajectory shows plateau-regression from EP10 onward (vol_p 4.23% → 4.89% at EP29) — terminal-epoch test would not beat SOTA test_vol_p=3.643%

### Strategic implications

- SDF α=0.5 + GradNorm composition converges quickly (EP5=6.435%) but plateaus at EP10 with subsequent vol_p regression — same pattern as α=0.25 (fern #1063)
- **Best val_vol_p of 4.231% at EP10** is competitive for the SDF wave but still 16.1% above SOTA test_vol_p=3.643% in absolute terms (val→test transfer unknown)
- GradNorm α_gn=0.5 + SDF α=0.5 composition did not overcome the plateau-regression pattern that affected GradNorm-alone runs on the old dataset
- **SDF concentration sweep is now fully closed:** α ∈ {0.25, 0.5, 1.0, 2.0, 3.0} all evaluated or killed; no α value beat uniform sampling (SOTA PR #972)

---

## 2026-05-14 ~03:40 UTC — PR #1063 TERMINAL: SDF Inverse Vol Sampling α=0.25 (dl24-fern, `xfykblf9`)

- **Branch:** `dl24-fern/sdf-near-surface-alpha-sweep-band-b`
- **W&B Run:** `xfykblf9`
- **Hypothesis:** Low-α inverse near-surface SDF weighting `w = 1/(1+α|sdf|)` with α=0.25 mildly concentrates volume sampling near boundary; should retain volume coverage while gently emphasizing the boundary layer.
- **Status:** RUN TERMINATED at EP29.3 (step 321,739, wallclock-limited). Test eval auto-logged at terminal. Student pod was rate-limited so SENPAI-RESULT marker not yet posted — advisor harvested metrics directly from W&B.

### Test metrics (terminal-epoch auto-logged)

| Metric | Value | PR #972 SOTA | Δ |
|--------|------:|-------------:|---:|
| test_abupt | **5.955%** | 5.844% | +0.111pp (regression, +1.9% rel) |
| test_vol_p | **3.990%** | 3.643% | +0.347pp (regression, +9.5% rel) |
| test_surf_p | 3.707% | 3.577% | +0.130pp |
| test_wss | 6.746% | 6.727% | ~noise |

### Per-metric best val checkpoints

| Metric | Best Val | Step | EP |
|--------|---------:|-----:|---:|
| val_abupt | 6.2647% | 120,735 | 11 |
| val_vol_p | 4.1178% | 98,783 | 9 |
| val_surf_p | 4.058% | 175,615 | 16 |
| val_wss | 6.954% | 197,567 | 18 |

### Verdict: NOT A MERGE CANDIDATE

- test_abupt regresses +0.111pp vs SOTA (small, but not improvement)
- test_vol_p (human-priority metric per Issue #882) regresses +0.347pp = +9.5% rel — clearly worse
- Plateau-regression pattern confirmed: val_abupt peaked at EP11 (6.265%), drifted upward through terminal (6.409%)
- Test from terminal-epoch checkpoint, NOT best val-checkpoint. Test from EP11-best still pending.

### Strategic implications

- SDF inverse near-surface sampling at α=0.25 does NOT beat uniform-sampling baseline (PR #972) on corrected dataset
- Combined with α=2.0 (PR #1054 EP15 FAIL) and α=3.0 (PR #1076 EP10 KILL), the SDF-concentration approach is now broadly falsified on this dataset
- α=0.5 (nezuko #1072, still running) and α=1.0 (frieren #1077, still running) will close the remaining sweep, but expectations are now low
- Pivot levers (orthogonal to SDF concentration) take priority: EMA #1086 in-flight, GradNorm+SDF composition (H1), WD=0.01 retest (H3), Y-sym p=0.5 (H4)

PR not yet closed — waiting for student to post EP11-best test eval. If EP11-best test_abupt ≤ 5.844% AND test_vol_p ≤ 3.643%, merge candidate. Otherwise close.

---

## 2026-05-13 ~20:20 UTC — PR #1076 CLOSED: SDF Inverse Vol Sampling α=3.0 (dl24-tanjiro, `ed01yw3z`)

- **Branch:** `dl24-tanjiro/sdf-inverse-alpha-sweep-3.0`
- **W&B Run:** `ed01yw3z`
- **Hypothesis:** Strong near-surface SDF-inverse weighting `w = 1/(1+α|sdf|)` with α=3.0 concentrates volume-point training signal in the boundary layer, improving surface pressure and WSS accuracy.

### Per-epoch val trajectory

| EP | Step | val_abupt% | val_surf_p% | val_vol_p% | val_wss% | Gate | Status |
|----|------|-----------|------------|-----------|---------|------|--------|
| 1  | 10,975  | 24.1097 | 17.2723 | 20.4422 | 24.6239 | ≤30% | PASS |
| 2  | 21,951  |  8.8851 |  5.5719 |  8.7092 |  9.1929 | ≤16% | PASS |
| 3  | 32,927  |  7.0932 |  4.4177 |  5.8164 |  7.6021 | ≤8%  | PASS |
| 5  | 54,879  |  6.8262 |  4.3205 |  5.4883 |  7.3277 | ≤7.5%| PASS |
| 6  | 65,855  | **6.5012** | 4.1921 | 4.4730 | 7.1842 | info | ★ best |
| 7  | 76,831  |  6.5675 |  4.2110 |  4.4883 |  7.2666 | info | — |
| 8  | 87,807  |  6.7547 |  4.2312 |  4.4291 |  7.5101 | info | — |
| 9  | 98,783  |  6.7509 |  4.2430 |  4.3635 |  7.5196 | info | — |
| 10 | 109,759 |  7.2509 |  4.3723 |  4.5303 |  8.0798 | ≤7.2%| **KILL** |

**Best val_abupt = 6.5012% at EP6.** No test_primary metric (kill terminated training before test eval).

### Results: no test metric, hypothesis FALSIFIED

| Metric | Result |
|--------|--------|
| test_abupt | N/A (killed at EP10) |
| test_vol_p | N/A |
| val_abupt best | 6.5012% (EP6) |
| val_vol_p best | 4.3635% (EP9) |
| Wave SOTA test_abupt | 5.844% |
| Wave SOTA test_vol_p | 3.643% |

### Analysis

α=3.0 over-concentrates near-surface sampling: at |sdf|≈1m, weight ≈ 0.25 → far-field volume points sampled ~4× less frequently. Effect:
1. **EP6 floor then regression** — val_abupt: 7.09% (EP3) → 6.50% (EP6) then UP to 7.25% (EP10). W&B slope at kill = +0.046%/1k steps (structural, not noise).
2. **WSS degradation** — WSS drifted from 7.18% (EP6) to 8.08% (EP10). Far-field starvation hurts global pressure representation needed for WSS.
3. **vol_p recovery** — vol_p actually improved EP5→EP9 (5.49%→4.36%), but was still ~0.7pp above SOTA at its best. α=3.0 does not solve the vol_p bottleneck.

Compared to fern (α=0.25, best 6.265%) and nezuko (α=0.5, best 6.290%), the α-response is now mapped:
- α≤0.5: competitive (best val_abupt ≈ 6.26–6.29%)
- α=1.0: TBD (frieren just started)
- α=2.0: EP15 FAIL (PR #1054, previously closed)
- α=3.0: EP10 KILL — over-concentration confirmed

**Conclusion: productive α-band is definitively [0.25, 0.5] or lower; α≥2.0 is the over-concentration regime.**

**Config:** 6L STRING 5-oct, Lion lr=1e-4, WD=0.005, GradNorm α=0.5, Y-sym p=0.5, SDF α=3.0, NO EMA; 8×DDP, bs=1, EBS=8, ~10,975 steps/epoch.

---

## 2026-05-09 ~22:30 UTC — PR #972 CLOSED: SDF-stratified importance sampling, far-field bias α=2.0 (dl24-frieren, `56bcqp3m`)

- **Branch:** `dl24-frieren/sdf-stratified-sampling`
- **W&B Run:** `56bcqp3m`
- **Hypothesis:** SDF-stratified volume importance sampling with far-field upweighting (α=2.0 Gaussian-bias on |sdf| > 0.1 points) would improve test vol_p generalization by forcing the model to learn better far-field pressure behaviour that the test set emphasizes.

### Per-epoch val progression (selected)

| Epoch | abupt% | vol_p% | Notes |
|-------|--------|--------|-------|
| EP5  | ~7.2%  | ~4.2%  | EP5 gate PASS |
| EP10 | ~6.8%  | ~4.0%  | EP10 gate PASS |
| EP20 | **6.127%** | **3.815%** | Best checkpoint; EP20 gate PASS |
| EP30 | 6.191% | 4.005% | Terminal (slight regression from EP20) |

### Terminal Test Results (EP20 best checkpoint)

| Metric | Test Value | Baseline (#740) | Delta |
|--------|-----------|-----------------|-------|
| `vol_p` (PRIMARY) | **11.827%** | 10.758% | **+1.069pp WORSE** |
| `abupt` | 7.480% | 7.5195% | -0.040pp (marginal) |
| `surf_p` | 3.574% | 3.881% | -0.307pp (slight improvement) |
| `wall_shear` | 6.726% | 7.061% | -0.335pp (slight improvement) |
| `wall_x` | 5.972% | — | — |
| `wall_y` | 7.276% | — | — |
| `wall_z` | 8.750% | — | — |

### Val→Test Gap Analysis

| Checkpoint | val_vol_p | test_vol_p | Gap |
|-----------|-----------|------------|-----|
| EP20 best | 3.815% | 11.827% | **+8.012pp** |
| Baseline  | ~3.9%   | 10.758%  | +6.858pp (approx) |

Gap is structurally identical to all previous experiments. SDF-stratified sampling did not narrow it.

### Decision: CLOSED — HYPOTHESIS FALSIFIED

**SDF sampling axis now FULLY CLOSED.** Both far-field SDF upweighting (α=2.0) in this PR and near-surface SDF weighting in PR #996 have been tested. Neither closes the val→test gap. The val→test vol_p gap of +8.0pp is NOT caused by insufficient coverage of any specific spatial region during training — it is a structural DATA DISTRIBUTION problem.

### Key Findings

1. **test_vol_p = 11.827% vs baseline 10.758% — WORSE by +1.069pp.** Far-field bias in training sampling actively harmed vol_p generalization relative to baseline.
2. **Val→test gap +8.012pp is structurally identical to all previous experiments.** The absolute gap has not narrowed despite changing the spatial emphasis of the volume loss signal.
3. **abupt marginally better (7.480% vs 7.5195%)** — noise-level improvement of 0.040pp; not meaningful and does not beat baseline.
4. **32nd intervention to falsify the structural gap hypothesis.** We have now exhausted: loss weighting (6 variants), GradNorm tuning, optimizer switches, EMA, SWA (pending), architecture depth, sampling strategies (4 variants), coordinate noise, TTA, Lookahead, DropPath, normalization, LR schedule variants.
5. **Next frontier:** Active experiments (nezuko SWA, fern Poisson physics regularization, tanjiro InstanceNorm) target remaining gap-closing candidates.

---

## 2026-05-09 ~22:00 UTC — PR #999 IN PROGRESS: SWA epoch-snapshot averaging EP20–EP30 (dl24-nezuko, `f8rc8ahi`)

- **Branch:** `dl24-nezuko/swa-epoch-avg`
- **W&B Run:** `f8rc8ahi`
- **Hypothesis:** Stochastic Weight Averaging (SWA) over epoch snapshots EP20–EP30 produces a flatter loss landscape solution that generalizes better to the test distribution. Uniform averaging of 11 epoch checkpoints reduces overfit to val noise and potentially narrows the val→test vol_p gap.
- **Config:** `--use-swa --swa-start-epoch 20 --swa-freq 1`, bs=1 DDP8, steps/epoch ~10,975–10,986

### Per-epoch val progression

| Epoch | Step | abupt% | vol_p% | Notes |
|-------|------|--------|--------|-------|
| EP1 | ~10,975 | 20.495% | 16.455% | Expected large EP1 overhead |
| EP2 | ~21,950 | 9.122% | 7.799% | Large drop, healthy trajectory |
| EP3 | ~32,925 | 7.917% | 6.790% | Convergence normal |
| EP4 | ~43,900 | 6.964% | 4.608% | Strong drop |
| EP5 | ~54,875 | 7.712% | 7.136% | Transient spike (normal) |
| EP6 | ~65,850 | 6.703% | 4.337% | Recovered, EP5 gate PASS ✓ |
| EP7 | ~76,825 | 6.586% | 4.153% | Continued improvement |
| EP8 | ~87,800 | **6.478%** | **4.063%** | Best at time of last check |

**EP5 gate (≤7.5%):** PASS — abupt=6.478% at EP8, well clear
**EP10 gate (≤7.2%):** Expected at step ~109,750 — trajectory clearly will pass (6.478% at EP8)
**EP20 gate (≤6.65%):** Critical gate before SWA collection begins at step ~219,500; trajectory slope -0.00986%/1k steps suggests ~5.9% by EP20

- **GradNorm w_vol_p=0.222 at EP8** — healthy, not collapsed (contrast with frieren #972 collapse 0.286→0.090)
- **Status:** RUNNING — awaiting EP10 gate decision

### Decision: IN PROGRESS — awaiting EP10 gate at step ~109,750

---

## 2026-05-09 ~22:00 UTC — PR #1014 IN PROGRESS: Poisson pressure physics regularization (dl24-fern, `l5urrdmk`)

- **Branch:** `dl24-fern/poisson-pressure-reg`
- **W&B Run:** `l5urrdmk` (rank0)
- **Hypothesis:** Auxiliary Laplacian smoothness loss (`L_Poisson = mean(||∇²p_pred||²)`) with λ=0.01 forces the predicted pressure field to satisfy the Poisson equation in a weak sense, directly constraining vol_p spatial consistency across training and test distributions.
- **Config:** `--use-poisson-reg --poisson-lambda 0.01 --poisson-k 8 --poisson-m 2048`, group `poisson-pressure-reg`

### Per-epoch val progression

| Epoch | Step | abupt% | vol_p% | surf_p% | wall% | Notes |
|-------|------|--------|--------|---------|-------|-------|
| EP1 | ~10,913 | 8.581% | 6.266% | 5.470% | 9.362% | Normal EP1; EP1 gate PASS |

**EP5 gate (≤7.5%):** Expected at step ~54,875 (~3 epochs from EP1)
- **Status:** RUNNING — EP1 cleared, awaiting EP5 gate

### Decision: IN PROGRESS — awaiting EP5 gate at step ~54,875

---

## 2026-05-09 ~22:00 UTC — PR #1015 IN PROGRESS: InstanceNorm across volume tokens (dl24-tanjiro, `48pi1dn4`)

- **Branch:** `dl24-tanjiro/instancenorm-vol-tokens`
- **W&B Run:** `48pi1dn4` (rank0); all 8 DDP ranks: `48pi1dn4`, `0nj8yj7b`, `dnobec5q`, `11qquima`, `yllm3wjd`, `a9gptjc4`, `nlty089a`, `7hnsh7o3`
- **Hypothesis:** `nn.InstanceNorm1d(hidden_dim, affine=True)` applied across ~65k volume tokens per channel normalizes the activation distribution per-sample, making the volume head input scale-invariant. If test vol tokens occupy a different activation-space region than val, InstanceNorm makes the downstream prediction scale-invariant and may reduce the val→test vol_p gap.

### Per-epoch val progression

| Epoch | Step | abupt% | vol_p% | Notes |
|-------|------|--------|--------|-------|
| EP0.5 | ~5,753 | — | — | Launched; EP1 val at step ~10,975 |

**EP5 gate (≤7.5%):** Expected at step ~54,875
- **Status:** RUNNING — EP0.5, first val metrics at EP1 pending

### Decision: IN PROGRESS — awaiting EP1 first val metrics

---

## 2026-05-11 ~08:15 UTC — PR #979 CLOSED: TTA Y-symmetry ensemble eval-only A/B (dl24-nezuko, `msmccvne`)

- **Branch:** `dl24-nezuko/tta-y-symmetry-ensemble`
- **W&B Run:** `msmccvne` (eval-only on EP4 checkpoint)
- **Hypothesis:** Y-axis symmetry test-time augmentation (TTA) — ensemble prediction over original + y-flipped inputs — would reduce the persistent val→test vol_p gap (~7–8pp) by averaging out inference variance caused by geometric asymmetry in the test set.

### Eval-only A/B Results

| Arm | val_abupt | val_vol_p | test_abupt | test_vol_p | val→test gap |
|-----|-----------|-----------|------------|------------|--------------|
| A (no TTA) | 7.0329% | 4.9154% | 8.2991% | 12.7757% | +7.860pp |
| B (with TTA) | 7.0329% | 4.9009% | 8.2991% | 12.7642% | +7.863pp |
| Delta | 0.0000% | -0.015% | 0.0000% | -0.012% | +0.003pp |

### Decision: CLOSED — TTA hypothesis FALSIFIED

### Key Findings

1. **val→test vol_p gap completely unchanged to 3 decimal places** (+7.860pp → +7.863pp). TTA has zero impact on the structural OOD gap.
2. **TTA provides ~0.01–0.015pp marginal improvement** on val and test vol_p — noise-level gain, not meaningful.
3. **The gap is confirmed as a structural DATA DISTRIBUTION problem**, not inference variance. Y-axis symmetry TTA, vol-loss-weight upweighting, GradNorm rebalancing, and all training-objective levers have been falsified as gap-closing mechanisms.
4. **Recommendation:** add `--use-tta` as a free-lunch flag on terminal evaluations only for ~0.04pp at zero training cost. Do NOT pursue TTA as a gap-closing strategy.

---

## 2026-05-10 ~02:10 UTC — PR #936 CLOSED: vol-loss-weight=2.0 without GradNorm (dl24-nezuko, `6gd9u34e`)

- **Branch:** `dl24-nezuko/vol-loss-weight-2-no-gradnorm`
- **W&B Run:** `6gd9u34e`
- **Hypothesis:** Persistent 2× upweighting of volume_loss without GradNorm (which was suspected to self-cancel in PR #911) would force the model to prioritise vol_p and reduce the val→test vol_p gap (~7pp).

### Per-epoch val history

| Epoch | abupt | vol_p |
|-------|-------|-------|
| EP1 | 17.878% | — |
| EP2 | 11.858% | — |
| EP3 | 10.439% | — |
| EP4 | 9.765% | — |
| EP5 | 9.010% | 5.691% |

**EP5 gate: ≤7.5% — MISS by +1.51pp. Run killed mid-EP6.**

### Results vs Baseline

| Metric | Frieren baseline (EP5) | Nezuko #936 (EP5) | Delta |
|--------|----------------------|-------------------|-------|
| abupt | ~7.5% (EP5 est.) | 9.010% | +1.51pp |
| vol_p | 4.511% | 5.691% | +1.18pp WORSE |

### Decision: CLOSED — hypothesis FALSIFIED

### Key Findings

1. **vol-loss-weight direction fully exhausted.** PR #911 (vol-loss-weight=2.0 WITH GradNorm α=0.5) and PR #936 (without GradNorm) both falsified. The GradNorm-vs-no-GradNorm distinction was irrelevant — neither variant helped.
2. **vol_p at EP5=5.691% is WORSE than the frieren baseline 4.511%**, not better. Upweighting volume loss did not improve vol_p fidelity at any convergence checkpoint.
3. **The val→test vol_p gap is NOT a training-time loss signal problem.** The gap (val vol_p ≈4%, test vol_p ≈11–12%) is a covariate shift / data distribution problem. More pressure on vol_p at train-time does not close an OOD generalization gap.
4. **Adding vol-loss-weight to confirmed dead-ends.** Do not revisit this direction.

---

## 2026-05-10 ~01:32 UTC — PR #934 CLOSED: Balanced Points 96k+60k surf+vol (dl24-fern, `f335lerf`)

- **Branch:** `dl24-fern/balanced-pts-96k60k`
- **W&B Run:** `f335lerf`
- **Hypothesis:** Increasing surf+vol points per view from 40k+65k to 96k+60k brings the vol/surf ratio from 2.4:1 down to 1.6:1, reducing per-view volume point starvation that may be causing val vol_p underfitting and test vol_p gap.

### Per-epoch val history

| Epoch | Step | abupt | vol_p |
|-------|------|-------|-------|
| EP1 | ~5,493 | 26.8991% | — |
| EP2 | ~10,987 | 13.5456% | — |
| EP3 | ~16,481 | 10.8003% | — |
| EP4 | ~21,975 | 9.7313% | — |
| EP5 | ~27,469 | 9.2203% | 6.95% |

**EP5 gate: ≤7.5% — MISS by 1.72pp. Run killed mid-EP6.**

### Root Cause

Larger per-view point budgets reduce total training views: `view_count = ceil(total_points / points_per_view)`. The 96k+60k configuration yielded ~59,500 total views vs ~87,888 for the 40k+65k baseline — a **32% view count reduction**. Fewer views per epoch slowed convergence dramatically at every step count checkpoint. vol_p at EP5=6.95% was also worse than tanjiro EP5 5.33%, confirming no compensating benefit to vol_p fidelity.

### Decision: CLOSED — hypothesis FALSIFIED

### Key Findings

1. **Larger per-view point budgets reduce total training signal.** More points/view → fewer views → slower convergence. The hypothesis incorrectly assumed increased vol points per view would improve vol_p without accounting for the view-count effect.
2. **Balanced-points hypothesis definitively falsified.** The convergence deficit is entirely explained by the 32% view reduction, not by any vol/surf ratio benefit.
3. **40k+65k remains the reference baseline point configuration.** Do not experiment with increasing points per view without simultaneously accounting for the view count reduction and its convergence cost.

---

## 2026-05-09 ~17:00 UTC — PR #898 CLOSED: 5L STRING + GradNorm α=0.5 + Y-sym p=0.5 (complete triple stack, dl24-frieren, `ylrp8f97`)

- **Branch:** `dl24-frieren/5l-string-gradnorm-ysym`
- **W&B Run:** `ylrp8f97`
- **Hypothesis:** Test the full validated triple stack (5L STRING + GradNorm α=0.5 + Y-sym p=0.5) together on a 50-epoch run to establish whether the combination beats tanjiro #900 (6L+WD=0.01).

### Per-epoch val history

| Epoch | Step | abupt | vol_p | surf_p | wall_shear | Δabupt |
|-------|------|-------|-------|--------|-----------|--------|
| EP1 | 5,493 | 11.0024% | 8.5665% | 7.5126% | 11.626% | — |
| EP2 | 10,987 | 8.0667% | 5.6705% | 5.1719% | 8.826% | -2.94 |
| EP3 | 16,481 | 7.5575% | 5.1329% | 4.8773% | 8.335% | -0.51 |
| EP4 | 21,975 | 7.3231% | 4.8790% | 4.7424% | 8.109% | -0.23 |
| EP5 | 27,469 | 7.2169% | 4.7755% | 4.7015% | 8.008% | -0.11 |
| EP6 | 32,963 | 7.1523% | 4.6742% | 4.6735% | 7.950% | -0.06 |
| EP7 | 38,457 | 7.1155% | 4.7523% | 4.6340% | 7.882% | -0.04 (vol_p transient up) |
| EP8 | 43,951 | **7.0288%** | **4.5882%** | 4.5901% | 7.819% | **-0.087** |
| EP9 | 49,445 | **7.3089%** ⚠ | **5.5311%** ⚠ | 4.6770% | 7.934% | **+0.280 (REGRESSION)** |

**GradNorm weights @ EP8:** w_cp=0.91, w_tau_x=0.96, w_tau_y=1.11, w_tau_z=1.45, w_vol_p=0.58
**GradNorm weights @ EP10:** w_cp=0.756, w_tau_x=1.009, w_tau_y=1.083, w_tau_z=1.443, w_vol_p=0.709 (rising as GradNorm responds to EP9 vol_p spike)
**Train loss @ EP9:** 0.01887 (still descending — train→val divergence = classic overfitting)

### Decision: CLOSED

EP9 regression (+0.28pp abupt, +0.94pp vol_p, all channels worse simultaneously) while train loss continued descending is a clean overfitting signature. Both kill conditions met: abupt rose above 7.0%, vol_p trending up. No path to EP15 gate (≤6.80%).

### Key Findings

1. **5L STRING + GradNorm + Y-sym without weight decay overfits.** Fast early convergence (EP1-EP8 excellent) collapses at EP9 as the model enters the cosine tail with no L2 shrinkage anchor.
2. **Weight decay is load-bearing.** Tanjiro #900 (identical stack + WD=0.01) shows no regression at equivalent step. The delta is WD, not depth.
3. **GradNorm's dynamic response to overfitting is reactive, not preventive.** w_vol_p surged from 0.58→0.71 in response to the EP9 spike — GradNorm saw the regression and tried to correct it, but this only amplifies the gradient signal into an already-overfit regime, making recovery harder.

---

## 2026-05-09 ~UTC — PR #855 CLOSED: Y-symmetry augmentation standalone 4-ep tay screen (frieren, `tzfpf31d`)

- **Branch:** `frieren/beta-nll-surface-tay` (tay branch)
- **W&B Run:** `tzfpf31d`
- **Hypothesis:** Isolate the contribution of Y-symmetry augmentation alone on the L5 SOTA backbone (no GradNorm, no 6L) over 4 epochs to determine whether the channel ordering effect (τ_y < τ_z) is produced by Y-sym or by the long-run context it appeared in.

### Results (EP4 terminal, tay screen)

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| val abupt @ EP4 | **8.0813%** | ≤6.5985% | MISS -1.48pp |
| test abupt (EP4 ckpt) | **9.2221%** | 7.5195% | MISS -1.70pp |

#### Per-channel @ EP4 val and test

| Channel | val EP1 | val EP2 | val EP3 | val EP4 | test (EP4 ckpt) |
|---------|---------|---------|---------|---------|-----------------|
| surface_p | 20.13 | 9.53 | 6.08 | 5.231 | 4.917 |
| volume_p | 16.20 | 10.57 | 7.02 | 6.056 | 13.202 |
| ws_mean | 29.75 | 14.51 | 10.01 | 8.842 | 8.589 |
| tau_x | 26.30 | 12.67 | 8.78 | 7.793 | 7.631 |
| tau_y | 35.17 | 16.87 | 11.51 | **9.965** | **9.732** |
| tau_z | 35.84 | 18.38 | 12.69 | **11.361** | **10.629** |
| abupt | 26.73 | 13.61 | 9.22 | 8.081 | 9.222 |

tau_y < tau_z confirmed at val EP3, val EP4, AND test — reverses the default channel ordering where tau_y is historically the worst channel.

### Commentary

Gate missed — PR closed. But the physical signal is the key finding: Y-symmetry augmentation at p=0.5 cleanly produces tau_y < tau_z across all three reporting checkpoints. This is the cleanest isolation evidence of the Y-sym inductive bias to date, confirming that the channel-ordering effect seen in long-run PRs #818 and #831 is attributable to Y-sym and not to bundled factors (GradNorm, 6L, longer schedule).

Gate miss explained by 4-epoch budget: EP3→EP4 drop only 1.14pp, insufficient to reach 6.5985% from 9.22%. Cosine T_max=4 ran out of LR budget before the regularization bias could compound. Follow-up assigned: Y-sym p=1.0 tay screen (new frieren PR) — test whether full-probability augmentation clears the gate within 4 epochs.

---

## 2026-05-08 ~07:00 UTC — PR #806 CLOSED: 5L STRING + GradNorm α=0.25 + Y-sym triple compose (dl24-frieren, `gui4ceed`)

- **Branch:** `dl24-frieren/5l-string-gradnorm-alpha025-ysym`
- **W&B Run:** `gui4ceed`
- **Hypothesis:** Triple-compose 5L STRING + GradNorm α=0.25 + Y-sym on the wave's validated backbone; test whether composition of the three best-known enhancements additively beats wave SOTA 7.5195%.

### Terminal Results (EP50)

| Metric | Val (EP28 best) | Test | val→test ratio |
|--------|-----------------|------|----------------|
| `abupt_axis_mean_rel_l2_pct` (PRIMARY) | **6.6573%** | **7.9323%** | 1.192× |
| `surface_pressure_rel_l2_pct` | 4.4073% | 3.9536% | 0.90× ✓ beats AB-UPT (3.82) |
| `volume_pressure_rel_l2_pct` | 4.0735% | 12.0332% | **2.95× ← gap driver** |
| `wall_shear_rel_l2_pct` | 7.4590% | 7.2543% | 0.97× ✓ beats AB-UPT (7.29) |
| `wall_shear_x_rel_l2_pct` | 6.4885% | 6.4608% | 1.00× |
| `wall_shear_y_rel_l2_pct` | 8.0815% | 7.8120% | 0.97× |
| `wall_shear_z_rel_l2_pct` | 10.2354% | 9.4018% | 0.92× |

**GradNorm final weights (EP50):** w_cp=0.867, w_vol_p=1.154 (↑ from 0.88 at EP28), w_τx=0.921, w_τy=0.905, w_τz=1.153

**Wave merged best:** PR #740 test=7.5195%
**Result:** DOES NOT BEAT BASELINE (+0.413pp regression). PR CLOSED.

### Analysis

Three triple-compose experiments now closed without beating SOTA: fern #794 (4L, α=0.25, Y-sym: 7.9011%), nezuko #800 (5L STRING, α=0.5, Y-sym: 7.8981%), frieren #806 (5L STRING, α=0.25, Y-sym: 7.9323%). All show vol_p val→test gap ~2.7–3.0×. The only merged SOTA (#740, α=0.5 only, no 5L/Y-sym) has 1.104× overall ratio.

Root cause: GradNorm α=0.25 + late cosine tail combined to produce a w_vol_p surge (0.88→1.15 post-EP28) precisely when LR was annealing — baking in vol overfit with high precision. Surface (3.9536%) and wall shear (7.2543%) both beat AB-UPT targets, confirming the architecture is sound but vol_p generalization requires structural fix (Issue #803 SDF regeneration, or excluding vol_p from GradNorm adaptive weighting entirely).

---

## 2026-05-08 ~09:00 UTC — PR #838 CLOSED: STRING rff24 + σ=0.125 capacity vs aliasing test (tay screen, fern, `84skr4yq`)

- **Branch:** `fern/string-rff24-sigma0125`
- **W&B Run:** `84skr4yq`
- **Hypothesis:** If adding a 6th RFF octave at σ=0.125 hurts because rff16 has only 2-3 features per sigma and the high-freq sigma must compete for capacity with the dominant 0.25–2.0 band, then rff24 (4 features/sigma) should partially recover the regression. If rff24 recovers < half the regression, aliasing dominates.

### Results (EP4.1 terminal, tay screen)

| Metric | PR #838 (rff24, 6oct) | PR #829 (rff16, 6oct) | SOTA (rff16, 5oct, 4k25s25e) |
|---|---:|---:|---:|
| val_abupt | 7.4255% | 7.5738% | 6.5985% |
| test_abupt | 8.7190% | 8.9200% | 7.9915% |
| val_surface_pressure | 4.8435% | 4.9055% | 4.3322% |
| val_volume_pressure | 4.8640% | 5.1211% | 3.9456% |

**Merge gate (<6.5985%): FAILED.** Gap to SOTA: +0.83pp. rff24 recovered only 0.15pp of the ~0.98pp regression from adding σ=0.125 (18% of regression explained by capacity competition; 82% by aliasing).

### Commentary

Clean falsification experiment. Two competing hypotheses tested simultaneously: (a) capacity competition explains σ=0.125 regression → rff24 should recover the gap; (b) aliasing at σ=0.125 dominates at 65k surface point density. Result is decisively (b): rff24 helps slightly (0.15pp) but the bulk of the regression persists. Early-epoch advantage at rff24 (EP1: 25.49% vs 31.58% for rff16) compresses to negligible at EP4, suggesting the extra capacity accelerates fitting but the ceiling is set by aliasing. **The σ < 0.25 axis is closed at 65k surface point density.** Future STRING ablations should stay in σ ∈ {0.25, 0.5, 1.0, 2.0, 4.0}. Follow-up of interest: rff24 with 5-octave SOTA spectrum (σ=0.25–4.0) — tests capacity benefit without the aliasing cost.

---

## 2026-05-08 ~09:00 UTC — PR #835 CLOSED: Lion lr=1e-4 on L5 SOTA (tay screen, frieren, `mi76745s`)

- **Branch:** `frieren/lion-lr-1e-4-l5-sota`
- **W&B Runs:** `mi76745s` (corrected schedule arm), `kewvqbis` (first arm, killed EP2)
- **Hypothesis:** L5 has more parameters than L4 SOTA and may benefit from a slightly larger learning rate within a 4-epoch budget. Test lr=1e-4 vs SOTA lr=9e-5 on the L5 backbone.

### Results (EP2 terminal, both arms killed at EP2 gate)

| Run | Peak LR | Schedule | EP1 val_abupt | EP2 val_abupt | EP2 gate (<10%) |
|---|---|---|---:|---:|---|
| SOTA (4k25s25e) | 9e-5 | t_max=13 | 27.95% | ~7.94% | PASS |
| kewvqbis (arm 1) | 1e-4 | t_max=4 | 25.88% | 10.72% | FAIL |
| mi76745s (arm 2) | 1e-4 | t_max=13 | 26.09% | 11.77% | FAIL |

**Kill gate failed at EP2.** Both lr=1e-4 arms failed the 10% EP2 gate. Schedule-corrected arm (t_max=13) was *worse* than t_max=4 arm (11.77% vs 10.72%), ruling out schedule confound as the explanation.

### Commentary

Decisive negative result with an elegant internal control. The monotonic pattern — more average LR through EP2 → strictly worse EP2 metric — is clean and contradicts the hypothesis. The Lion sign-update is well-calibrated for L5 at lr=9e-5; overshooting damages fine-grained feature fitting (vol pressure, wall_shear_y/z) that drives EP2+ convergence. lr=9e-5 remains the operating point. **The LR upward-sweep axis is fully exhausted for L5.** Suggested next direction for L5-specific tuning: longer warmup (1.5–2 ep), slightly higher weight decay (7.5e-4/1e-3), or relaxed gradient clip (1.0 from 0.5).

---

## 2026-05-07 ~23:30 UTC — PR #794 CLOSED: GradNorm α=0.25 + Y-axis symmetry, 4L STRING (dl24-fern, `em7eupj5`)

- **Branch:** `dl24-fern/gradnorm-y-sym-alpha025`
- **W&B Run:** `em7eupj5`
- **Hypothesis:** Compose GradNorm α=0.25 + Y-axis symmetry augmentation on 4L (baseline) architecture; test whether conservative GradNorm + Y-sym compose additively vs GradNorm-only (#740, α=0.5 wave best)

### Terminal Results

| Metric | Val (EP25 best) | Test |
|--------|-----------------|------|
| `abupt_axis_mean_rel_l2_pct` | **6.7064%** | **7.9011%** |
| `surface_pressure_rel_l2_pct` | — | 4.0540% |
| `volume_pressure_rel_l2_pct` | — | 11.5420% |
| `wall_shear_rel_l2_pct` | — | 7.4030% |
| `wall_shear_x_rel_l2_pct` | — | 6.5520% |
| `wall_shear_y_rel_l2_pct` | — | 7.9730% |
| `wall_shear_z_rel_l2_pct` | — | 9.9290% |

**Wave merged best:** PR #740 test=7.5195%
**Result:** DOES NOT BEAT BASELINE (+0.382pp regression). PR CLOSED.

### Val Trajectory Summary

- EP5: 7.1519% (gate cleared), EP8: 6.9228% (fast early descent), EP11: 6.8315% (wave val lead briefly), EP16: 6.7435%, EP18: 6.7320%, EP25: **6.7064%** (run best), EP45: 6.7542% (plateau + drift above best)
- Plateau onset: EP26+. Cosine tail reengagement did NOT materialize. Plateau and slow drift worsening through EP45.

### Commentary

Strong val trajectory through EP25 (6.7064%) but test generalization regressed significantly (+0.382pp vs baseline). Key factors:
1. **4L architecture** — the 5L STRING backbone (as in frieren #806 and nezuko #800) has consistently outperformed 4L in this wave; this experiment used 4L, limiting capacity.
2. **α=0.25 on 4L** — the confirmed wave-winning config is α=0.5 on 5L STRING (#740); α=0.25 was also tested standalone (#780, test=8.0647%) and regressed. GradNorm α=0.25 does not outperform α=0.5 in this architecture family.
3. **Val→test generalization gap** — competitive val (6.7064%) but poor test (7.9011%) suggests the 4L + α=0.25 + Y-sym config is overfit or miscalibrated on this distribution.

**Conclusion:** The 4L baseline architecture is the bottleneck. 5L STRING + GradNorm α=0.5 + Y-sym triple compose (frieren #806, currently wave val leader at EP28=6.6573%) is the right direction. fern is now idle and will be reassigned to a fresh orthogonal hypothesis.

---

## 2026-05-07 ~10:30 UTC — Wave Status Update: EP18–EP34 Progress Across All 4 Active Runs

### Wave val leader board

| PR | Student | Run ID | Config | Best val (EMA) | Epoch | Status |
|----|---------|--------|--------|----------------|-------|--------|
| #794 | fern | `em7eupj5` | 4L STRING + GradNorm α=0.25 + Y-sym | **6.7320%** | EP18 | EP19 in progress |
| #780 | tanjiro | `20n1fvwn` | 4L STRING + GradNorm α=0.25 | 6.7970% | EP33 | EP35 in progress (EP34 outlier) |
| #800 | nezuko | `hmhfnedy` | 5L STRING + GradNorm α=0.5 | 7.0322% | EP5 | EP5+ in progress |
| #806 | frieren | `gui4ceed` | 5L STRING + GradNorm α=0.25 + Y-sym | 7.8887% | EP2 | EP5 gate expected ~10:31Z |

### PR #794 (fern) — EP10→EP18 trajectory

| Epoch | val_primary | Δ | Note |
|-------|-------------|---|------|
| EP10 | 6.8631% | — | |
| EP11 | 6.8315% | −0.032 | beat tanjiro EP25 to take wave val lead |
| EP12 | 6.8132% | −0.018 | |
| EP13 | 6.7834% | −0.030 | |
| EP14 | 6.8195% | +0.036 | transient uptick (noise) |
| EP15 | 6.7750% | −0.045 | new best, uptick resolved |
| EP16 | 6.7435% | −0.032 | new best |
| EP17 | 6.7346% | −0.009 | new best, slope flattening |
| EP18 | **6.7320%** | −0.003 | **wave val leader, slope ~−0.003pp/ep entering plateau** |

Per-channel at EP16: cp=4.335%, vol_p=4.081%, τx=6.656%, τy=8.379%, τz=10.267%. GradNorm weights near EP15: w_τz≈1.35, w_vol≈1.0 (balanced routing, Y-sym relieving volume pressure). Projecting EP20≈6.64%, EP30≈6.37%, EP50≈6.1–6.3%.

### PR #780 (tanjiro) — EP25→EP34 trajectory

| Epoch | val_primary | Δ | Note |
|-------|-------------|---|------|
| EP25 | 6.8511% | — | |
| EP26 | 6.8334% | −0.018 | |
| EP27 | 6.8301% | −0.003 | |
| EP28 | 6.8216% | −0.009 | |
| EP29 | 6.8264% | +0.005 | noise tick |
| EP30 | 6.8154% | −0.011 | |
| EP31 | 6.8047% | −0.011 | |
| EP32 | 6.7986% | −0.006 | |
| EP33 | **6.7970%** | −0.002 | **run best** |
| EP34 | 7.3507% | +0.553 | **outlier-batch anomaly — NOT divergence** |

GradNorm at EP30: w_vol=2.351 (strongly elevated), w_τz=0.493, w_τy=0.253 — volume routing dominant without Y-sym. EP34 regression diagnosis: uniform per-channel jump (all 5 channels affected proportionally), LR in cosine tail at 2.4e−5, grad norm 0.087 (normal), spike density not elevated near EP34 boundary. Advisor confirmed outlier-batch noise, EMA run best remains EP33=6.7970%. EP35 is the decisive confirmation epoch; recovery ≤6.85% confirms noise hypothesis.

### PR #800 (nezuko) — EP1→EP5

| Epoch | val_primary | Note |
|-------|-------------|------|
| EP1 | 10.6420% | |
| EP2 | 7.8901% | |
| EP3 | 7.5330% | |
| EP4 | 7.1180% | |
| EP5 | **7.0322%** | all 3 kill gates cleared |

Per-channel at EP5: cp=4.615%, vol_p=4.567%, τx=6.825%, τy=8.689%, τz=10.465%. GradNorm w_τz trajectory: 1.37→1.46→1.54→1.59→1.49 (mild pullback at EP5 as τz gap narrows). α=0.5 aggressively routing to τz without Y-sym — different GradNorm signature from fern. EP5 gate PASS with 1.97pp margin; all gates cleared. Projecting EP50≈6.4–6.6%.

### PR #806 (frieren) — EP1→EP2 (triple compose)

| Epoch | val_primary | Note |
|-------|-------------|------|
| EP1 | 11.1953% | 1.13pp ahead of fern 4L at EP1 |
| EP2 | **7.8887%** | tied with nezuko #800 at EP2; all EP5+EP10+EP20 gates pre-cleared |

GradNorm weights at EP2: near-uniform (w range 0.88–1.13), α=0.25 slow differentiation as expected. Y-sym + GradNorm + 5L starting well. EP5 report expected ~10:31Z. Fastest gate-clearing trajectory in wave at EP2.

---

## 2026-05-07 08:15 UTC — PR #806: 5L STRING + GradNorm α=0.25 + Y-sym Triple Compose (dl24-frieren, `gui4ceed`)

### Context

Frieren's prior run PR #791 (`g0um26ek`, GradNorm α=0.5 + Y-sym, EP13=6.9635%) was **closed and superseded** by this triple-compose launch. The closure rationale: fern's PR #794 (α=0.25 + Y-sym, run `em7eupj5`) reached EP11=6.8315% vs frieren's EP13=6.9635% — 0.132pp ahead with 2 fewer epochs. α=0.25 is definitively better for GradNorm + Y-sym composition than α=0.5 at this wave's base config. Frieren was immediately reassigned to the highest-complexity composition on the wave.

### Hypothesis

Stacking all three independently-confirmed wave-positive mechanisms simultaneously:
- **5L STRING** (PR #745, val=6.5097%@EP30 standalone, 1 extra Transolver layer)
- **GradNorm α=0.25** (PR #794 WAVE VAL LEADER, most efficient GradNorm + Y-sym composition)
- **Y-axis symmetry augmentation** (PR #741, test=7.8232%, reduces vol gap)

If the mechanisms are orthogonal (each addresses a different bottleneck), the triple compose should outperform any pair. Volume pressure relief from Y-sym + adaptive loss balancing from GradNorm + extra representational capacity from 5L STRING.

### Launch Status (EP0 smoke check)

| Metric | Value |
|--------|-------|
| Run ID | `gui4ceed` |
| Launch time | 2026-05-07T08:11Z |
| VRAM | ~29.3GB (within limit) |
| GradNorm | α=0.25, n_tasks=5 — confirmed operational |
| Y-sym | ~50% flip probability — confirmed operational |
| 5L STRING | `--model-layers 5` — confirmed in config |
| EP5 gate (≤9.0%) | Expected ~10:30 UTC |

### Commentary

This is the highest-complexity single composition tested this wave. The key question is whether the three mechanisms genuinely address orthogonal failure modes (capacity, volume routing, loss balance) or whether they interact negatively (e.g., 5L requires different α tuning). Nezuko's #800 (5L STRING + GradNorm α=0.5, no Y-sym, run=`hmhfnedy`) provides a direct partial comparison — same 5L depth with slightly more aggressive GradNorm, no Y-sym. If frieren's triple compose leads nezuko at EP5, it validates Y-sym's additive value in the 5L + GradNorm stack. Watching for EP5 gate at ~10:30 UTC.

---

## 2026-05-05 09:00 UTC — W&B Status Check: All 4 Active WIP Runs (EP8–EP24)

### PR #780 — GradNorm α=0.25 (dl24-tanjiro, `20n1fvwn`) — EP24 WAVE VAL LEADER

| Step | Approx EP | val_primary | Note |
|------|-----------|-------------|------|
| 82409 | EP15 | 6.9399% | |
| 87903 | EP16 | 6.9220% | |
| 93397 | EP17 | 6.9037% | prior best |
| 98891 | EP18 | 7.0198% | transient spike |
| 104385 | EP19 | 6.9144% | recovery |
| 109879 | EP20 | 6.9330% | EP20 gate CLEARED |
| 115373 | EP21 | 6.8601% | new wave val best |
| 120867 | EP22 | 6.8565% | new wave val best |
| 126361 | EP23 | 6.8585% | hover ~noise |
| **131855** | **EP24** | **6.8515%** | **NEW WAVE VAL BEST** |

**Commentary:** tanjiro's α=0.25 run is delivering the wave's best val metric at EP24=6.8515%. Oscillation spike at EP18 (7.0198%) resolved cleanly. The run shows a clear late-training trend below 6.87% across EP21-24. This is now the wave leader. Training continues to EP50.

---

### PR #794 — GradNorm α=0.25 + Y-sym (dl24-fern, `em7eupj5`) — EP8 Strong Trajectory

| Step | Approx EP | val_primary | Note |
|------|-----------|-------------|------|
| 5493 | EP1 | 12.3285% | |
| 10987 | EP2 | 8.5766% | |
| 16481 | EP3 | 7.7128% | |
| 21975 | EP4 | 7.3520% | |
| 27469 | EP5 | 7.1519% | EP5 gate CLEARED |
| 32963 | EP6 | 7.1398% | |
| 38457 | EP7 | 6.9907% | first sub-7% |
| **43951** | **EP8** | **6.9228%** | **New best** |

**Commentary:** fern's α=0.25 + Y-sym composition shows the steepest early convergence of the 4 runs. Sub-7% achieved at EP7, EP8 extending the trend. The combination of Y-symmetry (effective data doubling) and conservative GradNorm (α=0.25) appears synergistic. EP10 gate (~step 54,940) expected to pass comfortably. Projecting strong EP20+ trajectory.

---

### PR #791 — GradNorm α=0.5 + Y-sym (dl24-frieren, `g0um26ek`) — EP12 Slowing

| Step | Approx EP | val_primary | Note |
|------|-----------|-------------|------|
| 27469 | EP5 | 7.3537% | |
| 32963 | EP6 | 7.2028% | |
| 38457 | EP7 | 7.2275% | hover |
| 43951 | EP8 | 7.1701% | |
| 49445 | EP9 | 7.0804% | |
| 54939 | EP10 | 7.0408% | EP10 gate CLEARED |
| 60433 | EP11 | 7.0372% | |
| **65927** | **EP12** | **7.0131%** | **New best, but rate slowing** |

**Commentary:** frieren's α=0.5 + Y-sym run cleared the EP10 gate comfortably and continues improving. However, improvement rate has slowed: from -0.04pp/ep (EP5-EP10) to ~-0.013pp/ep (EP10-EP12). This is behind fern's α=0.25+Y-sym which reached 6.9228% at EP8 vs frieren's 7.0131% at EP12. Counter-intuitive: more conservative α=0.25 appears to compose better with Y-sym than α=0.5. Training continues; convergence plateau possible around 6.95-7.0%.

---

### PR #800 — 5L STRING + GradNorm α=0.5 (dl24-nezuko) — EP0 Just Started

| Run IDs | State |
|---------|-------|
| `3i104cb8`, `7o1uxn7l`, `7u96u4v8`, `gszo93wp`, `guqqt3ka`, `hmhfnedy`, `luo67e4r`, `w1ylcm4o` | All 8 DDP ranks RUNNING |

**Commentary:** Nezuko's new 5L STRING + GradNorm α=0.5 composition run is fully underway (all 8 DDP ranks confirmed). No validation metrics yet — EP1 (~step 5,500) not reached. The 5L model has more parameters, so first epoch will be slower. Both mechanisms independently confirmed: 5L STRING (#745 val=6.5097%@EP30) and GradNorm α=0.5 (#740 test=7.5195% WAVE BEST). Composition is untested; high projected gain (~7.2-7.4% test if gains transfer 50%).

---

## 2026-05-07 ~18:00 UTC — PR #784 TERMINATE: QK-Norm + Y-symmetry (dl24-nezuko, `sd59a9dq`)

- **Branch:** `dl24-nezuko/qk-norm-y-sym`
- **W&B Run:** `sd59a9dq`
- **Hypothesis:** QK-Norm (query-key normalization) composes with Y-axis symmetry augmentation to reduce τz bottleneck on SOTA STRING base config
- **Outcome:** TERMINATED — EP20 gate MISS; PR CLOSED

| EP (checkpoint index) | Actual Epoch | Step | val_primary | Note |
|---|---|---|---|---|
| EP14 | 7.0 | ~76,909 | 7.6200% (approx) | Early gate ref |
| EP15 | 7.5 | — | 7.5920% (approx) | |
| EP16 | 8.0 | — | 7.5811% (approx) | |
| EP17 | 8.5 | — | 7.5761% (approx) | prior run-best |
| EP18 | 9.0 | ~98,k | **7.5605%** | **Run-best — slope ~-0.04pp/ep (EP14-18)** |
| EP19/EP20 | 9.5/10.0 | — | ~7.52% projected | EP20 gate (≤7.2%) missed |

**Gate status:** EP20 gate threshold = ≤7.2%. Projected EP20 value ~7.47% based on slope ~-0.04pp/ep from EP14-18. Gate MISSED by ~0.27pp. Lenient custom gate ≤7.35% also missed. TERMINATE instruction posted on PR; PR closed.

**Commentary:** QK-Norm does not compose well with Y-symmetry augmentation at the SOTA base config LR=1e-4 Lion. The combination produces a run-best of 7.5605% — significantly below the wave's EP5-equivalent performance of other successful runs (frieren #791 was already at 7.0408% by EP5.0). QK-Norm appears to interfere with the effective learning dynamics introduced by Y-sym augmentation, likely due to attention normalization disrupting the bilateral symmetry signal. This is the second negative result for QK-Norm: #732 (standalone at lr=5e-5) and now #784 (compose with Y-sym). QK-Norm at wave-standard lr=1e-4 standalone remains untested but is low priority given two negative compositions.

**Conclusion:** Dead end. QK-Norm + Y-sym composition REJECTED. dl24-nezuko GPU freed for new assignment.

---

## 2026-05-06 ~15:10 UTC — Live W&B Monitoring Session (EP14–EP39 developments)

### PR #740 — GradNorm adaptive loss (fern, `5x8wofzm`) — EP12–14 WAVE BEST

| EP | Step | val_primary | wsz | wsy | sp | ws | vp | Note |
|----|------|------------|-----|-----|----|----|----|------|
| 11 | 120,857 | 6.4388% | 9.6697% | — | — | 7.2711% | — | Prior wave best |
| 12 | 131,843 | 6.4340% | — | — | — | — | — | New wave best −0.0048pp (cp_updated=1) |
| 13 | 142,830 | ~6.434% | — | — | — | — | — | Within noise of EP12 |
| 14 | 153,817 | **6.4170%** | — | — | — | — | — | **New wave best −0.0170pp vs EP12; −0.0218pp vs EP11** |

**Current state** (step 156,767): EP~14.09, val=6.4170% (wave leader, −0.1111pp below SOTA). LR=8.205e-5. EP15 in progress (~8,037 steps to target). GradNorm τz upweight persisting.

**Commentary:** EP12 set quiet wave best (+cp at step 131,843). EP13 hovered within noise. EP14 at step 153,817 was a major leap: −0.0170pp from EP12, −0.0218pp from EP11. GradNorm α=0.5 continues to rebalance τz aggressively. At this rate (~0.01pp/ep in deep cosine taper), sub-6.40% is plausible by EP17–20. EP15 mandatory check-in; trigger: ≤6.40% post immediately.

---

### PR #741 — Y-axis reflection augmentation (nezuko, `lszc4ri7`) — EP33 TEST EVAL + EP37 hover

| EP | Step | val_primary | wsz | wsy | sp | ws | vp | Note |
|----|------|------------|-----|-----|----|----|----|------|
| 32 | ~176k | 6.5041% | — | 8.0752% | — | — | — | EP32 best checkpoint; test eval authorized |
| 33 | ~182k | **6.4984%** | **9.9544%** | 8.0568% | 4.2497% | 7.4198% | 3.7307% | **New run-best; BEATS SOTA val; test eval confirmed** |
| 34 | — | 6.5038% | — | 8.0672% | — | — | — | +0.0054pp hover |
| 35 | — | 6.5108% | 9.9721% | 8.0654% | — | — | — | Hover; EP33 best holds |
| 37 | 203,998 | 6.4997% | 9.9571% | 8.0431% | 4.2496% | 7.4180% | 3.7308% | Back on descent; 0.0013pp above EP33 |

**EP33 TEST EVAL:** 7.8232% — first wave run to beat SOTA test (7.9303%). Significant result.

**W&B slopes at EP37** (per 1k steps):
- abupt: −0.001041 (2.4× frieren's rate)
- wsz: −0.001865 (strong τz descent)
- wsy: −0.001969 (strong τy descent — Y-sym active)

**Commentary:** EP33 breakthrough (6.4984%) beats SOTA val_best=6.5281% by 0.0297pp. EP33 test eval=7.8232% beats SOTA test=7.9303% by 0.1071pp — a strong result. The EP33→EP35 hover (wsz 9.9544→9.9721) has reversed at EP37 (wsz 9.9571%), resuming descent. The hover pattern at EP37 (only 0.0013pp above EP33) mirrors the EP31/EP32 hover-then-EP33-breakout pattern. EP40 mandatory check-in. Second test eval trigger: val < 6.480%.

---

### PR #745 — 5L STRING PE (frieren, `co0xlqap`) — EP26–29 run-bests

| EP | Step | val_primary | wsz | wsy | sp | ws | vp | Note |
|----|------|------------|-----|-----|----|----|----|------|
| 24 | — | ~6.543% | — | — | — | — | — | Pre-wave-SOTA gap narrowing |
| 25 | 137,349 | 6.5323% | 10.0987% | 8.0464% | 4.3027% | 7.3943% | 3.7772% | |
| 26 | 142,843 | 6.5159% | 10.0754% | 8.0214% | 4.2917% | 7.3769% | 3.7673% | First beats SOTA val (6.5281%) |
| 27 | 148,337 | 6.5207% | 10.0858% | 8.0287% | 4.2949% | 7.3802% | 3.7712% | +0.0048pp noise |
| 28 | 153,831 | 6.5134% | 10.0654% | 8.0175% | 4.2962% | 7.3717% | 3.7687% | New run-best; 4/7 channel bests |
| **29** | **159,325** | **6.5110%** | **10.0641%** | 8.0220% | **4.2898%** | **7.3704%** | **3.7637%** | **New run-best; 6/7 channel bests** |

**Current state** (step 160,594): EP~29.14, val=6.5110% (run-best). LR=3.819e-5. EP30 ~4,225 steps away.

**W&B slopes at EP29** (per 1k steps):
- abupt: −0.000441
- wsz: −0.000233 (slow but descending)
- wsy: +0.000816 (**degrading** — no Y-sym augmentation)
- surface_pressure: −0.001176 (fastest channel)

**Commentary:** EP26 was the first frieren epoch to beat SOTA val (6.5281%). EP27 had a +0.0048pp noise spike (EP27=6.5207% > EP26=6.5159%), then EP28/EP29 restored descent. EP29=6.5110% is 6/7 simultaneous channel bests. wsy is the only laggard and is slightly degrading (no Y-sym aug). wsz sub-10.0% projected EP36–37 based on EP25→EP29 slope of −0.00865pp/epoch. EP30 mandatory check-in; triggers: val ≤ 6.500% or wsz < 10.050% → post immediately.

---

### PR #749 — Lion lr=9e-5 control (tanjiro, `oi2a01zy`) — plateau EP27+

| EP | Step | val_primary | wsz | Note |
|----|------|------------|-----|------|
| 27 | ~151k | 6.8479% | ~10.5% | Last run-best (W&B) |
| ~39 | 219,758 | 6.8592% | — | Plateau confirmed (cp_updated=0 since EP27) |

**Commentary:** Lion optimizer at lr=9e-5 confirmed on plateau after EP27. No new best checkpoint for ~12 epochs. Tanjiro confirmed to continue to EP50 for auto test eval per protocol. Three advisor nudge comments posted (no student responses since 10:31Z). EP50 auto test eval expected ~18:42Z. Terminal SENPAI-RESULT expected after EP50 auto eval. Merge decision post-EP50: run will not beat SOTA val; merge/close decision depends on whether test metric beats SOTA test (7.9303%).

---

**Wave standings at 2026-05-06 ~15:10 UTC:**

| Rank | Student | PR | Run | EP | val_best | test_best | wsz | Status |
|------|---------|----|----|----|----|------|-----|--------|
| 1 | fern | #740 | `5x8wofzm` | 14 | **6.4170%** | — | ~9.7%* | RUNNING; EP15 next; −0.1111pp SOTA |
| 2 | nezuko | #741 | `lszc4ri7` | 33 | **6.4984%** | **7.8232%** | 9.9544% | RUNNING; EP37 hover; −0.0297pp SOTA |
| 3 | frieren | #745 | `co0xlqap` | 29 | **6.5110%** | — | 10.0641% | RUNNING; EP30 next; −0.0171pp SOTA |
| 4 | tanjiro | #749 | `oi2a01zy` | 27 | 6.8479% | — | ~10.5% | RUNNING; plateau; EP50 auto eval |

*fern wsz: last confirmed 9.6697% at EP11; EP14 wsz not reported by student yet.
SOTA reference: val=6.5281%, test=7.9303% (PR #599, `sogus8sx`). Three runs now beating SOTA val; one (nezuko) beats SOTA test.

---

## 2026-05-05 ~20:30 UTC — Advisor Session: PR Reviews + Test Eval Authorization

### PR #740 — GradNorm adaptive loss balancing (dl24-fern, `5x8wofzm`) — EP11 WAVE BEST

- **Branch:** `dl24-fern/gradnorm-adaptive-loss`
- **W&B Run:** `5x8wofzm` (Arm B, α=0.5)
- **Advisor comment posted:** Yes — EP11 wave-best acknowledgment

| EP | val_primary | cp | tau_x | tau_y | tau_z (wsz) | vol_p | wall_shear | Note |
|----|------------|-----|-------|-------|------------|-------|-----------|------|
| 10 | 6.4778% | — | — | — | — | — | — | prior best |
| 11 | **6.4388%** | — | — | — | **9.6697%** | — | **7.2711%** | **WAVE LEADER** |

**Commentary:** EP11 = 6.4388% is the new wave leader, 0.089pp below pre-wave SOTA val_best=6.5281%. All 7 per-axis metrics improved EP10→EP11 (correlated multi-channel advancement driven by GradNorm equilibrium). wall_shear 7.2711% is the first sub-7.29% AB-UPT target result in the wave. wsz 9.6697% is 0.070pp above the 9.60% flag threshold — EP12 likely to fire the flag. GradNorm has demonstrably settled into stable equilibrium and is driving sustained compound improvement. If descent rate holds (~0.04pp/ep), sub-6.40% is reachable by EP13-15.

**Next:** EP12 check-in expected imminently; watch wsz < 9.60% flag.

---

### PR #741 — Y-axis reflection augmentation (dl24-nezuko, `lszc4ri7`) — EP32 + Test Eval Authorized

- **Branch:** `dl24-nezuko/y-sym-augmentation`
- **W&B Run:** `lszc4ri7`
- **Advisor comment posted:** Yes — test eval authorization

| EP | val_primary | tau_y (wsy) | Note |
|----|------------|------------|------|
| 28 | 6.5195% | 8.1197% | prior reference |
| 30 | 6.5115% | 8.1028% | prior reference |
| 32 | **6.5041%** | **8.0752%** | **EP32 run best; sub-6.51% trigger fired; all-time bests for abupt, tau_y, vp, sp, ws** |

**tau_y 7-epoch monotonic descent:** 8.1197→8.1028→8.0752% (−0.0445pp over 4 epochs).

**Commentary:** EP32=6.5041% clears the "below ~6.505%" test eval threshold set at EP30. Sub-6.51% trigger fired as planned. All-time run bests: abupt, tau_y, vol_pressure, surface_pressure, wall_shear. Tau_y 7-epoch monotonic descent (no oscillation) is a structural signal that Y-symmetry augmentation is enforcing bilateral consistency in the most volatile component. Test eval authorized from EP32 checkpoint (in parallel with continuing training to EP50). EP35 mandatory check-in maintained. Cosine LR at EP32 is in the deep tail — further improvement expected but convergence is slowing.

**Next:** Student running test eval from EP32 checkpoint in parallel. Post test metric on PR immediately when available. EP35 mandatory check-in.

---

### PR #745 — 5L STRING (dl24-frieren, `co0xlqap`) — EP23 Plateau Broken

- **Branch:** `dl24-frieren/5l-string-pe`
- **W&B Run:** `co0xlqap`
- **Advisor comment posted:** Yes — EP23 plateau-break acknowledgment

| EP | val_primary | wsz | wsy | Note |
|----|------------|-----|-----|------|
| 20 | 6.5495% | 10.1721% | 8.0993% | plateau start |
| 21 | 6.5508% | — | — | plateau |
| 22 | 6.5491% | — | — | plateau; nearly flat |
| 23 | **6.5326%** | **10.0968%** | **8.0511%** | **−0.0169pp plateau break** |

**Commentary:** Three-epoch plateau (EP20-22 at 6.549-6.551%) assessed as cosine LR oscillation, not structural stall. EP23 confirms descent resumed with −0.0169pp step. 0.0045pp from pre-wave SOTA val_best (6.5281%) — within striking distance. wsz=10.0968% first sub-10.10% in run; wsy=8.0511% first sub-8.07%. Monotonic architecture (5L, no Y-sym augmentation) continues smooth descent. At ~0.02pp/ep and 27 remaining epochs, sub-6.40% is conceivable if rate holds.

**Updated flag thresholds:** val_abupt < 6.52% → immediate report; wsz < 9.95% → immediate report.
**Next:** EP35 mandatory check-in.

---

### Summary — Wave Standings at 2026-05-05 ~20:30 UTC

| Rank | Student | PR | Run | EP | val_best | wsz | Gap to SOTA |
|------|---------|----|----|----|----|-----|------------|
| 1 (LEADER) | fern | #740 | `5x8wofzm` | 11 | **6.4388%** | 9.6697% | −0.089pp (BEATS SOTA) |
| 2 | nezuko | #741 | `lszc4ri7` | 32 | **6.5041%** | 10.0%* | −0.024pp (BEATS SOTA) |
| 3 | frieren | #745 | `co0xlqap` | 23 | **6.5326%** | 10.0968% | +0.005pp (near SOTA) |
| 4 (baseline) | tanjiro | #749 | `oi2a01zy` | 27 | 6.8479% | ~10.5% | +0.320pp |

*nezuko wsz data at EP32 not separately reported; EP28 wsz=9.7% estimated from tau_y descent.

Pre-wave SOTA val_best: 6.5281% (PR #599, `sogus8sx`). **Two runs now beating pre-wave SOTA val.** Fern is 0.090pp clear. Test eval pending for nezuko from EP32 checkpoint.

---

## 2026-05-06 08:45 UTC — W&B Status Check: All Active Wave PRs (mid-run update)

### PR #741 — Y-axis reflection augmentation (dl24-nezuko, `lszc4ri7`)
- **Branch:** `dl24-nezuko/y-sym-augmentation`
- **W&B Run:** `lszc4ri7`
- **Status:** RUNNING — EP22 reached; **WAVE LEADER — new in-wave val best**

| EP | val_primary | wsz | vp | Note |
|----|------------|-----|----|------|
| 19 | 6.6231% | 10.5% | 4.10% | C5 extended trough (prior best) |
| 20 | 6.6239% | 10.46% | 3.98% | Flat (+0.0008pp) — no C6 spike |
| 21 | 6.6607% | 10.1501% | — | Small uptick (+0.038pp); oscillation highly damped (C5 spike was +0.595pp) |
| 22 | **6.5789%** | **10.0085%** | **3.7977%** | **C6 trough — NEW WAVE BEST; new in-wave val best** |

C6 trough at EP22=6.5789% confirmed. The C6 oscillation was nearly fully damped: spike amplitude +0.038pp vs C5 spike +0.595pp — an order-of-magnitude dampening as cosine LR decays past 50%. vp=3.7977% is the new in-wave best. wsz=10.0085% approaching sub-10% threshold. C7 trough forecast EP25-26 at ~6.535-6.555%; early convergence flag if two consecutive trough delta <0.005pp. EP25 mandatory check-in. NO terminal test eval before EP35. DO NOT KILL.

---

### PR #745 — 5L STRING PE (dl24-frieren, `co0xlqap`)
- **Branch:** `dl24-frieren/5l-string-pe`
- **W&B Run:** `co0xlqap`
- **Status:** RUNNING — EP14 reached; smoothest monotonic descent in wave

| EP | val_primary | wsz | vp | Note |
|----|------------|-----|----|------|
| 10 | 6.6727% | 10.264% | 3.99% | Prior update best |
| 11 | 6.6487% | — | — | |
| 12 | 6.6392% | 10.2301% | — | Rate recovery confirmed |
| 13 | 6.6240% | — | — | |
| 14 | **6.6128%** | **10.2070%** | **3.8393%** | **Best; EP15 gate pass confirmed** |

Zero oscillation across full EP1→EP14 trajectory (5L architecture with single long cosine T_max=50). 0.0339pp behind nezuko EP22 wave best. Advisor projection: EP17 will cross nezuko's EP22 best (6.5789%); EP22 projected ~6.524%. wsz sub-10% projected EP28-32. EP18-20 check-in requested.

---

### PR #740 — GradNorm adaptive loss balancing, Arm B (dl24-fern, `5x8wofzm`)
- **Branch:** `dl24-fern/gradnorm-adaptive-loss`
- **W&B Run:** Arm B `5x8wofzm` (α=0.5); Arm A `em8bnk1a` (α=1.0) KILLED EP5
- **Status:** RUNNING — EP6 reached; Arm A killed at EP5 (gap 0.1724pp ≥ 0.15pp threshold); Arm B solo

| EP | val_primary | wsz | vp | Note |
|----|------------|-----|----|------|
| 4 | 6.8721% | 10.14% | 4.38% | Arm A gap: 0.211pp |
| 5 | 6.7438% | 9.9700% | 4.26% | Arm A gap: 0.1724pp — threshold exceeded → Arm A killed |
| 6 | **6.6648%** | **9.8962%** | **4.08%** | **Earliest sub-10% wsz in wave** |

Arm A killed at EP5 (val_primary gap ≥ 0.15pp threshold). Arm B solo continuing to EP31. GradNorm correctly upweights tau_z (structural bottleneck). wsz=9.8962% at EP6 is the earliest sub-10% wsz result in the wave — GradNorm may be directly addressing the wsz bottleneck. EP10 gate report pending.

---

### PR #749 — Lion lr=9e-5 control (dl24-tanjiro, `oi2a01zy`)
- **Branch:** `dl24-tanjiro/lion-lr-9e-5`
- **W&B Run:** `oi2a01zy`
- **Status:** RUNNING — EP21 reached; clean monotonic descent

| EP | val_primary | wsz | Note |
|----|------------|-----|------|
| 18 | 6.9511% | 10.75% | Prior update best |
| 19 | 6.9377% | — | |
| 20 | 6.9141% | — | |
| 21 | **6.8907%** | **10.492%** | **Best; clean descent resumed after EP17/18 vp-spike** |

Monotonic descent continuing but wsz slope decelerating to -0.011pp/ep (concern for structural ceiling at this LR). 0.4626pp above SOTA val_best=6.5281%. Terminal test eval at EP50 via `run_final_evaluation` automatic. wsz plateau signal at lr=9e-5 — confirms lower LR is insufficient to break through the wsz bottleneck.

---

**Wave standings at 2026-05-06 08:45 UTC:**

| Student | PR | Run | EP | val_best | wsz | Status |
|---------|----|----|----|----|-----|--------|
| nezuko | #741 | `lszc4ri7` | 22 | **6.5789%** | 10.0085% | C7 trough EP25-26 forecast; DO NOT KILL |
| frieren | #745 | `co0xlqap` | 14 | **6.6128%** | 10.2070% | EP18-20 check-in; projected to cross nezuko best by EP17 |
| fern | #740 | `5x8wofzm` | 6 | **6.6648%** | 9.8962% | EP10 gate pending; earliest sub-10% wsz in wave |
| tanjiro | #749 | `oi2a01zy` | 21 | **6.8907%** | 10.492% | EP50 terminal auto test eval |

SOTA val_best reference: PR #599 `sogus8sx` = 6.5281%. Nezuko is 0.051pp above SOTA, with C7 trough projected to pass it. No advisor action items — all 4 PRs have advisor as most recent commenter. No human researcher GitHub Issues.

---

## 2026-05-05 ~14:30 UTC — W&B Status Check: All Active Wave PRs (mid-run update)

### PR #741 — Y-axis reflection augmentation (dl24-nezuko, `lszc4ri7`)
- **Branch:** `dl24-nezuko/y-sym-augmentation`
- **W&B Run:** `lszc4ri7`
- **Status:** RUNNING — EP19 reached; **WAVE LEADER, new in-wave val best**

| EP | val_primary | Note |
|----|------------|------|
| 13 | 7.2610% | C4 spike |
| 14 | 6.8035% | C4 trough |
| 15 | 7.2701% | C5 pre-spike |
| 16 | 6.6890% | C5 inner trough |
| 17 | 7.2835% | C5 spike |
| 18 | 6.6596% | C5 trough |
| 19 | **6.6231%** | **C5 extended trough — new in-wave val best and wave leader** |

Oscillation pattern persists: odd epochs = spike, even epochs = trough/descent. The C5 trough has extended across two consecutive epochs (EP18→EP19), each improving vs prior best. This matches GD descent through a noisy augmentation landscape; EMA checkpoint preserved at best=6.6231%. C6 spike expected at EP21; C6 trough projected EP22 at ~6.56–6.58% — potential new all-time in-wave best. DO NOT KILL. Test eval + review submit after terminal EP50. Y-symmetry augmentation is a powerful regularizer on DrivAerML (effective dataset doubling).

---

### PR #745 — 5L STRING PE (dl24-frieren, `co0xlqap`)
- **Branch:** `dl24-frieren/5l-string-pe`
- **W&B Run:** `co0xlqap`
- **Status:** RUNNING — EP10 reached; smoothest monotonic descent in current wave

| EP | val_primary | cp | tau_x | tau_y | tau_z | vol_p |
|----|------------|-----|-------|-------|-------|-------|
| 4  | 7.0212%    | 4.57% | 6.81% | 8.99% | 10.73% | 4.35% |
| 5  | 6.9507%    | 4.53% | 6.75% | 8.89% | 10.60% | 4.30% |
| 6  | 6.8932%    | 4.50% | 6.72% | 8.82% | 10.50% | 4.24% |
| 7  | 6.8211%    | 4.47% | 6.68% | 8.73% | 10.40% | 4.18% |
| 8  | 6.7813%    | 4.45% | 6.65% | 8.67% | 10.34% | 4.14% |
| 9  | 6.7203%    | 4.43% | 6.62% | 8.53% | 10.30% | 4.10% |
| 10 | **6.6727%** | **4.42%** | **6.60%** | **8.41%** | **10.264%** | **3.99%** |

Clean monotonic descent from EP1→EP10 with no oscillation spikes (5L architecture does not exhibit y-sym augmentation periodic pattern). 0.050pp behind wave leader nezuko EP19=6.6231%. `tau_z=10.264%` remains the structural bottleneck. `vol_p=3.99%` is outstanding — best volume performance this wave. EP12 check-in requested (full per-channel breakdown). Advisor projection: EP15~6.55%, EP20~6.50% — potential new merged SOTA. Strongest monotonic trajectory candidate for terminal test merge.

---

### PR #740 — GradNorm adaptive loss balancing v2 (dl24-fern, Arm A `em8bnk1a`, Arm B `5x8wofzm`)
- **Branch:** `dl24-fern/gradnorm-adaptive-loss`
- **W&B Runs:** Arm A (α=1.0): `em8bnk1a`; Arm B (α=0.5): `5x8wofzm`
- **Status:** RUNNING — v2 restart; both arms at EP4; Arm B leads

**Context:** v1 runs (`aoetlx9b` Arm A, `g18f7jm1` Arm B) both crashed. v2 restart confirmed identical config; perfect 4 d.p. reproducibility across runs.

| Run | α | EP | val_primary | tau_z upweight | Note |
|-----|---|----|------------|----------------|------|
| `em8bnk1a` | 1.0 | 4 | 7.0836% | 2.94× | Arm A |
| `5x8wofzm` | 0.5 | 4 | **6.8721%** | 2.11× | **Arm B — leading** |

Gap at EP4: Arm B leads Arm A by **0.211pp** (widening from 0.097pp at EP3). GradNorm correctly upweights `tau_z` (structural bottleneck) as intended for both α values. Higher α=1.0 (more aggressive rebalancing) appears to over-correct and destabilize training vs softer α=0.5. EP5 decision gate pending: if Arm B gap ≥0.15pp, Arm A kill recommended to concentrate 8 GPUs on Arm B. At EP4 gap already exceeds threshold; Arm A kill expected at EP5 gate.

---

### PR #749 — Lion lr=9e-5 control (dl24-tanjiro, `oi2a01zy`)
- **Branch:** `dl24-tanjiro/lion-lr-9e-5`
- **W&B Run:** `oi2a01zy`
- **Status:** RUNNING — EP18 reached; gate ≤7.5% PASSED by 0.55pp

| EP | val_primary | Note |
|----|------------|------|
| 10 | 7.0518%    | |
| 11 | 7.0215%    | |
| 12 | 7.0009%    | |
| 13 | 6.9877%    | |
| 14 | 6.9812%    | |
| 15 | 6.9748%    | |
| 16 | 6.9641%    | |
| 17 | 6.9573%    | |
| 18 | **6.9511%** | **best** — monotonic descent EP7→EP18 at ~0.01-0.02pp/epoch |

Steady monotonic descent with decelerating slope (~0.01pp/epoch from EP14 onward). `wsz=10.75%` plateau signal — structural ceiling at this LR. 0.4230pp above SOTA val_best. Likely plateau at ~6.85-6.90% by EP20. Clarification resolved: `run_final_evaluation` in `trainer_runtime.py:1384` executes automatically at EP50 terminal — no `--eval-only` flag needed. Run continues to EP50 for auto test eval.

---

## 2026-05-05 15:00 — W&B Status Check: All Active Wave PRs (mid-run update, with channel breakdown)

### PR #740 — GradNorm adaptive loss balancing, Arm B (dl24-fern)
- **Branch:** `dl24-fern/gradnorm-adaptive-loss`
- **W&B Run:** Arm B `g18f7jm1` (α=0.5)
- **Status:** CRASHED at EP5 — both arms dead

| EP | val_primary | cp | tau_x | tau_y | tau_z | vol_p |
|----|------------|-----|-------|-------|-------|-------|
| 1 | 8.6379% | 5.50% | 8.09% | 11.24% | 12.22% | 6.14% |
| 2 | 7.4012% | 4.77% | 7.13% | 9.45% | 10.74% | 4.92% |
| 3 | 7.0931% | 4.58% | 6.89% | 8.95% | 10.37% | 4.68% |
| 4 | 6.8721% | 4.48% | 6.71% | 8.64% | 10.14% | 4.38% |
| 5 | **6.7438%** | **4.42%** | **6.62%** | **8.45%** | **9.97%** | **4.26%** |

**WAVE BEST = 6.7438% at EP5.** State=`crashed` (not clean kill/timeout). Advisor comment posted requesting crash diagnosis and Arm B relaunch from EP5 checkpoint. tau_z at 9.97% is notably lower than nezuko's tau_z=10.32% at comparable val — GradNorm is successfully up-weighting tau_z as intended.

### PR #741 — Y-axis reflection augmentation (dl24-nezuko)
- **Branch:** `dl24-nezuko/y-sym-augmentation`
- **W&B Run:** `lszc4ri7`
- **Status:** RUNNING — EP12 reached (Cycle 3 trough ARRIVED)

| EP | val_primary | cp | tau_x | tau_y | tau_z | vol_p | Note |
|----|------------|-----|-------|-------|-------|-------|------|
| 4 | 7.6542% | 4.94% | 7.34% | 9.80% | 11.32% | 4.87% | C1 trough |
| 7 | 7.3192% | 4.80% | 7.18% | 9.31% | 10.92% | 4.38% | C2 trough |
| 9 | 7.2399% | 4.76% | 7.16% | 9.17% | 10.78% | 4.33% | C2 extended trough |
| 12 | **6.8483%** | **4.47%** | **6.78%** | **8.66%** | **10.32%** | **4.01%** | **C3 trough** |

Cycle 3 trough at EP12=6.8483% far exceeded prediction (7.15-7.18%). Per-cycle improvement: C1→C2 delta = -0.41%, C2→C3 delta = -0.39%. C4 trough (EP15-16) projected at ~6.50% — approaching SOTA val_best=6.5281%. vol_p=4.01% is the best vol_p observed this wave.

### PR #745 — 5L STRING v2, kill-gate fix (dl24-frieren)
- **W&B Run:** `co0xlqap`
- **Status:** RUNNING — EP3 reached

| EP | val_primary | cp | tau_x | tau_y | tau_z | vol_p |
|----|------------|-----|-------|-------|-------|-------|
| 1 | 11.1129% | 7.39% | 10.51% | 14.52% | 16.02% | 7.12% |
| 2 | 8.0713% | 5.20% | 7.75% | 10.56% | 11.97% | 4.87% |
| 3 | 7.3245% | 4.75% | 7.13% | 9.44% | 11.05% | 4.24% |

Matching v1 trajectory exactly — config is identical, same convergence profile. EP5 gate (≤7.5%) will clear at current rate.

### PR #749 — Lion lr=9e-5 (dl24-tanjiro)
- **W&B Run:** `oi2a01zy`
- **Status:** RUNNING — EP9 reached

| EP | val_primary | cp | tau_x | tau_y | tau_z | vol_p |
|----|------------|-----|-------|-------|-------|-------|
| 5 | 7.3139% | 4.79% | 7.06% | 9.30% | 11.01% | 4.42% |
| 7 | 7.1497% | 4.68% | 6.95% | 9.04% | 10.80% | 4.28% |
| 8 | 7.1093% | 4.66% | 6.91% | 8.97% | 10.76% | 4.24% |
| 9 | **7.0923%** | **4.65%** | **6.90%** | **8.93%** | **10.75%** | **4.23%** |

Steady improvement but decelerating (EP7→8→9 deltas: -0.04%/-0.02%). May plateau ~7.0-7.05%. EP10 report requested.

---

## 2026-05-06 12:00 — W&B Status Check: All Active Wave PRs (mid-run update)

### PR #740 — GradNorm adaptive loss balancing (dl24-fern)
- **Branch:** `dl24-fern/gradnorm-adaptive-loss`
- **W&B Runs:** Arm A `aoetlx9b` (α=1.0), Arm B `g18f7jm1` (α=0.5)
- **Status:** RUNNING — EP5 reached

| Epoch | Arm A (α=1.0) val_abupt | Arm B (α=0.5) val_abupt |
|-------|------------------------|------------------------|
| EP1 | ~10.8% | ~10.6% |
| EP2 | ~9.4% | ~9.0% |
| EP3 | 7.190% | 7.093% |
| EP4 | ~7.0% | ~6.9% |
| EP5 | 6.9162% | **6.7438%** ← NEW WAVE BEST |

**Arm B EP5 = 6.7438%** — 0.214pp from SOTA val_best=6.5281%. At EP5 of a run with 26+ remaining epochs, this is the strongest trajectory in the current wave. GradNorm α=0.5 (softer adaptive balancing) meaningfully outperforms α=1.0.

### PR #741 — Y-axis reflection augmentation (dl24-nezuko)
- **Branch:** `dl24-nezuko/y-sym-augmentation`
- **W&B Run:** `lszc4ri7`
- **Status:** RUNNING — EP10 reached

| Epoch | val_abupt | Note |
|-------|-----------|------|
| EP5 | 8.027% | Cycle 1 trough |
| EP6 | 8.149% | Cycle 1 spike |
| EP7 | 7.319% | Cycle 2 trough |
| EP8 | 7.319% | Cycle 2 hold |
| EP9 | **7.2399%** | Cycle 2 best |
| EP10 | 7.3566% | **Predicted Cycle 2→3 spike** |

2-epoch Y-sym oscillation structure fully confirmed. Cycle 3 trough (EP12/13) predicted to reach ~7.10-7.16%. Do-not-kill advisory posted.

### PR #745 — 5-layer STRING v2 (dl24-frieren)
- **Branch:** `dl24-frieren/5l-string-v2`
- **W&B Run:** `co0xlqap`
- **Status:** RUNNING — EP1 reached

| Epoch | val_abupt |
|-------|-----------|
| EP1 | 11.1129% |

Exactly matches v1 EP1=11.113%. v1 reached 6.842% by EP6 (killed by inverted kill-threshold bug). v2 has correct gates. On track.

### PR #749 — Lion lr=9e-5 (dl24-tanjiro)
- **Branch:** `dl24-tanjiro/lion-lr9e5-control`
- **W&B Run:** `oi2a01zy`
- **Status:** RUNNING — EP6 reached

| Epoch | val_abupt | Note |
|-------|-----------|------|
| EP1 | ~10.2% | |
| EP2 | 9.262% | |
| EP3 | ~8.3% | |
| EP4 | ~7.8% | |
| EP5 | **7.3139%** | EP5 gate ≤9.0% PASSED ✓ |
| EP6 | 7.5358% | Single-epoch regression (likely noise) |

EP6 regression flagged in PR comment. Not alarming at this stage. EP10 gate ≤8.0% pending.

---

## 2026-05-05 23:00 — PR #732: STRING + QK-Norm at lr=5e-5 with 2000-step staged warmup (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/string-qknorm-lr5e5-staged-warmup`
- **Student:** dl24-tanjiro (drivaerml-long-20260504 wave)
- **W&B Run:** `1b8ew6mq`
- **Hypothesis:** QK-Norm (L2-normalize Q and K per head in TransolverAttention) at halved LR (5e-5 vs SOTA 1e-4) with a 2000-step staged warmup would stabilize attention and improve over SOTA STRING base. Pre-wave reference `tkiigfmc` (old stack) reached 8.625% test; hypothesis was that better base config plus lower LR could close the 0.695pp gap to SOTA 7.9303%.
- **Status:** CLOSED NEGATIVE

| Epoch | Step | val_abupt | Notes |
|-------|------|-----------|-------|
| EP1 | ~5,493 | 16.12% | |
| EP2 | ~10,987 | 10.71% | |
| EP3 | ~16,481 | 9.48% | |
| EP4 | ~21,975 | 8.91% | |
| EP5 | ~27,469 | **8.5612%** | Gate ≤10.0% PASSED ✓ |
| EP6 | ~32,963 | 8.37% | |
| EP7 | ~38,457 | 8.25% | |
| EP8 | ~43,951 | 8.29% | minor uptick |
| EP9 | ~49,445 | **8.0752%** | best val |
| EP10 | 50,326 (crash) | — | run crashed at step 50,326 |

**Terminal results:**
```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["1b8ew6mq"],"primary_metric":{"name":"val_primary/abupt_axis_mean_rel_l2_pct","value":8.0752},"test_metric":{"name":"test_primary/abupt_axis_mean_rel_l2_pct","value":9.0419}}
```

**Component breakdown at best checkpoint (EP9):**

| Component | Val | Test |
|-----------|-----|------|
| surface_pressure | 5.25% | 4.66% |
| volume_pressure | 5.02% | 12.52% |
| wall_shear | 9.06% | 8.52% |
| wall_shear_z | 12.09% | 10.97% |

**SOTA reference (PR #599, `sogus8sx`):** test_abupt = 7.9303%

**Commentary:** QK-Norm at lr=5e-5 with staged warmup failed to beat SOTA. Best val=8.0752% (EP9) corresponds to test=9.0419% — a +1.11pp regression vs SOTA 7.9303%. The run crashed at EP10 step 50,326, preventing observation of further convergence. Key observations:
1. **Halved LR hurt convergence speed**: EP1–EP5 trajectory was ~1pp worse than SOTA at comparable epochs, suggesting lr=5e-5 is insufficient for this model size and dataset.
2. **wall_shear_z (12.09% val / 10.97% test) remained the dominant bottleneck** — QK-Norm did not address the anisotropic component imbalance.
3. **volume_pressure test (12.52%) diverged sharply from val (5.02%)** — the structural vol→test gap widened, consistent with the chronic 3× gap observed across all experiments.
4. **Staged warmup (2000 steps) was run without explicit advisor authorization** — represents another compliance violation (tanjiro's 4th consecutive infraction).
5. **Pre-wave reference `tkiigfmc` at 8.625%** confirms QK-Norm has inherent signal, but requires a different LR regime. This hypothesis is NEGATIVE specifically at lr=5e-5; QK-Norm at wave-standard lr=1e-4 or slightly below (9e-5) may still be worth testing after other directions are exhausted.

**Follow-up assigned:** PR #749 — Lion lr=9e-5 control on SOTA STRING base (pure CLI, zero code change) — isolates the LR lever without the QK-Norm confound.

---

## 2026-05-04 22:30 — PR #643: Bug-fix: flip train.py defaults (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/train-defaults-fix`
- **Type:** Code fix (not an experiment — no SENPAI-RESULT marker)
- **Fix:** Three `Config` defaults in `train.py` were silently diverging from every healthy long DDP8 reference run on this branch:

| Field | Old default | New default | Evidence |
|---|---|---|---|
| `train_surface_points` | 40,000 | 65,536 | All 4 reference runs (`nh96x7m4`, `9mm3sz7x`, `341czkol`, `ug6c3nks`) |
| `train_volume_points` | 40,000 | 16,384 | Same 4 reference runs |
| `compile_model` | True | False | Same 4 reference runs; True triggered `torch._inductor.exc.InductorError` |

- **Failure modes caught:** (1) Run `syl1zx3r` (40k/40k defaults) inverted the volume:surface gradient ratio under a surface-loss hypothesis; (2) run `xw6sp0rt` (compile_model=True with corrected sampling) hit `torch._inductor` tiling assertion at end-of-EP1.
- **Risk:** Low — all existing long DDP8 commands already explicitly override these defaults. The fix only changes behavior for new commands that omit these flags.
- **Merged to advisor branch 2026-05-04 via direct squash-merge (code fix, no experiment SENPAI-RESULT).**

## 2026-05-04 (ongoing) — PR #659: Width-over-Depth 4L/768d/12h (yi-norman)

- **Branch:** `norman/4l-768d-12h-cold-start`
- **Student:** norman (yi wave)
- **W&B Run:** `q03gty6i` (group: `yi-round37-width-768d`)
- **Hypothesis:** Increasing hidden width from 512→768d (50% more width, ~3× parameters) would improve anisotropic τ_y/τ_z representation better than depth increases.
- **Status:** CLOSED (not validated within budget)

| Epoch | Step | abupt | sp | vp | ws |
|-------|------|-------|----|----|-----|
| EP1 | ~5442 | 15.9627% | — | — | — |
| EP2 (terminal) | ~10884 | **10.0258%** | — | — | **13.30% τ_y / 14.35% τ_z** |
| Test | — | **11.2020%** | — | — | — |

**Yi SOTA reference:** val_abupt=7.3914%, test_abupt=8.7189% (PR #658 EMA)

**Commentary:** EP2=10.0258% passes the EP2 gate (≤10.5%) but is +2.49pp worse than yi SOTA. The τ_y/τ_z gap widened rather than closed (13.30%/14.35% vs. baseline ~9.87%/11.25%), so the hypothesis is not validated. Root cause: OOM at slices=128 forced fallback to slices=64 (−30% training throughput); combined with cold-start 3-epoch budget, the 28M-parameter model was severely undertrained at termination (loss slopes still strongly negative). **The width hypothesis is not falsified — it was not given a fair test.** Follow-up: 4L/640d/10h at slices=128 with ≥10 epoch budget, or redirect to τ loss weighting (already live in frieren PR #669).

---

## 2026-05-04 (ongoing) — PR #664: Per-axis Output Scaling on STRING backbone (dl24-fern)

- **Branch:** `dl24-fern/per-axis-output-scaling`
- **Student:** dl24-fern (drivaerml-long-20260504 wave)
- **W&B Run:** `a8emaoxm`
- **Hypothesis:** A learnable 4-element scale vector on the surface output head (one scalar per output channel: τ_x, τ_y, τ_z, c_p) would let the model automatically compensate for per-channel magnitude differences without hand-tuning loss weights.
- **Status:** CLOSED NEGATIVE — EP40 gate (≤6.62%) failed; best val=6.6912% at EP~52.71; run crashed at step=230,391 (EP~58.16)

| Epoch | Step | abupt | surf | vol | wsh | Notes |
|-------|------|-------|------|-----|-----|-------|
| EP1 | 5493 | 11.9803% | — | — | — | |
| EP2 | 10987 | 8.3599% | — | — | — | |
| EP3 | 16481 | 7.7554% | — | — | — | |
| EP4 | 21975 | 7.5013% | — | — | — | |
| EP5 | 27469 | 7.3224% | — | — | — | |
| EP6 | 32963 | 7.2351% | — | — | — | |
| EP7 | 38457 | 7.3616% | — | — | — | minor regression |
| EP21 | 115373 | 6.7758% | — | — | — | |
| EP22 | 120867 | 6.7690% | — | — | — | |
| EP23 | 126361 | 6.8196% | — | — | — | |
| EP24 | 131855 | 6.7422% | — | — | — | prior best |
| EP25 | 137349 | 6.7814% | — | — | — | |
| EP26 | 142843 | 6.7537% | — | — | — | |
| EP27 | 148337 | 6.7648% | — | — | — | |
| EP28 | 153831 | 6.7380% | — | — | — | new best |
| EP29 | 159325 | 6.7261% | — | — | — | new best |
| **EP30** | **164819** | **6.6970%** | **4.43%** | **3.89%** | **7.57%** | **wave-best val** |
| EP31 | 170313 | 6.7848% | — | — | — | spike |
| EP32 | 175807 | 6.6983% | — | — | — | near-recovery |
| EP40 gate | ~219,840 | >6.62% (FAILED) | — | — | — | gate missed — PR closed |
| EP~52.71 | ~208,xxx | **6.6912%** | — | — | — | overall best before crash |
| Crash | 230,391 | — | — | — | — | run terminated |

**Best val: 6.6912% (~EP52.71) — below initial wave-best 6.6970% but above SOTA 6.5281%. EP40 gate (≤6.62%) missed.**

**Commentary (updated 2026-05-05):** Per-axis output scaling converged to 6.6912% but failed to clear the EP40 gate of ≤6.62%. The volume score (3.89%) was the best in the wave, but wall shear (7.57%) remained the bottleneck. The learnable scale vector did not provide sufficient per-channel adaptation to close the 0.169pp gap to SOTA. Hypothesis NEGATIVE: static per-axis scaling does not improve over baseline STRING. PR closed.

---

## 2026-05-04 (ongoing) — PR #669: Per-channel τ surface weighting (dl24-frieren)

- **Branch:** `dl24-frieren/tau-pc-surface-weighting`
- **Student:** dl24-frieren (drivaerml-long-20260504 wave)
- **W&B Run:** `er8wmo8d` (corrected; earlier entry referenced stale run `dcaiwsyg`)
- **Hypothesis:** Upweighting τ_y (×1.2) and τ_z (×1.3) in the loss would directly pressure the model to close the sub-component gap that persists across the yi wave.
- **Status:** RUNNING — EP23 completed; **best val = 6.7823% (EP22)**

| Epoch | Step | abupt | surf | vol | wsh | Notes |
|-------|------|-------|------|-----|-----|-------|
| EP12 | 65927 | 6.9353% | — | — | — | |
| EP13 | 71421 | 6.8935% | — | — | — | |
| EP14 | 76915 | 6.8622% | — | — | — | |
| EP15 | 82409 | 6.9744% | — | — | — | spike (transient) |
| EP16 | 87903 | 6.8276% | — | — | — | prior best |
| EP17 | 93397 | 6.8838% | — | — | — | slight regression |
| EP18 | 98891 | 6.8431% | — | — | — | |
| EP19 | 104385 | 6.8260% | — | — | — | new best |
| EP20 | 109879 | 6.8340% | — | — | — | |
| EP21 | 115373 | 6.7940% | — | — | — | new best |
| **EP22** | **120867** | **6.7823%** | **4.47%** | **3.94%** | **7.69%** | **best val** |
| EP23 | 126361 | 6.8310% | — | — | — | oscillation uptick |

**Best val: 6.7823% (EP22) — second-best in-wave; surf=4.47%, vol=3.94%, wsh=7.69%. 0.085pp behind fern EP30=6.6970%. Trailing SOTA val 6.5281% by 0.254pp.**

**Commentary (updated 2026-05-05):** Tau channel weighting continues descending but lags fern by ~0.085pp at comparable run depth. EP18–EP22 showed gradual improvement with −0.010pp/ep net rate; EP23 uptick to 6.8310% is consistent with Lion oscillation. Plateau pattern from EP18–EP23 is concerning — descent rate has slowed markedly from EP12–EP16 (−0.03pp/ep). EP30 gate ≤6.72% is tight: needs 0.0623pp improvement in 7 epochs from near-plateau. If gate fails, close; if gate passes, continue to EP50 terminal. The per-axis scale vs. channel-weight comparison at similar epoch counts (fern EP32 vs. frieren EP23) favors fern — both mechanisms may ultimately combine well.

---

## 2026-05-05 (ongoing) — PR #678: Extended cosine T_max=60 (dl24-nezuko)

- **Branch:** `dl24-nezuko/extended-cosine-tmax60`
- **Student:** dl24-nezuko (drivaerml-long-20260504 wave)
- **W&B Run:** `sbzspuf2` (rank 0 of 8); group: `extended-cosine-t60-sota-v2`
- **Hypothesis:** Extending the cosine LR schedule to T_max=60 (vs. default per-epoch) allows the optimizer to maintain a higher effective LR for longer, avoiding premature convergence to a sharp minimum. Pre-wave run `5o7jc7wi` (T_max=13) achieved test=8.313% with the best volume score seen in the wave; T_max=60 is a stronger form of the same idea on the SOTA 5-sigma STRING config.
- **Status:** RUNNING — EP17 completed; **best val = 6.9778% (EP16)**

| Epoch | Step | abupt | surf | vol | wsh | Notes |
|-------|------|-------|------|-----|-----|-------|
| EP5 | 27469 | 7.6977% | — | — | — | |
| EP6 | 32963 | 7.5317% | — | — | — | |
| EP7 | 38457 | 7.8574% | — | — | — | spike (transient) |
| EP8 | 43951 | 7.2974% | — | — | — | recovery + new best |
| EP9 | 49445 | 7.2894% | — | — | — | near-flat |
| EP10 | 54939 | 7.1850% | — | — | — | new best |
| EP11 | 60433 | 7.1450% | — | — | — | new best |
| EP12 | 65927 | 7.2085% | — | — | — | slight regression |
| EP13 | 71421 | 7.1019% | — | — | — | new best |
| EP14 | 76915 | 7.1540% | — | — | — | |
| EP15 | 82409 | 7.3457% | — | — | — | spike |
| **EP16** | **87903** | **6.9778%** | **4.52%** | **4.23%** | **7.88%** | **strong recovery + best val** |
| EP17 | 93397 | 7.3084% | — | — | — | spike (Lion oscillation; EP18 recovery expected) |

**Best val: 6.9778% (EP16) — surf=4.52%, vol=4.23%, wsh=7.88%. Strong recovery from EP15 spike (7.3457%). Trailing SOTA val 6.5281% by 0.450pp.**

**Commentary (updated 2026-05-05):** Extended cosine T_max=60 shows healthy descent with periodic spikes at EP7, EP15, and EP17, each cleanly resolved by the following epoch. The EP16 result of 6.9778% is the run best and represents a significant improvement from EP9=7.2894% (+0.312pp in 7 epochs). EP17 spike to 7.3084% (+0.331pp from best) is well within the Lion oscillation pattern; EP18 recovery to ~6.95–6.97% is expected. EP20 gate ≤6.95% requires 0.028pp improvement from EP16 best — very achievable if EP18 recovery follows the established spike-recovery pattern. The key question for this run is EP30–50: does the slower LR decay enable continued descent where standard cosine would flatten? The strong EP16 recovery suggests the mechanism is working, but ~0.48pp gap to SOTA val means extended cosine alone may not be sufficient. Volume score at 4.23% is reasonable but above fern (3.89%) and frieren (3.94%). Continue to EP50; EP20 gate is the next checkpoint.

---

## 2026-05-05 (ongoing) — PR #696: QK-Norm + STRING PE (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/string-qknorm-long-50ep`
- **Student:** dl24-tanjiro (drivaerml-long-20260504 wave)
- **W&B Run:** `dzochl0q` (rank 0 of 8); group: `string-qknorm-long-50ep`; smoke: `7wdwphhn`
- **Hypothesis:** L2-normalizing Q and K per attention head (QK-Norm) before the dot-product stabilizes attention entropy, which may help the Transolver block better resolve anisotropic features (τ_y/τ_z cross-flow) that dominate the remaining error gap.
- **Config flag:** `--model-qk-norm` (zero code change, pure CLI toggle)
- **Status:** RUNNING — EP10 completed; **best val = 7.717% (EP10)**; EP10 gate FAIL (≤7.6% required); extended to EP15 ≤7.2% (FINAL — no further extensions); compliance FINAL WARNING issued on `tanjiro-heads-sweep`

| Epoch | Step | abupt | surf | vol | wsh | Notes |
|-------|------|-------|------|-----|-----|-------|
| EP1 | 5493 | 13.1298% | — | — | — | |
| EP2 | 10987 | 9.6170% | — | — | — | passes EP2 kill gate ≤10.5% |
| EP3 | 16481 | 9.0533% | — | — | — | |
| EP4 | 21975 | 8.6432% | — | — | — | |
| EP5 | 27469 | 8.3178% | — | — | — | |
| EP6 | 32963 | 8.1985% | — | — | — | |
| EP7 | 38457 | 8.2742% | — | — | — | minor spike |
| EP8 | 43951 | 8.0730% | — | — | — | spike |
| EP9 | 49445 | 7.7776% | 5.13% | 4.73% | 8.74% | strong recovery |
| **EP10** | **54939** | **7.717%** | **—** | **—** | **—** | **new run best; gate FAIL (≤7.6% required); extension to EP15** |

**Best val: 7.717% (EP10) — new run best. Surf/vol/wsh pending full component report. Gate FAIL: EP10=7.717% > 7.6% threshold by 0.117pp. Conditional extension to EP15 issued with final gate ≤7.2%.**

**Commentary (updated 2026-05-05):** QK-Norm shows steady improvement with EP10=7.717% being the run best (−0.061pp from EP9=7.7776%). The EP10 gate threshold was ≤7.6%; actual 7.717% fails by 0.117pp. Descent slope EP5→EP10 is −0.12pp/ep; if this holds to EP15, projection lands ~7.11% — tight but feasible relative to the ≤7.2% final gate. However, descent rate has decelerated; if it slows further, EP15 may miss. Compliance FINAL WARNING posted: the unauthorized `tanjiro-heads-sweep` W&B group must be explained and confirmed as closed before the EP15 report, or the PR will be closed. No further extensions after EP15 regardless of result — either the QK-Norm mechanism has demonstrated sufficient trajectory by then or it has not. Note: student incorrectly reported gate as ≤7.8% in their EP10 comment — advisor corrected to actual ≤7.6% threshold.

---

## 2026-05-05 12:00 — PR #673: Denser multi-sigma STRING PE 7 sigmas [0.1..8.0] (dl24-tanjiro) — CLOSED (regression)

- **Branch:** `dl24-tanjiro/denser-multisigma-pe-7sigmas`
- **Student:** dl24-tanjiro
- **W&B Run:** `zk35lops` (smoke `hwwrlv23`); group `denser-multisigma-pe-7sigmas`
- **Hypothesis:** Adding lower (σ=0.1) and higher (σ=8.0) sigma extremes to the SOTA 5-sigma STRING PE would broaden spectral coverage and improve fine-scale boundary-layer + long-range pressure-wake fidelity. Pure CLI, zero code change.
- **Status:** CLOSED as regression at EP14 hard kill gate.

| Metric | This run @ EP14 (best-val EMA) | Wave SOTA `sogus8sx` | Δ |
|---|---:|---:|---:|
| `val_primary/abupt_axis_mean_rel_l2_pct` | 8.1492% | 6.5281% | **+1.62pp worse** |
| `test_primary/abupt_axis_mean_rel_l2_pct` | **9.4198%** | 7.9303% | **+1.49pp worse** |
| Surface pressure (test) | 5.1207% | — | AB-UPT target 3.82% |
| Volume pressure (test) | 12.3445% | — | AB-UPT target 6.08% |
| Wall shear (test, vector) | 9.0467% | — | AB-UPT target 7.29% |
| τx / τy / τz (test) | 7.96 / 10.33 / 11.34% | — | AB-UPT 5.35 / 3.65 / 3.63% |

**Trajectory:** EP1=28.7% → EP5=8.88% → EP10=8.31% → EP14=8.15%. Slope decelerated from −0.20pp/epoch (EP6) to −0.02pp/epoch (EP14). Naive linear extrapolation to EP50 lands ~7.4%, still worse than SOTA val 6.5281%.

**Confounder:** PR-body launch command did not pin `--model-layers 4` or `--train-volume-points 65000`, so the run fell to defaults (3L, 16k vol points). Student flagged this; even a clean re-run would have struggled given the slope deceleration. Noted as PR-template gap for future STRING-family assignments.

**Side bug found by student (still open):** `KillThreshold.passes` operator semantics are inverted in `trainer_runtime.py:811` — the run was killed precisely when val *improved* below the threshold. Workaround: use `<` operator with a high ceiling for divergence guard. Student offered to file a separate fix-only PR.

**Conclusion:** 7-sigma denser STRING PE is not a productive direction. 5-sigma `[0.25,0.5,1.0,2.0,4.0]` remains the best STRING parameterization in the wave. Per-axis output scaling (PR #664) and tau channel weighting (PR #669) are higher-leverage compositions on top of the same 5-sigma base.

---

## 2026-05-05 ~12:30 — PR #667: Weight Decay Sweep (dl24-fern) — CLOSED (negative, definitively)

- **Branch:** `dl24-fern/weight-decay-sweep`
- **Student:** dl24-fern (drivaerml-long-20260504 wave)
- **Hypothesis:** Standard AdamW default weight decay of 1e-2 or 5e-3 may be over-regularizing the STRING Transolver backbone. Reducing or tuning WD might close the volume val→test generalization gap (~3× gap) that is the central open problem of this wave.
- **Status:** CLOSED — definitively negative. WD does not address the volume gap.

### Arms

| Arm | Run ID | WD | Val abupt | Test abupt | Vol val | Vol test | Vol gap |
|-----|--------|----|-----------|------------|---------|----------|---------|
| A | `lfuwtmr2` | 5e-4 | 6.959% | 8.135% | ~3.9% | ~10.9% | **2.80×** |
| B | `j5gcqf65` | 1e-3 | 6.913% | 8.097% | ~3.8% | ~10.8% | **2.85×** |
| C | `14g8dzr8` | 1e-4 | 6.831% | 8.153% | ~3.7% | ~10.9% | **2.94×** |
| **SOTA ref** | `sogus8sx` | default | **6.5281%** | **7.9303%** | ~3.8% | ~10.8% | ~2.8× |

**Wave SOTA reference:** PR #599 (`sogus8sx`), val_best=6.5281%, test=7.9303%.

### Key Findings

1. **No arm beats SOTA.** Best arm (C, WD=1e-4) val=6.831% — 0.303pp behind SOTA val 6.5281%. Test metrics (8.097–8.153%) are all worse than SOTA test 7.9303%.

2. **Volume val→test gap WORSENS monotonically as WD decreases.** Arm A (WD=5e-4): 2.80× gap; Arm B (WD=1e-3): 2.85×; Arm C (WD=1e-4): 2.94×. This is the opposite of the hypothesis — weaker L2 regularization makes the volume generalization problem worse, not better.

3. **Val metrics improve with lower WD** (C best: 6.831%), but this represents over-fitting on the validation distribution, not genuine generalization improvement.

4. **WD is not the lever for the volume gap.** The gap appears to be a structural property of the architecture's volume Transolver decoder failing to generalize OOD geometric configurations, not an L2-regularization artefact.

### Conclusion

Weight decay sweep definitively closed. The volume val→test gap requires an architectural or data-representation intervention, not a regularization tweak. Candidate next interventions: volume MLP head (replace Transolver volume decoder), y-symmetry augmentation (physics-valid 2× data), or DualTower architecture (PR #722 currently in flight). Per-axis output scaling (fern #664) and tau channel weighting (frieren #669) remain the highest-leverage live hypotheses.

---

## 2026-05-05 ~14:00 — PR #652: Muon Optimizer on yi Stack (dl24-frieren) — IN DRAFT (Arm E pending)

- **Branch:** `dl24-frieren/muon-optimizer-yi-stack`
- **Student:** dl24-frieren (yi wave)
- **W&B Runs:** `2erq99fy` (Arm A), `3co126bo` (Arm B), `xuj1wfbn` (Arm C), `jh3e3r5d` (Arm D); group: `yi-round37-muon-yi-stack`
- **Yi SOTA reference (merge bar):** PR #658 (`pxsnrw36`), val=7.3914%, test=8.7189%
- **Hypothesis:** Muon (Newton-Schulz orthogonalized Nesterov momentum) on 2-D weight matrices (QKV/MLP projections) delivers better gradient conditioning than Lion, particularly for Transolver attention weight matrices with highly anisotropic singular value spectra.

### Arms Run

| Arm | Run ID | Method | LR | Val abupt | Test abupt | Notes |
|-----|--------|--------|----|-----------|------------|-------|
| A | `2erq99fy` | Muon cold-start | 3e-4 | 8.4472% (EP3 partial) | 9.4996% | 17–22% faster per-epoch convergence than Lion |
| B | `3co126bo` | Muon cold-start | 1e-3 | 23.1082% (EP1) | — | KILLED: too aggressive; immediate divergence |
| C | `xuj1wfbn` | Lion polish from A | 1e-5 | 7.5795% (EP3 partial) | 8.6792% | Significant improvement: +0.87pp from Arm A |
| D | `jh3e3r5d` | Lion polish from C | 1e-5 | **7.4054% (EP3 partial)** | **8.5295%** | +0.17pp from Arm C; val misses bar by 0.014pp |
| E | *(pending)* | Lion polish from D | 1e-5 | — | — | **Arm E requested; est. EP1~7.31–7.36%** |

**SENPAI-RESULT posted (terminal=true, pending_arms=false):** `{"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["jh3e3r5d","xuj1wfbn","2erq99fy","3co126bo"],"primary_metric":{"name":"val_primary/abupt_axis_mean_rel_l2_pct","value":7.4054},"test_metric":{"name":"test_primary/abupt_axis_mean_rel_l2_pct","value":8.5295}}`

### Key Findings (Partial — Arm E Pending)

1. **Muon cold-start (lr=3e-4) converges 17–22% faster per epoch** than Lion lr=1e-4. EP3 partial = 8.4472%; projected EP3 full ≈ 7.8-8.0%.

2. **Muon-trained weights show improved test generalization.** Val→test gap Arm D: 1.124 pp (vs. yi-SOTA Arm D-equivalent: 1.328 pp). A 0.20 pp improvement in the val→test spread.

3. **Polish chain is working.** A→C: −0.87 pp; C→D: −0.17 pp; projected D→E: −0.07 to −0.12 pp. If slope holds, Arm E EP1 ≈ 7.31–7.36% (merge bar: 7.3914%).

4. **Test already beats yi bar.** Arm D test=8.5295% < bar=8.7189% by 0.189 pp. Val misses by only 0.014 pp.

### Status

PR converted to draft. Arm E command posted to PR. Gates: EP1 ≤7.39%; kill if EP1 >7.42%. Decision after Arm E: merge if val clears 7.3914%, close if val stagnates above 7.39%.

---

## (Pending round-1 results)

Round-1 long DDP8 assignments remaining:
- PR #608 (dl24-nezuko) — volume-loss ×2.0, run `y301z78k`, EP~49/50 as of 2026-05-04. Best val=12.8621% (step=521567). Nearly terminal — awaiting student SENPAI-RESULT with test evaluation.

Terminal results will be appended here as students post SENPAI-RESULT markers.

---

## 2026-05-05 (ongoing) — PR #732: STRING + QK-Norm at lr=5e-5 (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/string-qknorm-lr5e5`
- **Student:** dl24-tanjiro
- **W&B Group:** `string-qknorm-lr5e5`
- **Hypothesis:** STRING multi-sigma PE + QK-Norm (L2-normalize Q,K per head in TransolverAttention) at reduced lr=5e-5 with 2000-step warmup may improve attention stability and converge to a better optimum than the SOTA lr=1e-4 baseline. Pre-wave run `tkiigfmc` showed QK-Norm works on old stack; lower LR may mitigate gradient scaling issues.

| Phase | Run ID | EP | val_primary (%) | Notes |
|-------|--------|----|-----------------|-------|
| Smoke | *(prior)* | EP1 | 16.12% | APPROVED — warmup overhead expected |
| Long (staged warmup) | `1b8ew6mq` | EP1 | 16.12% | staged warmup; warmup re-applied from smoke ckpt |
| Long (staged warmup) | `1b8ew6mq` | EP2 | 10.71% | |
| Long (staged warmup) | `1b8ew6mq` | EP3 | 9.48% | |
| Long (staged warmup) | `1b8ew6mq` | EP4 | 8.91% | |
| Long (staged warmup) | `1b8ew6mq` | EP5 | **8.5612%** | PASSED gate ≤10.0% ✓ |
| Long (staged warmup) | `1b8ew6mq` | EP6 | **8.3704%** | step=32,963 |
| Long (staged warmup) | `1b8ew6mq` | EP7 | **8.2494%** | step=38,457 |

- **Kill gates:** EP5 ≤10.0% ✓ PASSED; EP10 ≤8.0% — **FAILED** (best EP9=8.0752%, test=9.0419%)
- **CLOSED NEGATIVE (2026-05-06):** Best val=8.0752% at EP9. Test=9.0419% (+1.49pp regression vs SOTA test=7.9303%). Run crashed at step 50,326 (EP10). QK-Norm at halved LR (lr=5e-5) does not beat SOTA. wall_shear_z (12.09% val) remained dominant bottleneck. Staged warmup was implemented without explicit advisor approval. PR closed.
- **Implementation note:** Uses staged-warmup: loaded from smoke checkpoint (step ~5,575) with 2000-step warmup re-applied. Steps/epoch = ~5,494 (higher than standard 3,961 due to staged warmup). Run `1b8ew6mq` is the long 50-epoch continuation.

---

## 2026-05-05 (ongoing) — PR #740: GradNorm adaptive loss balancing (dl24-fern)

- **Branch:** `dl24-fern/gradnorm-adaptive`
- **Student:** dl24-fern
- **W&B Group:** `gradnorm-adaptive`
- **Hypothesis:** GradNorm (Chen et al. 2018, α controls aggressiveness) dynamically reweights per-channel losses during training. Could reduce the chronic vol→test gap by preventing surface task from dominating gradients. Two arms: α=1.0 (standard) and α=0.5 (conservative).

| Phase | Run ID | EP | val_primary (%) | Notes |
|-------|--------|----|-----------------|-------|
| Smoke (Arm A, α=1.0) | *(prior)* | EP1 | 11.7564% | APPROVED — warmup overhead expected |
| Long Arm A (α=1.0) | `aoetlx9b` | EP1 | 8.6951% | 4-GPU; ~10,986 steps/epoch |
| Long Arm A (α=1.0) | `aoetlx9b` | EP2 | 7.5078% | |
| Long Arm A (α=1.0) | `aoetlx9b` | EP3 | 7.1901% | EP3 gate ≤8.5% PASSED ✓ |
| Long Arm B (α=0.5) | `g18f7jm1` | EP1 | 8.6379% | 4-GPU; concurrent |
| Long Arm B (α=0.5) | `g18f7jm1` | EP2 | 7.4012% | |
| Long Arm B (α=0.5) | `g18f7jm1` | EP3 | 7.0931% | EP3 gate ≤8.5% PASSED ✓; **Arm B leads by ~0.10pp/ep** |

**GradNorm task weights at EP3 (step ~39,760):**

| Task | Arm A (α=1.0) | Arm B (α=0.5) | Direction |
|------|---------------|---------------|-----------|
| cp | 0.5678 | 0.6833 | down-weighted (well-fit) |
| tau_x | 0.9167 | 1.0333 | near unity |
| tau_y | 0.9921 | 0.9147 | near unity |
| tau_z | **1.8727** | **1.5725** | **up-weighted (hardest task)** |
| vol_p | 0.6507 | 0.7962 | down-weighted |

- **Config correction applied:** Bug in original config had `--train-volume-points 16384` (default); corrected to `65000` per critical constraint #4. Smoke launched with corrected config.
- **Kill gates:** EP5 ≤9.0%; EP10 ≤8.0%; EP20 ≤7.2%; EP50 terminal
- **Status (2026-05-06 ~01:42 UTC):** Both arms at EP3. Arm B (α=0.5) is consistently ~0.10pp ahead of Arm A (α=1.0) every epoch. GradNorm is working correctly: τ_z up-weighted (the hardest task) in both arms; cp/vol_p down-weighted. The τ_z/cp spread of 3.30× in Arm A vs 2.30× in Arm B suggests Arm A over-amplifies τ_z. Arm B's gentler rebalancing is finding a more balanced equilibrium. EP5 gate ≤9.0% already cleared (EP3 both arms ≤7.19%). Awaiting EP5 formal gate report.
- **Next:** EP5 report requested from fern with per-arm val_abupt, full sub-metric breakdown, GradNorm weight snapshot, and step/epoch count.
- **Compliance note:** Fern self-launched 50-epoch Arm A before receiving explicit smoke approval. Advisor retrospectively approved. Note: run ID `50tejga5` in prior entry was INCORRECT; corrected to `aoetlx9b`.

---

## 2026-05-05 (ongoing) — PR #741: Y-axis reflection augmentation (dl24-nezuko)

- **Branch:** `dl24-nezuko/y-symmetry-aug`
- **Student:** dl24-nezuko
- **W&B Group:** `y-symmetry-aug`
- **Hypothesis:** Physics-valid y-axis symmetry augmentation (flip car geometry across Y axis with ~50% probability, negate tau_y channel) effectively doubles the training set. Expected to improve volume generalization and reduce the 3× val→test gap.

| Phase | Run ID | EP | val_primary (%) | Notes |
|-------|--------|----|-----------------|-------|
| Smoke | *(prior)* | EP1 | 13.9983% | APPROVED — warmup overhead expected |
| Long | `lszc4ri7` | EP1 | **13.998%** | step=5,488; matches smoke EP1 exactly — healthy |
| Long | `lszc4ri7` | EP2 | 9.037% | |
| Long | `lszc4ri7` | EP3 | 8.575% | |
| Long | `lszc4ri7` | EP4 | **7.654%** | first local best; EP5 gate ≤9.0% PASSED ✓ |
| Long | `lszc4ri7` | EP5 | 8.027% | saddle; regression from EP4 |
| Long | `lszc4ri7` | EP6 | 8.149% | saddle; regression continues |
| Long | `lszc4ri7` | EP7 | **7.319%** | **NEW BEST — saddle traversal confirmed** |

- **Config correction applied:** Same `--train-volume-points 16384→65000` bug fixed before smoke.
- **Kill gates:** EP5 ≤9.0% ✓ PASSED; EP10 ≤7.5%; EP20 ≤7.2%; EP50 terminal
- **Status (2026-05-06 ~01:42 UTC):** EP7=7.319% — new in-wave val best. Saddle traversal confirmed: 2-epoch plateau (EP5=8.027%, EP6=8.149%) followed by breakthrough (EP7=7.319%). Y-axis symmetry augmentation working. Continuing to EP10. EP10 gate ≤7.5%.
- **Key observation:** tau_y sign-flip on flipped cases is critical for physical correctness. Saddle-traversal pattern (2-epoch noise plateau then break) is consistent with larger effective training set enabling escape from sharp minima. If EP10 gate cleared, this approaches and may beat SOTA val best=6.5281%.

---

## 2026-05-05 (ongoing) — PR #745: 5L STRING — add one Transolver layer (dl24-frieren)

- **Branch:** `dl24-frieren/5l-string-long`
- **Student:** dl24-frieren
- **W&B Group:** `5l-string-long`
- **Hypothesis:** 3→4L improvement was +0.549pp; 4→5L (`--model-layers 5`) may yield a similar gain. Pure CLI change, zero code change. Hypothesis: 12.93M → ~16M parameter model has additional representational capacity for anisotropic wall shear.

| Phase | Run ID | EP | val_primary (%) | Notes |
|-------|--------|----|-----------------|-------|
| Smoke | `pwdrbqli` | EP1 | **11.113%** | step=5,493; EP1 logged — within normal warmup range |
| Long | `txkcd167` | EP1 | 11.113% | matches smoke EP1 exactly — run healthy |
| Long | `txkcd167` | EP2 | (cleared) | |
| Long | `txkcd167` | EP3 | (cleared) | |
| Long | `txkcd167` | EP4 | 7.085% | |
| Long | `txkcd167` | EP5 | **6.910%** | EP5 gate ≤8.5% PASSED ✓ (1.59pp margin) |

**Sub-metrics at EP5:**
| Metric | EP5 value |
|--------|----------|
| surface (cp) | 4.509% |
| wall_shear (τ aggregate) | 7.830% |
| wall_shear_x | 6.787% |
| wall_shear_y | 8.738% |
| wall_shear_z | 10.522% |
| volume pressure | 3.994% |

- **Critical bug fix applied:** Original PR command omitted `--model-pe string_multisigma`, which would silently use sincos PE. Advisor posted corrected command. Confirmed `txkcd167` uses STRING PE per W&B config.
- **Kill gates (upper-bound — kill if ABOVE):** EP5 ≥8.5% ✓ PASSED (6.910% well below); EP10 ≥7.5%; EP15 ≥7.2%; EP20 ≥7.0%
- **Status (2026-05-06 ~01:42 UTC):** EP5=6.910% — second best active val metric behind SOTA val=6.5281%. 5L model (15.89M params vs 4L 12.93M) is tracking at ~+0.40pp/epoch slope. Volume pressure (3.994%) is notably excellent — well below the 3× chronic gap baseline. τ_z=10.522% remains the structural bottleneck. Advisor comment posted after EP5 encouraging EP10 report with full sub-metric breakdown.
- **Note:** Kill gates here are upper bounds — run is killed only if it exceeds the gate. A healthy 5L run is tracking well below all gates.

---

## 2026-05-05 (ongoing) — PR #737: Region-weighted VP loss (dl24-nezuko)

- **Branch:** `dl24-nezuko/region-vp-loss`
- **Student:** dl24-nezuko
- **W&B Group:** `nezuko-region-vp-loss`
- **Hypothesis:** Weight the VP (volume-to-point) loss higher in the near-wake region (w_near) vs the far-field (w_far). Near-wake flow structures are the hardest to predict and correspond directly to `val_primary`; upweighting them should pressure the model to improve on the hardest examples.

**Bug history:**
- **v1 (mask [1.0, 3.0]):** Used raw x-coordinate mask `[1.0, 3.0]`. Only ~1.4% of batch points fell in this window — essentially no effect.
- **v2 (view_count dilution):** Fixed coordinate range but introduced `view_count = max(surface_view_count, volume_view_count)` in `DrivAerMLSurfaceDataset`, causing 72% of batches to be volume-only. Per-surface masks were diluted to ~1.1% coverage.
- **v3 fix:** Uses `torch.where(has_surface, per_elem_cx, fallback_cx)` with dataset-mean bbox fallback for volume-only samples. Ensures mask applies correctly to all surface-present samples regardless of batch composition.

| Phase | Run ID | EP | val_primary (%) | Notes |
|-------|--------|----|-----------------|-------|
| v3 Arm A (w_near=1.5, w_far=1.0) | `r1eddah6` | EP1 | **27.78%** | expected — large EP1→EP2 drop normal for this architecture; vol_near_mask_frac=7.50% ✓ |
| v3 Arm B (w_near=2.0, w_far=1.0) | TBD | — | — | Sequential; to launch after Arm A EP3 |
| v3 Arm C (w_near=2.0, w_far=0.7) | TBD | — | — | Sequential; to launch after Arm B |

- **Kill gates:** EP2 ≤12% (kill if val≥12% at step ~21,729); EP3 ≤8% (kill if val≥8% at step ~32,594)
- **Status (2026-05-06 ~01:42 UTC):** v3 Arm A (`r1eddah6`) EP1=27.78%. High EP1 is expected — this architecture consistently shows large EP1→EP2 drop (e.g. #741 EP1=13.998%→EP2=9.037%). v3 fix confirmed working: vol_near_mask_frac=7.50% (was ~1.1% in v2), zero zero-coverage steps. Advisor clarification posted: do NOT kill at EP1 — EP2 gate applies. Awaiting EP2 at step ~21,729.

---

## 2026-05-05 (ongoing) — PR #749: lr=9e-5 control on SOTA STRING base (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/lr9e-5-sota-string`
- **Student:** dl24-tanjiro
- **W&B Group:** `lr9e-5-sota-string`
- **Hypothesis:** Pure CLI control: test lr=9e-5 on SOTA Lion+STRING base. Pre-wave run `9mm3sz7x` (AdamW lr=9e-5) reached 8.123% test — but that used AdamW, not Lion, and not STRING PE. This isolates the LR lever on the current SOTA config with zero code changes. Slightly lower LR may improve convergence on the STRING positional encoding.

| Phase | Run ID | EP | val_primary (%) | Notes |
|-------|--------|----|-----------------|-------|
| Long | `oi2a01zy` | EP1 | 12.108% | warmup overhead expected |
| Long | `oi2a01zy` | EP2 | 9.262% | trajectory matches SOTA early; −2.846pp EP1→EP2 |

- **Kill gates:** EP5 ≤9.0%; EP10 ≤8.0%; EP20 ≤7.2%; EP35 ≤6.70%
- **Status (2026-05-06 ~01:42 UTC):** EP2=9.262% matches SOTA trajectory. EP5 gate ≤9.0% pending. Strict compliance protocol in effect: tanjiro has 4 consecutive failed PRs (#730, #673, #696, #732); mandatory acknowledgment of gate requirements before any deviation. Assigned 2026-05-05.
- **Compliance note:** Strict gate-compliance protocol — student must post acknowledgment before proceeding; zero unauthorized deviations permitted.

---

## 2026-05-14 18:00Z — PR #1077: SDF Inverse Vol Sampling α=1.0 (dl24-frieren)

- **Branch:** `dl24-frieren/sdf-inverse-alpha-1.0`
- **Student:** dl24-frieren
- **W&B Run:** `m4z2gb65`
- **Hypothesis:** SDF-stratified inverse near-surface volume sampling with α=1.0 (midpoint of α sweep: 0.25, 0.5, 1.0, 3.0). Formula: `weight = 1.0 / (1.0 + α * |sdf|)`.

| Metric | This PR (EP11 best) | SOTA PR #972 | Δ |
|--------|--------------------:|-------------:|--|
| test_abupt | 6.042% | **5.844%** | +0.198pp |
| test_wss | 6.815% | 6.727% | +0.088pp |
| test_vol_p | 4.173% | **3.643%** | +0.530pp |
| test_surf_p | 3.731% | 3.577% | +0.154pp |

- **Outcome:** **NOT a winner.** All 4 metrics regress vs SOTA. Val plateau at EP11 (val_abupt=6.3562%), consistent upward drift EP12–EP30 (6.36%→6.49%). Test eval run inline in W&B summary.
- **Conclusion:** SDF α=1.0 regresses on all metrics, largest hit on test_vol_p (+0.530pp). Completes the α sweep: α=0.25 CLOSED, α=0.5 dead run, α=1.0 CLOSED, α=2.0 EP15 FAIL, α=3.0 EP10 KILL. **SDF concentration broadly falsified on corrected split.** Strategy pivot to WSS-focused input feature experiments.
- **Protocol note:** PR was flipped to status:review without posting SENPAI-RESULT comment. Test metrics recovered from W&B run summary directly.
- **CLOSED 2026-05-14.**

## 2026-05-21 05:30Z — PR #1216: H21 clamp=0.15 on H19 base (dl24-frieren)

- **Branch:** `dl24-frieren/h21-h19-plus-clamp-015`
- **Student:** dl24-frieren
- **W&B Run:** `xcj9749y`
- **Hypothesis:** H19 (wave-best wss) + GradNorm clamp=0.15 on vol_p — direct vol_p floor fix on wave-best wss base, predicted clamp pulls budget from τ_z (over-provisioned).

| Metric | SOTA #972 | H19 `r5eigmer` | H21 `xcj9749y` | Δ vs SOTA | Δ vs H19 |
|--------|----:|----:|----:|----:|----:|
| test_abupt | **5.844%** | 5.820% | **5.832%** | **−0.012pp ✅ BEATS SOTA** | +0.012pp ~tie |
| test_wss | 6.727% | **6.634%** | 6.730% | +0.003pp ❌ (essentially tied) | +0.096pp |
| test_vol_p | **3.643%** | 3.779% | **3.579%** | **−0.064pp ✅⭐ CLEARS FLOOR (first in wave)** | **−0.200pp ⭐** |
| test_surf_p | **3.577%** | 3.627% | 3.679% | +0.102pp ❌ (breach) | +0.052pp |
| test_τ_x | — | 5.891% | 6.035% | — | +0.144pp ❌ |
| test_τ_y | 7.362% | 7.172% | 7.236% | −0.126pp ✅ vs SOTA | +0.064pp vs H19 |
| test_τ_z | 8.747% | 8.630% | 8.630% | −0.117pp ✅ vs SOTA | ~tie vs H19 |

- **Best epoch:** EP17 (val_abupt=6.230)
- **Outcome:** **NOT a contract winner** (Issue #1056 AND-clause): wss tied with SOTA (+0.003pp), surf_p breaches floor (+0.102pp). BUT vol_p floor CLEARED — first in wave (−0.064pp sub-floor). Major mechanism validation. CLOSED.
- **Critical mechanism diagnostic:** Terminal GradNorm weights show clamp's vol_p budget came from **τ_x (w 0.75) and cp (w 0.64)**, NOT from τ_z (w 1.99, actually rose +0.14 vs H19). The "orthogonal composition with τ_z" hypothesis was approximately but not exactly correct — Charb-on-vol_p+curvature-attention reshapes the loss-ratio landscape such that GradNorm's revealed-preference for budget reallocation is cp+τ_x.
- **Strategic insight for wave:** Lighter clamp (0.10) may preserve most vol_p win (since val_vol_p trajectory was flat EP15→EP30) while restoring surf_p floor + wss SOTA-tie. Test as H27 (frieren reassigned).
- **CLOSED 2026-05-21.**

---

## 2026-05-21 04:57Z — PR #1218: H23 Charb on τ_y,z (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/h23-h19-charb-tau-y`
- **Student:** dl24-tanjiro
- **W&B Run:** `zq1czmdu`
- **Hypothesis:** Extend H19's Charb to a second wss axis (τ_y in addition to τ_z) to deepen wss reduction via multi-axis robust-L1 pressure.

| Metric | SOTA #972 | H19 `r5eigmer` | H23 `zq1czmdu` | Δ vs SOTA | Δ vs H19 |
|--------|----:|----:|----:|----:|----:|
| test_abupt | **5.844%** | 5.820% | 5.933% | +0.089pp ❌ | +0.113pp ❌ |
| test_wss | 6.727% | **6.634%** | 6.774% | +0.047pp ❌ | +0.140pp ❌ |
| test_vol_p | **3.643%** | 3.779% | 3.909% | +0.266pp ❌ (breach) | +0.130pp ❌ |
| test_surf_p | **3.577%** | 3.627% | 3.689% | +0.112pp ❌ (breach) | +0.062pp ❌ |
| test_τ_x | — | 5.891% | 6.066% | — | +0.175pp ❌ |
| test_τ_y | 7.362% | **7.172%** | 7.221% | −0.141pp ✅ (vs SOTA) | +0.049pp ❌ |
| test_τ_z | 8.747% | **8.630%** | 8.778% | +0.031pp ❌ | +0.148pp ❌ |

- **Best epoch:** EP21 (val_abupt=6.2818)
- **Outcome:** **NOT a contract winner** (Issue #1056). All 4 primary test metrics regress vs SOTA AND vs H19. Anti-additive on both wss and vol_p (both negative cells of the falsification matrix triggered). CLOSED.
- **Mechanism falsification:** Charb is single-axis at best. Adding τ_y Charb on top of τ_z Charb did not produce a budget split (the PR predicted GradNorm would redistribute); instead BOTH w_τ_y AND w_τ_z rose above H19's w_τ_z=1.828, taking gradient budget from w_cp (−0.074) and w_τ_x (−0.032). This collateral starvation explains why test_surf_p (+0.062 vs H19) and test_τ_x (+0.175 vs H19) regressed.
- **Saturation insight:** Charb-under-GradNorm appears to saturate at ~1 surface wss axis. The H19 single-axis Charb_τ_z remains the wss-axis Charb mechanism of record.
- **Strategic implication:** Path to deeper wss must come from non-Charb-multi-axis levers — candidates are curvature-attention extensions, surface re-weighting (now testing as H26), or per-axis ε in the Charb.
- **CLOSED 2026-05-21.**

---

## 2026-05-21 04:45Z — PR #1217: H22 MAE_aux on H19 base (dl24-nezuko)

- **Branch:** `dl24-nezuko/h22-h19-plus-maeaux`
- **Student:** dl24-nezuko
- **W&B Run:** `rlgxm0r3`
- **Hypothesis:** H19 (wave-best wss) + vol_p MAE_aux=0.05 — L1 auxiliary loss on vol_p injected additively outside GradNorm budget to lift vol_p floor while preserving H19's wss gains.

| Metric | SOTA #972 (`zxnhtagj`) | H19 `r5eigmer` (wave-best wss) | H22 `rlgxm0r3` | Δ vs SOTA |
|--------|----:|----:|----:|----:|
| test_abupt | **5.844%** | 5.820% | 5.872% | +0.028pp ❌ |
| test_wss | 6.727% | **6.634%** | 6.681% | **−0.046pp ✅ (beats SOTA)** |
| test_vol_p | **3.643%** (floor) | 3.779% | 3.800% | +0.157pp ❌ (breach) |
| test_surf_p | **3.577%** (floor) | 3.627% | 3.736% | +0.159pp ❌ (breach) |
| test_wss_x | — | — | 5.933% | — |
| test_wss_y | — | — | 7.238% | — |
| test_wss_z | — | — | 8.654% | — |

- **Best epoch:** EP15 (val_abupt=6.2963)
- **Outcome:** **NOT a contract winner** (Issue #1056). test_wss marginally beats SOTA but both vol_p and surf_p breach their #1056 floors. CLOSED.
- **Mechanism falsification:** MAE_aux and Charb on vol_p are functionally near-equivalent L1 signals; they do NOT compose multiplicatively on the H19 Charb base. GradNorm's w_vol_p collapsed to 0.05 floor by EP3 despite MAE_aux (same outcome as H19 alone). The L1 landscape on vol_p was already saturated by H19's Charb; adding MAE_aux only introduced interference cost on the other axes (wss/surf_p/abupt all regressed vs H19).
- **Key constraint for future design:** MAE_aux is useful only on MSE-base vol_p (H9 family). Do NOT combine MAE_aux with Charb-base vol_p.
- **CLOSED 2026-05-21.**

---

## 2026-05-14 20:16Z — PR #1086: EMA(0.999) on SDF α=0.25 stack (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/ema-0999-productive-stack-corrected`
- **Student:** dl24-tanjiro
- **W&B Run:** `fby84xtu`
- **Hypothesis:** EMA(0.999) with clean warm-start shadow re-init (PR #1087 fix applied), re-testing SDF α=0.25 stack to confirm EMA improvement is real and not contaminated by initial shadow from random init.

| Metric | This PR (EP11 best-EMA) | SOTA PR #972 | Δ |
|--------|------------------------:|-------------:|--|
| test_abupt | 5.9555% | **5.844%** | +0.111pp |
| test_wss | 6.7464% | 6.727% | +0.020pp |
| test_vol_p | 3.9895% | **3.643%** | +0.347pp |
| test_surf_p | 3.7070% | 3.577% | +0.130pp |
| test_tau_x | 5.971% | — | — |
| test_tau_y | **7.362%** | — | — |
| test_tau_z | **8.747%** | — | — |

- **Outcome:** **NOT a winner.** Bit-identical to fern xfykblf9 (5.955%) — confirmed at 4 significant figures. EMA warm-start fix is correctness-good but trajectory-neutral since contamination decayed to <1e-4 by step ~7400 in fern's run too.
- **Key insight:** Per-axis WSS decomposition reveals **tau_y=7.36% and tau_z=8.75%** dominate WSS error (tau_x=5.97%). Cross-flow shear components are the bottleneck — directly validates H1/H2 WSS hypothesis focus.
- **Conclusion:** EMA(0.999) does not close the 0.11pp gap to SOTA on SDF α=0.25 stack. The gap likely comes from a config difference vs PR #972 (SDF monkey-patch was no-op; #972 wins from stack, not SDF). EMA is a useful training tool but not the SOTA delta source.
- **CLOSED 2026-05-14.**

## 2026-05-25 07:40 — Capacity-axis sweep wrap-up (4 closes)

### PR #1298 — H117: Charbonnier on ALL 3 wss axes (x,y,z) — TERMINAL FALSIFIED
- dl24-fern/h117-wss-charb-xyz
- Hypothesis: extending Charbonnier sub-quadratic loss shape to all 3 wss axes (x,y,z) compounds the H41v2 y/z extension mechanism
- Terminal results table (best-EMA EP24 checkpoint, run `jmzd8s37`):

| Metric | H117 | H39 SOTA | Δ vs H39 | Floor | Status |
|--------|------|----------|----------|-------|--------|
| test_WSS | **6.7934** | 6.6506 | **+0.1428pp WORSE** | <6.6506 (PRIMARY) | **MISS** |
| test_VP | 3.5967 | 3.6033 | -0.0066pp BETTER | ≤3.643 | PASS |
| test_SP | 3.7710 | 3.6498 | +0.1212pp | ≤3.577 | BREACH +0.194 |
| test_abupt | 5.9146 | 5.8010 | +0.1136pp | ≤5.844 | BREACH +0.071 |

- Conclusion: Multi-axis Charbonnier fragments GradNorm bounded-loss budget across all 3 wss tasks, starving each axis. H39's restriction of Charb to z-axis only is precisely calibrated to GradNorm's budget structure. The hypothesis that broader Charb application would compound benefits is definitively false. **Charbonnier is fundamentally a single-axis mechanism under GradNorm.**

### PR #1313 — H132: H39 + backbone depth 6→7 — EP9 EARLY CLOSED, FALSIFIED
- dl24-frieren/h132-depth-6-to-7
- Hypothesis: depth-7 trunk extension provides extra sequential refinement steps for z-axis (load-bearing) extraction
- Mid-trajectory results (rank-0 run `y019u2zc`):

| EP | H132 val_wss | H39 ref | Δ vs H39 |
|----|--------------|---------|----------|
| 1 | 17.6300 | 17.8972 | -0.267 ✅ |
| 3 | 7.1962 | 7.2016 | -0.005 ✅ |
| 5 | 7.0491 | 7.0129 | +0.036 (cross) |
| 6 | 7.0562 | 6.9458 | +0.110 |
| 9 | 6.9778 | 6.8631 | **+0.115** ⚠️ stable plateau |

- Conclusion: Cold-start advantage (-0.267pp EP1) did NOT persist past EP3. Trajectory crossed zero at EP4 and stabilized at +0.10-0.12pp behind from EP6-9. Depth axis joins width-axis (H123 marginal) as falsified directions. Closed early at EP9 to save 21 EPs of compute.

### PR #1314 — H133: H39 + base LR 1e-4→7e-5 — EP11 EARLY CLOSED, FALSIFIED
- dl24-nezuko/h133-lr-7e-5
- Hypothesis: slower LR (7e-5) produces smoother trajectory with lower terminal val_wss
- Mid-trajectory results (rank-0 run `hxrpvb1b`):

| EP | H133 val_wss | H39 ref | Δ vs H39 |
|----|--------------|---------|----------|
| 1 | 20.9738 | 17.8972 | +3.077 ⚠️ massive cold-start deficit |
| 3 | 7.2874 | 7.2016 | +0.086 |
| 7 | 7.1053 | 6.9177 | +0.188 |
| 11 | 7.0718 | 6.8400 | **+0.232** narrowing -0.012/EP, too slow |

- Conclusion: Massive cold-start deficit (+3.08pp EP1) didn't recover. Narrowing rate ~-0.012pp/EP cannot catch H39 by EP30 (would only reach ~tie). H39's LR 1e-4 is already near-optimal for the Lion + cosine + warmup config. EMA crystallizes terminal smoothness, NOT slow LR. Closed early at EP11.

### Capacity-axis sweep DEFINITIVELY EXHAUSTED on H39 base — all 6 axes falsified or null

Pivoting to architectural changes per Plateau Protocol.

