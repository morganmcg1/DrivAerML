# SENPAI Research State
- 2026-05-03 03:15 UTC (Round 30 mid-cycle — frieren #474 closed (3h silence after correction), reassigned PR #420 STRING-sep PE port to frieren as #490, fern #491 EMA soup; all 16 yi students now WIP)

## Most Recent Human Researcher Directives (Issue #18)

1. Be bold with architecture — complete backbone replacements encouraged
2. Cross-branch inspiration (check noam/radford branches)
3. Priority experiments: surface-tangent frame, Perceiver-IO, asinh/log norm, RANS div-free, 1-cycle LR
4. Use all W&B signals (gradient norms, weight histograms, loss slopes) to identify epoch-limited vs broken runs
5. Current architecture focus: structural wall_shear_y/z gap (2.5–2.9×), volume_pressure (2.0×)

## Current SOTA on yi

| Metric | Yi baseline (PR #309 thorfinn) | Aspirational (PR #311 tay) | AB-UPT Target |
|--------|-------------------------------|---------------------------|---------------|
| val_abupt (primary) | **9.039%** | 7.546% | — |
| test_abupt | ~10.2% | 8.771% | — |
| surface_pressure | ~5.5% | 4.485% | 3.82% |
| wall_shear (vector) | ~10.5% | 8.227% | 7.29% |
| wall_shear_y | ~10.5% | 9.233% | 3.65% |
| wall_shear_z | ~11.8% | 10.449% | 3.63% |
| volume_pressure | ~13.5% | 12.438% | 6.08% |

**Active yi merge bar: val_abupt < 9.039%**
**Aspirational once PR #420 (STRING-sep PE on yi) lands: < 7.546%**

## Infrastructure Status
- **DDP**: Working (PR #355 emma merged). All students use `torchrun --standalone --nproc_per_node=4`.
- **Muon optimizer**: Landed on yi (PR #377 edward merged 2026-05-02). Use `--optimizer muon --lr 1e-3`.
- **Huber loss**: `--wallshear-huber-delta` flag available (PR #317 violet merged).
- **STRING-sep PE**: NOT on yi yet — PR #420 (fern) is the port, in progress.
- **Per-step `train/lr` logging**: MISSING from yi `train.py`. PR #474 (frieren) will add it.
- **Key infra gap**: `--lr-warmup-epochs` and per-step cosine scheduling both absent on yi.

## Dominant Gaps vs AB-UPT (test metrics, post PR #311 aspirational)
- wall_shear_z: 2.88× (LARGEST GAP — top priority)
- wall_shear_y: 2.53×
- volume_pressure: 2.05×
- surface_pressure: 1.17×

## Active WIP Fleet (Round 30 — all 16 yi students WIP)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #491 | fern | EMA model soup (last-K=4 ckpt avg) — preserves PR #474 hypothesis | NEW (post-frieren-close) |
| #490 | frieren | STRING-sep PE port to yi (cherry-pick `6f2e991`) — PR #420 reassign | NEW (HIGH PRIORITY) |
| #485 | kohaku | asinh wallshear normalization (3-arm screen) | WIP — corrected wd 02:54 UTC |
| #478 | chihiro | Per-step cosine LR schedule (fix epoch-step decay — effective constant LR bug) | WIP |
| #473 | alphonse | Per-axis coord normalization (x/2, y/0.5, z/0.5) to fix PE anisotropy | WIP |
| #472 | edward | Muon@1e-3 vs AdamW full-budget 4-GPU DDP (PR #377 promotion) | WIP |
| #479 | thorfinn | Stochastic depth + dropout sweep (val→test gap attack, 2×2 grid) | NEW |
| #449 | norman | Log-x coordinate compression for isotropic sincos PE | WIP — advisor responded |
| #440 | violet | Huber δ sweep without tangential loss — SENT BACK for full-budget DDP | WIP |
| #436 | noam | Slices sweep 64/128/192 on STRING-sep stack | WIP |
| #435 | senku | 5L+STRING-sep full-budget 4-GPU DDP | WIP |
| #431 | askeladd | Grad clip norm sweep (0/0.1/0.3/0.5) | WIP |
| #430 | emma | Cosine EMA decay ramp 0.99→0.9999 — SENT BACK for DDP rerun | WIP |
| #425 | stark | tau_z channel upweight sweep z=2→5 | WIP |
| #421 | kohaku | Dual-stream Transformer with cross-attention bridge | WIP |
| #420 | fern | STRING-sep PE on yi branch (learnable log_freq/phase) | WIP — HIGH PRIORITY (CONFLICTING, results pending) |
| #370 | tanjiro | Cross-flow exposure index as input feature | WIP |
| #367 | haku | Theta-conditioned wall-shear loss weight | WIP |
| #366 | gilbert | Volume-pressure vol_weight=0.5 DDP promotion | WIP — 3x advisor pings, branch CLEAN |
| #262 | nezuko | Linear-warmdown LR (WSD-style) | WIP — rebased CLEAN, Arm A ep1 done |

## Current Research Themes

1. **STRING-sep PE port to yi** (PR #420 fern): The single most important in-flight work. When merged, it resets the merge bar from 9.039% to 7.546% and unlocks the multi-scale STRING-sep follow-up (PR #385 finding: k=4 beats k=1 with corrected init).

2. **Coordinate anisotropy attacks** (PRs #473 alphonse, #449 norman): Two complementary approaches — per-axis range normalization (alphonse) and log-x compression (norman) — both targeting the same root cause of tau_y/z gap (x spans ±2.0, y/z only ±0.5). These are cheap, zero-architecture changes that could yield significant channel-specific gains. Norman advisor-responded: separate surface vs volume asymmetry key to interpretation.

3. **tau_y/z direct attacks** (PRs #370 tanjiro, #367 haku, #425 stark): Three complementary angles — cross-flow exposure index, theta-conditioned loss weighting, channel upweighting. #419 chihiro tangent-frame input feature approach rejected (input redundancy).

4. **Optimizer frontier** (PR #472 edward): Muon@1e-3 showed −24.8% val_abupt at matched-compute vs AdamW. Now promoted to full-budget DDP run. This is the highest-leverage short-term win candidate.

5. **LR scheduling fix** (PR #478 chihiro): Current `CosineAnnealingLR(T_max=max_epochs)` stepped per epoch effectively gives constant LR for 9-epoch runs (T_max=50 default). Per-step cosine with `T_max=total_steps` would provide proper full-budget decay. Potentially a meaningful convergence improvement with a small code change.

6. **Post-training free wins** (PR #474 frieren): EMA-soup (last-K checkpoint averaging) adds zero training cost. Literature predicts 0.5–2pp gain when checkpoint-to-checkpoint variance is high.

7. **Architecture bold** (PR #421 kohaku): Dual-stream Transformer with O(NK) register-token cross-attention. If it works, blueprint for full Perceiver-IO replacement.

8. **Loss rebalancing** (PRs #440 violet, #366 gilbert, #367 haku): Huber loss for wall-shear tail, vol_weight=0.5 for pressure balance, theta-conditioned loss weighting.

9. **Generalization regularization** (PR #479 thorfinn): PR #450 confirmed val→test gap (~0.81pp) is dataset-shift-driven, not wd-driven. PR #479 tests stochastic depth (`--stochastic-depth-prob 0.05`) + dropout (`--model-dropout 0.05`) in a 2×2 grid — both are already-wired flags with zero code risk, targeting smoother representations that generalize better under covariate shift.

## Recently Reviewed (this session — 2026-05-02/03)

| PR | Student | Result | Decision |
|----|---------|--------|----------|
| #474 | frieren | EMA-soup (orig) — closed for non-engagement after 3h silence (2 advisor pings ignored); hypothesis preserved & reassigned to fern as #491 | CLOSED (no-engagement) |
| #429 | frieren | OneCycleLR vs cosine — screen structurally broken (sub-epoch, cosine never decays) | CLOSED |
| #430 | emma | EMA ramp — ramp never exercised (1.15% of schedule), only fixed-low-EMA snapshot | SENT BACK for DDP |
| #419 | chihiro | Tangent frame input features — +2pp regression; Arm A (input only) worse on all channels; Arm B (local-frame pred) killed ep1 | CLOSED |
| #450 | thorfinn | Lion WD sweep (1e-4/5e-4/2e-3/5e-3) — B(wd=5e-4) wins by 0.31pp, below 0.5pp DDP threshold; Arms C/D catastrophic; wd ≥ 1e-3 ruled out forever for Lion | CLOSED INFORMATIVE NEGATIVE |

## Dead Ends (do not re-assign on yi)

| Approach | PR | Reason |
|----------|-----|--------|
| Asinh tau-target normalization | #374, #249 | Grad-clip interaction: asinh-space MSE 3-4× inflation, wrong gradient direction |
| Area-weighted MSE | #17 | Heavy clipping erases physics signal |
| RANS divergence constraint | #124 | CFD pressure not smooth |
| Perceiver-IO backbone | #122, #212 | Cross-attn bottleneck loses fine CFD spatial structure |
| Learned Fourier / RFF PE | #298, #7 | Warmup confound confirmed; at matched warmup, no signal vs sincos |
| Surface-only point masking | #391 | Shared encoder propagates masking to all heads |
| K-NN local surface attention | #197 | Locality bias falsified; tau_y/z gap is not receptive-field issue |
| FFT spectral loss | #288 | Below practical bar; geometrically meaningless on unstructured mesh |
| OneCycleLR (pct_start=0.3) | #429 | Screen compromised; retry with pct_start=0.05-0.10 and DDP needed first |
| Surface-tangent frame input features | #419 | Input redundancy (t1/t2 deterministic from n — already an input); normalization bug in local-frame prediction |
| Lion wd ≥ 1e-3 | #450 | Catastrophic: wd=2e-3 → vol_p 30.98% (3.9×), wd=5e-3 → ws_y/z 43-44%. Sign-update collapses under high wd. Keep Lion wd in [1e-4, 5e-4]. |

## Key Research Insights

1. **Coordinate anisotropy is root cause of tau_y/z gap**: x spans ±2.0, y/z span ±0.5. STRING-sep learnable PE fixed this on tay (PR #311). Three different approaches to address it now in flight: STRING-sep (PE learns it), coord normalization (pre-PE equalization), log-x compression.

2. **Muon optimizer landed and shows strong early signal**: −24.8% rel at matched compute. PR #472 will validate at convergence. If it holds, Muon replaces AdamW as the canonical optimizer.

3. **DDP is critical for credible screening**: Single-GPU at this config takes ~7.5h/epoch. Sub-epoch evaluations are noisy and schedule-dependent. All meaningful screens should use 4-GPU DDP (5,441 steps/epoch, ~31 min/epoch, 8+ epochs in budget).

4. **Grad-clip + aggressive target normalization = hostile interaction**: Closed both asinh PRs. Don't reopen without disabling per-channel grad-clip.

5. **Lion at lr=1e-4 + 1-epoch warmup + clip=0.5 is confirmed SOTA optimizer baseline** on 4L/512d — until Muon DDP result arrives.

## Potential Next Research Directions (beyond current WIP)

1. **Muon on 5L architecture (if #472 wins)**: Muon + 5L STRING-sep could be the next large compounding win.

2. **Multi-scale STRING-sep k=4 with corrected init**: PR #385 (alphonse) confirmed k=4 beats k=1 by −1.14% val. Corrected linspace(−1.6, 3.0, 4) init for proper high-freq coverage. Port to yi/bengio stack with STRING-sep as base.

3. **SWA with multiple seeds (soup across seeds, not epochs)**: Train 3 models with different random seeds, average weights. Orthogonal to EMA-soup (PR #474 frieren).

4. **Fourier neural operator (FNO) decoder**: Replace MLP output heads with a small FNO. Well-motivated for CFD outputs with spatial structure.

5. **Data augmentation**: Small streamwise-pitch rotations (±5°) preserve physics but requires rotating normals AND tau vectors correctly — nontrivial rotation equivariance, defer until architecture settles.

6. **Volume-only coordinate normalization** (follow-up to #449 norman): If Arm B of #449 shows mixed (surface wins, volume regresses), split into log-x surface-only and log-x volume-only to isolate the best transform per stream.

7. **Mixed-precision BF16 → FP32 fallback for last N epochs**: If near-convergence instability limits final val_abupt floor, try finishing with FP32 for the last 1-2 epochs.
