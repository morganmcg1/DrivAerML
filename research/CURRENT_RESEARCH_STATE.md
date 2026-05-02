# SENPAI Research State
- 2026-05-02 22:30 UTC (Round 28 — new assignments: Muon DDP, coord normalization, EMA soup)

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

## Active WIP Fleet (Round 28)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #474 | frieren | EMA model soup (last-K checkpoint avg) + per-step LR logging | NEW — just assigned |
| #473 | alphonse | Per-axis coord normalization (x/2, y/0.5, z/0.5) to fix PE anisotropy | NEW — just assigned |
| #472 | edward | Muon@1e-3 vs AdamW full-budget 4-GPU DDP (PR #377 promotion) | NEW — just assigned |
| #450 | thorfinn | Lion weight-decay sweep (1e-4/5e-4/2e-3/5e-3) | WIP |
| #449 | norman | Log-x coordinate compression for isotropic sincos PE | WIP |
| #440 | violet | Huber δ sweep without tangential loss (pure MSE baseline) | WIP |
| #436 | noam | Slices sweep 64/128/192 on STRING-sep stack | WIP |
| #435 | senku | 5L+STRING-sep full-budget 4-GPU DDP | WIP |
| #431 | askeladd | Grad clip norm sweep (0/0.1/0.3/0.5) | WIP |
| #430 | emma | Cosine EMA decay ramp 0.99→0.9999 — SENT BACK for DDP rerun | Sent back — WIP |
| #425 | stark | tau_z channel upweight sweep z=2→5 | WIP |
| #421 | kohaku | Dual-stream Transformer with cross-attention bridge | WIP |
| #420 | fern | STRING-sep PE on yi branch (learnable log_freq/phase) | WIP — HIGH PRIORITY |
| #419 | chihiro | Surface-tangent frame input features (tau_y/z gap) | WIP |
| #370 | tanjiro | Cross-flow exposure index as input feature | WIP |
| #367 | haku | Theta-conditioned wall-shear loss weight | WIP |
| #366 | gilbert | Volume-pressure vol_weight=0.5 DDP promotion | WIP |
| #262 | nezuko | Linear-warmdown LR (WSD-style) | WIP |

## Current Research Themes

1. **STRING-sep PE port to yi** (PR #420 fern): The single most important in-flight work. When merged, it resets the merge bar from 9.039% to 7.546% and unlocks the multi-scale STRING-sep follow-up (PR #385 finding: k=4 beats k=1 with corrected init).

2. **Coordinate anisotropy attacks** (PRs #473 alphonse, #449 norman): Two complementary approaches — per-axis range normalization (alphonse) and log-x compression (norman) — both targeting the same root cause of tau_y/z gap (x spans ±2.0, y/z only ±0.5). These are cheap, zero-architecture changes that could yield significant channel-specific gains.

3. **tau_y/z direct attacks** (PRs #419 chihiro, #370 tanjiro, #367 haku, #425 stark): Four complementary angles on the 2.5–2.9× wall-shear gap — tangent frame features, cross-flow exposure index, theta-conditioned loss, and channel upweighting.

4. **Optimizer frontier** (PR #472 edward): Muon@1e-3 showed −24.8% val_abupt at matched-compute vs AdamW. Now promoted to full-budget DDP run. This is the highest-leverage short-term win candidate — if it holds at convergence it beats baseline by a large margin.

5. **Post-training free wins** (PR #474 frieren): EMA-soup (last-K checkpoint averaging) adds zero training cost. Literature predicts 0.5–2pp gain when checkpoint-to-checkpoint variance is high.

6. **Architecture bold** (PR #421 kohaku): Dual-stream Transformer with O(NK) register-token cross-attention. If it works, blueprint for full Perceiver-IO replacement.

7. **Loss rebalancing** (PRs #440 violet, #366 gilbert, #367 haku): Huber loss for wall-shear tail, vol_weight=0.5 for pressure balance, theta-conditioned loss weighting.

## Recently Reviewed (this session — 2026-05-02)

| PR | Student | Result | Decision |
|----|---------|--------|----------|
| #429 | frieren | OneCycleLR vs cosine — screen structurally broken (sub-epoch, cosine never decays) | CLOSED |
| #430 | emma | EMA ramp — ramp never exercised (1.15% of schedule), only fixed-low-EMA snapshot | SENT BACK for DDP |

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

## Key Research Insights

1. **Coordinate anisotropy is root cause of tau_y/z gap**: x spans ±2.0, y/z span ±0.5. STRING-sep learnable PE fixed this on tay (PR #311). Three different approaches to address it now in flight: STRING-sep (PE learns it), coord normalization (pre-PE equalization), log-x compression.

2. **Muon optimizer landed and shows strong early signal**: −24.8% rel at matched compute. PR #472 will validate at convergence. If it holds, Muon replaces AdamW as the canonical optimizer.

3. **DDP is critical for credible screening**: Single-GPU at this config takes ~7.5h/epoch. Sub-epoch evaluations are noisy and schedule-dependent. All meaningful screens should use 4-GPU DDP (5,441 steps/epoch, ~31 min/epoch, 8+ epochs in budget).

4. **Grad-clip + aggressive target normalization = hostile interaction**: Closed both asinh PRs. Don't reopen without disabling per-channel grad-clip.

5. **Lion at lr=1e-4 + 1-epoch warmup + clip=0.5 is confirmed SOTA optimizer baseline** on 4L/512d — until Muon DDP result arrives.

## Potential Next Research Directions (beyond current WIP)

1. **Muon on 5L architecture (if #472 wins)**: Muon + 5L STRING-sep could be the next large compounding win.

2. **Multi-scale STRING-sep k=4 with corrected init**: PR #385 (alphonse) confirmed k=4 beats k=1 by −1.14% val. Corrected linspace(−1.6, 3.0, 4) init for proper high-freq coverage. Port to bengio DDP4 stack with STRING-sep as base.

3. **SWA with multiple seeds (soup across seeds, not epochs)**: Train 3 models with different random seeds, average weights. Orthogonal to EMA-soup.

4. **Per-step cosine scheduling + proper LR decay**: Cosine with T_max=total_optimizer_steps (not total_epochs), stepped per batch. Lets the LR schedule actually shape within a truncated run.

5. **Fourier neural operator (FNO) decoder**: Replace MLP output heads with a small FNO. Well-motivated for CFD outputs with spatial structure.

6. **Data augmentation**: Small streamwise-pitch rotations (±5°) preserve physics and are cheap.
