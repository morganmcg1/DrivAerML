# SENPAI Research State
- 2026-05-03 09:25 UTC (Round 31 cycle open — closed #485 kohaku asinh-target (Jacobian-amplification rejection) and #431 askeladd grad-clip (structural finding: clip = effective-LR scaler at 97.5% clip rate). Reassigned 6 idle students with fresh single-question hypotheses; all 16 yi students now WIP again, zero idle GPUs.)

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

## Active WIP Fleet (Round 31 — all 16 yi students WIP)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #522 | thorfinn | BF16 → FP32 last-2-epoch fine-final-precision policy | NEW (R31) |
| #521 | senku | sigmoid-l1 slice gating (replace softmax slice) | NEW (R31) |
| #520 | norman | volume-only coord transform (signed-log / asinh) — #449 follow-up | NEW (R31) |
| #519 | gilbert | layer-wise LR decay (LLRD) for transformer blocks | NEW (R31) |
| #518 | kohaku | loss-side asinh wallshear (no inverse pipeline) — #485 corrected | NEW (R31) |
| #517 | askeladd | Lion (lr, clip) joint sweep at fixed lr·clip product | NEW (R31, follow-up to #431) |
| #491 | fern | EMA model soup (last-K=4 ckpt avg) | WIP |
| #490 | frieren | STRING-sep PE port to yi (cherry-pick `6f2e991`) | WIP — HIGH PRIORITY |
| #478 | chihiro | Per-step cosine LR schedule (fix epoch-step decay — effective constant LR bug) | WIP |
| #473 | alphonse | Per-axis coord normalization (x/2, y/0.5, z/0.5) to fix PE anisotropy | WIP |
| #472 | edward | Muon@1e-3 vs AdamW full-budget 4-GPU DDP (PR #377 promotion) | WIP |
| #449 | norman | Log-x coordinate compression for isotropic sincos PE | WIP — paired with #520 |
| #440 | violet | Huber δ sweep without tangential loss | WIP |
| #436 | noam | Slices sweep 64/128/192 on STRING-sep stack | WIP |
| #430 | emma | Cosine EMA decay ramp 0.99→0.9999 | WIP |
| #425 | stark | tau_z channel upweight sweep z=2→5 | WIP |
| #421 | kohaku | Dual-stream Transformer with cross-attention bridge | WIP |
| #420 | fern | STRING-sep PE on yi branch (learnable log_freq/phase) | WIP — HIGH PRIORITY |
| #370 | tanjiro | Cross-flow exposure index as input feature | WIP |
| #367 | haku | Theta-conditioned wall-shear loss weight | WIP |
| #262 | nezuko | Linear-warmdown LR (WSD-style) | WIP — rebased CLEAN, Arm A ep1 done |

## Round 31 fresh assignments (2026-05-03 09:25 UTC)

Six new single-question hypotheses just kicked off, broadening the theme space:

- **#517 askeladd** — Lion (lr, clip) joint sweep at fixed `lr·clip` product. Tests his own #431 finding that at 97.5% clip-rate, clip is just an effective-LR scaler.
- **#518 kohaku** — loss-side asinh wallshear (residual transform inside loss only, no inverse pipeline). Re-tests #485's underlying gradient-rebalancing hypothesis the structurally correct way.
- **#519 gilbert** — layer-wise LR decay (LLRD) for the 4 transformer blocks + PE + head. Standard Kaggle/BERT lever, completely untested in this programme.
- **#520 norman** — volume-only signed-log/asinh coordinate transform, paired with his existing #449 (which tests the global form). Together they isolate the surface↔volume asymmetry.
- **#521 senku** — sigmoid-l1 slice gating to replace softmax slice attention. Architectural A/B test on Transolver winner-take-all dynamics.
- **#522 thorfinn** — BF16 → FP32 last-2-epoch fine-final-precision policy. Post-training free-win category alongside #491 (EMA soup).

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
| #485 | kohaku | asinh wallshear target normalization (3-arm screen) — best 18.09% vs 9.04% bar; monotonic *wrong* direction; mechanism: inverse-pipeline Jacobian amplifies heavy-tail errors | CLOSED NEGATIVE — followed up by #518 (loss-side asinh, no inverse pipeline) |
| #431 | askeladd | gradient-clip sweep — single-GPU phase 1 found interior optimum at clip=0.3 but 100% clip-rate; DDP-4 phase 2 had multi-axis confound (optimizer+wd+clip differed). Lion+clip=0.5 = 10.92% val_abupt (3-epoch DDP-4 budget). Structural finding: at 97.5% clip-rate, clip is an effective-LR scaler. | CLOSED INFORMATIVE — followed up by #517 (Lion lr·clip joint sweep at iso-product) |
| #474 | frieren | EMA-soup (orig) — closed for non-engagement after 3h silence; reassigned to fern as #491 | CLOSED (no-engagement) |
| #429 | frieren | OneCycleLR vs cosine — screen structurally broken (sub-epoch, cosine never decays) | CLOSED |
| #430 | emma | EMA ramp — ramp never exercised (1.15% of schedule); SENT BACK for DDP | (in flight) |
| #419 | chihiro | Tangent frame input features — +2pp regression on all channels | CLOSED |
| #450 | thorfinn | Lion WD sweep — B(wd=5e-4) wins by 0.31pp, below DDP threshold; wd ≥ 1e-3 ruled out forever for Lion | CLOSED INFORMATIVE NEGATIVE |

## Dead Ends (do not re-assign on yi)

| Approach | PR | Reason |
|----------|-----|--------|
| Asinh tau-target normalization | #374, #249, #485 | Grad-clip interaction (#374, #249) AND inverse-pipeline Jacobian amplifies heavy-tail errors when metric is in original space (#485). Loss-side asinh (no inverse) is the corrected framing — see #518. |
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
