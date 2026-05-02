# SENPAI Research State
- 2026-05-02 19:00 UTC (Round 27 — baseline correction + new assignments)

## CRITICAL BASELINE CORRECTION (2026-05-02 19:00 UTC)

**PR #311 (edward, STRING-sep PE, val 7.546%) was merged into the `tay` branch, NOT `yi`.** The `yi` train.py only has the original fixed-omega `ContinuousSincosEmbed`. BASELINE.md corrected in commit `96a2345`.

- **True yi merge bar TODAY: val_abupt 9.039%** (PR #309 thorfinn, grad-clip=0.5)
- **Aspirational target once PR #420 lands STRING-sep PE on yi: 7.546%** (PR #311 tay run `gcwx9yaa`)

## Most Recent Human Researcher Directives (Issue #18)

1. Be bold with architecture — complete backbone replacements encouraged
2. Cross-branch inspiration (check noam/radford branches)
3. Priority experiments: surface-tangent frame, Perceiver-IO, asinh/log norm, RANS div-free, 1-cycle LR

## Current SOTA on yi (corrected)

| Metric | Yi baseline (PR #309 thorfinn) | Aspirational (PR #311 tay) | AB-UPT Target |
|--------|-------------------------------|---------------------------|---------------|
| val_abupt (primary) | **9.039%** | 7.546% | — |
| test_abupt | ~10.2% | 8.771% | — |
| surface_pressure | ~5.5% | 4.485% | 3.82% |
| wall_shear_y | ~10.5% | 9.233% | 3.65% |
| wall_shear_z | ~11.8% | 10.449% | 3.63% |
| volume_pressure | ~13.5% | 12.438% | 6.08% |

## Dominant Gaps vs AB-UPT (after STRING-sep lands on yi)
- wall_shear_z: 2.88× (LARGEST GAP — top priority)
- wall_shear_y: 2.53×
- volume_pressure: 2.05×
- surface_pressure: 1.44×

## Infrastructure Status
- **DDP**: Restored by PR #355 (emma), merged 2026-05-02 17:05 UTC. All students can use `torchrun --standalone --nproc_per_node=4`. Yi pods have 4 GPUs (not 8 — base config in BASELINE.md uses 8 GPUs, adjust to 4).
- **Huber loss**: PR #317 (violet) merged — `--wallshear-huber-delta` flag available.
- **Volume loss weight**: `--volume-loss-weight` flag available on yi.
- **STRING-sep PE**: NOT on yi yet — PR #420 (fern) is the port, commit `6f2e991` has the implementation, running.

## Active WIP Fleet (Round 27 — 18 PRs)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #449 | norman | Log-x coordinate compression for isotropic sincos PE | NEW — just assigned |
| #440 | violet | Huber δ sweep without tangential loss (pure MSE baseline) | Nudged — awaiting launch |
| #436 | noam | Slices sweep 64/128/192 on STRING-sep stack | Nudged — awaiting launch |
| #435 | senku | 5L+STRING-sep full-budget 4-GPU DDP | Running (budget-aware schedule `3hxdbk9p`) |
| #431 | askeladd | Grad clip norm sweep (0/0.1/0.3/0.5) | Running (4-GPU parallel) |
| #430 | emma | Cosine EMA decay ramp 0.99→0.9999 | Nudged — awaiting launch |
| #429 | frieren | 1-cycle LR (OneCycleLR) vs cosine | Running (Arm A GPU0 + Arm B GPU1 parallel) |
| #425 | stark | tau_z channel upweight sweep z=2→5 | Nudged — awaiting launch |
| #421 | kohaku | Dual-stream Transformer (register-token cross-attn) | Rewriting (O(N²)→O(NK) fix approved) |
| #420 | fern | STRING-sep PE on yi branch (learnable `log_freq/phase`) | Running — Arms A+B parallel 2+2 GPUs |
| #419 | chihiro | Surface-tangent frame input features (tau_y/z gap) | Running — Arm A then Arm B sequential |
| #385 | alphonse | Multi-scale STRING-sep PE k=1/4/8 | Running — 3-arm screen ~50% done (21:00 UTC ETA) |
| #377 | edward | Muon vs AdamW — final val results pending | Awaiting val completion |
| #370 | tanjiro | Cross-flow exposure index as input feature (R2 3-arm) | Running — R2 relaunched post-rebase |
| #367 | haku | Theta-conditioned wall-shear loss weight (3-arm) | Running — relaunched post-rebase |
| #366 | gilbert | Volume-pressure Huber + vol_loss_weight=0.5 DDP run | SENT BACK — await 4L/512d Lion DDP promotion |
| #262 | nezuko | Linear-warmdown LR (WSD-style) | Conflict — pending rebase (Arm A ep1: 18.84%) |

## Recently Closed (this session)
- **PR #374 (thorfinn) CLOSED**: Asinh tau-target normalization fails monotonically. Grad-clip at norm=1.0 interacts badly with asinh-space MSE (3-4× inflation, wrong gradient direction). All 4 arms worse than control. Dead end confirmed.
- **PR #391 (norman) CLOSED**: Surface-only point masking hypothesis disproved. All 3 masking arms regressed vs control (shared encoder propagates Bernoulli masking to volume predictions).

## Pending Reviews (waiting for student results)
- PR #385 alphonse: results expected ~21:00 UTC
- PR #431 askeladd: running
- PR #377 edward: final val pending (~20:00 UTC for Muon@1e-3)

## Current Research Themes

1. **STRING-sep PE port to yi** (PR #420 fern): The single most important in-flight work. Fixes the yi baseline miscalibration. Commit `6f2e991` has the implementation. When PR #420 merges, the aspirational bar becomes the real merge bar.

2. **Coordinate preprocessing / anisotropy** (PR #449 norman): Log-x compression approximates STRING-sep's per-axis effect for free, using the existing fixed PE. Complementary test even if PR #420 succeeds.

3. **tau_y/z direct attacks** (PRs #419 chihiro tangent frame, #370 tanjiro cross-flow index, #367 haku theta-conditioning, #425 stark z-upweight): Six complementary angles on the 2.53-2.88× wall-shear gap.

4. **Architecture bold changes** (PR #421 kohaku dual-stream with register-token cross-attn): Register-token bottleneck is O(NK) — makes dual-stream tractable at N=65536.

5. **Volume pressure** (PR #366 gilbert vol_loss_weight=0.5): DDP promotion of confirmed mechanism. -1.28pp val_abupt in 1-epoch screen without breaching vol_p constraint.

6. **Optimizer / LR** (PRs #377 edward Muon, #430 emma EMA ramp, #429 frieren 1-cycle, #262 nezuko WSD, #431 askeladd clip, #435 senku 5L DDP): Full optimization parameter sweep after DDP restored.

## Dead Ends (do not re-assign on yi)

| Approach | PR | Reason |
|----------|-----|--------|
| Asinh tau-target normalization | #374, #249 | Grad-clip interaction: asinh-space MSE 3-4× inflation causes wrong gradient direction |
| Area-weighted MSE | #17 | "Heavy clipping erases the physics signal" |
| RANS divergence constraint | #124 | CFD pressure is NOT smooth |
| Perceiver-IO backbone | #122, #212 | Cross-attn bottleneck loses fine CFD spatial structure |
| Fourier/RFF position encoding | #298 (fern), #7 (fern) | Warmup-confound; at matched warmup, no signal vs sincos |
| Surface-only point masking | #391 | Shared encoder propagates masking to all heads |
| K-NN local surface attention | #197 | Locality bias falsified; tau_y/z gap is not receptive-field |
| FFT spectral loss | #288 | Signal below practical bar; geometrically meaningless on unstructured mesh |

## Key Research Insights

1. **Coordinate anisotropy is the root cause** of tau_y/z gap: x spans [-2, 2], y/z span [-0.5, 0.5]. STRING-sep learnable PE (PR #311 on tay) fixed this by per-axis log_freq tuning. Port via PR #420 is the priority.

2. **Grad-clip + aggressive target normalization = hostile interaction**: clip_grad_norm=1.0 + asinh-space MSE = wrong gradient direction. Closed both asinh PRs. Don't reopen without disabling grad-clip first.

3. **Volume pressure is near-solved**: PR #309 baseline has vol_p ~13.5%. PR #311 tay shows 12.44%. Reducing it costs val_abupt gains (PR #366 shows vol_w=0.3 trips vol_p constraint).

4. **DDP is working and critical**: Single-GPU runs are 4× slower; Arm comparisons on single-GPU need budget-aware schedule (e.g., `--max-steps-per-epoch 2721`).

5. **Lion at lr=1e-4 + 1-epoch warmup is the confirmed SOTA optimizer** on 4L/512d. Muon stability ceiling = 1e-3 peak LR (PR #377 edward).

## Potential Next Research Directions (beyond current WIP)

1. **Perceiver-IO with register tokens**: The kohaku dual-stream work (PR #421) is building toward O(NK) cross-attention. If it works, it's a blueprint for a full Perceiver-IO with efficient cross-attention.

2. **Multi-checkpoint soup averaging**: Train 3+ models with different random seeds, average weights. If predictions decorrelate (expected given different random init), free ~1–2pp gain.

3. **Curriculum point sampling**: Start with high-curvature surface points (tau_y/z concentrated there per PR #193 finding), gradually add uniform points. Biases early training toward the hard cases.

4. **Physics-based coordinate frame**: Instead of raw xyz, use car-aligned coordinate system (streamwise/lateral/vertical) — more meaningful physics decomposition for the anisotropic PE.

5. **Fourier neural operator (FNO) decoder**: Replace MLP output heads with a small FNO that operates on the spatial distribution of surface points. Well-motivated for CFD outputs with periodic structure.

6. **Data augmentation via pitch/roll perturbation**: Small rotations around x and z axes (±5°) as an augmentation that's cheap and preserves physics (pressure distribution changes predictably with pitch).
