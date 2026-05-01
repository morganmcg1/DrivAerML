# SENPAI Research State
- 2026-05-02 (Round 15/16/17 in flight — 17 WIP PRs, 0 idle students)

## Most Recent Research Direction from Human Researcher Team

**Issue #252** (open, Morgan, 2026-05-01): "Get inspired by Modded-NanoGPT". Directs the advisor to review the modded-nanogpt world record history table and reason carefully about applicability before assigning experiments. Already addressed by Round 15 PRs (see below).

**Issue #248** (open, Morgan): senpai-yi-stark pod requires manual provisioning (orchestrator service account lacks RBAC create/patch on configmaps/deployments). Comment posted directing Morgan to launch via cluster admin. Blocks PR #227 (surface-tangent frame).

**Issue #18** (earlier): Stop incremental tuning. Rip out the model architecture and try completely new approaches. Most priority experiments are assigned or closed:
1. Surface-tangent frame wall-shear prediction — PR #227 (stark), blocked on pod provisioning
2. Perceiver-IO backbone replacing Transolver — closed as dead end (PRs #122, #212)
3. asinh/log target normalization for wall shear — PR #249 (tanjiro), in progress
4. Physics-informed RANS divergence constraint — closed as dead end (PR #124)
5. 1-cycle LR schedule with higher peak (1e-3) — closed as dead end (PR #191)

## Modded-NanoGPT Mapping (Round 15, Issue #252 response)

| modded-nanogpt technique | PR | Student | Branch | Reasoning |
|---|---|---|---|---|
| Muon optimizer (Newton-Schulz orthogonalized momentum, record 3) | #261 | norman | yi | Lion stable at lr=1e-4 (PR #222). Muon is a strictly better Newton-step optimizer in weight space — direct fit for our Transolver. |
| Linear warmdown LR (WSD, records 28/41) | #262 / #269 | nezuko / norman | yi | Cosine cuts LR too early on 9-epoch budget; WSD keeps LR high longer. |
| Post-attn RMSNorm / sandwich-LN-style highway | #266 | stark | bengio | U-net record 11 analogue: gradient highway across layers helps tau_y/z multi-scale. |
| tanh output soft-cap (record 18 analogue) | #270 | violet | yi | Bounds physical predictions, prevents NaN-prone runaway in epoch 1. |
| Tighter grad clip (record 28 stability) | #267 | haku | yi | Reduces variance of crash distribution at lr=5e-4. |
| Larger width with muP-scaled LR | #271 | senku | yi | hidden_dim=768 with appropriate LR scaling. |
| LR-min lower bound (1e-5) | #272 | violet | yi | WSD-style minimum LR for plateau gain. |

## Current Baseline: PR #222 (fern) — yi branch — val abupt 9.291%

| Metric | yi best (val) | AB-UPT target | Ratio |
|---|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **9.291** | — | — |
| `surface_pressure_rel_l2_pct` | **5.8707** | 3.82 | 1.54× |
| `wall_shear_rel_l2_pct` | **10.3423** | 7.29 | 1.42× |
| `volume_pressure_rel_l2_pct` | **5.8789** | 6.08 | **0.97× (SOLVED)** |
| `wall_shear_x_rel_l2_pct` | — | 5.35 | — |
| `wall_shear_y_rel_l2_pct` | — | **3.65** | **~3.7× (major gap)** |
| `wall_shear_z_rel_l2_pct` | — | **3.63** | **~4.0× (major gap)** |

**Volume pressure is solved (0.97×). Wall_shear_y/z remain the dominant challenge. Surface pressure at 1.54× is the #2 priority.**

**Merge bar: 9.291% — any PR must beat this val_abupt to merge.**

## Active WIP PRs (as of 2026-04-29 — 17 WIP PRs)

| PR | Student | Hypothesis | Round |
|---|---|---|---|
| #273 | edward | Focal-loss per-point surface weighting for tau_y/z (γ sweep) | 15 |
| #270 | violet | tanh output soft-cap for wsy/wsz tail bounding | 15 |
| #262 | nezuko | Linear-warmdown LR (modded-nanogpt WSD-style) | 15 |
| #261 | norman | Muon optimizer (Newton-Schulz orthogonalized momentum) | 15 |
| #249 | tanjiro | asinh normalization for wall-shear targets | 14 |
| #288 | gilbert | Spectral Fourier loss on wall-shear tau_y/z channels | 16 |
| #244 | emma | Sweep surface-loss-weight (1.5/2.0) | 13 |
| #243 | chihiro | Sweep aux-rel-l2-weight (0.1/0.5/1.0) | 13 |
| #230 | senku | SWA uniform weight averaging for flat-minima generalization | 13 |
| #227 | stark | Wall-shear in local surface tangent frame **[BLOCKED — no pod, Issue #248]** | 14 |
| #297 | haku | symm-aug Arm C (include-both bs=4) on stable lr=1e-4/wu=1ep base — follow-up to #225 | 17 |
| #224 | fern | Learned Fourier embeddings for tau_y/z gap (per-axis freq learning) | 13 |
| #210 | kohaku | Gradient accumulation eff_bs=32 for smoother tau_y/z grads | 13 |
| #286 | frieren | Bilateral-symmetry TTA (y→-y reflection at inference) | 15 |
| #284 | alphonse | 6L/512d depth+width scaling — DDP port in progress (awaiting relaunch) | 15 |
| #208 | askeladd | Sandwich-LN to unlock 8L/256d depth (stability fix) | 13 |
| #193 | thorfinn | Curvature-biased surface point sampling (tau_y/z gap fix) | 13 |

## Key Architecture Configuration (PR #222 winning base config)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --optimizer lion \
  --lr 1e-4 \
  --weight-decay 5e-4 \
  --no-compile-model \
  --batch-size 4 \
  --model-layers 4 \
  --model-hidden-dim 512 \
  --model-heads 8 \
  --model-slices 128 \
  --ema-decay 0.999 \
  --lr-warmup-epochs 1
```

Note: Lion optimizer (lr=1e-4 with 1-epoch warmup) is now confirmed stable via PR #222. This resolves the earlier Lion instability observed at higher LRs.

## Fleet-Wide Stability Constraints

- **lr=5e-4 is the hard stability ceiling** for AdamW with clip=1.0/bf16.
- **`--lr-warmup-epochs 1` (or `--lr-warmup-steps 500`) is the dominant stability lever**.
- **Lion optimizer confirmed stable at lr=1e-4** with 1-epoch warmup (PR #222).
- **Adam v-saturation ceiling confirmed**: Lion at lr>1e-4 diverges; AdamW at lr=5e-4 NaN-instable.
- **Kill threshold**: gnorm<300 (not 100).
- **Per-axis static weight ceiling**: W_y=W_z < 3.0 at lr=5e-4/clip=1.0. W_y=W_z=2 stable (PR #66).

## Operational Notes (current round)

- **Fleet-wide stochastic lr=5e-4 instability**: PRs #243, #244, #245 all hit gradient explosions (gnorm → 100k+) in early epoch 1 caused by seed-dependent init/data orderings, NOT by the experimental interventions. Mitigation: kill at gnorm>300, relaunch at lr=3e-4 or with fresh seed. PR #243 Arms A-r3 (`v4mdrc2h`) and B-r2 (`f2oca4ee`) running healthily at lr=3e-4.
- **Stale Round-4 PRs closed 2026-05-01**: #75 (fern lr sweep), #79 (emma 60k points), #80 (tanjiro surface-loss-weight=2.0). All three superseded by current 4L/512d SOTA stack.

## Closed Dead Ends (do not re-assign)

| PR | Result | Reason |
|---|---|---|
| #75 fern | Old Round-4 LR sweep on 4L/256d+no-EMA | Architecture superseded by 4L/512d SOTA |
| #79 emma | 60k points sweep on 4L/256d+no-EMA | Architecture superseded by 4L/512d SOTA |
| #80 tanjiro | surface-loss-weight=2.0 on 4L/256d | Superseded by emma PR #244 on 4L/512d |
| #122 emma | Perceiver-IO: 2× worse than baseline | Cross-attn bottleneck loses fine CFD spatial structure |
| #212 noam | Perceiver-IO: closed (no pod) | No senpai-yi-noam deployment existed |
| #132 violet | Decoupled wallshear mag+dir: +12.7% worse | Cosine loss scales by sin(θ), not helpful for small-magnitude axes |
| #127 nezuko | Stochastic depth: all 3 arms worse on tau_y/z | Incoherent layer signal hurts boundary-layer features |
| #135 tanjiro | T_max=100 cosine LR: +4.74% vs PR #115 SOTA | Schedule lever closed; lr-change dominates |
| #167 tanjiro | W_y=W_z=3.5 + 1k LR warmup: NaN'd | Adam v-saturation at high static loss weights |
| #119 edward | RFF encoding: 56% worse | Fixed Gaussian B + non-isotropic coords = unstable |
| #124 gilbert | RANS divergence: all non-zero λ NaN'd | CFD pressure is NOT smooth — physical mismatch |
| #197 gilbert | K-NN local surface attention: all arms worse | Locality bias hypothesis falsified; tau_y/z gap is NOT a receptive-field problem |
| #196 edward | Lion optimizer (high LR): all 12 arms diverged | Lion unstable at lr>=1e-4 (old test, pre-warmup) |
| #191 haku | 1-cycle LR super-convergence: best 18.43 | OneCycleLR incompatible with time-limited regime |
| #171 norman | Snapshot ensemble with cosine restarts: V1+V2 failed | Cyclic LR snapshots don't give free gain |
| #199 stark | Surface-tangent frame: pod never launched | Zero compute attached; reassigned as PR #227 |
| #144 edward | β2=0.95 sweep: best 11.803 vs baseline 10.69 | β2 not a primary stability lever |
| #45 | Mamba-2 SSM: diverged | — |
| #15/#36 | SDF-gated volume attention: no improvement | — |
| #7/#17 | Area-weighted loss: non-viable | — |

## Key Research Insights

1. **Coordinate anisotropy** (PR #183): pos_max_wavelength=1000 gave +4.5% improvement vs 10000. DrivAerML vehicle bbox is ~8m×2.5m×2m — denser frequency sampling critical.

2. **Bilateral symmetry** of DrivAerML vehicles under y→-y reflection: tau_y anti-symmetric, tau_x/z unchanged. 50% free augmentation. PR #225 confirmed ep1 signal (−28% abupt, −29.4% tau_y/z for include-both Arm C), but lr=5e-4 instability prevented convergence. PR #297 (haku) re-tests Arm C on stable lr=1e-4/wu=1ep base.

3. **The y/z gap is a feature-resolution problem**, not a capacity problem. 6L/256d previously beat 4L/512d; depth beats width. However PR #222 found that 4L/512d with proper LR (1e-4 Lion + warmup) outperforms prior 6L/256d best.

4. **Volume pressure is solved** (0.97× AB-UPT). All future experiments should avoid sacrificing p_v for tau gains.

5. **LR warmup is mandatory for stability** at any optimizer. Single-epoch warmup (PR #222) works better than step-based warmup for Lion.

6. **Deep architecture risk**: 8L stability is untested; askeladd (#208) is testing sandwich-LN as stabilizer.

## Potential Next Research Directions

After current round completes:

- **Ensemble/averaging**: Train 2+ models with different seeds, average predictions. Free ~1–2% compounding win if predictions decorrelate (minimal code complexity, high expected impact).
- **Deeper investigation into what the model is getting wrong**: Visualize worst-predicted surface regions; are errors concentrated in specific geometric features (wheel arches, side mirrors)?
- **Bigger effective batch size via gradient accumulation** (#210 kohaku) — if eff_bs=32 shows gains, push to eff_bs=64/128.
- **Architecture width/depth scaling** — with 1-epoch warmup confirmed working, try 6L/512d or 8L/512d to see if the architecture capacity is still the ceiling.
- **Multi-scale feature aggregation** — hierarchical point encoding to capture both local surface topology and global geometry simultaneously.
- **Deformable convolution-style learned sampling** — instead of fixed sampling points, learn which regions to query for each prediction.
- **Spectral/frequency-domain loss** — optimize in frequency space for tau_y/z to address the heavy-tail distribution issue without asinh heuristics.
- **Physics-based regularization** (beyond RANS, which failed): symmetry constraints, continuity constraint at boundaries, divergence-free surface pressure gradients.
- **Cross-validation study**: Are current best metrics stable across folds? A variance analysis of the test set would indicate if further tuning helps or if we're noise-fitting.
