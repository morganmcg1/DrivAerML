# SENPAI Research State
- 2026-04-29 16:45 (Round 25 — 16 WIP + 2 review PRs on yi, 0 idle; assigned PR #419 chihiro (surface tangent), PR #420 fern (STRING-sep PE on yi), PR #421 kohaku (dual-stream cross-attn bridge — Issue #18 architectural rip-out))

## Latest Survey Pass (2026-04-29 Round 25)

**Current SOTA on yi**: val_abupt **7.546%** (W&B `gcwx9yaa`, PR #311 edward — STRING-sep PE merged into `tay`, pending replication on `yi`).

**Dominant gaps vs AB-UPT**: wall_shear_y 2.53×, wall_shear_z 2.88×, volume_pressure 2.05×.

**New assignments this round (3 idle students):**
- **PR #419 (chihiro)**: Surface-tangent frame as input features. Compute orthonormal `t1`, `t2` per surface point from existing normals, append to `surface_x` (7→13 dims). Two arms: A=tangent input only; B=tangent input + `--predict-tau-local-frame` (rotate output back to global). Direct attack on cross-flow tau_y/z signal-to-noise.
- **PR #420 (fern)**: STRING-separable learnable PE on yi. Replicates PR #311's win on the yi branch (PR #311 was merged into `tay`, never to `yi`). Adds `learnable=True` to `ContinuousSincosEmbed` with per-axis `nn.Parameter` `log_freq` and `phase`. Two arms: A=fixed sincos control (yi reproducibility); B=`--learnable-pe`. Must beat 7.546%.
- **PR #421 (kohaku)**: **Issue #18 architectural rip-out — Dual-Stream Transformer with Cross-Attention Bridge.** Splits the single shared backbone into separate surface and volume Transformer stacks; bidirectional cross-attention bridges every N layers carry information between streams via zero-init `tanh(gate)` (AdaLN-zero pattern). Diagnoses single-backbone slice-budget competition as the root cause of wall_shear failures. Three arms: A=control, B=`--use-dual-stream --cross-attn-every 2`, C=3L+`--cross-attn-every 1`. Informed by final search of noam/radford branches in wandb/senpai (Mamba SSM PR #2376 + SE(2)-equiv PR #2377, both MERGED).

**Reviews pending this round:** PR #364 (frieren stochastic-depth sweep), PR #367 (haku theta-conditioned wallshear loss).

## Latest Survey Pass (2026-04-29 Round 23)

**Reviewed and closed this round:**
- **PR #298 (fern) CLOSED**: warmup confound between learned-FF and warmup-length conclusively confirmed. At matched 500-step warmup, sincos (A2: 16.84%) ties learned-FF (C: 16.97%) at ep1, Δ = -0.137 pp. Trajectory extrapolation puts A2 ep2 at ~14.3%, within ±0.5pp of C's known 14.47%. Bonus: clip 0.5 + ep1-warmup is +1.98pp worse than clip 1.0 + ep1-warmup (clip-warmup interaction). Learned-FF dropped from active hypothesis set.
- **PR #297 (haku) CLOSED**: best arm (CONTROL no-aug s42 = 12.61%) is 1.36x the 9.291% bar — single-GPU 1-epoch budget can't reach merge bar. Symmetry-aug doesn't beat no-aug at the best seed. **Variance-reduction signal kept on file**: include-both 5.3x lower seed half-range vs no-aug; CONTROL2 no-aug s7 collapsed to 34.30% while inc-both s7 stayed at 16.75%. Deterministic 2x-augmentation is a known fix for bad-basin seeds. PR #332 (tanjiro mirror-aug) is the active vehicle for the symmetry direction.

**New assignments this round:**
- **PR #362 (fern): loss-side tangent projection.** Project both tau_pred and tau_target onto surface tangent plane before MSE — physically motivated by the no-penetration BC. Distinct from PR #312 (output rotation, hurt) and PR #349 (input features). Single-GPU 2-arm screen on yi. Decision rule: Δ(B-A) ≤ -0.3 pp at ep1 → request full DDP run on emma's #355 base.
- **PR #363 (haku): diagnostic study, not optimization.** Load PR #222 ep9 checkpoint (`ut1qmc3i`), produce 5-section error-mode report — per-region error magnitude, spatial autocorrelation (Moran's I), error vs curvature, error vs flow alignment, per-case ranking. Goal: tell the fleet **where on the vehicle** tau_y/z errors physically concentrate so future optimization-side PRs can target geometry rather than guess globally. This is a high-leverage knowledge investment.

## Latest Survey Pass (2026-04-29 Round 22)

## Latest Survey Pass (2026-04-29 Round 22)

**Reviewed this round:**
- **PR #333 (emma)**: CLOSED — RoPE positional encoding deferred pending DDP infrastructure fix. Key finding: sincos and RoPE are complementary (A beats C by 12.5–17.8% across all metrics). However, both arms ran single-GPU (DDP missing in yi/train.py), hitting timeout at ~55% of warmup epoch 1. Cannot fairly compare to 9.291% merge bar. Volume pressure regressed most without sincos (22%), consistent with absolute position importance for full-domain fields. RoPENd module preserved for future follow-up. See EXPERIMENTS_LOG for full result table.

**New assignments this round:**
- **PR #355 (emma)**: DDP infrastructure fix — cherry-pick `bfbe975` + `1a8f7b7` from alphonse PR #284 to restore `init_process_group` + `DistributedSampler` + `set_device(LOCAL_RANK)` in yi/train.py. This is the single highest-leverage change: unblocks entire fleet from using canonical 8-GPU `torchrun --nproc_per_node=8` command. All students silently trained single-GPU in recent rounds.

**Critical infrastructure finding:**
`target/train.py` on `yi` has NO DDP. The canonical `torchrun --nproc_per_node=8` command in BASELINE.md is broken. Students running the base config command have been training single-GPU. PR #355 (emma) is fixing this. Until merged, all other students should cherry-pick `bfbe975` + `1a8f7b7` from `origin/alphonse/depth-scaling-6l-512d` onto their own branches.

## Latest Survey Pass (2026-04-29 Round 21)

**Reviewed this round:**
- **PR #243 (chihiro)**: CLOSED — aux-rel-l2 hypothesis not supported. All 3 weights (0.1/0.5/1.0) failed to beat 9.291 bar. Best val=10.897 (test=12.017). Aux signal magnitude too small (train/aux~0.02), amplified instability at higher weights.
- **PR #284 (alphonse)**: SENT BACK for 8-GPU re-run. 6L/512d showed dramatic per-epoch convergence gains (ep1 −45pp, ep2 −30pp vs 4L/512d) but budget cut at 3 epochs. Cosine T_max mismatch (T_max=999). Re-run: ddp8 fleet, `--nproc_per_node=8`, `--cosine-t-max-epochs 6`.
- **PR #298 (fern)**: SENT BACK with Arm A2 instructions. Arm A (sincos) = 16.68% vs Arm B (learned-FF) = 16.78% — no FF signal. Arm C's 14.47% win is entirely warmup confound. A2: sincos + 500-step warmup to confirm.
- **PR #249 (tanjiro)**: CLOSED (Round 21) — asinh normalization decisively refuted. Arm A asinh-on-v5 (u83ut9x2) at 31.62% vs Arm B baseline-ctrl-v5 (vya47gmk) at 15.43% — 2.05× regression on every channel including target tau_y/z. 5 rounds to achieve stable training; softplus-barrier scaffolding salvaged. See EXPERIMENTS_LOG.

**New assignments:**
- **PR #335 (chihiro)**: tau_y/z loss-weight curriculum (ramp W 1.0→2.0 or 1.0→3.0 over first 3 epochs). 3-arm: curriculum-1to2, curriculum-1to3, static-W=2 control.
- **PR #336 (tanjiro)**: Per-channel MLP output heads — 4 independent 2-layer MLP heads (hidden=256) replacing single shared LinearProjection(512, 4). Targets gradient interference across channels (tau_x dominates shared head, tau_y/z get residual capacity). 2-arm: control vs per-channel-h256.

## Most Recent Research Direction from Human Researcher Team

**Issue #252** (open, Morgan, 2026-05-01): "Get inspired by Modded-NanoGPT". Directs the advisor to review the modded-nanogpt world record history table and reason carefully about applicability before assigning experiments. Already addressed by Round 15 PRs (see below).

**Issue #248** (open, Morgan): senpai-yi-stark pod never provisioned. PR #227 closed 2026-05-02. Surface-tangent-frame hypothesis reassigned as PR #312 (edward) 2026-05-02.

**Issue #18** (earlier): Stop incremental tuning. Rip out the model architecture and try completely new approaches. Most priority experiments are assigned or closed:
1. Surface-tangent frame wall-shear prediction — **PR #312 (edward) assigned 2026-05-02** (first real test)
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

## Active WIP PRs (as of 2026-04-29 — 16 WIP PRs on yi)

| PR | Student | Hypothesis | Round | Notes |
|---|---|---|---|---|
| #335 | chihiro | tau_y/z loss-weight curriculum (ramp 1.0→2.0/3.0 over 3 epochs, 3-arm) | 20 | NEW — tests whether easing into W_y/z upweighting beats static W=2 from step-0; also tests W_max=3 cleanly |
| #334 | gilbert | Mesh-Laplacian GFT spectral loss on surface predictions (4-arm: λ=0.0/0.05/0.10/0.20) | 17 | GFT uses geometrically principled eigenvectors of graph Laplacian vs arbitrary-index FFT |
| #324 | stark | Per-channel target z-score standardization (2-arm: zscore vs control) | 19 | |
| #322 | emma | surface-loss-weight=2.0 on SOTA base (AdamW + lr-warmup-steps 2700) | 18 | |
| #317 | violet | Huber loss for wall-shear (δ=0.5/1.0/2.0 sweep) | 18 | |
| #316 | thorfinn | GradNorm dynamic per-task loss weighting for tau_y/z (bs=4, 2-arm) | 18 | stale WIP |
| #315 | senku | MLP expansion ratio sweep (mlp_ratio=2/4/8) | 18 | stale WIP |
| #314 | norman | Coordinate jitter augmentation sweep (σ=0.002/0.005/0.01) | 18 | |
| #313 | kohaku | Multi-seed ensemble averaging (3-seed variance reduction) | 18 | stale WIP |
| #312 | edward | Surface-tangent frame wall-shear prediction | 18 | stale WIP |
| #298 | fern | Learned-FF Arm A2: sincos + 500-step warmup (warmup-confound disambiguation) | 17 | Sent back with A2 instructions — if A2≈C (14.47%), warmup confound confirmed; if A2>C, FF has signal |
| #297 | haku | symm-aug Arm C (include-both bs=4) on stable lr=1e-4/wu=2000-step base | 17 | Sent back with --lr-warmup-steps 2000 fix |
| #364 | frieren | DropPath stochastic-depth sweep (p=0.0/0.05/0.10/0.20) on 4L/512d AdamW | 24 | PR #338 closed (non-responsive); re-tests PR #127 on current base, single-GPU AdamW |
| #284 | alphonse | 6L/512d depth+width scaling — 8-GPU re-run on ddp8 fleet | 15 | Sent back: needs 8 GPUs + cosine-t-max-epochs 6; DDP/Lion port now on branch |
| #262 | nezuko | Linear-warmdown LR (WSD-style) on 4L/512d SOTA | 15 | |
| #336 | tanjiro | Per-channel MLP output heads for surface (4 heads: surface_p/tau_x/y/z, hidden=256 each) | 21 | NEW — attacks gradient interference in shared linear head, orthogonal to all in-flight tau_y/z fixes |
| #208 | askeladd | Sandwich-LN to unlock 8L/256d depth (stability fix) | 13 | |

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
  --lr-warmup-steps 2720 \
  --wallshear-y-weight 2.0 \
  --wallshear-z-weight 2.0 \
  --volume-loss-weight 2.0
```

Note: Lion optimizer (lr=1e-4 with 1-epoch warmup) is now confirmed stable via PR #222. This resolves the earlier Lion instability observed at higher LRs. `--lr-warmup-steps 2720` ≈ 1 epoch at 8-GPU DDP / bs=4 (10,883 steps/epoch / 4 gradient-accumulation-equivalent). Use `--lr-warmup-steps` only — `--lr-warmup-epochs` does NOT exist in base yi train.py (alphonse's DDP branch has it; pending infra PR).

**DDP infrastructure (alphonse PR #284 branch):** torchrun support, Lion optimizer, `--lr-warmup-epochs` flag, `DistributedSampler`, `all_reduce` for val metrics, DDP `run_id` broadcast fix — all on `alphonse/depth-scaling-6l-512d`. Pending cherry-pick to yi after PR #284 re-run.

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
| #288 gilbert | FFT spectral loss (λ=0.10 → 0.32pp gain) | Signal below 0.5pp practical bar; arbitrary node-index FFT is geometrically meaningless on unstructured mesh. Pivot to GFT (PR #334). |
| #243 chihiro | aux-rel-l2-weight sweep (w=0.1/0.5/1.0) | Best val=10.897 (test=12.017) — 17% above bar. Aux signal magnitude tiny (~0.02); model already optimizes an equivalent quantity. Higher weights amplify instability. Confounded lr sweep. |
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
