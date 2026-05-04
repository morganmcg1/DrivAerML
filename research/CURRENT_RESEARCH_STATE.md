# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-04 (4 new experiment PRs assigned: nezuko PR #602 ensemble pool expansion, tanjiro PR #603 EMA decay sweep, fern PR #604 inference-time mirror-y TTA, edward PR #605 surface curvature features; fleet at full utilization 8/8)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Fleet:** 0 idle students, 8 WIP PRs (full GPU utilization)
- **Tay-deployed students:** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn (8 total)

## ENSEMBLE SOTA — PR #562 nezuko greedy K=7 forward selection — val_abupt **6.2345%**

Beats prior K=5 ensemble (PR #556 val=6.2681%) by Caruana 2004 greedy forward selection from 22 candidates.

## SINGLE-MODEL SOTA — PR #571 askeladd tau_y×1.5 / tau_z×2.0 weight intensification — val_abupt **6.7644%** / test_abupt **8.171%**

Beats prior SOTA PR #516 (6.8701%) by −0.106pp (−1.54% relative). W&B run `nh96x7m4`, group `askeladd-tau-sweep`, runtime ~4.7h.

All single-model PRs must beat val_abupt < **6.7644%** with test_abupt ≤ ~8.20%.

**Key insight:** Moderate tau weight intensification (tau_y×1.5, tau_z×2.0) on the SOTA stack further closes the tau_y/tau_z gap. tau_y: 8.663%→8.489% (−0.174pp), tau_z: 10.266%→9.997% (−0.269pp). GradNorm-EMA NOT used — pure fixed-weight reweighting outperformed GradNorm on this stack.

### Previous Single-Model SOTA: PR #516 — val_abupt 6.8701% / test_abupt 8.1229%

### Noise-floor calibration (askeladd PR #571 rebased SOTA repro — now MERGED)
- Identical-config rerun of SOTA stack: val=6.9226% vs claimed 6.8701% (+0.052pp on identical code)
- **Treat improvements within ±0.05pp of SOTA as noise**, not signal
- Genuine win threshold: val_abupt < **6.71%** (was 6.82% vs PR #516; now recalibrated against 6.7644%)
- Borderline (need replicate): 6.71–6.81%
- No signal / regression: > 6.81%

| Metric | PR #523 SOTA val EP6 | AB-UPT |
|---|---:|---:|
| `abupt` | **6.9246%** | — |
| `surface_pressure` | 4.5840% | 3.82% |
| `wall_shear` | 7.7457% | 7.29% |
| `volume_pressure` | **4.3040%** | 6.08% (BEATEN) |
| `tau_x` | 6.7193% | 5.35% |
| `tau_y` | 8.7197% | 3.65% |
| `tau_z` | 10.2960% | 3.63% |

Test (best-val checkpoint): test_abupt=8.2355%, test_tau_y=8.4656%, test_tau_z=9.6720%.
W&B run: `wyz68o8r`, group `thorfinn-gradnorm-r2`, runtime 4.71h.

### Canonical SOTA reproduce command

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <student> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --lr-cosine-t-max 13 --epochs 13 \
  --use-gradnorm --gradnorm-mode ema_proxy \
  --gradnorm-alpha 0.5 --gradnorm-ema-beta 0.9 --gradnorm-min-weight 0.7
```

Note: `--use-gradnorm` overrides `--surface-loss-weight` / `--volume-loss-weight` (legacy scalars are ignored when gradnorm is enabled).

---

## Latest research direction from human researcher team

No new directives as of 2026-05-04 (issues #285, #252, #48 all have current advisor responses).

---

## Currently in-flight (8 active WIP PRs on tay, ZERO idle)

| PR | Student | Lever | Status (2026-05-04) |
|---|---|---|---|
| #552 | thorfinn | GradNorm-EMA min_weight floor sweep | Arm A (floor=0.5) DONE val=6.9602%; Arm B (floor=0.3) DONE val=6.9569%; alpha=1.5 follow-up running — awaiting results |
| #592 | alphonse | Model depth sweep — layers=5 vs layers=6 on SOTA stack | In progress (assigned 2026-05-01) |
| #593 | frieren | lr-min cosine floor sweep — lr-min=1e-6 vs lr-min=1e-5 | In progress (assigned 2026-05-01) |
| #594 | askeladd | RFF spectral budget sweep — 8 vs 32 features on PR #571 SOTA | In progress (assigned 2026-05-01) |
| #602 | nezuko | Ensemble pool expansion: add PR #571 run nh96x7m4 (pool 22→23), re-run greedy K≤15 | **NEW — assigned 2026-05-04** |
| #603 | tanjiro | EMA decay sweep {0.9993, 0.9997, 0.9999} on full SOTA stack | **NEW — assigned 2026-05-04** |
| #604 | fern | Inference-time Y-mirror TTA on SOTA checkpoint nh96x7m4 (zero training cost) | **NEW — assigned 2026-05-04** |
| #605 | edward | Surface curvature features (mean H + Gaussian K via k-NN PCA k=20) concatenated to surface inputs | **NEW — assigned 2026-05-04** |

Round-12/13 focus: build on PR #571 new SOTA (6.7644%); further close tau_y/tau_z gap (tau_y still 8.489% vs AB-UPT 3.65%, tau_z still 9.997% vs AB-UPT 3.63%); expand ensemble pool to capture new SOTA run; sweep EMA decay and RFF budget for orthogonal gains; probe surface geometric priors. Gate: val_abupt < 6.7644% (single-model), < 6.2345% (ensemble).

---

## Recent merges / closures (Round 12 → Round 13 boundary; updated 2026-05-04)

### PR #571 askeladd tau_y/tau_z weight intensity Arm A — MERGED (NEW SINGLE-MODEL SOTA)
- val_abupt=6.7644% (−0.106pp vs PR #516), test_abupt=8.171%
- W&B run `nh96x7m4`, group `askeladd-tau-sweep`
- tau_y: 8.663%→8.489% (−0.174pp), tau_z: 10.266%→9.997% (−0.269pp)
- New gate: val_abupt < 6.7644%
- Arm B (tau_y×2.0/tau_z×2.5, run `62yojciu`) still running — watch for further improvement

### PR #553 alphonse coord-jitter regularization — CLOSED NEGATIVE
- Hypothesis: Gaussian coordinate noise (sigma=0.001–0.01) as Tikhonov regularizer would improve tau_y/tau_z via preventing surface-coordinate overfitting
- Sigma=0.0 baseline arm reproduced SOTA at val=6.9511% (clean repro, confirms stack intact)
- Sigma=0.001 arm: val_abupt=9.603% at EP2.3 — **+38% relative regression** vs baseline
- Verdict: STRING-sep/RFF relies on precise coordinate structure; injecting coordinate noise destroys positional encoding quality. Falsified. Added to negative results catalog.

### PR #562 nezuko greedy K=7 ensemble — MERGED (ENSEMBLE SOTA)
- val_abupt=6.2345% via Caruana 2004 forward selection from 22-run pool

### PR #516 askeladd tau-weight v2 — MERGED (SINGLE-MODEL SOTA)
- val_abupt=6.8701%, test_abupt=8.1229% — defines current single-model gating threshold

### PR #142 thorfinn vol_w=2.0 — closeout logged (+10.78%)
### PR #146, #141 — closeouts logged

### Completed / closed in this cycle (since 2026-05-01)
- PR #568 fern NorMuon — completed and closed; released fern for PR #604 (TTA assignment)
- PR #572 nezuko Lion β1 sweep — completed and closed; released nezuko for PR #602 (ensemble expansion)
- PR #573 edward EMA decay sweep (prior round) — completed and closed; released edward for PR #605 (surface curvature)
- PR #574 tanjiro RFF spectral density (prior round) — completed and closed; released tanjiro for PR #603 (EMA decay sweep redux)
- PR #571 Arm B (tau_y×2.0/tau_z×2.5) — watch for results; may trigger another merge if val < 6.7644%

---

## Active research themes

### 1. tau_y/tau_z gap closure (primary open problem)
- **Gap status:** tau_y=8.489%, tau_z=9.997% vs AB-UPT 3.65%/3.63% (2.3–2.7× above; values from PR #571 Arm A best-val — improvement from PR #516: tau_y −0.174pp, tau_z −0.269pp)
- **Active attacks (this round):**
  - GradNorm-EMA tighter floor (#552 thorfinn) — alpha=1.5 follow-up after floor-never-binds result
  - Model depth sweep (#592 alphonse) — layers=5 vs 6 on full SOTA stack (prior 6L negative was on older stack)
  - lr-min cosine floor (#593 frieren) — preserve late-epoch signal via nonzero cosine floor
  - RFF spectral budget (#594 askeladd) — 8 vs 32 features to probe capacity-noise tradeoff
  - EMA decay sweep (#603 tanjiro) — {0.9993, 0.9997, 0.9999}; longer EMA windows (effective 1429, 3333, 10000 steps) may reduce tau_y/tau_z variance
  - Surface curvature features (#605 edward) — mean H + Gaussian K (k-NN PCA k=20) as geometric prior for near-wall tau prediction
  - Inference-time mirror-y TTA (#604 fern) — zero-cost: average pred(x,y,z) and pred(x,-y,z) with tau_y sign-flip; distinct from training aug (#536 which failed)

### 2. Negative-direction confirmed (do not retry on current stack)
- **Static channel reweighting**: Prior round showed 4× negative (#142, #454, #467, #531); however askeladd PR #571 Arm A (tau_y×1.5, tau_z×2.0 on SOTA GradNorm+Lion stack) now shows val=6.7644% — **GENUINE WIN**. Prior negatives used simpler stacks. Intensity sweep Arm B (×2.0/×2.5) in progress to find optimal point.
- **Y-mirror data augmentation** (#536) — gap is structural
- **Direction loss on tau** (#531) — gap is not direction-prediction

### 3. Composition opportunities (next round, when winners emerge)
- EMA decay winner (#603) + surface curvature features (#605) — orthogonal; can stack
- RFF budget winner (#594) + lr-min floor winner (#593) — spectral × schedule
- Ensemble pool expansion (#602) is independent of all training PRs — always applies
- Model depth winner (#592) + EMA decay winner (#603) — architecture × regularization
- Inference TTA (#604) is composable with any single-model winner at zero training cost

---

## Potential next research directions

1. **Flow-aligned coordinate frame for tau outputs** — predict tau in (t̂, n̂, ŝ) tangent-frame basis instead of (x,y,z), freeing the model from coordinate-axis-aligned bias (would explain why tau_x is easier than tau_y/tau_z)
2. **EMA + surface curvature composition** — if both #603 and #605 win, combine on a single run to check additive gain
3. **Knowledge distillation from SOTA ensemble into single model** — ensemble of 7+ models as teacher, smaller student with surface auxiliary targets
4. **Volumetric query-point density curriculum** beyond 65536 — push to 98304 or 131072 with VRAM headroom check; SOTA stack may have VRAM to spare with batch=4
5. **Loss-aware sampling** — over-sample query points in regions of historically high tau_y/tau_z error during training (boosting-style); requires per-point error tracking
6. **Tau uncertainty decomposition** — separate aleatoric (noise-floor) from epistemic uncertainty in tau predictions using MC-dropout; identifies irreducible vs reducible error
7. **Sliced Wasserstein distance** as auxiliary distribution-matching loss on tau channels (vs L2 only)
8. **Two-stage training: warmup-only on volume, then unfreeze surface heads** — attacks the joint-loss imbalance from a different angle than GradNorm
9. **Spectral lion with adaptive beta sweep** — sweep Lion β2 momentum (currently fixed at 0.99); longer second-moment windows may improve tau channels similarly to EMA
10. **Greedy ensemble with test-time TTA members** — if #604 TTA wins, build ensemble pool of (original + TTA-flipped) predictions for every pooled run, effectively doubling pool at zero training cost

---

## Negative results catalog (do not retry on current stack)

| Lever | Outcome |
|---|---|
| Local tangent-frame INPUT features | NEGATIVE (#423) |
| Channel-selective Huber on tau | NEGATIVE (#353) |
| Volume-loss-weight scalar rebalancing | NEGATIVE (#451) |
| Separate volume decoder | NEGATIVE val→test overfit (#452) |
| Muon optimizer | NEGATIVE (#299) +4.09pp |
| Sandwich-norm | NEGATIVE diverged |
| U-net skips | NEGATIVE (+0.555pp) |
| 256d / 768d hidden | NEGATIVE |
| 6L / 8L depth (pre-STRING-sep stack) | NEGATIVE on old stack — retrying 5L/6L on full SOTA stack (PR #592 alphonse) |
| Per-axis output head scaling (#467) | NEGATIVE — gap is upstream |
| TTA mirror-y inference (#499) | NEGATIVE +1.18pp on old stack — retrying on PR #571 SOTA stack (PR #604 fern) with correct tau_y sign-flip; prior attempt may have lacked sign-flip correction |
| **Y-mirror training aug (#536)** | **NEGATIVE — gap is structural, not symmetry-addressable** |
| 2× surface point density (#506) | NEGATIVE — slower/epoch beats density |
| tau_yz scalar loss-weight reweight (#142, #454, #467, #531) | Previously 4× NEG on simpler stacks; **PR #571 Arm A (tau_y×1.5/tau_z×2.0 on SOTA Lion+GradNorm stack) SUCCEEDED — MERGED** |
| mlp_ratio=6/8 wider FFN (#458) | NEGATIVE — mlp4 is optimal |
| Signed-log target transform (#471 arm-b) | NEGATIVE |
| log1p target transform (#481) | NEGATIVE |
| AdamW vs Lion (#532) | NEGATIVE — Lion optimal, confirmed |
| Full GradNorm (5× autograd overhead) | NEGATIVE operationally — crashes in budget |
| Unit-vector cosine direction loss on tau (#531) | NEGATIVE — direction is not the bottleneck |
| **slw=2.0 13-epoch full (PR #537)** | **NEGATIVE vs current SOTA — within noise of #510 prior win, doesn't beat #523** |
