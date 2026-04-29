# SENPAI Research State

- **Date:** 2026-04-29
- **Branch:** `yi`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B:** `wandb-applied-ai-team/senpai-v1-drivaerml`
- **Most recent direction from human team:** 2026-04-28 — morganmcg1 (Issue #19)

## Key directives from human research team (Issue #18, 2026-04-28)

1. **Be bolder with architecture changes.** Don't be afraid to completely replace the model backbone. Students can handle radical departures from the reference `train.py` as long as logging, validation, and checkpointing are maintained.
2. **Cross-branch inspiration.** Before finalizing new hypothesis assignments, scan PRs from the `noam` and `radford` branches in wandb/senpai for prior art on similar techniques — useful for refinement ideas even if the dataset context differs. Similar work on a different dataset is *not* a reason to skip an idea.
3. **Empower students.** Frame assignments to give students the latitude to make big changes rather than conservative tweaks. Trust students to make great leaps.

## Current research focus

Round 1 calibration on a clean slate. The W&B project has zero prior runs and
`yi` has no merged baseline; the first wave must both establish a strong baseline
and surface the strongest single-delta improvements over it.

Next wave will prioritize **bold architectural ideas**: completely new model
backbones, transformer variants, neural operators, equivariant architectures —
not incremental tuning. Reference `noam` and `radford` branches before
finalizing hypotheses to avoid duplicating work and to draw inspiration.

## Known prior art (from outside the `yi` W&B project)

- `codex/optimized-lineage` branch ships a heavier baseline: 4L/256d/4h/128sl,
  lr 2e-4, wd 5e-4, 65k points, EMA 0.9995. Treated as the proven floor.
- `wandb/senpai` `radford` branch converged on a 4L/512d/8h champion with
  fourier features, cosine-T_max=36 LR schedule, EMA 0.9995, metric-aware
  rel-L2 auxiliary loss, and DomainLayerNorm. Best surface val ≈ 3.6%.
  The hardest known levers were `wall_shear` and `volume_pressure`.

## Round 1 — active PRs (all 16 students assigned 2026-04-28)

### Stream 1 — exploit existing evidence

| PR | Student | Hypothesis |
|---|---|---|
| #2 | alphonse | Stock defaults baseline — calibration floor |
| #3 | askeladd | codex/optimized-lineage config (4L/256d/4h, 65k pts, lr=2e-4) |
| #4 | chihiro | Large model 4L/512d/8h — radford champion scale-up |
| #5 | edward | Cosine LR + 5% warmup (proven radford winner family) |
| ~~#6~~ | ~~emma~~ | ~~Metric-aware MSE + rel-L2 aux loss~~ — CLOSED (sqrt instability) |

### Stream 2 — fresh targeted ideas

| PR | Student | Hypothesis | Primary target |
|---|---|---|---|
| #7 | fern | Gaussian random Fourier features for coordinates | p_s / tau |
| #8 | frieren | Per-case geometry FiLM conditioning | all |
| #9 | gilbert | Volume loss weight sweep 2.0x vs 3.0x | p_v (6.08%) |
| #10 | haku | Per-axis wall-shear channel loss weights (2x vs 3x) | tau (7.29%) |
| #11 | kohaku | Tangential wall-shear projection loss (physics-aware) | tau axes |
| #12 | nezuko | Stochastic depth / DropPath regularization | generalization |
| #13 | norman | Progressive EMA decay anneal 0.99→0.9999 | test checkpoint |
| #14 | senku | Deeper model 5L/256d/4h (depth ablation) | all |
| #15 | tanjiro | SDF-gated volume attention bias (near-wall emphasis) | p_v (6.08%) |
| #16 | thorfinn | Test-time bilateral symmetry TTA (xz-plane) | tau_y esp. |
| #17 | violet | Surface-area-weighted MSE loss (physics-consistent) | p_s / tau |

## Round 2 plan — bold architecture replacements

Full hypothesis pool: `research/RESEARCH_IDEAS_2026-04-28_ROUND2_ARCHITECTURES.md`
(16 ideas, generated after mining `wandb/senpai` `noam` and `radford` per Issue #18).

### Top-priority findings to act on

1. **A01 — ANP cross-attention surface decoder is a near-certain win.**
   noam PR #2379 swapped one head and got -70% in-domain pressure / -48% OOD on a
   different dataset. Should be among the first Round-2 assignments.
2. **The Transolver backbone has never been challenged on DrivAerML.** All 200 PRs
   on `radford` were tuning, not architecture swaps. Backbone-replacement frontier
   is wide open.
3. **50 unlabelled test geometries are free pretraining data.** B01 (denoising),
   B02 (MAE masking), C01 (DPOT transfer) can exploit this — no prior PR has.

### Ranked Round-2 candidates (top 8)

| # | Idea | Backbone change | Target |
|---|---|---|---|
| A01 | ANP cross-attention surface decoder | Replace surface MLP head | p_s, tau |
| A02 | SE(3)-invariant coord features (12-d) | Input augmentation only | all |
| B04 | Mamba-2 SSM surface decoder (Morton sort) | Replace surface head | p_s, tau |
| B05 | Soft MoE FFN (4 experts, learned dispatch) | Replace every FFN | all |
| C02 | Deep Evidential Regression head (NIG) | Replace MSE objective | all |
| A03 | Perceiver-IO encoder+decoder (1024 latents) | Full backbone swap | all |
| B01 | Denoising pretraining on geometry then fine-tune | Pretrain stage | all |
| B02 | MAE 75%-mask point pretraining | Pretrain stage | all |

Round 2 will assign these once Round 1 results come in (so we know which
loss/optim/EMA/data-weighting wins to compose with the new backbone).

## Round 2 — active assignments (2026-04-29)

| PR | Student | Hypothesis |
|---|---|---|
| ~~#20~~ | ~~nezuko~~ | ~~EMA decay sweep~~ — CLOSED, verdict (B) confirmed |
| #21 | kohaku | Normal-component suppression on top of tangential projection |
| #22 | gilbert | Add gradient clipping to train.py + 4-arm sweep |
| #23 | frieren | Full composition: FiLM + projection + vol_w=2.0 + bs=8 |
| #24 | emma | Squared rel-L2 aux loss (drop sqrt, smooth backward) |
| #26 | nezuko | A01 — ANP cross-attention surface decoder (architecture swap) |

**Closed in error 2026-04-29:** PR #25 (assigned to non-existent student `stark`) —
SE(3) local-frame coordinate features. Hypothesis remains a top-priority Round-2
candidate (A02) and will be reassigned to a real idle student.

## Round 1 — reviewed results (2026-04-29)

### VERIFIED WIN (pending merge) + MERGED — yi baseline progression

- **PR #8 (frieren, per-block FiLM conditioning) — VERIFIED WIN, pending merge.**
  `abupt_axis_mean = 16.53` (vs 17.39 baseline = −4.9%). Run `hltti2ec`,
  state=finished, 1 epoch, best_epoch=1, bs=2 only. Beats baseline in every
  test_primary axis. Apples-to-apples vs PR #3 (no FiLM, same config): 30.47 → 16.53
  = 46% reduction. FiLM mechanistically confirmed (token norm 70× growth, FiLM weights
  1.8–3.6×). Merge blocked on rebase conflict — frieren rebasing now.
  **Will be the new yi best once merged.**
  - Follow-up PR #23 (frieren): full composition — FiLM + vol_w=2.0 + projection + bs=8.
- **PR #9 (gilbert, vol_w=2.0 + protocol fixes) MERGED 03:57 UTC — current yi best.**
  `abupt_axis_mean = 17.39` (vs prior 35.12 = 50.5% reduction). Run `y2gigs61`,
  state=finished, 6 epochs reached, best_epoch=3.
  Win came primarily from **protocol fixes** (bs=8, validation-every=1, log-cadence).
  - **Infrastructure bug flagged:** `train.py` has no gradient clipping.
    Follow-up PR #22 (gilbert) adds it.
- **PR #11 (kohaku, tangential wall-shear projection) — MERGED earlier.**
  Code remains on yi (default off). Expected to compose for further gains.
- **Follow-up PR #21 (kohaku, normal-component suppression sweep)** —
  λ ∈ {0.0, 0.01, 0.1, 1.0} on top of projection.
- **Follow-up PR #22 (gilbert, gradient clipping)** — adds
  `torch.nn.utils.clip_grad_norm_` + 4-arm sweep. Infrastructure win
  blocking high-LR / high-weight / high-batch sweeps.

**Three independent wins now compounding (next big leap from stacking all):**
1. Tangential projection (PR #11, default off, `--use-tangential-wallshear-loss`)
2. Protocol fixes: vol_w=2.0, bs=8, validation-every=1 (PR #9)
3. Per-block FiLM conditioning (PR #8, pending merge)
→ PR #23 (frieren) will test all three together.

### CLOSED

- **PR #12 (nezuko, DropPath p=0.1) closed.** 81.21 vs 64.66 norman
  comparator. Underfitting regime (best_epoch=1 on both runs); any
  regularizer hurts. Wrong tool for the binding constraint.
- **Critical infrastructure win from PR #12.** nezuko shipped a per-step
  timeout fix (`train.py`); cherry-picked onto `yi` as `af92e9a`. Reserves
  `SENPAI_VAL_BUDGET_MINUTES` (default 90). Without this, every 65k-pts run
  silently times out without producing `test_primary/*`.
- **PR #20 (nezuko, EMA decay sweep) — CLOSED 2026-04-29 with diagnostic value.**
  4-arm sweep across `--ema-decay ∈ {0.99, 0.999, 0.9995, 0.99995}`. Verdict:
  **(B) genuine post-epoch-1 instability**, not EMA lag. Even the most aggressive
  decay (0.99, window ~100 steps) peaks at epoch 1; train loss is non-monotonic
  across all four seeds (5–7× higher in epoch 2). Per-step spikes hit 6–22× the
  median around steps 45–60k — exactly where missing gradient clipping bites.
  Best arm 0.9995 → abupt 24.74 (above yi baseline; closed because diagnostic
  not competitive). Confirms `--ema-decay 0.9995` as right default; PR #22
  (gradient clipping) is the cure for the binding constraint.
- **Follow-up: nezuko assigned A01 (ANP cross-attention surface decoder)** — the
  largest architectural win from noam (PR #2379 MERGED, −70% in-domain p_s on
  TandemFoil). Direct port to DrivAerML.

### Cross-cutting directives broadcast to all active PRs

1. Rebase onto `yi` to pick up `af92e9a` + projection code (default off).
2. **New recommended base config (PR #9 winner):**
   `--volume-loss-weight 2.0 --batch-size 8 --validation-every 1
    --gradient-log-every 100 --weight-log-every 100 --no-log-gradient-histograms`
3. Wall-shear targeted PRs (haku #10, tanjiro #15, thorfinn #16) should
   compose with `--use-tangential-wallshear-loss` so their delta stacks on
   the merged baseline.
4. **Training-stability bug flagged:** until PR #22 lands gradient clipping,
   sudden train-loss spikes followed by best_epoch lock-in are the bug, not
   the hypothesis.
5. Report any train→val divergence observed.

### Resolved question (2026-04-29)

**Train→val divergence after epoch 1 = (B) genuine post-epoch-1 instability**,
diagnosed by PR #20 (nezuko EMA sweep). Train loss explodes 5–7× from epoch 1
to epoch 2 across all 4 EMA arms; per-step train_loss spikes hit 6–22× median.
The cure is **gradient clipping** (PR #22 in flight) and likely **LR warmup**
(PR #5 edward in flight). Both fixes should land before drawing conclusions
from any other Round-1 PR with `best_epoch=1` lock-in.

## Constraints

- `SENPAI_MAX_EPOCHS` and `SENPAI_TIMEOUT_MINUTES` are fixed by harness — do
  not override. Throughput improvements (compile, AMP, point-count tuning,
  attention scaling) are the main lever for "more update budget".
- Read-only files: `data/*`, `pyproject.toml`, `instructions/*`. All edits
  go in `train.py`.
- **Logging cadence (Issue #19, 2026-04-28):** every Round-2+ assignment
  MUST include `--gradient-log-every 100 --weight-log-every 100` (or 250
  if needed) in the reproduce command. Per-step gradient/weight logging
  bottlenecks training to ~0.44 it/s on the 4L/256d/65k-pts/bs=2 base,
  preventing the run from reaching epoch 1 inside the 6 h timeout. At
  every-50 the same config runs at ~6.8 it/s. Slope cadence
  (`--slope-log-fraction 0.05`) is already efficient and stays.
- **Per-step timeout fix (commit `af92e9a` on `yi`, 2026-04-29):**
  `SENPAI_VAL_BUDGET_MINUTES` (default 90) is reserved out of
  `SENPAI_TIMEOUT_MINUTES` for in-loop val + post-loop full_val + test.
  All new student work must rebase onto `yi` to pick this up.
