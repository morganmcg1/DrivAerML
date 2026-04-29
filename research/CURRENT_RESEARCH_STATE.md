# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-29 20:00 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Most recent direction from human team:** _(none on tay yet — yi advisor
  carried Issue #18: "be bolder; replace the backbone; mine `noam`/`radford`
  branches; trust students to take big leaps")_

## Calibration status — PR #40 MERGED 2026-04-29 ~20:00 UTC

**New tay SOTA: test_abupt 17.25 (−2.9% vs PR #33, −12.9% vs PR #30)**

| PR | Student | test_abupt | Key lever |
|---|---|---:|---|
| #30 | alphonse | 19.81 | yi calibration config (4L/512d/8h) |
| #33 | fern | 17.77 | RFF coord features sigma=1.0, no-compile |
| **#40** | **alphonse** | **17.25** | **torch.compile fix → 12 epochs vs 9; every axis improved** |

**torch.compile bug is now FIXED** (`trainer_runtime.py:293,301,929`). All future runs should use compile by default (no `--no-compile-model`). Throughput: ~16 min/epoch compiled → 270-min budget yields 12 epochs.

Key insight from PR #40: compile fix alone (no RFF) already beats RFF-without-compile. Volume pressure improved most: 16.13 → 14.71. Val curve still descending at epoch 12 — more budget would improve further.

**Update (2026-04-29 16:25 UTC) — Round 1 first wave RESULTS CONFIRMED:**

Test results on 50-case test split (best-val checkpoint), all under the
torch.compile-bug-limited 9-epoch budget (alphonse calibration baseline
established at PR #30, test_abupt 19.81):

| Student | Hypothesis | val_abupt | test_abupt | Δ vs baseline | Outcome |
|---|---|---:|---:|---:|---|
| fern | RFF coord features (sigma=1.0) | 17.06 | **17.77** | **−2.04** | **NEW LEADER**; pushed comment to mark ready (kill sigma=2.0 sweep) |
| thorfinn | Per-axis tau weights (1/1.5/1.5) | 18.52 | **19.44** | −0.37 | Win in W&B; **code not pushed** (asked to push) |
| alphonse | Calibration (yi PR #4) | 18.70 | 19.81 | baseline | MERGED PR #30 |
| frieren | FiLM AdaLN-zero | 19.86 | 20.07 | +0.26 | CLOSED (1.3% noise floor) |
| edward | Cosine LR + warmup | 19.96 | 20.99 | +1.18 | Loss; comment sent to close on Arm A |
| askeladd | Full composition stack | 36.x | 38.02 | +18.21 | CLOSED (tangential proj broken tau_z) |
| nezuko | ANP cross-attn decoder | **17.85*** | — | — | Running, val tracking close to fern; slope still descending |
| tanjiro | SDF gate / Lion | — | — | — | Pod stuck on stale assignment file |

*nezuko val 17.85 at step 20591, runtime 4.6h — still running; if test tracks val,
this would be a tay leader near-tie with fern.

**Key learnings**:
1. **RFF features (fern) are a real win** — drop test_abupt by 10% over
   calibration. Compose orthogonally with everything in queue.
2. **FiLM did not transfer to 4L/512d at 9 epochs** — reproduces yi's mechanism
   (geom_token L2 ×180, scale_w L2 ×38) but no metric lift. Revisit after
   compile fix when we have ~16-22 compiled epochs.
3. **Tangential projection has structural tau_z bug** — z-aligned panels
   lose gradient signal because GT wall-shear is mostly normal-direction
   on roof/hood/floor. Tau_z monotonically *increased* during training in
   PR #31. Eval-time projection is the correct fix (PR #41 askeladd).
4. **Cosine EMA budget miscalibration** — schedule is over max_epochs but
   we only run 17% of it in the budget. Effective decay ~0.991 vs
   alphonse's 0.9995. Cosine EMA was effectively neutralized.
5. **Compile + drop_last=False bug** — the largest single throughput lever.
   Fixing this in trainer_runtime.py recovers 50% of training budget for
   ALL future experiments. PR #40 alphonse working on this now.

**Round 1b assignments (after first wave):**
- alphonse #40: Fix compile bug (drop_last=True) + recalibrate — running (step 834)
- askeladd #41: Eval-time tangential projection of wall-shear — running (step 2156)
- frieren #42: Squared rel-L2 loss (drop outer sqrt for smooth backward) — running (step 2374)
- tanjiro #39: Lion optimizer — pod stuck on stale assignment file (#36 still showing)
- thorfinn: pod alive but in-pod claude session never reported PR #37 results

**Round 1b/2 disposition:**
- fern #33: MERGED 16:39 UTC — tay SOTA test_abupt **17.77** (−10.3%)
- edward #32: CLOSED — test 20.99
- alphonse #40: **MERGED ~20:00 UTC** — new tay SOTA test_abupt **17.25** (−2.9% vs #33)

**Round 2 active assignments:**
- alphonse #46: Compose RFF (sigma=1.0, 32 feats) + compile-fix — most important next composition test; if additive, projects to ~15.5 abupt below yi's 15.82
- fern #43: Multi-scale RFF 3-band {0.1,1.0,10.0} — 3rd attempt running (crashed at step ~12980 twice), currently step ~5442
- edward #44: Progressive cosine EMA (0.99→0.9999) + RFF — running, early training
- askeladd #41: Eval-time tangential wall-shear projection — running step ~20832
- frieren #42: Squared rel-L2 loss — running step ~23459, val 18.04 (descending)
- thorfinn #47: Bilateral symmetry **train-time** augmentation — fresh assignment after closing #37
- nezuko #35: ANP cross-attn decoder + RFF + vol_w=3 — running step ~16031
- tanjiro #39: Lion optimizer — running step ~18210

**Round 1b/2 dispositions:**
- thorfinn #37: CLOSED ~20:30 UTC — bilateral-tta-eval-only finished test_abupt 19.71 (regression vs SOTA 17.25); per-axis weights code never pushed. Reassigned to PR #47.

Round 1 on tay therefore has two simultaneous purposes:

1. **Calibrate** the tay/DDP8 baseline against the yi numbers (16.64 merged,
   15.82 pending) so we can claim genuine improvements.
2. **Compose orthogonal wins** that yi never finished merging together
   (FiLM × cosine-EMA × 512d × tangential-loss is the obvious composition).

A subset of students push beyond yi with bold architecture or training
changes that yi only got as far as Round-2 assignments for.

## Active research themes

1. **Composition of yi orthogonal wins on DDP8.** Width 512d (chihiro),
   FiLM AdaLN-zero (frieren), cosine EMA 0.99→0.9999 (norman), vol_w=2.0
   (gilbert), tangential wall-shear projection (kohaku). Yi predicted ~12-13
   abupt for the full stack; we run that on DDP8 and verify.
2. **Schedule and feature engineering.** Cosine LR + warmup, Gaussian
   Fourier coordinate features. Both were Round-1 ideas in yi that never
   produced clean comparable results because of stability bugs that have
   since been patched (gradient clipping is now default in the trainer).
3. **Architectural swap.** Replace the Transolver surface MLP head with an
   ANP-style cross-attention decoder. Largest single architectural win
   identified from `noam`'s prior work; never tested on DrivAerML.
4. **Volume-targeted attention.** SDF-gated volume attention bias for
   near-wall p_v emphasis. p_v is one of the hardest targets to move and
   AB-UPT's 6.08 reference is a real gap from yi's 14.21.
5. **Wall-shear axis disparity.** tau_y and tau_z are systematically worse
   than tau_x. Per-axis loss weighting + bilateral-symmetry TTA both
   directly attack this gap with low-risk single deltas.

## Round 1 — assignments (8 students, all DDP8) — opened 2026-04-29

| PR | Student | Stream | Hypothesis |
|---|---|---|---|
| #30 | alphonse | calibrate | Reproduce yi PR #4 winning config (4L/512d/8h, lr=5e-5, bs=4) on DDP8 — establishes tay baseline |
| #31 | askeladd | exploit | Full yi composition stack: 512d × cosine-EMA × tangential proj × vol_w=2.0 |
| #32 | edward | exploit | Cosine LR + 5% warmup on top of 512d composition |
| #33 | fern | exploit | Gaussian random Fourier features + 512d composition |
| #34 | frieren | exploit | AdaLN-zero per-block FiLM + 512d composition |
| #35 | nezuko | explore | A01 — ANP cross-attention surface decoder (replace head) |
| #36 | tanjiro | explore | SDF-gated volume attention bias — **CLOSED** (5+ deterministic eval-path crashes at step 2719); reassigned to **PR #39 Lion optimizer** (single-delta train.py-only). Pod stuck on iteration 9 since 11:53 UTC; PR #39 not yet picked up. |
| #37 | thorfinn | explore | Per-axis wall-shear loss weighting + bilateral-symmetry TTA |

## Next research directions (Round 2 candidates, queued)

These are the strongest ideas to try after Round 1 lands:

- **Backbone replacement** — Perceiver-IO, Mamba-2 SSM surface decoder, full
  GeoTransolver/Transolver-3 backbone, GINO/FNO operator on the volume
  channel.
- **Pretraining** — denoising on 50 unlabelled test geometries; MAE 75%
  point masking; DPOT transfer.
- **Loss reformulation** — squared rel-L2 (no sqrt, smooth backward);
  deep-evidential-regression head with NIG; uncertainty-aware loss.
- **MoE / routing** — Soft-MoE FFN with 4 experts; dispatcher conditioned
  on geometry token.
- **Data representation** — Morton/Hilbert ordering for SSM-friendly
  sequences; multi-resolution patch tokens; equivariant local frames.
- **Optimization** — µP scaling so width sweeps don't need per-width LR
  retuning; AdamW + Sophia/Lion comparisons.

## Constraints

- `SENPAI_MAX_EPOCHS` and `SENPAI_TIMEOUT_MINUTES` set by harness —
  do not override. Throughput improvements (compile, AMP, point-count,
  attention scaling, DDP8 communication tuning) are the lever for "more
  update budget" inside the budget.
- DDP8 = 8 GPUs per student × 96 GB each. Effective batch size at 512d/bs=4
  per-GPU = 32. Use `--wandb_group` whenever a hypothesis needs multiple
  arms.
