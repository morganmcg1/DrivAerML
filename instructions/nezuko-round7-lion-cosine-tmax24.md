# nezuko Round 7: Lion+cosine T_max=24 nocompile (schedule sweep midpoint)

## Hypothesis

Single-lever experiment: add a cosine LR schedule with **T_max=24** to the SOTA Lion configuration (PR #50). Everything else is identical to the PR #50 baseline.

**Motivation:** Askeladd's PR #57 is testing T_max=16 simultaneously. PR #57 reached val 10.13 at ep9, confirming cosine schedule is stable with Lion uncompiled. This PR provides the **T_max sweep midpoint** — we need to know whether T_max=16 decays too aggressively before the model is ready.

## Why T_max=24?

At the 270-minute training budget (approximately ep9), different T_max values leave very different fractions of peak LR remaining:

| T_max | LR at ep9 | % of peak | Assessment |
|-------|-----------|-----------|------------|
| 16    | ~2.0e-5   | ~40%      | Aggressive decay — may cut LR too early |
| **24** | **~3.5e-5** | **~70%** | **Sweet spot — meaningful decay, substantial LR** |
| 50    | ~4.6e-5   | ~92%      | Barely any decay — schedule has minimal effect |

T_max=24 finds the balance: the model still has substantial learning rate in the final epochs, but the schedule has enough effect to help settle into a better minimum.

## Current SOTA Baseline

| Metric | Value | PR |
|--------|-------|-----|
| test_abupt (primary) | **11.208** | PR #50 |
| val metric (best) | ~10.08 | PR #50 (still descending at cutoff) |

Config: Lion uncompiled, lr=5e-5, wd=5e-4, NO schedule, 4L/512d/8h/128slices, ema-decay=0.9995, batch=4, surface/volume points=65536.

## Reproduce Command

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --agent nezuko \
  --wandb-name "nezuko/round7-lion-cosine-tmax24-rank0" \
  --wandb-group "nezuko-round7-lion-cosine-tmax" \
  --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 5e-5 --weight-decay 5e-4 \
  --no-compile-model \
  --lr-cosine-t-max 24 \
  --volume-loss-weight 2.0 --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.9995 \
  --gradient-log-every 100 --weight-log-every 100 \
  --no-log-gradient-histograms
```

**Single change from PR #50 baseline:** `--lr-cosine-t-max 24` added. All other flags identical.

## Context

- **Askeladd PR #57** is testing T_max=16 simultaneously. Compare results once both complete.
- This experiment is the T_max sweep midpoint between T_max=16 (too aggressive?) and T_max=50 (no schedule benefit).
- If T_max=24 beats T_max=16, try T_max=32 or T_max=40 next round.
- If T_max=16 beats T_max=24, the sweet spot is in the shorter range (T_max=12 could be worth testing).
