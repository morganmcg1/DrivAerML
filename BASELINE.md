# DrivAerML Baseline

**Branch:** `yi` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml`

## Status update — 2026-05-03 18:38 UTC

**STRING-sep learnable PE is now on yi.** PR #490 (frieren, `frieren/string-sep-pe-yi-v2`) merged at 15:48 UTC 2026-05-03. The `--learnable-pe` flag is available in yi `train.py`. The frieren resumed run `zwh9qzjw` achieved val_abupt=8.087% using STRING-sep + Lion lr=1e-4 clip=0.5 (from a checkpoint resume — non-canonical merge candidate).

**Current active merge bar: val_abupt 9.032%** (PR #517 askeladd, `brat65z4`, Lion lr=1e-4 clip=0.5 without STRING-sep — launched before PR #490 merged). The bar has NOT yet been updated to reflect STRING-sep because PR #517's run predates the merge.

**Expected next bar: ~8.1–8.5%** once PR #539 (frieren, from-scratch canonical STRING-sep + Lion run) completes.

Compounding-wins entries 1–14 below remain valid (all merged on yi).

## Mechanism flag landed: PR #317 violet (Huber wall-shear loss) — 2026-04-29

PR #317 (violet, `--wallshear-huber-delta`) adds opt-in Huber loss on standardized
residuals for surface τ channels (1..3 = τ_x/y/z). Default delta=0.0 → exact MSE,
zero behavior change. The mechanism was validated in a 2-seed parity comparison:
mean Δ val_abupt = −1.45pp (ctrl 37.15% → d10 35.70%), both seeds agree (−1.15pp and
−1.75pp), per-channel gains match heavy-tail hypothesis (ws_z −3.39pp, ws_x −3.16pp,
ws_y −0.57pp). Volume backbone uncontaminated (vol_loss ~0.014–0.016 flat).

Note: Absolute merge bar (7.546% from PR #311) was not contested — paired comparison
ran under 1-epoch tangential-loss conditions (SENPAI_TIMEOUT_MINUTES=360 limits to
1 epoch on single-GPU). The flag is available for composition with asinh (#374) and
full-budget DDP runs. W&B runs: ctrl `g1s45tbt`/`6649fm5e`, d10 `52urviip`/`zni9if9p`.

**Active merge bar on yi as it actually exists today: val_abupt 8.861% (PR #583 edward, β-NLL beta=0.5, W&B run `5xovw2si`).**
**Aspirational target once STRING-sep PE is fully on yi: val_abupt ~7.0–7.5% (tay SOTA PR #511, `5o7jc7wi`).**

## Status: edward PR #311 wins — new baseline 2026-05-02

PR #311 (edward, STRING-separable learnable position encoding) reduced
`val_primary/abupt_axis_mean_rel_l2_pct` from 9.039% (PR #309 thorfinn grad-clip=0.5) to
**7.546%** — a 16.5% improvement on the headline metric, and a 13.9% improvement
on `test_primary/abupt_axis_mean_rel_l2_pct` (10.190% → **8.771%**). W&B run: `gcwx9yaa`,
group: `tay-round18-grape-ablation`, Arm B.

Key finding: replacing isotropic RFF-32 (or no spectral encoding) with a
**STRING-separable** position encoding — axis-aligned frequencies with per-axis
**learnable** `log_freq` and `phase` parameters — yields the cleanest
architectural win in many rounds. Test deltas vs prior SOTA: surface_pressure
−0.98pp, wall_shear −1.68pp, volume_pressure −0.22pp. All val slopes still
negative at terminal epoch (model still converging) → more epochs would likely
extend the gain. This is the **first new-architecture (vs HP) win** in several
rounds and should compound with the existing 4L/512d Lion+warmup stack.

## Previous: fern PR #222 wins — baseline 2026-05-01

PR #222 (fern, 1-epoch LR warmup before cosine decay) reduced
`val_primary/abupt_axis_mean_rel_l2_pct` from 9.484% (fern PR #208 / prior best) to
**9.2910%** — a 2.0% improvement on the headline metric. W&B run: `ut1qmc3i`, group: `tay-round12-lr-warmup-1ep`.

Key finding: adding a single epoch of linear LR warm-up before cosine annealing stabilises
Lion training on the 4L/512d architecture (lr=1e-4, batch=4, torchrun 8-GPU DDP).
Smooth loss descent across all 9 epochs with no instability. Best-epoch=9.
Operates on the **4L/512d** architecture (not 6L/256d), which had been the
previous SoTA architecture.

**Compounding wins so far (all landed on `yi`):**
1. PR #11 kohaku — tangential wall-shear projection loss code
2. PR #9 gilbert — protocol fixes (bs=8, vol_w=2.0, validation-every=1)
3. PR #4 chihiro — width scale-up to 512d/8h
4. PR #14 senku — depth scale-up to 6L/256d
5. PR #58 alphonse — NaN-safe checkpoint guard (bugfix)
6. PR #66 thorfinn — per-axis tau_y/z loss upweighting W_y=2, W_z=2
7. PR #99 fern — LR peak 5e-4 (5× base)
8. PR #169 thorfinn — NaN/Inf-skip safeguard, --seed, --lr-warmup-steps (infra utilities, no metric regression)
9. PR #183 fern — pos_max_wavelength=1000 (omega-bank sincos positional encoding)
10. PR #222 fern — lr_warmup_epochs=1 (Lion stability, 4L/512d architecture)
11. PR #309 thorfinn — gradient clipping max_norm=0.5 (val 9.039%)
12. ~~PR #311 edward — STRING-separable learnable position encoding (val 7.546%, test 8.771%)~~ **NOT ON YI: PR #311 merged to `tay`, not `yi`. Yi `train.py` still has fixed-omega `ContinuousSincosEmbed`. Port via PR #420 (fern) is in flight.**
13. PR #355 emma — DDP infrastructure restored (cherry-pick bfbe975+1a8f7b7: init_process_group, DistributedSampler, DDP wrap, Lion optimizer wiring; unblocks full fleet 4/8-GPU torchrun; metrics bar unchanged)
14. PR #517 askeladd — Lion optimizer lr=1e-4 clip=0.5 confirmed optimal (val 9.032%, W&B run brat65z4). Iso-product hypothesis FALSIFIED: Arm B lr=5e-5 clip=1.0 (same lr·clip=5e-5) reached 9.860% — lr and clip are NOT redundant controls.

**New recommended base config (PR #222 winning arm):**

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --agent fern \
  --optimizer lion \
  --lr 1e-4 \
  --weight-decay 5e-4 \
  --no-compile-model \
  --batch-size 4 \
  --validation-every 1 \
  --train-surface-points 65536 \
  --eval-surface-points 65536 \
  --train-volume-points 65536 \
  --eval-volume-points 65536 \
  --model-layers 4 \
  --model-hidden-dim 512 \
  --model-heads 8 \
  --model-slices 128 \
  --ema-decay 0.999 \
  --lr-warmup-epochs 1
```

**PR #222 epoch-by-epoch metrics (W&B run `ut1qmc3i`):**

| Epoch | Step  | val_abupt  | surf_pres | vol_pres  | wall_shear |
|-------|-------|-----------|-----------|-----------|------------|
| 1     | 2720  | 67.7263%  | 52.1452%  | 59.6851%  | 68.9568%   |
| 2     | 5441  | 41.9288%  | 31.5194%  | 25.0452%  | 46.0780%   |
| 3     | 8162  | 19.3033%  | 13.3635%  | 11.4773%  | 21.4948%   |
| 4     | 10883 | 13.7327%  | 9.2451%   | 8.0774%   | 15.3581%   |
| 5     | 13604 | 11.6016%  | 7.6196%   | 6.8622%   | 13.0241%   |
| 6     | 16325 | 10.5020%  | 6.7791%   | 6.2569%   | 11.7950%   |
| 7     | 19046 | 9.8759%   | 6.3077%   | 6.0145%   | 11.0603%   |
| 8     | 21767 | 9.4516%   | 6.0019%   | 5.7614%   | 10.5847%   |
| **9** | **23544** | **9.2910%** | **5.8707%** | **5.8789%** | **10.3423%** |

---

## Previous: thorfinn PR #66 — baseline 2026-04-30

PR #66 (thorfinn, per-axis tau_y/z loss upweighting W_y=2, W_z=2 on 6L/256d base) reduced
`test_primary/abupt_axis_mean_rel_l2_pct` from 13.15 (senku PR #14) to
**12.74** — a 3.1% improvement on the headline metric. Tau_y dropped from 16.23→15.15
(−6.7%) and tau_z from 16.75→15.05 (−10.2%). W&B run: `gvigs86q`.

Key finding: upweighting the two hardest wall-shear axes (tau_y and tau_z) by 2×
improves the composite metric without hurting surface_pressure or volume_pressure.
W_y=2, W_z=2 beats W_y=1.5, W_z=1.5 and the equal-weight arms. Thorfinn's code
adds `--wallshear-y-weight` and `--wallshear-z-weight` flags to `train.py`.

---

## Previous: senku PR #14 — baseline 2026-04-29

PR #14 (senku, 6L/256d depth scale-up) reduced
`test_primary/abupt_axis_mean_rel_l2_pct` from 16.64 (chihiro PR #4) to
**13.15** — a 21.0% improvement on the headline metric. Both 5L (13.52, −18.7%)
and 6L (13.15, −21.0%) beat all pending PRs. W&B runs: `t5tv01ch` (5L) and
`et4ajeqj` (6L). Key finding: depth is more parameter-efficient than width —
6L/256d (4.73M params) crushes 4L/512d (12.7M params).

---

## Previous: chihiro PR #4 — baseline 2026-04-29

PR #4 (chihiro, 4L/512d/8h large-model scale-up) reduced
`test_primary/abupt_axis_mean_rel_l2_pct` from 17.39 (gilbert PR #9) to
**16.64** — a 4.3% improvement on the headline metric. Run `pejudvyd`,
state=finished, 3 best epochs, params ~12.7M. Width upgrade used `lr=5e-5`
(3 prior runs at 2e-4 diverged) and `bs=4` (largest power-of-2 fitting 96GB).
Standout gain: `volume_pressure` 14.37 vs 15.21 — orthogonal to FiLM and
cosine-EMA wins still pending merge (PRs #8, #13).

---

## Previous: gilbert PR #9 — baseline 2026-04-29 03:57 UTC

PR #9 (gilbert, vol_w=2.0 + protocol fixes) reduced
`test_primary/abupt_axis_mean_rel_l2_pct` from 35.12 (kohaku PR #11) to
**17.39** — a 50.5% improvement on the headline metric. Wall-shear axes saw
~50–70% reductions. Surface pressure regressed slightly (+1 pp). Run
`y2gigs61`, state=finished, 6 epochs reached, best_epoch=3.

PR #9 was a CLI-flag-only change (no code diff). The win compounds the
existing PR #11 projection-loss code on `yi` with: `--volume-loss-weight 2.0`,
`--batch-size 8`, `--validation-every=1`, `--gradient-log-every 100
--weight-log-every 100`. **Future PRs should adopt this base config.**

**Important caveat** — gilbert's run did **not** include
`--use-tangential-wallshear-loss`, yet still beat kohaku's projection-loss
run by 50%. This means the bulk of the win came from the protocol fixes
(bs=8 + validation-every=1 + log cadence), not the loss form. A follow-up
combining all three (projection + vol_w=2.0 + protocol) should be even
better.

## Reference baseline targets (must beat — AB-UPT public reference)

| Target | This-repo metric | AB-UPT |
|---|---|---:|
| Surface pressure `p_s` | `test_primary/surface_pressure_rel_l2_pct` | **3.82** |
| Vector wall shear `tau` | `test_primary/wall_shear_rel_l2_pct` | **7.29** |
| Volume pressure `p_v` | `test_primary/volume_pressure_rel_l2_pct` | **6.08** |
| Wall shear `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | **5.35** |
| Wall shear `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | **3.65** |
| Wall shear `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | **3.63** |

Lower is better. Final claims must come from `test_primary/*` after best-validation
checkpoint reload.

## Current best on `tay`

**Updated 2026-05-03: PR #511 (edward, extended cosine T_max=13) — new tay SOTA**

| Metric | Best (val) | Best (test) | PR | W&B run | Date |
|---|---:|---:|---|---|---|
| `abupt_axis_mean_rel_l2_pct` | **7.0134** | **8.3130** | #511 | 5o7jc7wi | 2026-05-03 |
| `surface_pressure_rel_l2_pct` | 4.5104 | **4.2709** | #511 | 5o7jc7wi | 2026-05-03 |
| `wall_shear_rel_l2_pct` | 7.9649 | **7.7863** | #511 | 5o7jc7wi | 2026-05-03 |
| `volume_pressure_rel_l2_pct` | 4.2168 (val) | 11.8673 | #511 | 5o7jc7wi | 2026-05-03 |
| `wall_shear_x_rel_l2_pct` | 7.0052 | **6.9184** | #511 | 5o7jc7wi | 2026-05-03 |
| `wall_shear_y_rel_l2_pct` | 8.7717 | **8.5819** | #511 | 5o7jc7wi | 2026-05-03 |
| `wall_shear_z_rel_l2_pct` | 10.5629 | **9.9267** | #511 | 5o7jc7wi | 2026-05-03 |

**Key finding (PR #511):** Extended cosine T_max=13 (vs T_max=11 in PR #488 baseline) bought 2 additional training epochs at near-zero LR (EP12 6.67e-6, EP13 2.44e-6). Monotone descent every epoch EP1→EP13. The anisotropic τ_y/τ_z axes improved the fastest at near-floor LR (τ_y −0.191pp, τ_z −0.132pp EP11→EP13), confirming these axes were still under-trained. Wins on 6 of 7 test metrics vs PR #488. Only `volume_pressure` regressed on test (+0.364pp) despite val improvement — likely test-split outlier cases.

**Previous tay SOTA (PR #488 alphonse EP11):** val_abupt=7.3672%, test_abupt=8.4791%
**Fleet leader before PR #511 (PR #489 thorfinn):** val_abupt=7.1792%

## Infrastructure merge — 2026-05-04: PR #580 (haku, principal surface curvatures)

PR #580 (haku, `--surface-curvature-features {none,h_k,k1_k2}`) merged as an infrastructure PR. The feature is now available on `yi`. Best arm: Arm C (κ₁, κ₂), val_abupt=9.7225%, test_abupt=10.795% (W&B runs: Arm A `k5kp1rdp`, Arm B `hdpl64cf`, Arm C `c5685y3e`).

Arm C edges out Arm B (H, K) by a hair on every primary metric. Both arms confirm the mechanism: τ_y/τ_z improvement is monotone with curvedness (per-decile analysis, τ_y −8.7%, τ_z −6.5% at the sharpest decile 10). The feature adds zero measurable per-step training overhead.

**Metric bar unchanged: val_abupt 9.032% (PR #517).** The 3-epoch compute budget limited curvature runs to val_abupt=9.7225% — above the merge bar. The plumbing is clean and ready for full-budget composition runs.

**Reproduce (Arm C):**
```bash
cd target/
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --no-compile-model --validation-every 1 \
  --optimizer lion --lr 1e-4 --weight-decay 5e-4 --grad-clip 0.5 \
  --surface-curvature-features k1_k2
```

15. PR #580 haku — principal surface curvatures (κ₁, κ₂) as 9-channel input features (infrastructure, `--surface-curvature-features k1_k2`; metric bar unchanged)

## 2026-05-04 — PR #583: Round 33 — β-NLL Heteroscedastic Loss (edward)

β-NLL (Seitzer 2022) with `--beta-nll-beta 0.5` merged on yi. Training from EP4 checkpoint
of run `nr9oenyl`, resumed as run `5xovw2si` (EP5 = global EP5 of full run).

**Result:** val_abupt improved from 9.032% → **8.861%** — a 1.9% relative gain.

Note: the log_var heads collapsed to ~-6.8 across all channels by EP5, meaning active
heteroscedastic reweighting was minimal — the gain came primarily from extra training time.
The β-NLL flag is in `train.py` and available for composition runs.

**Reproduce:**
```bash
cd /workspace/senpai/target
torchrun --standalone --nproc_per_node=4 train.py \
  --resume-from outputs/drivaerml/run-nr9oenyl/checkpoint.pt \
  --agent edward --wandb-group yi-round33-beta-nll \
  --wandb-name "edward/beta-nll-resume-nr9oenyl-ep4" \
  --learnable-pe --optimizer lion --lr 1e-4 --weight-decay 5e-4 --clip-grad-norm 0.5 \
  --lr-warmup-epochs 0 --ema-decay 0.999 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --batch-size 4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --beta-nll-beta 0.5 --epochs 20 \
  --validation-every 1 --no-compile-model \
  --gradient-log-every 100 --weight-log-every 100
```

## 2026-05-04 — PR #590: Round 34 — Gradient EMA Smoothing α=0.5 (thorfinn)

Post-backward gradient EMA (`--grad-ema-alpha 0.5`) applied before the Lion optimizer step.
Arm A (α=0.9) over-damped: grad-norm collapsed to ~30% of raw, clip-grad-norm=0.5 never engaged →
EP3=13.60%, +4.20pp regression. Arm B' (α=0.5) restored effective clipping and improved training
stability → val_abupt=**8.686%** — a 2.0% relative gain vs prior bar (8.861%).

τ_y improvement was the strongest single-axis result this round (−0.957pp). Wall-shear across all
axes improved. Volume pressure shows an anomalous test/val gap (5.648% val vs 12.354% test) — flagged
for follow-up. W&B run: `86lxu1w0`.

**Reproduce:**
```bash
cd target/
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --agent thorfinn --wandb-group yi-round34-grad-ema \
  --learnable-pe --optimizer lion --lr 1e-4 --weight-decay 5e-4 --clip-grad-norm 0.5 \
  --grad-ema-alpha 0.5 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --batch-size 4 --validation-every 1 --no-compile-model \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536
```

## 2026-05-04 — PR #576: STRING-sep PE + Lion lr=1e-4 clip=0.5 (frieren) — NEW yi SOTA

PR #576 (frieren, STRING-sep learnable PE composed with Lion lr=1e-4 clip=0.5) merged 2026-05-04.
Cold-start anchor run `ym8x8301` + resume continuation `t4qaysur`. Best val at step 12,315 (beyond EP2).

**Result:** val_abupt improved from 8.686% → **8.2528%** — a 4.9% relative gain.

Key finding: STRING-sep learnable PE (PR #490) composes orthogonally with Lion + grad-ema.
τ_y wall_shear_y improved dramatically vs the prior yi SOTA (PR #590, run `86lxu1w0`).
All 7 primary channels beat the PR #590 baseline. Test metric: 9.5339%.

W&B runs: `ym8x8301` (cold-start), `t4qaysur` (resume continuation).

**Reproduce:**
```bash
cd target/
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --agent frieren --wandb-group yi-round35-string-sep-lion \
  --learnable-pe --optimizer lion --lr 1e-4 --weight-decay 5e-4 --clip-grad-norm 0.5 \
  --grad-ema-alpha 0.5 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --batch-size 4 --validation-every 1 --no-compile-model \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536
```

## Current best on `yi`

| Metric | Best (val) | Best (test) | PR | W&B run | Date |
|---|---:|---:|---|---|---|
| `abupt_axis_mean_rel_l2_pct` | **8.2528** | **9.5339** | #576 | t4qaysur | 2026-05-04 |

Note: PR #576 (frieren, STRING-sep learnable PE + Lion lr=1e-4 clip=0.5) merged 2026-05-04 —
new yi SOTA at val_abupt=8.2528%. Prior bar was 8.686% (PR #590 thorfinn, gradient EMA α=0.5).
Prior compounding wins: PR #590 (thorfinn, grad-ema α=0.5), PR #583 (edward, β-NLL), PR #580 (haku, surface curvatures),
PR #517 (askeladd, Lion lr=1e-4 clip=0.5), PR #490 (frieren, STRING-sep PE infrastructure).
**Merge bar: val_abupt 8.2528% on the yi codebase as it exists today (PR #576 frieren, STRING-sep PE + Lion).**
**Aspirational target: val_abupt ~7.0–7.5% (tay SOTA PR #511, `5o7jc7wi`).**

**Distance from AB-UPT targets (test, multiple of target):**

| Metric | tay SOTA (PR #511) | yi best (PR #311) | AB-UPT | tay Ratio |
|---|---:|---:|---:|---:|
| surface_pressure | 4.2709 | 4.485 | 3.82 | 1.12× |
| wall_shear | 7.7863 | 8.227 | 7.29 | 1.07× |
| volume_pressure | 11.8673 | 12.438 | 6.08 | 1.95× |
| wall_shear_x | 6.9184 | 7.253 | 5.35 | 1.29× |
| wall_shear_y | 8.5819 | 9.233 | 3.65 | 2.35× |
| wall_shear_z | 9.9267 | 10.449 | 3.63 | 2.73× |
| abupt_axis_mean | 8.3130 | 8.771 | — | — |

PR #511 closed the gap vs PR #488 on surface_pressure (1.12× vs ~1.16×) and wall_shear (1.07× vs ~1.12×).
The dominant remaining gaps are **wall_shear_y/z (2.4×, 2.7×)** and
**volume_pressure (2.0×)** — these are the key research targets for upcoming
rounds.

## Reference config (`train.py` defaults on `yi`)

```
lr=3e-4  weight_decay=1e-4  batch_size=2  epochs=50
train_/eval_ surface_points=40_000  train_/eval_ volume_points=40_000
model: 3 layers · 192 hidden · 3 heads · 96 slices · mlp_ratio=4
amp=bf16  ema_decay=0.999  validation_every=10
```
