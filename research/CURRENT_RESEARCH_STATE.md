# SENPAI Research State

**Updated**: 2026-05-29 05:45Z | Branch: `tay` | **SOTA: H185+TTA PR #1382** | Round 4 PIVOTED: 8 eval-only TTA variants

---

## Round 4 Pivot Notice

Original Round 4 (PRs #1389–#1396) was assigned with two systemic errors:
1. **Recipe spec mismatched actual yw2a5dyl config** (tau_y=1.3 actual vs 3.0 cited; mirror_augmentation boolean vs `--mirror-augment-p`; grad-clip-norm 0.5 vs grad-clip 1.0; many flags non-existent on tay)
2. **Budget infeasible**: yw2a5dyl ran 874.4 min (~14.6h) — full H185 retrain incompatible with 6h SENPAI_TIMEOUT_MINUTES cap

All 7 affected PRs CLOSED with apology + reassignment. New Round 4 = 8 parallel eval-only TTA variants, all building on `eval_tta_h209.py` infrastructure, no training, no recipe dependence.

BASELINE.md corrected with actual yw2a5dyl recipe (commit pending).

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| Prior SOTA H112 (PR #1283) | 6.1358% | 5.839% | 6.752% | 3.421% | 3.695% |
| **SOTA H185+TTA (PR #1382)** | **5.9755%** | **5.8221%** | **6.7214%** | 3.4400% | 3.6806% |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Gap to test_WSS target**: 0.87pp (13% relative)

**Merge gate**: val_abupt < **5.9755%** AND test_abupt < **5.8221%**
**Paper floor**: test_VP ≤ 3.421, test_SP ≤ 3.577, test_WSS ≤ 6.727

Source: W&B `bx3t1vdw` | H185 checkpoint: `yw2a5dyl` EP13 EMA

---

## H185 Recipe (ACTUAL — verified from W&B config)

- optimizer=lion, β1=0.9, β2=0.99
- lr=9e-5, weight_decay=5e-4, batch_size=4 per GPU (DDP×8)
- 13 epochs, lr_cosine_t_max=13, lr_warmup_epochs=1, lr_min=1e-6
- **tau_y_loss_weight=1.3** (NOT 3.0), **tau_z_loss_weight=1.67** (NOT 2.0)
- surface_loss_weight=2.0, volume_loss_weight=0.5
- **mirror_augmentation=True** (boolean — not `--mirror-augment-p`)
- ema_decay=0.999, **grad_clip_norm=0.5** (NOT 1.0)
- vol_points_schedule=`0:16384:3:32768:6:49152:9:65536`
- use_qk_norm=True, rff_num_features=16, pos_encoding_mode=string_separable
- model: 5 layers, hidden 512, heads 4, slices 128
- Runtime: **874.4 min total (~14.6h on 8 GPUs)** — incompatible with current 6h cap

**Mirror augmentation NOT on tay** — lives on `askeladd/h148-mirror-augmentation` and `fern/h183-mirror-aug-tau-y-3p0-compound` only.

---

## Round 4 Active Fleet (PIVOTED — all 8 eval-only TTA variants)

| PR | Student | Hypothesis | Mechanism | ETA |
|---|---|---|---|---|
| #1390 | askeladd | H223: TTA rot_x ±2° | rotation around x-axis | ~45min |
| #1397 | alphonse | H224: TTA coordinate scale ±2% | y,z scaling | ~45min |
| #1398 | edward | H225: TTA rot_x angle sweep {1°,2°,4°,8°} | optimal rotation angle | ~60min |
| #1399 | fern | H226: TTA-mirror on H112 (PR #1283) | Finding N N=4 extension | ~45min |
| #1400 | frieren | H227: TTA rot_z ±2° | rotation around z-axis | ~45min |
| #1401 | nezuko | H228: 4-pass TTA stack (mirror + rot_x±2°) | additive stacking test | ~45min |
| #1402 | tanjiro | H229: TTA Gaussian noise σ=0.001 | smooth equivariance | ~45min |
| #1403 | thorfinn | H230: TTA on H183/H190/H148 | Finding Q N=5 extension | ~45min |

**Expected first result**: ~30-45min (depends on which student dispatches first)

---

## Strategic Logic

The Round 4 pivot exploits a key insight from Finding Q: TTA on mirror-aug-trained models gives +4-5bp uniform gain by averaging over decorrelated noise on equivariance axes the model has absorbed.

**New axes to explore** (each could give additional gain on top of H209):
- **askeladd/edward** (H223/H225): rotation around x-axis (vertical pitch) — model NOT trained for this axis
- **frieren** (H227): rotation around z-axis (yaw) — orthogonal axis
- **alphonse** (H224): coordinate scale — geometric scale equivariance
- **tanjiro** (H229): Gaussian noise — smooth equivariance (no specific axis)
- **nezuko** (H228): 4-pass stack — tests if multiple TTA axes compound

**Transfer tests** (apply existing TTA to other checkpoints):
- **fern** (H226): TTA-mirror on non-mirror-trained H112 — extends Finding N control floor
- **thorfinn** (H230): TTA-mirror on other mirror-trained checkpoints (H148/H183/H190) — extends Finding Q to N=5

Any winner from #1390/#1397/#1398/#1400/#1401/#1402 → IMMEDIATE merge candidate
Any non-trivial result from #1399/#1403 → strengthens program finding

---

## Findings Banked (program-permanent)

| Finding | Cycle | Summary |
|---|---|---|
| E-K | prior | slope-pres compound failures, mirror-aug load-bearing, tau_y stacking threshold |
| L | prior | Mirror-aug bimodal: p=0.25 collapses slope, p=0.5 canonical |
| M | prior | Mid-EP EMA checkpoints never saved program-wide |
| N | this cycle | TTA on non-mirror-trained = +1.27-1.38pp control floor (N=3) |
| O | this cycle | Cross-recipe SWA destroys models (permutation symmetry) |
| Q | this cycle | TTA on mirror-aug = +4-5bp gain, WSS_x slope NOT recovered |
| R | this cycle | Linear mode connectivity absent for tay-track checkpoints (N=2 pairs) |
| S | this cycle | delta_mirror_WSS diagnostic: p=0.5 → ~0, p=0.25 → +0.002pp |
| T | this cycle | Permutation barrier at sub-block granularity |
| U | this cycle | H112 basin radius < 0.005 in H183 direction |

---

## Diagnostic Invariants

- No capacity additions (≥+1% param overhead → slope flattening)
- **No ensembles** (Morgan directive). TTA is NOT ensembling.
- DDP 8 GPUs every run
- z-axis tau_z LOCKED at the trained value
- WSS_x slope sign = BASIN-DISRUPTION DIAGNOSTIC
- data/loader.py, data/preload.py, data/split_manifest.json — READ-ONLY
- SENPAI_TIMEOUT_MINUTES=360 hard cap — incompatible with full H185 retrain (14.6h)

---

## Human Researcher Directives

- **Morgan (Issue #1056, 2026-05-28 15:27Z)**: Ensembles BANNED. "PUSH HARD". test_WSS < 5.85% target.
- Last advisor update to Morgan: 2026-05-29 05:45Z (Round 4 pivot — apology + reassignment)
