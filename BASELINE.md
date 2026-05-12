# DrivAerML Baseline — `tay`

**Branch:** `tay` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

---

## ENSEMBLE SOTA: nezuko PR #1030 greedy ensemble pool-33 refresh K=3 (Caruana 2004) — 2026-05-12

**val_abupt=5.9170%** / **test_abupt=7.3192%** — −0.1119pp val (−1.86% relative) vs prior K=6 (#880, 6.0289%); −0.0501pp test (−0.68% relative)

Pool refresh adds PR #958 Arm A run `29nohj67` (single-model SOTA, val=6.2868%) to the prior K=6 pool. Pre-#958 pool members re-cached at git worktree commit `5b28c2d` (PR #958 replaced surface_out/volume_out with deeper MLPs causing load_state_dict failures). Greedy forward selection stopped at K=3 (next delta at K=4 was −0.0012pp < 0.005pp threshold). Note: test_vol_p regressed slightly (+0.30pp to 11.6492%) but net test_abupt improved.

**W&B run:** `wpji54h7` (group `nezuko-ensemble-pool33-refresh`)
**PR:** #1030
**K=3 members:** 29nohj67, ghh0s4ne, 4k25s25e

**Val per-channel (K=3):** surface_pressure=3.8379%, volume_pressure=3.5136%, wall_shear=6.7232%
**Test per-channel (K=3):** surface_pressure=3.5439%, volume_pressure=11.6492%, wall_shear=6.5461%

**Greedy selection path:**
- Step 1: val=6.2869% (seed: 29nohj67, PR #958 vol_p aux head)
- Step 2: val=6.0021% (delta=0.2848pp; added: ghh0s4ne, PR #823 surf→vol xattn)
- Step 3: val=5.9170% (delta=0.0852pp; added: 4k25s25e, PR #880 pool)
- (Step 4: next delta=−0.0012pp → stopped, below 0.005pp threshold)

**Ensemble gate:** val_abupt < **5.9170%**

---

## Prior Ensemble SOTA: nezuko PR #880 greedy ensemble pool-32 refresh K=6 (Caruana 2004) — 2026-05-01

**val_abupt=6.0289%** / **test_abupt=7.3693%** — −0.1462pp val (−2.37% relative) vs prior K=7 (#612, 6.1751%); −0.1654pp test (−2.19% relative)

Pool expanded from 24→32 by adding PR #823 surf→vol cross-attn run `ghh0s4ne` (val=6.4407%, new single-model SOTA) and other recent single-model runs. Greedy forward selection re-run from scratch; `ghh0s4ne` selected as seed (best single-model). Greedy stopped at K=6 (next delta <0.005pp threshold). Significant improvement on both val and test vs prior K=7 pool-24 ensemble.

**W&B run:** `zst3y2mp` (group `nezuko-ensemble-greedy-pool32`)
**PR:** #880
**K=6 members:** ghh0s4ne, 4k25s25e, d777epep, 5o7jc7wi, 3s9di6sg, bubrguoh

**Val per-channel (K=6):** surface_pressure=3.8550%, volume_pressure=3.5678%, wall_shear=6.8719%
**Test per-channel (K=6):** surface_pressure=3.6007%, volume_pressure=11.3478%, wall_shear=6.6939%

**Greedy selection path:**
- Step 0: val=6.4408% (seed: ghh0s4ne, PR #823 surf→vol xattn)
- Step 1: val=6.1177% (delta=0.3231pp)
- Step 2: val=6.0681% (delta=0.0496pp)
- Step 3: val=6.0516% (delta=0.0165pp)
- Step 4: val=6.0373% (delta=0.0143pp)
- Step 5: val=6.0289% (delta=0.0085pp — stopped at K=6, next delta <0.005pp threshold)

**Key finding:** Pool-32 with PR #823 surf-xattn run as seed delivers +2.37% val / +2.19% test improvement over pool-24 K=7. Volume_pressure test-vs-val gap persists (val≈3.6%, test≈11.3%, ~3.2×) — primary systematic issue.

**Ensemble gate:** val_abupt < **6.0289%** (superseded by PR #1030)

---

## Prior Ensemble SOTA: nezuko PR #602 greedy forward ensemble selection K=7 (pool 22→23, Caruana 2004) — 2026-05-04

**val_abupt=6.2062%** / **test_abupt=7.5164%** — −0.0283pp val (−0.45% relative), −0.0269pp test vs prior K=7 (#562)

Pool expanded from 22→23 runs by adding PR #571 run `nh96x7m4` (askeladd, val=6.7644%). Greedy forward selection re-run with max-k=15. New K=7 selection retains strong diversity across students.

**W&B run:** `ydw7rxl2` (group `nezuko-ensemble-greedy-v2`)
**PR:** #602
**K=7 members:** nh96x7m4, 5o7jc7wi, wyz68o8r, 9mm3sz7x, 49aimdiz, 19qf6di1, nh2ke150

---

## Prior Ensemble SOTA: nezuko PR #562 greedy forward ensemble selection K=7 (Caruana 2004) — 2026-05-01

**val_abupt=6.2345%** / **test_abupt=7.5433%** — **−0.54% relative on val, −0.50% on test vs prior K=5 top-val ensemble (#556)**

Greedy forward ensemble selection (Caruana et al. 2004) from a candidate pool of 22 runs (all val≤7.5% single-model runs). Algorithm iteratively adds the candidate that maximally reduces ensemble val_abupt at each step, naturally accounting for error correlation between members. Selected K=7 members vs K=5 for naive top-val — better diversity yields better test generalization.

**W&B run:** `18oalu1h` (group `nezuko-ensemble-greedy-v1`, name `greedy-k12-pool22`)
**PR:** #562

### Greedy K=7 ensemble — per-channel test metrics

| Metric | Greedy K=7 (#562) | Top-val K=5 (#556) | Single-model SOTA (#516) | AB-UPT |
|---|---:|---:|---:|---:|
| `abupt` | **7.5433** | 7.5813 | 8.1229 | — |
| `surface_pressure` | **3.6964** | ~3.70 | 4.515 | 3.82 (BEATEN) |
| `wall_shear` | **6.8835** | ~7.1xxx | 7.757 | 7.29 (BEATEN) |
| `volume_pressure` | **11.4088** | — | — | 6.08 |

### Reproduce greedy ensemble

```bash
cd target/
# Phase 1: cache predictions from candidate pool (22 runs, val≤7.5%)
uv run python ensemble_eval.py \
  --greedy --cache-only \
  --candidate-run-ids <pool-run-ids> \
  --pred-cache-dir /tmp/ensemble_cache \
  --split val test

# Phase 2: greedy selection + eval
uv run python ensemble_eval.py \
  --greedy \
  --pred-cache-dir /tmp/ensemble_cache \
  --max-k 12 \
  --wandb-group nezuko-ensemble-greedy-v1 \
  --wandb-name greedy-k12-pool22
```

**Policy:** Ensemble SOTA gates ensemble-tier evaluation. Single-model training PRs continue to gate against val_abupt < 6.8701% (single-model SOTA #516). When a new single-model winner emerges, run `ensemble_eval.py --greedy` to check if it improves the greedy pool.

---

## Prior Ensemble SOTA: nezuko PR #556 K=5 inference-time ensemble — 2026-05-01

**val_abupt=6.2681%** / **test_abupt=7.5813%** — First result to beat AB-UPT reference on surface_pressure AND wall_shear simultaneously on test.

**W&B runs (K=5 members):** `9mm3sz7x` (askeladd), `49aimdiz` (alphonse), `wyz68o8r` (thorfinn/SOTA), `qqtdnlwq` (alphonse), `5o7jc7wi` (edward) — ensemble group `ensemble-inference-v1`
**PR:** #556

**Policy:** Ensemble SOTA gates ensemble-tier evaluation. Single-model training PRs continue to gate against val_abupt < 6.2868% (single-model SOTA #958). When a new single-model winner emerges, run `ensemble_eval.py` to check if it improves the K=3 pool.

---

## SINGLE-MODEL SOTA: nezuko PR #958 dedicated vol_p aux decoder head (Arm A) — 2026-05-09

**val_abupt=6.2868%** / **test_abupt=7.7445%** — −0.154pp val (−2.39% relative) vs prior single-model SOTA (PR #823, 6.4407%); +0.045pp test regression. Hypothesis on vol_p OOD gap NEGATIVE.

3-layer MLP auxiliary decoder head for volume_pressure (independent gradient path from volume tokens), stacked on full PR #823 SOTA config (surf→vol cross-attention). vol_loss_weight=1.0 (Arm A). Val_abupt improvement is broad-based (wall_shear −0.297pp val). Hypothesis that dedicated head would reduce OOD test_vol_p gap NEGATIVE — test_vol_p=12.0063% is worse than baseline 11.6704% (+0.336pp). The OOD gap (~3× val/test multiplier) is a data-distribution problem, not a model-capacity problem.

**W&B run:** `29nohj67` (group `nezuko-vol-aux-decoder-head`)
**PR:** #958

**Val metrics (EP13 best checkpoint):** val_abupt=6.2868%, surface_pressure=4.1766%, volume_pressure=3.9152%, wall_shear=7.0476%, tau_x=6.1726%, tau_y=7.6648%, tau_z=9.5053%
**Test metrics:** test_abupt=7.7445%, surface_pressure=3.9100%, volume_pressure=12.0063%, wall_shear=6.9848%, tau_x=6.2092%, tau_y=7.5689%, tau_z=9.0280%

**Single-model training gate:** val_abupt < **6.2868%** (previously 6.4407% from PR #823)

### Reproduce (L=5 + surf→vol xattn + vol_p aux head, Lion lr=9e-5, 13ep, vol-curriculum)

```bash
cd target/ && torchrun --standalone --nproc_per_node=8 train.py \
  --agent nezuko --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --lr-cosine-t-max 13 --epochs 13 \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
  --no-compile-model \
  --model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --use-surf-to-vol-xattn \
  --use-vol-pressure-aux-head --volume-loss-weight 1.0 \
  --wandb-group nezuko-vol-aux-decoder-head
```

---

## Prior Single-Model SOTA: nezuko PR #823 surface→volume cross-attention — 2026-05-08

**val_abupt=6.4407%** / **test_abupt=7.6992%** — −0.158pp val (−2.4% relative) vs prior single-model SOTA (PR #592, 6.5985%); −0.292pp test (−3.6% relative).

One `nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)` after the post-backbone LayerNorm split. Q=vol_hidden, K=V=surf_hidden. Zero-init out_proj (identity-at-init); post-norm residual. Improvement broad-based; OOD test/val ratio preserved (3.027× vs 3.025×) — xattn is a general capacity boost.

**W&B run:** `ghh0s4ne` (group `nezuko-surf-vol-xattn`, name `nezuko/surf-vol-xattn-13ep`)
**PR:** #823

**Val metrics (EP13 best checkpoint):** val_abupt=6.4407%, surface_pressure=4.1836%, volume_pressure=3.8557%, wall_shear=7.3448%, tau_x=5.7782%, tau_y=7.5977%, tau_z=9.0116%
**Test metrics:** test_abupt=7.6992%, surface_pressure=3.8451%, volume_pressure=11.6704%, wall_shear=7.0429%, tau_x=6.2773%, tau_y=7.6657%, tau_z=9.0375%

---

## Prior Single-Model SOTA: alphonse PR #592 depth-L5 (model-layers=5) — 2026-05-04

**W&B run:** `4k25s25e` (alphonse DDP8, rank-0) — group `model-depth-sweep`, name `alphonse/depth-L5`, best val **6.5985%** (EP4, step 43,459)
**PR:** #592

**Val metrics (best-val checkpoint, EP4):** val_abupt=6.5985%, surface_pressure=4.3322%, volume_pressure=3.9456%, wall_shear_x=6.5420%, wall_shear_y=8.3631%, wall_shear_z=9.8099%
**Test metrics:** test_abupt=7.9915%
**Model params:** ~15.9M | VRAM ~52GB/96GB | Training time ~270.8 min

**Key insight:** Increasing model depth from L=4 to L=5 (5 transformer layers, hidden_dim=512, heads=4, slices=128) beats the prior single-model SOTA (PR #594, val=6.7258%) by −0.1273pp (−1.90% relative). Surface pressure improves to 4.3322% (PR #571 was 4.455%). Near-wall tau physics also improve. L=5 adds ~4M params (~15.9M vs ~12M for L=4) at the cost of ~52GB VRAM vs ~42GB.

**Single-model training gate:** val_abupt < **6.5985%** (previously 6.7258% from PR #594)

### Reproduce (L=5, Lion lr=9e-5, tau_y×1.5/tau_z×2.0, surface_w=2.0, no-compile)

```bash
cd target/ && torchrun --standalone --nproc_per_node=8 train.py \
  --agent alphonse --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --lr-cosine-t-max 13 --epochs 13 \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
  --no-compile-model \
  --model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --wandb-group model-depth-sweep --wandb-name alphonse/depth-L5
```

**Artifact:** `model-alphonse-depth-L5-4k25s25e`

---

## Prior Single-Model SOTA: askeladd PR #594 rff32 on PR #571 SOTA stack — 2026-05-04

**W&B run:** `d777epep` (askeladd DDP8, rank-0) — group `askeladd-rff32-pr571-sota`, best val **6.7258%** (EP4, step 43,462)
**PR:** #594

---

## Prior Single-Model SOTA: askeladd PR #571 tau_y×1.5 / tau_z×2.0 weight intensification — 2026-05-04

**W&B run:** `nh96x7m4` (askeladd DDP8, rank-0) — group `askeladd-tau-sweep`, best val **6.7644%** (Arm A: tau_y×1.5 / tau_z×2.0), runtime ~4.7h
**PR:** #571
**Val metrics (best-val checkpoint):** val_abupt=6.7644%, surface_pressure=4.455%, wall_shear=7.593%, volume_pressure=4.249%, tau_x=~6.7%, tau_y=8.489%, tau_z=9.997%
**Test metrics:** test_abupt=8.171%

**Key insight:** Moderate tau weight intensification (tau_y×1.5, tau_z×2.0) on the PR #516 SOTA stack further improves val_abupt by −0.106pp (−1.54% relative). tau_y improves from 8.663%→8.489% (−0.174pp), tau_z from 10.266%→9.997% (−0.269pp). The increased weights do not destabilize surface_pressure or volume_pressure — surface_pressure actually improves (4.515%→4.455%). Test gate: 8.171% < 8.25% threshold.

**Single-model training gate:** val_abupt < **6.7644%** (previously 6.8701% from PR #516)

### Reproduce (tau_y×1.5, tau_z×2.0, Lion lr=9e-5, no GradNorm)

```bash
torchrun --standalone --nproc_per_node=8 target/train.py \
  --agent askeladd --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 \
  --surface-loss-weight 2.0 \
  --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --no-compile-model \
  --lr-cosine-t-max 13 --epochs 13 \
  --wandb-group askeladd-tau-sweep \
  --wandb-name askeladd/tau-sweep-y1.5-z2.0
```

**NOTE:** `--no-compile-model` required — torch.compile + DDP NCCL deadlock at step 1.

---

## Previous Single-Model SOTA: askeladd PR #516 per-channel tau_y/tau_z loss reweighting — 2026-05-04

**W&B run:** `9mm3sz7x` (askeladd DDP8, rank-0) — group `askeladd-tau-reweight-micro`, best val **6.8701%** (EP5), runtime ~4.7h
**PR:** #516
**Val metrics (best-val checkpoint, EP5):** val_abupt=6.8701%, surface_pressure=4.515%, wall_shear=7.757%, volume_pressure=4.144%, tau_x=6.763%, tau_y=8.663%, tau_z=10.266%
**Test metrics:** test_abupt=8.1229%

**Key insight:** Simple fixed per-channel tau reweighting (tau_y×1.2, tau_z×1.3) WITHOUT GradNorm outperforms EMA-proxy GradNorm (6.8701% < 6.9246%). Lion optimizer, lr=9e-5, surface_loss_weight=2.0, ema_decay=0.999.

### Reproduce (tau_y×1.2, tau_z×1.3, Lion lr=9e-5, no GradNorm)

```bash
torchrun --standalone --nproc_per_node=8 target/train.py \
  --agent askeladd --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
  --tau-y-loss-weight 1.2 --tau-z-loss-weight 1.3 \
  --surface-loss-weight 2.0 \
  --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --no-compile-model \
  --lr-cosine-t-max 13 --epochs 13 \
  --wandb-group askeladd-tau-reweight-micro \
  --wandb-name askeladd/tau-reweight-w1.2-w1.3-lr9e-5
```

**NOTE:** `--no-compile-model` is required. The actual SOTA run `9mm3sz7x` had `compile_model: False` in its config. Using `torch.compile` with DDP on this cluster causes NCCL all_reduce deadlocks at step 1. Always include `--no-compile-model` in reproduce commands.

### Previous SOTA: thorfinn PR #523 EMA-proxy GradNorm alpha=0.5 EP6 — 2026-05-01

**W&B run:** `wyz68o8r` (thorfinn DDP8, rank-0) — group `thorfinn-gradnorm-r2`, best val **6.9246%** (EP6)
**PR:** #523
**Val metrics:** val_abupt=6.9246%, surface_pressure=4.5840%, wall_shear=7.7457%, volume_pressure=4.3040%, tau_x=6.7193%, tau_y=8.7197%, tau_z=10.2960%
**Test metrics:** test_abupt=8.2355%, surface_pressure=4.2712%, wall_shear=7.5043%, volume_pressure=12.2131%, tau_x=6.5557%, tau_y=8.4656%, tau_z=9.6720%

### Merged result: frieren PR #555 GradNorm alpha=1.0 tau-channel-reweighting-v2 — 2026-05-04

**W&B run:** `341czkol` (frieren DDP8, rank-0) — group `frieren-gradnorm-alpha-sweep`, Arm B alpha=1.0, best val **6.8738%**
**PR:** #555
**Val metrics:** val_abupt=6.8738%, surface_pressure=4.4965%, wall_shear=7.6909%, volume_pressure=4.3764%, tau_x=6.7109%, tau_y=8.5156%, tau_z=10.2697%
**Test metrics:** test_abupt=8.2433%, surface_pressure=4.2205%, wall_shear=7.5321%, volume_pressure=12.4069%, tau_x=6.6947%, tau_y=8.3054%, tau_z=9.5893%

**Note:** val_abupt=6.8738% does NOT beat PR #516 SOTA (6.8701%); beats PR #523 (6.9246%) by −0.051pp. Single-model SOTA remains PR #516.
**Key finding:** alpha=1.0 confirms tau_y/tau_z benefit from higher GradNorm exponent (tau_y −0.204pp val, −0.160pp test vs #523); volume_pressure regresses on test (+0.194pp). Merged as incremental improvement over #523 baseline.

### Previous SOTA: alphonse PR #510 surface-loss-weight slw=2.0 EP5 — 2026-05-03

**alphonse PR #510 (surface-loss-weight=2.0, heavier surface gradient emphasis) beats PR #511 by −0.007pp val, −0.021pp test (7.0063% val, 8.2921% test). W&B run `qqtdnlwq`, EP5 EMA. Delta −0.10% relative on val.**

**W&B run:** `qqtdnlwq` (alphonse DDP8, rank-0) — group `alphonse-slw-sweep`, best val **7.0063%** (EP5 EMA), runtime 4.71h
**PR:** #510
**Val metrics (best-val checkpoint, EP5 EMA):** val_abupt=7.0063%, surface_pressure=4.5994%, wall_shear=7.8939%, volume_pressure=4.1643%, tau_x=6.8150%, tau_y=8.9516%, tau_z=10.5010%
**Test metrics:** test_abupt=8.2921%, surface_pressure=4.2381%, wall_shear=7.6341%, volume_pressure=12.1047%, tau_x=6.6657%, tau_y=8.6452%, tau_z=9.8066%

**Prior SOTA: edward PR #511 extended cosine schedule EP13 — 2026-05-01 (updated)**

**edward PR #511 (cosine T_max extended 11→13 for 2 extra convergence epochs) beats PR #489 by −0.1658pp val (7.0134% vs 7.1792% val). W&B run `5o7jc7wi`, EP13. Delta −2.31% relative.**

Key insight: PR #488 SOTA run `ki2q9ko9` was still descending at EP11 — the cosine schedule was truncating too early. Extending `lr_cosine_t_max` from 11→13 gives 2 more epochs on the descending tail, yielding consistent per-epoch improvement (EP11=7.13% → EP12=7.06% → EP13=7.01%). Test result confirms: test_abupt=8.3130% (vs prior SOTA 8.497%). Surface pressure improves substantially (4.271% vs 4.783%). tau_y/tau_z remain the open problem.

**W&B run:** `5o7jc7wi` (edward DDP8) — group `edward-extended-cosine`, best val **7.0134%** (EP13), runtime 5.67h
**PR:** #511
**Val metrics (best-val checkpoint, EP13):** val_abupt=7.0134%, surface_pressure=4.5104%, wall_shear=7.9650%, volume_pressure=4.2168%, tau_x=7.0053%, tau_y=8.7717%, tau_z=10.5629%
**Test metrics:** test_abupt=8.3130%, surface_pressure=4.2709%, wall_shear=7.7863%, volume_pressure=11.8673%, tau_x=6.9184%, tau_y=8.5819%, tau_z=9.9267%

### tay current best — `val_primary/*` (PR #510 alphonse, run `qqtdnlwq`)

| Metric | **PR #510 alphonse (SOTA)** | PR #511 edward (prev) | AB-UPT |
|---|---:|---:|---:|
| `abupt` | **7.0063** | 7.0134 | — |
| `surface_pressure` | 4.5994 | **4.5104** | 3.82 |
| `wall_shear` | **7.8939** | 7.9650 | 7.29 |
| `volume_pressure` | **4.1643** | 4.2168 | 6.08 |
| `tau_x` | **6.8150** | 7.0053 | 5.35 |
| `tau_y` | 8.9516 | **8.7717** | 3.65 |
| `tau_z` | **10.5010** | 10.5629 | 3.63 |

**Key insight:** slw=2.0 (heavier surface gradient emphasis) wins on val_abupt at EP5, with substantial tau_x/tau_z/wall_shear improvements. tau_y and surface_pressure regress slightly vs PR #511 on val, but test metrics show improvements on 5/7 channels. Run timed out at EP5 (360-min limit); a full 13-epoch run should yield further improvement. tau_y/tau_z gap (8.95%/10.50% vs target 3.65%/3.63%) remains primary open problem.

### Reproduce new SOTA (Lion lr=1e-4, EMA=0.999, STRING-sep, QK-norm, feat16 RFF, multi-sigma init, vol-curriculum, slw=2.0)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --agent alphonse --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 \
  --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --lr-cosine-t-max 13 \
  --epochs 13 \
  --surface-loss-weight 2.0 \
  --wandb-group alphonse-slw-sweep --wandb-name alphonse-slw-2.0
```

(Note: multi-sigma STRING-sep init from PR #488 model.py is required — `--rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"`)

### Compounding wins so far (updated through PR #516)

| PR | Who | Delta | Lever |
|---|---|---:|---|
| #30 | alphonse | baseline (19.81) | yi calibration config (4L/512d/8h, vol_w=2.0) |
| #33 | fern | **−2.04 (−10.3%)** | RFF coord features (sigma=1.0, 32 feats) — uncompiled |
| #40 | alphonse | **−0.52 (−2.9%) vs #33** | torch.compile fix → 12 epochs vs 9 |
| #39 | tanjiro | **−1.82 (−10.5%) vs #40** | Lion optimizer lr=1.7e-5 |
| #46 | alphonse | **−0.88 (−5.7%) vs #39** | AdamW + RFF (sigma=1.0) + compile → epoch 16 |
| (no PR) | tanjiro arm B | **−3.25 (−22.3%) vs #46** | Lion lr=5e-5/wd=5e-4 — paper config was wrong |
| #50 | nezuko | **−0.10 (−0.84%) vs arm B** | Lion uncompiled lr=5e-5 reproduce |
| #110 | edward | **−0.04 (−0.34%) vs #50** | Lion + cosine T_max=50 (gentle 8% decay) |
| #111 | tanjiro | **−0.03 (−0.25%) vs #110** | EMA decay 0.999 (5× faster than 0.9995) |
| #115 | thorfinn | **−0.562 (−5.04%) vs #111** | Compound: lr=1e-4 + EMA=0.999 (both winners stacked) |
| #222 | fern | **−0.193 (−2.03%) vs #115** | lr_warmup_epochs=1 (1-epoch linear warmup on top of SOTA stack) |
| #232 | askeladd | **−0.226 (−2.44%) vs #222** | model-heads=4 (halving attention heads from 8 to 4) |
| #309 | thorfinn | **−0.064pp (−0.63%) vs #232** | grad-clip-norm=0.5 (Lion EMA momentum stabilization, avoids ep8 regression) |
| #311 | edward | **−1.355pp (−13.39%) vs #309** | STRING-separable pos encoding: learnable per-axis log_freq + phase — largest single gain since tanjiro arm B |
| #358 | thorfinn | **−0.154pp (−2.04%) vs #311** | QK-norm (RMSNorm on Q and K) stacked on STRING-sep — best val at EP11 (7.3921%) |
| #387 | alphonse | **−0.031pp (−0.36%) vs #358** | feat16 RFF (rff_num_features=16) stacked on STRING-sep + QK-norm config — best val at EP11 (7.3816%) |
| #488 | alphonse | **−0.0144pp (−0.195%) vs #387** | multi-sigma STRING-sep init: log_freq params distributed across frequency octaves at init — vp drops from 12.189% to 4.357% (beats AB-UPT ref!) |
| #489 | thorfinn | **−0.1880pp (−2.55%) vs #488** | vol-points curriculum 16k→32k→49k→65k: progressive coarse-to-fine volume sampling across training epochs — vp further improves to 4.207% |
| #511 | edward | **−0.1658pp (−2.31%) vs #489** | extended cosine T_max 11→13: 2 extra convergence epochs on descending tail — EP13 val_abupt=7.0134%, test_abupt=8.3130% |
| **#510** | **alphonse** | **−0.0071pp (−0.10%) vs #511** | **surface-loss-weight=2.0: heavier surface gradient emphasis closes tau_x/tau_z/wall_shear at EP5 — val_abupt=7.0063%, test_abupt=8.2921%** |
| **#523** | **thorfinn** | **−0.0817pp (−1.16%) vs #510** | **EMA-proxy GradNorm alpha=0.5, floor=0.7: dynamic tau_y/tau_z upweighting (1.16×/1.24×), closed-form EMA weight updates ~0% overhead — val_abupt=6.9246%, test_abupt=8.2355%** |
| **#516** | **askeladd** | **−0.0545pp (−0.79%) vs #523** | **Per-channel tau_y×1.2 / tau_z×1.3 fixed loss reweighting (no GradNorm): simple static channel weights beat EMA-proxy GradNorm — val_abupt=6.8701%, test_abupt=8.1229%** |

---

## Prior SOTA record: thorfinn PR #489 volume-points curriculum 16k→65k ramp — 2026-05-03 (updated)

**PRIOR SOTA: thorfinn PR #489 (vol-points curriculum 16k→32k→49k→65k over 4 stages) beats PR #488 by −0.1880pp val (7.1792% vs 7.3672% val). W&B run `r5rw40rn`, EP11. Delta −2.55% relative.**

**W&B run:** `r5rw40rn` (thorfinn DDP8, rank 0) — group `thorfinn-vol-curriculum`, best val **7.1792%** (EP11)
**PR:** #489
**Val metrics (best-val checkpoint):** val_abupt=7.1792%, surface_pressure=4.783%, wall_shear=8.098%, volume_pressure=4.207%, tau_x=7.019%, tau_y=9.187%, tau_z=10.701%
**Test metrics:** test_abupt=8.497%

---

## Prior SOTA record: alphonse PR #488 multi-sigma STRING-sep init — 2026-05-03 (updated)

**PRIOR SOTA: alphonse PR #488 (multi-sigma RFF init across frequency octaves) beats PR #387 by −0.0144pp val (7.3672% vs 7.3816% val). W&B run `ki2q9ko9`, EP11.**

Multi-sigma STRING-sep init distributes `log_freq` parameters across frequency octaves at initialization via `--rff-init-sigmas`, giving the STRING-sep encoding a broader spectral coverage from the start. Dramatically improves volume_pressure (vp=4.357% vs SOTA 12.189% — a +7.832pp improvement), bringing it to near-target territory (AB-UPT ref: 6.08%). Surface pressure and wall shear see modest regression (+0.367pp and +0.348pp respectively), but the net val_abupt improvement confirms the octave-init approach is a genuine advance.

**W&B run:** `ki2q9ko9` (alphonse DDP8) — best val **7.3672%** (EP11)
**PR:** #488
**Val metrics (best-val checkpoint):** val_abupt=7.3672%, surface_pressure=4.805%, wall_shear=8.347%, volume_pressure=4.357%

---

## Prior SOTA record: alphonse PR #387 feat16 RFF + QK-norm + STRING-sep — 2026-05-01 (updated)

**PRIOR SOTA: alphonse PR #387 (feat16 RFF + QK-norm stacked on STRING-sep) beats PR #358 by −0.0105pp val (7.3816% vs 7.3921% val). W&B run `wj6mn6ve`, EP11 (Arm A: rff_num_features=16).**

RFF with rff_num_features=16 (feat16) stacks on top of the STRING-sep + QK-norm SOTA baseline. The feat16 encoding adds 16-feature Random Fourier Features on top of the learnable per-axis STRING-sep frequencies, providing richer spectral coverage at low compute cost. Both val and test improve over the prior SOTA.

**W&B run:** `wj6mn6ve` (alphonse DDP8) — group `alphonse-rff-sweep`, best val **7.3816%** (EP11)
**PR:** #387
**Test metrics (best-val checkpoint):** test_abupt=8.5936%, surface_pressure=4.4377%, wall_shear=7.9989%, volume_pressure=12.1885%, tau_x=6.9622%, tau_y=9.1058%, tau_z=10.2736%

---

## Prior SOTA record: thorfinn PR #358 STRING-sep + QK-norm stack — 2026-05-02 (updated)

**PRIOR SOTA: thorfinn PR #358 (STRING-sep + QK-norm stack) beat PR #311 by −0.154pp val (7.3921% vs 7.546% val). W&B run `tkiigfmc`, EP11.**

QK-norm adds `nn.RMSNorm(dim_head, elementwise_affine=True)` applied to Q and K projections immediately after the qkv chunk, before `F.scaled_dot_product_attention`. Stacks directly on top of PR #311 STRING-sep config. Convergence continued improving to EP11.

**W&B run:** `tkiigfmc` (thorfinn DDP8) — group `thorfinn-string-qknorm-r19`, best val **7.3921%** (EP11)
**PR:** #358
**Test metrics (best-val checkpoint):** test_abupt=8.625%, surface_pressure=4.462%, wall_shear=7.965%, volume_pressure=12.434%

---

## Prior SOTA record: edward PR #311 STRING-separable positional encoding — 2026-05-01 (updated)

**NEW SOTA: edward PR #311 (STRING-separable pos encoding) beats PR #309 by −1.493pp val / −1.355pp test (7.546% vs 9.039% val, 8.771% vs 10.126% test). This is a −13.93% relative improvement on test_abupt.**

STRING-separable replaces fixed isotropic Gaussian RFF with learnable per-axis frequency/phase (`log_freq` + `phase` as `nn.Parameter`, one per axis). The axis-separable factorization learns independent spectral emphasis per spatial axis, matching the anisotropic structure of automotive aerodynamics. All gradient diagnostics healthy (nonfinite_count: 0 throughout). Val slopes still negative at terminal epoch — model still converging, further gains possible.

**W&B run:** `gcwx9yaa` (rank 0) — group `tay-round18-grape-ablation`, STRING arm, best val 7.546%
**PR:** #311
**Test metrics:** test_abupt 8.771% (test_primary/abupt_axis_mean_rel_l2_pct)

### tay current best — `val_primary/*`

| Epoch | val_abupt |
|-------|-----------|
| **best** | **7.546%** |

### tay current best — `test_primary/*` (PR #311 edward, run `gcwx9yaa`)

| Metric | This-repo key | **PR #311 edward (NEW SOTA)** | PR #309 thorfinn (prev) | PR #232 askeladd | AB-UPT |
|---|---|---:|---:|---:|---:|
| `abupt` | `test_primary/abupt_axis_mean_rel_l2_pct` | **8.771** | 10.126 | 10.190 | — |
| `surface_pressure` | `test_primary/surface_pressure_rel_l2_pct` | **4.485** | 5.395 | 5.461 | 3.82 |
| `wall_shear` | `test_primary/wall_shear_rel_l2_pct` | **8.227** | 9.883 | 9.910 | 7.29 |
| `volume_pressure` | `test_primary/volume_pressure_rel_l2_pct` | **12.438** | 12.484 | 12.656 | 6.08 |
| `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | **7.253** | 8.402 | 8.432 | 5.35 |
| `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | **9.233** | 11.941 | 11.952 | 3.65 |
| `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | **10.449** | 12.407 | 12.447 | 3.63 |

**Wins over PR #309 on every axis.** Largest gains: tau_y −2.708pp (−22.68%), tau_z −1.958pp (−15.78%), surface_pressure −0.91pp (−16.87%), wall_shear −1.656pp (−16.76%).

**3-arm ablation comparison (tay-round18-grape-ablation):**

| Arm | Encoding | val_abupt | test_abupt | vs SOTA |
|-----|----------|-----------|------------|---------|
| A (RFF-32) | Fixed isotropic Gaussian | 9.710% | 10.721% | +0.595pp worse |
| **B (STRING-sep)** | **Learnable per-axis freq/phase** | **7.546%** | **8.771%** | **−1.355pp better** |
| C (GRAPE-M) | Minimal learned spectral proj | still running | — | — |
| SOTA (#309) | No spectral encoding (RFF-0) | 9.039% | 10.126% | baseline |

### Reproduce new SOTA (Lion lr=1e-4, EMA=0.999, heads=4, grad-clip-norm=0.5, STRING-sep)

**Note:** STRING-separable encoding uses `--pos-encoding-mode string_separable` (or equivalent flag). Stacks on the full PR #309 SOTA config.

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <STUDENT> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --grad-clip-norm 0.5 \
  --pos-encoding-mode string_separable
```

### Compounding wins so far

| PR | Who | Delta | Lever |
|---|---|---:|---|
| #30 | alphonse | baseline (19.81) | yi calibration config (4L/512d/8h, vol_w=2.0) |
| #33 | fern | **−2.04 (−10.3%)** | RFF coord features (sigma=1.0, 32 feats) — uncompiled |
| #40 | alphonse | **−0.52 (−2.9%) vs #33** | torch.compile fix → 12 epochs vs 9 |
| #39 | tanjiro | **−1.82 (−10.5%) vs #40** | Lion optimizer lr=1.7e-5 |
| #46 | alphonse | **−0.88 (−5.7%) vs #39** | AdamW + RFF (sigma=1.0) + compile → epoch 16 |
| (no PR) | tanjiro arm B | **−3.25 (−22.3%) vs #46** | Lion lr=5e-5/wd=5e-4 — paper config was wrong |
| #50 | nezuko | **−0.10 (−0.84%) vs arm B** | Lion uncompiled lr=5e-5 reproduce |
| #110 | edward | **−0.04 (−0.34%) vs #50** | Lion + cosine T_max=50 (gentle 8% decay) |
| #111 | tanjiro | **−0.03 (−0.25%) vs #110** | EMA decay 0.999 (5× faster than 0.9995) |
| #115 | thorfinn | **−0.562 (−5.04%) vs #111** | Compound: lr=1e-4 + EMA=0.999 (both winners stacked) |
| #222 | fern | **−0.193 (−2.03%) vs #115** | lr_warmup_epochs=1 (1-epoch linear warmup on top of SOTA stack) |
| #232 | askeladd | **−0.226 (−2.44%) vs #222** | model-heads=4 (halving attention heads from 8 to 4) |
| #309 | thorfinn | **−0.064pp (−0.63%) vs #232** | grad-clip-norm=0.5 (Lion EMA momentum stabilization, avoids ep8 regression) |
| **#311** | **edward** | **−1.355pp (−13.39%) vs #309** | **STRING-separable pos encoding: learnable per-axis log_freq + phase — largest single gain since tanjiro arm B** |

---

## Prior SOTA record: thorfinn PR #309 grad-clip-norm=0.5 — 2026-05-02 07:17 UTC

**PRIOR SOTA: thorfinn PR #309 grad-clip-norm=0.5 beats PR #232 by −0.026pp val / −0.064pp test (9.0389% vs 9.0650% val, 10.126% vs 10.190% test).**

**W&B run:** `ztdhodw1` (rank 0) — group `thorfinn-gradclip-r15`, ~270 min runtime, best val 9.0389% (ep11)
**PR:** #309
**Test metrics:** test_abupt 10.126% (test_primary/abupt_axis_mean_rel_l2_pct)

---

## Prior SOTA record: askeladd PR #232 model-heads=4 — 2026-05-01 21:06 UTC

**PRIOR SOTA: askeladd PR #232 heads=4 beat PR #222 by −0.226pp (9.0650% vs 9.2910% val).**

**W&B run:** `r8s2dtnq` (rank 0) — group `tay-round12-heads-4`, 282.5 min runtime, best val 9.0650% (ep11)
**PR:** #232
**Test metrics:** test_abupt 10.190%

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <STUDENT> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999
```

---

## Prior SOTA record: fern PR #222 lr_warmup_epochs=1 — 2026-05-01 19:30 UTC

**NEW SOTA: fern PR #222 lr warmup 1ep beats PR #115 by −2.03% (9.2910 vs 9.484 val).**

1-epoch linear LR warmup added to the SOTA stack (Lion lr=1e-4, EMA=0.999). Smooth convergence improvement with continuous descent across all 9 epochs — no instability. The warmup provides a gentler entry to steep descent, resulting in lower final val. ep1 inflated (warmup effect: LR still ramping) but ep2+ shows consistently better convergence than flat-LR baseline.

**W&B run:** `ut1qmc3i` (rank 0) — group `tay-round12-lr-warmup-1ep`, ~270 min runtime, 9 val epochs, best val 9.2910 (ep9)
**PR:** #222
**Test metrics:** **CONFIRMED — test_abupt 10.420% (beats prior PR #115 SOTA 10.580% by −0.16pp / −1.51%).** Updated 2026-05-01 from W&B run `ut1qmc3i` summary.

### Reproduce (Lion lr=1e-4, EMA=0.999, lr_warmup_epochs=1)

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <STUDENT> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.999 --lr-warmup-epochs 1
```

---

## Prior SOTA record: tanjiro Lion-arm-B — 2026-04-30 02:44 UTC

**MASSIVE NEW SOTA: Lion lr=5e-5/wd=5e-4 (NOT paper config) blows past PR #46 by −22.3%.**

This was a follow-up sweep run launched by tanjiro's pod after PR #39 was reviewed —
**not** an advisor-assigned PR experiment. The config is Lion at the AdamW-equivalent
LR/WD translation (lr=5e-5, wd=5e-4), no compile, no RFF. Despite being uncompiled,
the run trained for 4h50m past the 270-min budget cap and the val curve was still
descending at the end. Best-val checkpoint reload gave the test result.

**W&B run:** `vnb7oheo` (rank 0) — group `tanjiro-lion-lr-sweep`
**Best-val checkpoint:** val_abupt = 10.096 at last logged epoch
**Runtime:** 4h50m (290 min, past budget — likely launched without strict timeout)
**Advisor note:** No PR for this run. Result documented retroactively as the new SOTA.

### CRITICAL FINDING: Lion paper config is wrong for this task

PR #39 used Lion at `lr=1.7e-5, wd=5e-3` (paper config from Chen et al.) → test_abupt 15.43.
This run used `lr=5e-5, wd=5e-4` (the AdamW-equivalent translation tanjiro tested as
arm B of the original sweep) → test_abupt **11.30**. That is a **−27% improvement
just from changing the LR/WD constants on the same Lion optimizer**.

Why the paper config fails here:
- Lion paper used image classification with millions of training samples; we have 400 cars.
- Smaller datasets need more aggressive per-step movement (higher lr) to traverse the
  loss landscape within a 270-min budget.
- Higher wd in the paper helps regularize huge nets; our 4L/512d/8h is small enough
  that wd=5e-4 (AdamW-equivalent) is sufficient.

**All future Lion experiments must use `--lr 5e-5 --weight-decay 5e-4`, not the paper
config.** Queued PRs #50, #51, #52, #54 need their LR/WD updated.

### tay current best — `test_primary/*`

| Metric | This-repo key | tay best (Lion 5e-5) | PR #46 (RFF+compile) | PR #39 Lion (paper) | yi best | AB-UPT |
|---|---|---:|---:|---:|---:|---:|
| `abupt` | `test_primary/abupt_axis_mean_rel_l2_pct` | **11.303** | 14.550 | 15.43 | 15.82 | — |
| `surface_pressure` | `test_primary/surface_pressure_rel_l2_pct` | **6.216** | 8.628 | 9.45 | 9.99 | 3.82 |
| `wall_shear` | `test_primary/wall_shear_rel_l2_pct` | **11.315** | 14.882 | 16.28 | 16.60 | 7.29 |
| `volume_pressure` | `test_primary/volume_pressure_rel_l2_pct` | **12.755** | 15.032 | 13.83 | 14.21 | 6.08 |
| `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | **9.563** | 12.901 | 13.91 | 14.27 | 5.35 |
| `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | **13.831** | 17.281 | 19.58 | 19.49 | 3.65 |
| `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | **14.147** | 18.907 | 20.40 | 21.12 | 3.63 |

**Every axis improved by 15-29% vs PR #46. tau_y/tau_z gap to AB-UPT ref is now 3.8×
(was 5.4× and 5.6×).** This is the largest single jump in the project so far.

### Reproduce new SOTA (Lion lr=5e-5)

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 5e-5 --weight-decay 5e-4 \
  --volume-loss-weight 2.0 --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.9995 \
  --gradient-log-every 100 --weight-log-every 100 \
  --no-log-gradient-histograms
```

NOTE: `vnb7oheo` ran ~290 min (past budget). With strict 270-min budget the result
might land slightly higher (~11.5-12.0). Future reproduce runs should use the same
budget enforcement as standard PRs.

### Compounding wins so far

| PR | Who | Delta | Lever |
|---|---|---:|---|
| #30 | alphonse | baseline (19.81) | yi calibration config (4L/512d/8h, vol_w=2.0) |
| #33 | fern | **−2.04 (−10.3%)** | RFF coord features (sigma=1.0, 32 feats) — uncompiled |
| #40 | alphonse | **−0.52 (−2.9%) vs #33** | torch.compile fix → 12 epochs vs 9; beats #33 without RFF |
| #39 | tanjiro | **−1.82 (−10.5%) vs #40** | Lion optimizer lr=1.7e-5 — sign-based update, crosses yi frontier |
| #46 | alphonse | **−0.88 (−5.7%) vs #39** | AdamW + RFF (sigma=1.0) + compile → epoch 16; tau_y −11.7% |
| (no PR) | tanjiro arm B | **−3.25 (−22.3%) vs #46** | Lion lr=5e-5/wd=5e-4 (AdamW-equivalent translation, NOT paper config) — paper config was the culprit |

### INFRA NOTE — torch.compile bug FIXED (PR #40)

Two-line patch in `trainer_runtime.py`:
1. `drop_last=True` on `DistributedSampler` and `DataLoader` (lines 293, 301) — fixes partial-batch crash.
2. `unwrap_model(model)` in `accumulate_eval_batch` (line 929) — bypasses compile during eval because `pad_collate` produces variable-shape batches that trigger a symbolic-sum codegen bug in torch.inductor.

**All future runs should use `--compile-model` (the default) without `--no-compile-model`.**
Throughput: ~16 min/epoch compiled vs ~18 min uncompiled. 270-min budget → 12 epochs.

## Reference baseline targets — must beat (AB-UPT public reference)

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

## Yi reference results (different W&B project — for context only)

The current `yi` advisor SOTA (informational targets to match or beat on tay/DDP8):

| Metric | Best on yi | PR | W&B run | Date |
|---|---:|---|---|---|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **7.4861** | yi#657 | riy0bxtl | 2026-05-05 |
| `test_primary/abupt_axis_mean_rel_l2_pct` | **8.8110** | yi#657 | riy0bxtl | 2026-05-05 |

(yi#657 was fern's ultra-low-LR continuation (lr=1e-6) from PR #637 best ckpt `vzprvtaw`
at val_abupt=7.5373%. Got 3 epochs of refinement before timeout. Every sub-metric improved
on both val and test. Val slope still −0.0064%/1k steps at timeout — not fully converged.
Next: try lr=3e-7 or 1e-7 from this checkpoint with longer timeout or more epochs; also consider SWA.)

Previous yi best (yi#490, frieren's STRING-separable learnable PE port):
val_abupt=8.087, test_abupt=9.373 (W&B: zwh9qzjw, 2026-05-01)

Previous yi best (yi#13, norman's progressive cosine EMA 0.99→0.9999):
test_abupt=15.82, surf_p=9.99, wall_shear=16.60, vol_p=14.21 (W&B: wio9pqw2, 2026-04-29)

## Confirmed-orthogonal levers from yi (to compose on tay)

1. Width 512d / heads 8 / slices 128 (yi PR #4 chihiro) — the 256d→512d step
   moves volume_pressure 15.21 → 14.37. Needs `lr=5e-5`, `bs=4` at 512d.
2. Protocol fixes (yi PR #9 gilbert) — `--volume-loss-weight 2.0
   --batch-size 8 --validation-every 1`. Halved abupt mean over the
   pre-protocol baseline.
3. Tangential wall-shear projection loss (yi PR #11 kohaku) — denormalize →
   project onto surface tangent → renormalize. Default off; opt in with
   `--use-tangential-wallshear-loss` if/when ported.
4. AdaLN-zero per-block FiLM geometry conditioning (yi PR #8 frieren) —
   independent +5% on every axis at 256d.
5. Cosine EMA decay 0.99 → 0.9999 (yi PR #13 norman) — single largest
   non-architectural lever in yi (−9% on every axis).

## Update protocol

When a tay PR lands a new best `test_primary/abupt_axis_mean_rel_l2_pct`:
1. Update the Status header to the new PR + W&B run + date.
2. Replace the per-axis best table with the new run's `test_primary/*`.
3. Append a short "Compounding wins so far" entry naming the orthogonal lever.
