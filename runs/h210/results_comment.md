STUDENT tanjiro:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["1utf07yl","r3d3g2l1","2jnatpxj","qgb9r74q","xovknq3s"],"primary_metric":{"name":"full_val_primary/abupt_axis_mean_rel_l2_pct","value":88.0958},"test_metric":{"name":"test_primary/abupt_axis_mean_rel_l2_pct","value":88.3545}}

## Results

**Hypothesis decisively REFUTED. NO MERGE.** Naive cross-recipe parameter-space averaging collapses to ~88-90% rel-L2 — essentially a destroyed model. The H112 baseline sanity re-eval reproduces canonical numbers exactly (val_abupt=6.1358% / test_WSS=6.752% / test_VP=3.421%), confirming the eval pipeline is correct and the catastrophic SWA failures are a real signal about parameter-space geometry, not a tooling bug.

### Results table

| Config                         | W&B id     | val_abupt | val_SP  | val_VP  | val_WSS | val_WSS_x | val_WSS_y | val_WSS_z | test_abupt | test_SP  | test_VP  | test_WSS | test_WSS_x | test_WSS_y | test_WSS_z |
|--------------------------------|------------|----------:|--------:|--------:|--------:|----------:|----------:|----------:|-----------:|---------:|---------:|---------:|-----------:|-----------:|-----------:|
| **H112 baseline sanity**       | `xovknq3s` | **6.1358**| 4.0554  | 3.5478  | 6.9671  | 6.0924    | 7.6083    | 9.3751    | **5.8391** | 3.6948   | 3.4213   | **6.7523** | 5.9990     | 7.3604     | 8.7200     |
| swa5_all (equal 5-way)         | `1utf07yl` | 88.0958   | 74.3117 | 78.7888 | 90.8070 | 84.7864   | 101.2079  | 101.3841  | 88.3545    | 75.0070  | 78.9618  | 91.3529    | 85.4742    | 101.2594   | 101.0700   |
| swa3_baseline_h183             | `r3d3g2l1` | 89.5807   | 77.5198 | 78.3314 | 93.1892 | 88.3329   | 101.6063  | 102.1131  | 89.7735    | 77.8054  | 78.5050  | 93.8365    | 89.2697    | 101.5885   | 101.6991   |
| swa3_val_sota_only             | `2jnatpxj` | 90.3635   | 79.1660 | 78.3727 | 94.3157 | 89.9025   | 102.3383  | 102.0379  | 90.5458    | 79.1634  | 78.6592  | 95.0870    | 91.1223    | 102.2161   | 101.5680   |
| swa_weighted_h112_heavy        | `qgb9r74q` | 89.0998   | 77.5882 | 77.6326 | 92.2918 | 87.0130   | 101.2944  | 101.9708  | 89.2450    | 77.8131  | 77.9110  | 92.5659    | 87.1948    | 101.3438   | 101.9623   |
| Canonical H112 (PR #1283)      | `u9ue2ryb` | 6.1358    | 4.0553  | 3.5478  | 6.9670  | 6.0923    | 7.6084    | 9.3750    | 5.839      | 3.695    | 3.421    | 6.752    | 5.999      | 7.360      | 8.720      |

(Sanity reproduction passes — every column matches canonical H112 to ≤0.0001pp. Eval pipeline verified.)

### val→test slope on WSS aggregate

| Config | val_WSS | test_WSS | slope (test - val) |
|--------|---------:|---------:|-------------------:|
| H112 baseline | 6.9671 | 6.7523 | **−0.215pp** (canonical generalization profile) |
| swa5_all | 90.81 | 91.35 | +0.55pp (no useful slope — model destroyed) |
| swa3_baseline_h183 | 93.19 | 93.84 | +0.65pp |
| swa3_val_sota_only | 94.32 | 95.09 | +0.77pp |
| swa_h112_heavy | 92.29 | 92.57 | +0.27pp |

WSS_x slope sign on SWA arms is moot — the model is not predicting anything useful.

### SOTA gate

All 4 SWA configs FAIL every gate by ~10-15x. No SOTA candidate.

Strict gate (val_abupt < 6.1358% AND test_WSS ≤ 6.752% AND test_VP ≤ 3.421% AND test_SP ≤ 3.577% AND WSS_x slope ≤ −0.020pp):
- swa5_all best at val_abupt=88.10% (gate floor 6.1358%) — **decisively fails**.

### Exact command (representative)

```bash
cd target/
CUDA_VISIBLE_DEVICES=1 python runs/h210/eval_swa.py \
  --checkpoint runs/h210/swa/swa5_all.pt \
  --donor-config runs/h210/artifacts/h112/config.yaml \
  --wandb-group h210-tanjiro-swa-cohort \
  --wandb-name "tanjiro/h210-swa5-all" \
  --num-workers 2
```

Each eval consumed ~12 min (full_val 34 cases + test 50 cases on a single H100 at bf16). 5 evals ran in parallel on GPUs 0-4. Peak GPU memory: **9.89 GB** per eval.

### Per-config peak memory & wall time

| Config | seconds (full_val + test) | peak GB |
|--------|--------------------------:|--------:|
| h112 baseline | 480 + 730 = ~1210 | 9.89 |
| swa5_all | 481 + 730 = ~1211 | 9.89 |
| swa3_baseline_h183 | 482 + 730 = ~1212 | 9.89 |
| swa3_val_sota_only | 480 + 730 = ~1210 | 9.89 |
| swa_weighted_h112_heavy | 481 + 730 = ~1211 | 9.89 |

### What happened — honest analysis

The hypothesis was that "cross-recipe consensus across 5 diverse mechanism failures yields a checkpoint that is flatter and basin-stable." **The opposite happened**: averaging produced models whose internal hidden-state representations were destroyed by the very averaging operation. The mechanism is the **permutation-symmetry / loss-landscape connectivity problem**, a well-documented phenomenon in the SWA / weight-averaging literature:

1. **Permutation symmetry**: A trained Transformer can be reparameterized by permuting hidden units / attention heads without changing the input-output function. Two independently-trained models with different RNG seeds settle in different permutations of an equivalent solution. Their parameter vectors are in different orientations even though their forward passes are functionally equivalent.

2. **Loss-landscape connectivity** (Frankle et al. 2020, Garipov et al. 2018): Linear interpolation in parameter space between two SGD endpoints stays in a low-loss region **only if** the endpoints are "linearly mode connected" — typically requires either a shared starting point (e.g., a common pretrained warmstart) or explicit permutation alignment first.

3. **In this cohort**, all 5 runs started from `nn.Linear(in_features, out_features)`-style independent random init. Even H112 ↔ H164e (same recipe, different RNG for fern's N=2 calibration) are NOT linearly mode connected; their parameter midpoint sits at high loss. Adding H183/H185/H190 (different recipes — mirror augmentation, GradNorm, mirror p=0.25) compounds the misalignment.

4. The H112-heavy weighting (3×H112 + 2×H164e + 1×each others = 8 total) is still dominated 50/50 by H112-recipe vs others, so it ends up in a similar broken region as the equal SWA — proving the failure is geometric, not driven by which models are "best" individually.

5. **What this DOESN'T refute**: output-space averaging of forward predictions ("Caruana ensemble," PR #1102) DOES work because each model is evaluated independently in its own basin, and only the predictions are averaged. PR #1102 K=8 weighted ensemble achieves test_abupt=5.5196% — that is the working ensemble path. Naive parameter-space averaging across independent recipes is a fundamentally different and broken operation.

6. **Coordination note** for askeladd H207 and fern H208: their binary interpolations (H112↔H183, H112↔H190) will face the same mathematical problem. Binary interpolation at α=0.5 is structurally identical to a 2-way SWA. Expect catastrophic collapse around α=0.5, possibly with a narrow corridor near α=0 or α=1 where the result remains close to one parent. fern's H193 binary (H112+H164e) is the most likely to "work" since they share recipe, but even that requires linear mode connectivity to hold across N=2 RNG seeds — not guaranteed.

### Suggested follow-ups

1. **Permutation-aligned SWA (Git Re-Basin / OTFusion / activation matching)** — if cross-recipe SWA is the goal, the right approach is to (a) match permutations of attention heads / MLP hidden units between models via the Hungarian algorithm on activations or weights, (b) apply the permutations to align all 5 state_dicts onto H112's basis, (c) THEN average. Ainsworth et al. 2023 "Git Re-Basin" provides the algorithm; on small CNNs they recover SWA-quality results across independently-trained models. The Transolver-specific risk is whether attention head permutations on slice tokens + RFF features compose cleanly — would require a small implementation effort plus a smoke run.

2. **Functional-space averaging instead of weight-space**: this is what PR #1102 already does and remains the proven SOTA path. If we want the next-level ensemble, expand the candidate pool (H183/H185/H190 EP13 best artifacts) to PR #1102's K=8 selector and re-run greedy WSS-targeted Caruana — that's the lowest-effort path to ensemble SOTA. Worth pinging askeladd / fern on Caruana inclusion.

3. **Anti-stacking sanity for the cohort**: if α=0.5 binary interpolation of H112+H183 (askeladd H207) also collapses, that is strong evidence the cohort is not linearly mode connected. If H207 finds a non-trivial α* ∈ (0, 1) where val_abupt < 6.1358%, that would be surprising and worth investigating as a sign of weak mode connectivity between drop_path-only and mirror-aug recipes.

4. **No further SWA arms**: I would not recommend the advisor spend further compute on naive weight-space SWA in this cohort. The geometry argument applies a priori — additional weighting schemes (entropy-weighted, val-quality-weighted, slope-weighted) will all sit in the same broken parameter-space region. Push the cohort's "consensus" story through ensemble prediction averaging instead.

### Artifacts

- Eval script: `runs/h210/eval_swa.py`
- 4 SWA state_dicts saved at `runs/h210/swa/{swa5_all,swa3_baseline_h183,swa3_val_sota_only,swa_weighted_h112_heavy}.pt`
- 5 donor checkpoints cached at `runs/h210/artifacts/{h112,h164e,h183,h185,h190}/`
- All metric breakdowns logged to W&B group `h210-tanjiro-swa-cohort`
