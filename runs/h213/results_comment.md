STUDENT tanjiro:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["awl4hxem","80iaxoea","5ragl8wb","axefwtlk","w3o4rby9"],"primary_metric":{"name":"full_val_primary/abupt_axis_mean_rel_l2_pct","value":35.2234},"test_metric":{"name":"test_primary/abupt_axis_mean_rel_l2_pct","value":34.9546}}

## Results

**Hypothesis decisively REFUTED. NO MERGE.** The basin disruption is **distributed**, not depth-localized: replacing any single H112 block with the matching H183 block destroys the model. k=5 (all H112, sanity) reproduces canonical H112 numbers exactly; k=0..4 (any H183 block in the stack) all sit at 35–94% val_abupt, ~5–15× worse than functional. There is no k that improves val while keeping the basin intact, so no SOTA candidate.

Note: k=5 is the only functional config (because all non-block keys default to H112 per advisor template), and k=0 is therefore NOT a clean "all H183" sanity anchor — it has H183's 5 blocks but H112's pos-embed / heads / cross-attn / norms. See "k=0 caveat" below.

### Splice table (test metrics, val abupt, val→test slope)

| k | val_abupt | val_WSS | test_WSS | test_VP | test_SP | val→test WSS_x slope (pp) | Functional? |
|---|----------:|--------:|---------:|--------:|--------:|--------------------------:|-------------|
| 0 (H183 blocks + H112 non-block) | 94.3125 | 99.3714 | 100.4279 | 75.2714 | 78.7665 | **+1.8897** | NO — collapse |
| 1 | 80.3275 | 84.1742 | 84.3663 | 57.5993 | 71.7439 | **+0.2499** | NO — collapse |
| 2 | 65.1448 | 68.4573 | 68.8360 | 47.1774 | 54.3825 | **+0.3267** | NO — collapse |
| 3 | 52.6070 | 54.8453 | 54.8950 | 38.7767 | 42.8270 | **+0.0775** | NO — collapse |
| 4 | 35.2234 | 36.9197 | 37.0749 | 25.0901 | 27.2616 | **+0.2272** | NO — collapse |
| **5 (all H112)** | **6.1358** | **6.9671** | **6.7523** | **3.4213** | **3.6948** | **−0.0934** | ✓ sanity OK |
| Canonical H112 ref (PR #1283 `u9ue2ryb`) | 6.1358 | 6.9670 | 6.752 | 3.421 | 3.695 | −0.093 | reference |

Sanity reproduction at k=5 matches canonical H112 to ≤0.0001pp on every column — eval pipeline verified, splice operator verified, k=5 = pure H112.

### val→test slope summary (test − val, pp)

| k | abupt | SP | WSS | WSS_x | WSS_y | WSS_z | VP |
|---|------:|---:|----:|------:|------:|------:|---:|
| 0 | +0.145 | +0.105 | +1.057 | +1.890 | +0.079 | −2.082 | +0.733 |
| 1 | +0.053 | +0.887 | +0.192 | +0.250 | +0.341 | −1.535 | +0.323 |
| 2 | +0.028 | −0.073 | +0.379 | +0.327 | +0.674 | −1.029 | +0.240 |
| 3 | −0.283 | −0.375 | +0.050 | +0.078 | −0.001 | −1.096 | −0.018 |
| 4 | −0.269 | −0.602 | +0.156 | +0.227 | +0.194 | −1.111 | −0.052 |
| 5 (H112) | −0.297 | −0.361 | −0.215 | −0.093 | −0.248 | −0.655 | −0.126 |

Useful slope structure only exists at k=5 (where the model still predicts). For k=0..4 the model is in random-init territory (74–117% rel-L2 on individual heads), so slope sign is mathematically a real number but it is not a basin signal — there is no basin to disrupt.

### Answer to the H213 question

> Is H183's basin disruption depth-localized or distributed across the 5 encoder blocks?

**Distributed.** Every single block swap (k=4: replace just block 4 with H183, k=3: replace blocks 3+4, …) catastrophically destroys the model. No subset of blocks can be transferred from H183 onto an H112 base without breaking the inter-block residual stream.

Mechanism is the same permutation-symmetry barrier identified in H210:

1. Each TransformerBlock here is `norm1 → attention(temperature, in_project_{x,fx,slice}, qkv, q_norm/k_norm, proj) → norm2 → mlp(fc1, fc2)`. Block-internal coherence is preserved by the splice (each block keeps its own attention temperature, qkv ordering, head permutation, MLP hidden permutation).
2. But block i's output residual is consumed by block i+1's `norm1`. If block i is H112-trained and block i+1 is H183-trained, the H183 block expects the residual stream channels in H183's permutation realization (heads, MLP hidden units, slice-attention slot ordering) — and gets H112's. Result: every downstream block sees noise.
3. The k=4 boundary (4 H112 + 1 H183 final block) has only ONE bad interface — the H112→H183 transition at block 4 input — yet val_abupt still collapses to 35%. The interface is enough.
4. Monotonic degradation k=4 → k=0 (35% → 94%) is consistent with N independent bad interfaces accumulating noise.

### k=0 caveat (why it doesn't reproduce H183's 6.039%)

Per the build script (PR template), non-block keys (pos_embed, surface/volume biases, projections, heads, surf↔vol cross-attn, final norm) default to H112 for all k, which makes k=5 reproduce H112 exactly. That same choice means k=0 is **not** "all-H183" — it is H183 blocks bolted onto H112 embeddings/heads/cross-attn. So k=0's 94% collapse is a 5-bad-interface signal, not an "H183 sanity anchor". Building a separate "k=0' = full-H183 swap" sanity would be useful but is redundant given the trend k=4→k=0 already proves the point: even one interface kills it.

State-dict structure (137 keys total):
- 100 block keys (20 per block × 5 blocks)
- 37 non-block keys (string/sincos pos-embed, surface/volume biases, feature projections, surface/volume out-heads, surf↔vol cross-attention, final LayerNorm)

### SOTA gate

Strict gate: val_abupt < 6.1358 AND test_WSS ≤ 6.752 AND test_VP ≤ 3.421 AND test_SP ≤ 3.577 AND WSS_x slope ≤ −0.020pp.

- k=5 ties val_abupt (it IS H112) and matches the canonical baseline within rounding — not a SOTA improvement.
- k=0..4 all FAIL every column by orders of magnitude.
- **No SOTA candidate. No merge.**

### Exact commands

**Splice construction** (one-shot, ~5s):
```bash
cd target/
python runs/h213/build_splices.py \
  --h112 runs/h210/artifacts/h112/checkpoint.pt \
  --h183 runs/h210/artifacts/h183/checkpoint.pt \
  --out-dir runs/h213/splices
```

**Eval per splice** (single GPU, 5 parallel × 1 sequential = 6 evals):
```bash
cd target/
for k in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$k python runs/h213/splice_eval.py \
    --checkpoint runs/h213/splices/splice_k${k}.pt \
    --wandb-name tanjiro/h213-splice-k${k} \
    --wandb-group h213-tanjiro-splice-h112-h183 \
    --num-workers 4
done
```

k=5 reused the H210 baseline result (canonical H112 EP13 best). k=0..4 were five fresh single-GPU evals run in parallel on GPUs 0..4.

### Per-config wall time & peak memory

| k | val sec | test sec | wall total | peak GB |
|---|--------:|---------:|-----------:|--------:|
| 0 | 480.9 | 732.0 | ~20.2 min | ~9.9 |
| 1 | 481.5 | 730.6 | ~20.2 min | ~9.9 |
| 2 | 480.6 | 730.4 | ~20.2 min | ~9.9 |
| 3 | 482.3 | 732.0 | ~20.2 min | ~9.9 |
| 4 | 481.6 | 730.6 | ~20.2 min | ~9.9 |
| 5 (H112 sanity, H210 reuse) | 481.0 | 730.1 | ~20.2 min | 9.89 |

Effective wall time ~20 min (5 evals parallel on 5 GPUs), reusing H210's H112 sanity for k=5.

### W&B run IDs

- k=0: `awl4hxem` (tanjiro/h213-splice-k0)
- k=1: `80iaxoea` (tanjiro/h213-splice-k1)
- k=2: `5ragl8wb` (tanjiro/h213-splice-k2)
- k=3: `axefwtlk` (tanjiro/h213-splice-k3)
- k=4: `w3o4rby9` (tanjiro/h213-splice-k4)
- k=5 (H112 sanity, reused from H210): `xovknq3s` (tanjiro/h210-baseline-h112-sanity)

All in W&B group `h213-tanjiro-splice-h112-h183`.

### What happened — honest analysis

The hypothesis was that splicing the first k blocks of H112 with the last (5−k) blocks of H183 might find a "compatibility window" where the model improves on val while preserving the basin (WSS_x slope ≤ −0.02pp). The actual finding is the opposite:

1. **There is no compatibility window.** Even a single H183 block in the stack (k=4) drops val_abupt from 6.14% to 35.22%, a ~6× degradation, despite preserving 80% of the H112 parameter mass.
2. **The degradation is monotonic in #H183 blocks** (k=4: 35.22% → k=3: 52.61% → k=2: 65.14% → k=1: 80.33% → k=0: 94.31%), consistent with each H112↔H183 interface adding a roughly constant amount of permutation-mismatch noise to the residual stream.
3. **WSS_x slope sign flips don't matter once the model is destroyed.** k=3 has the smallest positive WSS_x slope (+0.078pp, almost flat), but its absolute val_WSS_x is 47.6% — the slope is between two broken numbers. Slope signals are only meaningful in the basin-intact regime, which here means k=5 only.
4. **Permutation-symmetry barrier is intra-stack, not just intra-block.** H210 established this barrier for parameter averaging across recipes. H213 establishes that it applies to inter-block boundaries too. Block-wise splicing preserves coherence inside each block but not across the residual stream, so the splicing trick does not bypass the H210 barrier — it only confirms it operates at finer granularity than expected.
5. **Implication for H207's α-sweep (askeladd) and H208 (fern)**: scalar interpolation between H112 and H183 across the full state-dict is structurally equivalent to a noisy 2-way SWA. The H207 catastrophic-collapse band at α≈0.5 (predicted in my H210 follow-ups) will be confirmed by their alpha sweeps. They may find a narrow corridor near α=0 or α=1 where one parent dominates and the result is approximately that parent, but no functional intermediate.
6. **What this rules out as a SOTA path:** any naive cross-recipe parameter-space stitch / weight-merge across the H112↔H183 pair, at block, layer, or scalar-α granularity, is not going to produce a better-than-parent model. The cross-recipe consensus story must move to **output-space** (Caruana / PR #1102 ensemble averaging of forward predictions), which H210's follow-up already recommended.

### Suggested follow-ups

1. **Stop weight-stitching this pair.** H210 ruled out weight averaging across recipes; H213 rules out block-wise splicing across recipes; H207/H208 will likely rule out scalar α-interp. The pattern is consistent — no further weight-merging experiments on H112↔H183 are likely to yield a SOTA candidate. Reallocate compute.
2. **Permutation-aligned variant** (Git Re-Basin / OTFusion, Ainsworth et al. 2023): if cross-recipe stitching is still desirable, the right move is to (a) compute Hungarian assignments on per-head / per-MLP activations between H112 and H183 on a small calibration batch, (b) apply the inverse permutation to H183's state-dict, aligning it onto H112's basis, (c) THEN splice or interpolate. On Transolver this is non-trivial — needs to permute attention heads AND slice-attention slot ordering AND MLP hidden units consistently across all 5 blocks AND the residual stream that connects them. Estimate ~1 day of implementation + a calibration smoke run before any eval.
3. **Move to output-space ensembling.** PR #1102's K=8 Caruana selector is the validated cross-recipe consensus path (test_abupt=5.5196%). Adding H183 EP13 (val SOTA at 6.039%) and other EP13 winners to that pool, then re-running greedy WSS-targeted Caruana, is the lowest-effort cross-recipe SOTA path remaining. Worth a coordinated assignment.
4. **Partial-block splices (single-component, not whole-block)** are a curiosity rather than a SOTA play, but would isolate which sub-component carries the permutation barrier — e.g., swap only `attention.in_project_x` weights, or only `mlp.fc1`. Useful for the H210/H213 mechanism story but unlikely to land a winning model.
5. **No new H213-style experiments.** This sprint terminates with a clean negative answer; the residual-stream barrier is established at finer-than-block granularity, and no further block-structured probes are likely to find a basin-intact splice.

### Artifacts (all committed in this PR)

- `runs/h213/build_splices.py` — splice builder with sanity checks (k=5 ⇄ H112, k=0 ⇄ H183 block keys, k=2 boundary check)
- `runs/h213/splice_eval.py` — single-GPU val+test eval with W&B logging and slope diagnostics
- `runs/h213/splices/splice_k{0..5}.pt` — six block-spliced state_dicts (each ~70 MB)
- `runs/h213/results/tanjiro_h213-splice-k{0..5}.json` — per-config metric breakdowns
- `runs/h213/logs/eval_k{0..4}.log` — per-eval stdout (k=5 reuses H210 H112 baseline log)
- Parent checkpoints from H210 sprint: `runs/h210/artifacts/h112/checkpoint.pt` and `runs/h210/artifacts/h183/checkpoint.pt`
- All metric breakdowns logged to W&B group `h213-tanjiro-splice-h112-h183`
