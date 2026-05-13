# SENPAI Research State
- 2026-05-13 ~07:00 UTC

## Human Research Directive (Issue #882)
**TOP PRIORITY — Volume Pressure Focus:**
- The **TEST volume pressure L2 error** is the only metric that matters for new experiment design
- Do NOT degrade surface error or wall shear stress metrics
- Published SOTA models show significantly better volume pressure test metrics are achievable — large headroom to close
- All new student assignments must be designed with volume pressure improvement as the singular focus

## DATASET ARTIFACT RESOLVED (2026-05-12) — Issue #1053

**CRITICAL — All prior val→test vol_p gap analysis is INVALIDATED.**

The persistent +7–8pp val→test volume pressure gap reported across all prior experiments was a **DATASET ARTIFACT** caused by a case-split bug in the pre-20260511 dataset. Under the corrected split (`rawcanon_20260511`), the gap disappears entirely.

**Corrected dataset path (MANDATORY for all new and active experiments):**
`/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`

**Corrected eval parameters:** `eval_surface_points=65536`, `eval_volume_points=65536` (chunk sizes, not caps), 34 val cases / 7,295 views, 50 test cases / 11,091 views.

**Impact on prior results:**
- PRs #972 and #968, previously labeled FALSIFIED, are the **TOP TWO PERFORMERS** on the corrected split.
- PR #740 (old SOTA at 7.5195%) re-evaluates to 8.165% — **rank 22** under the corrected split.
- All "AXIS FULLY CLOSED" labels on sampling strategies are **RETRACTED**.
- The 15+ "falsified" interventions were not tested on a valid dataset split.

## CRITICAL DISCOVERY (2026-05-12 ~22:00 UTC): SDF Monkey-Patch Was a No-Op

**PR #972 did NOT implement SDF-stratified sampling. It used uniform sampling.**

Two bugs were discovered in the `types.MethodType` monkey-patch used in PR #972 and copied to PRs #1054 and #1055:

### Bug 1: Python dunder method resolution no-op
`types.MethodType(__getitem__, dataset)` installs the function in the **instance dict**. Python resolves dunder methods (`__getitem__`) via **type MRO**, not instance dict. PyTorch DataLoader subscript access `dataset[idx]` calls the **class's** `__getitem__`, not the instance attribute. The monkey-patch is silently ignored.

**Fix**: Dynamically create a subclass and reassign `__class__`:
```python
class _SDFStratifiedDataset(type(train_dataset)):
    def __getitem__(self, idx):
        ...
train_dataset._sdf_alpha = args.sdf_stratified_alpha
train_dataset._sdf_n_vol = args.train_volume_points
train_dataset.__class__ = _SDFStratifiedDataset
assert type(train_dataset).__name__ == '_SDFStratifiedDataset', 'SDF patch not active'
```

### Bug 2: SDF formula direction inverted
`volume_sdf.npy` stores **unnormalized** SDF values with `max(|sdf|) ≈ 80m` per case. The original formula:
```
weight = 1.0 + α * |sdf[i]|
```
...upweights **far-field** points (weight ≈ 321 at α=4.0 and |sdf|=80), not near-surface. This is opposite to the stated hypothesis.

**Correct near-surface emphasis formula:**
```
weight = 1.0 / (1.0 + α * |sdf[i]|)
```
Surface points (sdf≈0) get weight≈1.0; far-field points get weight→0.

### Implication for SOTA table
**ALL three top corrected-split results used uniform sampling.** No true SDF-stratified experiment has run to date. The SDF-stratified hypothesis remains (essentially) untested. This is an opportunity, not a setback.

## Wave SOTA (Corrected Split — rawcanon_20260511)

**PR #972** (run `56bcqp3m`, eval `zxnhtagj`): `abupt_axis_mean_rel_l2_pct` = **5.844%**

| Metric | Value |
|--------|-------|
| test_abupt | **5.844%** |
| test_surf_p | 3.577% |
| test_vol_p | **3.643%** |
| test_wss | 6.727% |

> Note: PR #972 used uniform sampling (monkey-patch was a no-op). This is the baseline to beat.

**Top 5 on corrected split:**

| Rank | PR | test_ABUPT | test_vol_p | Notes |
|------|----|------------|------------|-------|
| 1 | #972 | 5.844% | 3.643% | Uniform (monkey-patch no-op) — CLOSED |
| 2 | #968 | 5.986% | 3.957% | Stochastic vol subsampling — CLOSED |
| 3 | #880 | 6.010% | 4.501% | CLOSED |
| 4 | #958 | 6.107% | 3.818% | CLOSED |
| 5 | #939 | 6.242% | — | CLOSED |

## Active Experiments (2026-05-13 ~07:00 UTC)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| tanjiro | #1055 | SDF near-surface sampling α=1.5 (corrected patch) | EP11 PASS ✓ (6.388%), EP15 gate (≤6.80%) approaching; MERGE CONFLICT |
| nezuko | #1072 | SDF near-surface sampling α=0.5 (corrected patch, softest regime) | JUST ASSIGNED — awaiting start |
| frieren | #1070 | Combined vol+surf SDF weighting | Awaiting start (dup #1069 closed) |
| fern | #1063 | [active experiment] | EP2 PASS ✓ (7.4836%), EP3 approaching |
| edward | #1050 | [active experiment] | EP9=7.0886%, EP12 gate (≤6.5%/≤5.0% vol_p) approaching |

**PR #1054 nezuko KILLED** — EP15 FAIL: val_abupt=6.9264% vs gate ≤6.80%. Confirmed dead at α=2.0 near-surface SDF.

**SDF α sweep status:**
- α=0.5 → nezuko PR #1072 (just assigned — softest, α < 1.0 regime, untested)
- α=1.5 → tanjiro PR #1055 (EP11 PASS, continuing)
- α=2.0 → nezuko PR #1054 (EP15 FAIL, KILLED — too aggressive)

## Tier 1 Follow-Ups (Current Priority)

1. **True near-surface SDF sampling vol-only** — nezuko (#1072, α=0.5) and tanjiro (#1055, α=1.5). Primary open question — true SDF near-surface emphasis has NEVER run successfully.
2. **Combined vol+surf SDF weighting** — frieren (#1070). Novel axis: apply SDF weighting to surface points as well.
3. **α=1.0 SDF after #1072 and #1055 results** — the midpoint of the α sweep. Only warranted once we understand α=0.5 and α=1.5 shape.

## Tier 2 Follow-Ups (After Tier 1 Results)

4. **SDF-stratified + dedicated vol decoder (6L vol tower)** — combine near-surface SDF sampling with independent vol tower architecture.
5. **Re-run vol-token LN (PR #1025 config) from scratch on corrected split** — showed val_vol_p=3.547% on old dataset; needs clean re-run.
6. **WD=0.005 re-test on corrected split** — previously falsified but could benefit from clean evaluation.
7. **EMA decay=0.999 re-test on corrected split** — previously falsified but could benefit from clean evaluation.
8. **Stochastic vol subsampling combined with SDF weighting** — PR #968 confirmed rank 2 on corrected split; combining with true SDF near-surface emphasis has not been attempted.

## Gate Schedule

| Gate | Standard Threshold | Steps (EBS=32) | Steps (EBS=8) |
|------|--------------------|----------------|----------------|
| EP1  | ≤30%               | 2,743          | 10,975         |
| EP2  | ≤16%               | 5,486          | 21,950         |
| EP3  | ≤8%                | 8,229          | 32,925         |
| EP5  | ≤7.5%              | 13,715         | 54,875         |
| EP10 | ≤7.2%              | 27,430         | 109,750        |
| EP15 | ≤6.80%             | 41,145         | 164,625        |
| EP20 | ≤6.70%             | 54,860         | 219,500        |
| EP25 | ≤6.65%             | 68,575         | 274,375        |
| EP30 | ≤6.60%             | 82,290         | 329,250        |

## Critical Config Constraints

1. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
2. **`--train-volume-points 65000` (or higher)** REQUIRED: default 16384 inverts volume:surface gradient ratio.
3. **`--lr-warmup-epochs 1` NOT `--lr-warmup-steps 500`**: epoch-based warmup is correct at 6L 65k.
4. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion optimizer.
5. **`--volume-loss-weight` BUG**: When `--use-gradnorm` is active, `--volume-loss-weight` is a no-op.
6. **`--model-pe string_multisigma` REQUIRED when using STRING PE on DL24**: omitting causes `--pe-init-sigmas` to be silently ignored.
7. **`--pe-init-sigmas` must be COMMA-separated**: `0.25,0.5,1.0,2.0,4.0` NOT space-separated.
8. **Kill threshold operator on DL24**: `<=` is correct (`passes()` returns True when observed<=threshold; kills when not passes).
9. **EMA decay=0.999 (NOT 0.9999)**: 0.9999 gives ~10,000-step lookback (too slow); 0.999 gives ~1,000-step lookback.
10. **Steps/epoch at bs=1 DDP8 (EBS=8)**: ~10,975.
11. **Steps/epoch at bs=4 DDP8 (EBS=32)**: ~2,743.
12. **SDF monkey-patch MUST use `__class__` reassignment**: `types.MethodType(__getitem__, dataset)` is a Python dunder no-op.
13. **SDF near-surface formula**: `weight = 1.0 / (1.0 + α * |sdf|)`. NOT `1.0 + α * |sdf|`.
14. **Branch flag incompatibility**: tay and DL24 branches use DIFFERENT CLI flags. Never mix.
15. **Dataset path**: `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`. NOT `/mnt/pvc/`.

## Confirmed Dead Ends (Hardware/Architecture — Not Dataset-Dependent)

- No weight decay: overfits at EP9 (PR #898)
- 7L depth without full regularization: catastrophic bounce (PR #873)
- 8L depth: EP11 bounce +0.222pp (PR #965)
- Dropout=0.1: consistent degradation (PR #899)
- vol-loss-weight=2.0 WITH GradNorm: self-cancelling (PR #911)
- vol-loss-weight=2.0 WITHOUT GradNorm: actively harmful (PR #936)
- vol-loss-weight=3.0 WITHOUT GradNorm: WORST EVER on old dataset (PR #964)
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- QK-Norm: consistent failures (multiple PRs)
- Y-sym p=1.0: over-augmentation (PR #866)
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (multiple PRs): terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16
- Extended cosine T_max=60 (PR #946): destabilizing in training tail
- DropPath regularization (PR #987): EP5 gate FAIL
- Lookahead Lion (PR #998): EP5 FAIL
- Online focal vol reweighting (#1026, #1033): scale=3 ceiling degeneration — EMA ratio approach AXIS CLOSED
- SDF near-surface sampling α=2.0 (nezuko PR #1054): EP15 FAIL (6.9264% > 6.80%) — too aggressive near-surface concentration

## Removed from Dead Ends (RETRACTED — Dataset Artifact)

- ~~SDF-stratified importance sampling (PR #972)~~: **NOW RANK 1 SOTA** (test_abupt=5.844%, but this was actually uniform)
- ~~Stochastic vol subsampling (PR #968)~~: **NOW RANK 2 SOTA** (test_abupt=5.986%)
- WD=0.01 (PR #900): needs re-test on corrected split
- WD=0.005 (PR #914): needs re-test on corrected split
- TTA Y-symmetry (PR #979): may still be neutral; re-test needed
- Vol coordinate noise (PR #990): EP5 gate FAIL — may be architecture-level; re-test needed
- Bbox normalization (PR #978): may need re-test
- EMA decay=0.999 (PR #954): needs re-test on corrected split

_Last updated: 2026-05-13 ~07:00 UTC. PR #1054 nezuko KILLED EP15 FAIL (6.9264% > 6.80%). PR #1072 assigned to nezuko (SDF α=0.5 near-surface, corrected patch + inverse formula). SDF α sweep: α=0.5 nezuko #1072 (new), α=1.5 tanjiro #1055 (EP11 PASS), α=2.0 DEAD._
