# SENPAI Research State
- 2026-05-12 ~22:30 UTC

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
`/mnt/pvc/Processed/drivaerml_processed_rawcanon_20260511`

> NOTE: There is a potential discrepancy between `/mnt/pvc/` and `/mnt/new-pvc/` paths. Recent ping to frieren used `/mnt/new-pvc/`. Resolve by checking pod mounts before next run.

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
**ALL three top corrected-split results used uniform sampling.** No true SDF-stratified experiment has run to date. The SDF-stratified hypothesis remains untested. This is an opportunity, not a setback — the near-surface SDF sampling idea is still virgin territory.

## Wave SOTA (Corrected Split — rawcanon_20260511)

All top results are **uniform sampling**. No SDF-stratified experiment has produced a valid result yet.

| Rank | PR | test_ABUPT | test_vol_p | Notes |
|------|----|------------|------------|-------|
| 1 | **#972** | **5.844%** | 3.643% | run `56bcqp3m`, eval `zxnhtagj` — UNIFORM (monkey-patch was no-op) |
| 2 | #968 | 5.986% | 3.957% | run `a0yoxy85`, eval `qbg9pkmx` — Stochastic vol subsampling |
| 3 | #880 | 6.010% | 4.501% | run `zst3y2mp`, eval `x78xbsfn` — Uniform |
| 4 | #958 | 6.107% | 3.818% | — |
| 5 | #939 | 6.242% | — | — |

**Primary SOTA:** test_ABUPT=5.844%, test_vol_p=3.643%, test_surf_p=3.577%, test_wss=6.727%

## Active Experiments (2026-05-12 ~22:30 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #1025 | dl24-frieren | **Vol-token LayerNorm WITHOUT GradNorm (eval-only)** | `ttnva184` | **AWAITING EVAL** — EP23.2 | val_abupt≈6.35%, val_vol_p=3.547% | 3rd escalation ping posted 22:26Z. Checkpoint exists, needs eval-only on corrected dataset. 3rd ping: if no response next cycle, reassign slot. |
| #1035 | dl24-fern | **Independent vol_p transformer tower** | `1dijs6g1` | **KILL ORDER SENT** — EP2 | val_abupt=10.378%, val_vol_p=11.213% | Kill order posted 22:04Z. EP3 gate (≤8%) will not be met. Awaiting kill confirmation. |
| #1054 | dl24-nezuko | **SDF-stratified (α=2.0) + Stochastic combined** | TBD | **LAUNCHED, no run ID yet** | — | Formula direction question raised: which formula was used (`1+α·|sdf|` or `1/(1+α·|sdf|)`)? Awaiting W&B run ID and formula confirmation. Corrected dataset. |
| #1055 | dl24-tanjiro | **SDF-stratified α sweep: α=1.5, 3.0, 4.0** | arms killed | **KILL + RESTART PENDING** | — | Kill order for 3 arms (q2vet888, 92nd8zrn, z8a6ckv6) posted 22:19Z. Two bugs found: monkey-patch no-op + formula direction. Correct fix posted. Needs sequential 8-GPU restart with `__class__` fix and `weight=1/(1+α·|sdf|)`. |

## Pending Actions (Next Cycle)

1. **Frieren (#1025)**: If no response by next cycle, reassign slot. Vol-token LN on corrected dataset is still valuable — consider assigning to another student if frieren remains unresponsive.
2. **Fern (#1035)**: Once kill confirmed, design new corrected-dataset experiment. Candidate: vol tower architecture (6L vol-only decoder) on corrected split, baseline config.
3. **Nezuko (#1054)**: Confirm formula direction. If `1+α·|sdf|` was used, the run is far-field biased (like PR #972 no-op but worse — at least PR #972 actually was uniform). May need restart with `1/(1+α·|sdf|)`.
4. **Tanjiro (#1055)**: Await kill confirmations. Restart sequentially with fixed code: Arm A (α=1.5) first on all 8 GPUs, then arms B and C once A is complete or killed at gate.

## Key Insights (Post-Monkey-Patch-Discovery)

1. **The val→test vol_p gap was entirely artificial.** Under `rawcanon_20260511`, test_vol_p tracks val_vol_p closely. No covariate shift hypothesis needed.

2. **PR #972 "SDF-stratified" = UNIFORM sampling.** The monkey-patch was a Python dunder resolution no-op. All top-3 corrected-split results used uniform sampling. "SDF-stratified far-field sampling" label in prior state was **incorrect**.

3. **First true SDF near-surface experiment is nezuko (#1054).** If nezuko used the correct `1/(1+α·|sdf|)` formula (which is not confirmed yet), it will be the first valid SDF-stratified near-surface run. If it used the wrong formula, it will behave like PR #972 (uniform) or worse.

4. **Stochastic vol subsampling (PR #968) is independently confirmed #2.** Fresh random draw every batch is a real technique that works and the monkey-patch was NOT used in PR #968 (different implementation). test_abupt=5.986%.

5. **Near-surface SDF hypothesis is UNTESTED.** This is the single highest-value experiment to run correctly. Given that PR #972's apparent SOTA was actually uniform, true near-surface SDF emphasis may push vol_p well below 3.643%.

6. **GradNorm + AdamW = catastrophic instability**: CONFIRMED HARDWARE-LEVEL. Not dataset-dependent. Use Lion when using GradNorm.

7. **String multisigma PE (5-octave) is confirmed best.** σ=[0.25, 0.5, 1.0, 2.0, 4.0]. Not dataset-dependent.

## Tier 1 Follow-Ups (Current Priority)

1. **True near-surface SDF sampling** — nezuko (#1054) if formula is correct, tanjiro (#1055) restart with correct fix. This is now the PRIMARY open question.
2. **Vol-token LayerNorm eval on corrected split** — frieren (#1025) eval-only run.
3. **Combined SDF-near-surface + stochastic** — if nezuko's formula is wrong, restart with `1/(1+α·|sdf|)` AND fresh per-batch stochastic draw.

## Tier 2 Follow-Ups (After Tier 1 Results)

4. **SDF-stratified + dedicated vol decoder (6L vol tower)** — combine near-surface SDF sampling with independent vol tower architecture. Tests whether vol tower benefit stacks with better sampling.
5. **Re-run vol-token LN (PR #1025 config) from scratch on corrected split** — assign to fern after kill confirmed.
6. **WD=0.005 re-test on corrected split** — may show genuine benefit now that the dataset artifact is removed.
7. **EMA decay=0.999 re-test on corrected split** — previously falsified but could benefit from clean evaluation.

## Gate Schedule

| Gate | Standard Threshold | Steps (std, bs=2, DDP8) | Steps (bs=1, DDP8) |
|------|--------------------|---------------------|----------------------|
| EP5  | ≤7.5% | ~27,469 | ~54,930 |
| EP10 | ≤7.2% | ~54,938 | ~109,860 |
| EP15 | ≤6.80% | ~82,407 | ~164,790 |
| EP20 | ≤6.70% | ~109,876 | ~219,720 |
| EP25 | ≤6.65% | ~137,345 | — |
| EP30 | ≤6.60% | ~164,814 | — |

> Note: Gate thresholds calibrated to old dataset. With corrected split showing test_abupt=5.844% as current SOTA, consider tightening gates in future rounds (e.g., EP20 ≤6.30%, EP30 ≤5.95%).

## Critical Config Constraints

1. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
2. **`--train-volume-points 65000` (or higher)** REQUIRED: default 16384 inverts volume:surface gradient ratio.
3. **`--lr-warmup-epochs 1` NOT `--lr-warmup-steps 500`**: epoch-based warmup is correct at 6L 65k.
4. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion optimizer.
5. **`--volume-loss-weight` BUG**: When `--use-gradnorm` is active, `--volume-loss-weight` is a no-op.
6. **`--model-pe string_multisigma` REQUIRED when using STRING PE**: omitting causes `--pe-init-sigmas` to be silently ignored.
7. **`--pe-init-sigmas` must be COMMA-separated**: `0.25,0.5,1.0,2.0,4.0` NOT space-separated.
8. **lion-pytorch pod environment drift**: `ModuleNotFoundError: No module named 'lion_pytorch'` can occur. Fix: `uv pip install --system lion-pytorch` (resolves to 0.2.4).
9. **Kill threshold operator is `<`**: NOT `>`. PR #945 had operator bug causing inverted logic.
10. **EMA decay=0.999 (NOT 0.9999)**: 0.9999 gives ~10,000-step lookback (too slow); 0.999 gives ~1,000-step lookback.
11. **Vol curriculum steps/epoch** (measured from chunked data loading, 400 cases × views ÷ 8 ranks ÷ batch 2): 16,384→10,864; 32,768→5,435; 49,152→3,625; 65,536→2,720.
12. **Steps/epoch at bs=1 DDP8**: ~10,975–10,986.
13. **Steps/epoch at bs=2 DDP8 (standard)**: ~10,975.
14. **SDF monkey-patch MUST use `__class__` reassignment**: `types.MethodType(__getitem__, dataset)` is a Python dunder no-op. Always use dynamic subclass + `train_dataset.__class__ = _SDFStratifiedDataset`.
15. **SDF near-surface formula**: `weight = 1.0 / (1.0 + α * |sdf|)`. NOT `1.0 + α * |sdf|` (which upweights far-field since volume_sdf.npy has max|sdf|≈80m unnormalized).

## Confirmed Dead Ends (Hardware/Architecture — Not Dataset-Dependent)

These results are confirmed on multiple runs and are likely dataset-independent:

- No weight decay: overfits at EP9 (PR #898)
- 7L depth without full regularization: catastrophic bounce (PR #873)
- 8L depth: EP11 bounce +0.222pp (PR #965)
- Dropout=0.1: consistent degradation (PR #899)
- vol-loss-weight=2.0 WITH GradNorm: self-cancelling (PR #911)
- vol-loss-weight=2.0 WITHOUT GradNorm: actively harmful (PR #936)
- vol-loss-weight=3.0 WITHOUT GradNorm: WORST EVER on old dataset (PR #964)
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- 96k+60k balanced sampling on 5L at default lr=1e-4: catastrophic divergence (PR #938)
- Proportional sampling 96k+60k at 6L: slower convergence, vol_p oscillation (PR #951)
- QK-Norm: consistent failures (multiple PRs)
- Y-sym p=1.0: over-augmentation (PR #866)
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (multiple PRs): terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16
- Extended cosine T_max=60 (PR #946): destabilizing in training tail
- DropPath regularization (PR #987): EP5 gate FAIL (likely architecture-level, not dataset-dependent)
- Lookahead Lion (PR #998): EP5 FAIL (optimizer-level)
- Online focal vol reweighting (#1026, #1033): scale=3 ceiling degeneration — EMA ratio approach AXIS CLOSED
- InstanceNorm across vol tokens (PR #1015): closed before terminal — vol-token LN (#1025) is successor

## Removed from Dead Ends (RETRACTED — Dataset Artifact)

The following were previously listed as dead ends but were tested on the broken pre-20260511 dataset split. They are **NOT confirmed dead ends** and some are now SOTA techniques:

- ~~SDF-stratified importance sampling (PR #972)~~: **NOW RANK 1 SOTA** (test_abupt=5.844%) — but was actually uniform sampling due to monkey-patch bug. True near-surface SDF is untested.
- ~~Stochastic vol subsampling (PR #968)~~: **NOW RANK 2 SOTA** (test_abupt=5.986%)
- WD=0.01 (PR #900): needs re-test on corrected split before concluding
- WD=0.005 (PR #914): needs re-test on corrected split before concluding
- TTA Y-symmetry (PR #979): may still be neutral; re-test on corrected split before concluding
- Vol coordinate noise (PR #990): EP5 gate FAIL — may be architecture-level; re-test on corrected split
- Bbox normalization (PR #978): may need re-test
- EMA decay=0.999 (PR #954): needs re-test on corrected split

_Last updated: 2026-05-12 ~22:30 UTC. MAJOR UPDATE: Discovered Python dunder no-op bug in types.MethodType monkey-patch — PR #972 "SDF-stratified" was actually uniform sampling. Also discovered SDF formula direction bug (far-field vs near-surface). All top-3 corrected-split results are uniform sampling. True near-surface SDF experiment is UNTESTED. Kill orders posted to fern (#1035) and tanjiro (#1055). Third escalation ping to frieren (#1025). Nezuko (#1054) formula direction question pending._
