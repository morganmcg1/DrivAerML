# SENPAI Research State
- 2026-05-13 ~00:30 UTC

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
| 1 | **#972** | **5.844%** | 3.643% | Uniform sampling (monkey-patch no-op) — CLOSED |
| 2 | #968 | 5.986% | 3.957% | Stochastic vol subsampling — CLOSED |
| 3 | #880 | 6.010% | 4.501% | — CLOSED |
| 4 | #958 | 6.107% | 3.818% | — CLOSED |
| 5 | #939 | 6.242% | — | — CLOSED |
| 22 | #740 | 8.165% | 13.660% | Old "SOTA" — NOW ARTIFACT |

## Active Experiments (2026-05-13 ~00:30 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #1050 | dl24-edward | **SDF-stratified near-surface sampling (DANN)** | `nc7lpobi` | **RUNNING** | EP12 val_abupt=6.71% | EBS=8 (bs=1, DDP8). Steps/epoch ~10,974. 30 epochs total. |
| #1052 | dl24-thorfinn | **DANN adaptive domain normalization** | Arm B: `yli6kbch` | **ARM B RUNNING** | — | EBS=32 (bs=4, DDP8). |
| #1054 | dl24-nezuko | **SDF-stratified (α=2.0) + Stochastic combined** | `yd8n1whr` | **RUNNING** | EP3=8.2002% (marginal miss, continuation granted) | Inverse formula + `__class__` reassignment confirmed. replacement=False. EBS=8. |
| #1055 | dl24-tanjiro | **SDF-stratified α sweep (α=1.5 → 3.0 → 4.0)** | Arm A: `58hk6r36` | **ARM A RUNNING** | EP1=22.9680% (✓ ≤30%) | Both bugs fixed: `__class__` reassignment + inverse formula. CORRECTION: `<=` IS correct kill operator (not `>=`). Arm A (α=1.5) live. |
| #1057 | dl24-fern | **Log-space vol_p loss (tay branch)** | TBD | **AWAITING RUN START** | — | Tay-compatible command posted (comment 4435673160). EBS=32. |
| #1062 | dl24-frieren | **SDF near-surface combined vol+surf sampling (α=2.0)** | TBD | **ASSIGNED** | — | New assignment. Apply inverse SDF weighting to BOTH volume AND surface point sampling simultaneously. DL24 branch. EBS=8. |

## CRITICAL: Kill-Threshold Operator Note

**For DL24-branch students (nezuko, tanjiro, frieren):**
The correct operator is `<=` as written in the kill-threshold string. The harness `passes()` function returns `True` when `observed <= threshold`; it kills when `not passes()` (i.e., metric still too high). `<=` means "pass/keep if metric is at or below threshold."

**For tay-branch students (fern) with EBS=32:**
Step-epoch mapping: 87,794 ÷ 32 = 2,743 steps/epoch.

**Step-to-epoch mapping (EBS=8, bs=1, DDP8):** 87,794 ÷ 8 = 10,975 steps/epoch
- EP1=10,975, EP2=21,950, EP3=32,925, EP5=54,875, EP10=109,750, EP15=164,625, EP20=219,500, EP25=274,375, EP30=329,250

**Kill gates for DL24 EBS=8:**
`10975:val_abupt<=30,21950:val_abupt<=16,32925:val_abupt<=8,54875:val_abupt<=7.5,109750:val_abupt<=7.2,164625:val_abupt<=6.80,219500:val_abupt<=6.70,274375:val_abupt<=6.65,329250:val_abupt<=6.60`

## Branch Architecture: tay vs DL24

Two incompatible branch architectures exist. Flag sets are NOT interchangeable:

**tay branch flags:**
- `--pos-encoding-mode string_separable`
- `--rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"`
- `--rff-num-features 16`
- `--vol-p-log-space-loss`
- `--gradnorm-mode ema_proxy`
- `--vol-points-schedule "1:65000"`

**DL24 branch flags:**
- `--model-pe string_multisigma`
- `--pe-init-sigmas "0.25,0.5,1.0,2.0,4.0"`
- `--model-layers 6`
- `--batch-size 1`

**Never mix flags across branches.** Sending DL24 flags to tay-branch students causes `unrecognized arguments` errors.

## Canonical Corrected SDF Monkey-Patch (DL24 branch)

```python
if args.use_sdf_stratified_vol_sampling:
    _ALPHA = args.sdf_stratified_alpha
    _N_VOL = args.train_volume_points

    class _SDFStratifiedDataset(type(train_dataset)):
        def __getitem__(self, idx):
            view = self.views[idx]
            counts = self.store.case_point_counts(view.case_id)
            surface_idx = self._indices(
                counts["n_surface"], self.max_surface_points, view,
                group_view_count=view.surface_view_count,
            )
            case = self.store.load_case(
                view.case_id,
                surface_rows=None if surface_idx is None else surface_idx.numpy(),
                volume_rows=None,
            )
            n_total = case.volume_x.shape[0]
            n_sample = min(self._sdf_n_vol, n_total)
            sdf_vals = case.volume_x[:, 3].abs()
            weights = 1.0 / (1.0 + self._sdf_alpha * sdf_vals)  # INVERSE = near-surface
            vol_idx = torch.multinomial(weights, n_sample, replacement=False)
            vol_idx = vol_idx.sort().values
            from data.loader import DrivAerMLCase
            return DrivAerMLCase(
                case_id=case.case_id,
                surface_x=case.surface_x, surface_y=case.surface_y,
                volume_x=case.volume_x[vol_idx], volume_y=case.volume_y[vol_idx],
                metadata={},
            )

    train_dataset._sdf_alpha = _ALPHA
    train_dataset._sdf_n_vol = _N_VOL
    train_dataset.__class__ = _SDFStratifiedDataset
    assert type(train_dataset).__name__ == '_SDFStratifiedDataset', 'SDF patch not active'
    print(f'[SDF] inverse near-surface vol sampling enabled: alpha={_ALPHA}, n_vol={_N_VOL}')
```

**THREE required elements:**
1. `__class__` reassignment (NOT `types.MethodType` — that's a Python dunder no-op)
2. Inverse formula `1/(1+α·|sdf|)` (NOT `1+α·|sdf|` — that upweights far-field)
3. `replacement=False` in `torch.multinomial` (prevents duplicate point sampling)

## Key Insights (Post-Artifact-Resolution)

1. **The val→test vol_p gap was entirely artificial.** Under `rawcanon_20260511`, test_vol_p tracks val_vol_p closely (~3.6–4.0% range).
2. **PR #972 "SDF-stratified" = UNIFORM sampling.** The monkey-patch was a Python dunder resolution no-op. All top-3 corrected-split results used uniform sampling.
3. **First true SDF near-surface experiment is nezuko (#1054).** Run `yd8n1whr` live with correct inverse formula + `__class__` reassignment + `replacement=False`. This is a virgin experimental axis.
4. **Stochastic vol subsampling (PR #968) is independently confirmed #2.** Fresh random draw every batch is a real technique. test_abupt=5.986%.
5. **Near-surface SDF hypothesis is ESSENTIALLY UNTESTED until now.** Given that PR #972's apparent SOTA was actually uniform, true near-surface SDF emphasis may push vol_p well below 3.643%.
6. **frieren is now assigned PR #1062**: combined vol+surf SDF weighting — novel axis not tested by any other student.

## Tier 1 Follow-Ups (Current Priority)

1. **True near-surface SDF sampling vol-only** — nezuko (#1054) `yd8n1whr` and tanjiro (#1055) Arm A `58hk6r36`. Primary open question.
2. **Combined vol+surf SDF weighting** — frieren (#1062). Novel axis.
3. **Log-space vol_p loss** — fern (#1057) on tay branch. Direct attack on primary metric.

## Tier 2 Follow-Ups (After Tier 1 Results)

4. **SDF-stratified + dedicated vol decoder (6L vol tower)** — combine near-surface SDF sampling with independent vol tower architecture.
5. **Re-run vol-token LN (PR #1025 config) from scratch on corrected split** — showed val_vol_p=3.547% on old dataset; needs clean re-run.
6. **WD=0.005 re-test on corrected split** — previously falsified but could benefit from clean evaluation.
7. **EMA decay=0.999 re-test on corrected split** — previously falsified but could benefit from clean evaluation.

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

## Removed from Dead Ends (RETRACTED — Dataset Artifact)

- ~~SDF-stratified importance sampling (PR #972)~~: **NOW RANK 1 SOTA** (test_abupt=5.844%, but this was actually uniform)
- ~~Stochastic vol subsampling (PR #968)~~: **NOW RANK 2 SOTA** (test_abupt=5.986%)
- WD=0.01 (PR #900): needs re-test on corrected split
- WD=0.005 (PR #914): needs re-test on corrected split
- TTA Y-symmetry (PR #979): may still be neutral; re-test needed
- Vol coordinate noise (PR #990): EP5 gate FAIL — may be architecture-level; re-test needed
- Bbox normalization (PR #978): may need re-test
- EMA decay=0.999 (PR #954): needs re-test on corrected split

_Last updated: 2026-05-13 ~00:30 UTC. frieren assigned PR #1062 (SDF combined vol+surf, α=2.0). Correction posted to PR #1055 (tanjiro kill-operator error). Nezuko EP3 marginal miss granted continuation. All 4 students active: edward/thorfinn/nezuko/tanjiro running, fern awaiting run start, frieren newly assigned._
