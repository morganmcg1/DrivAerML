# SENPAI Research State
- **Date:** 2026-05-07 06:58 UTC (Round 13 mid-flight — all 8 students active, EP gates resolving across thorfinn/tanjiro/askeladd/fern/edward/nezuko/alphonse, frieren #802 launching)
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## SDF pipeline artefact finding (2026-05-07 06:38 UTC — PR #797 edward, MERGED)

**CRITICAL DATA FINDING**: The 4 OOD test cases (run_133, 158, 203, 226) that account for ~92% of squared test_vol_p deviation are NOT geometrically novel — they are **data-pipeline artefacts**. Their `volume_sdf.npy` was regenerated through a different sampling pipeline that omits inside-body samples. They are exactly the `REQUIRED_RESTORED_CASE_IDS` in `target/data/loader.py`.

- All 4 cases have `sdf_min` ∈ [-0.015, -0.001] vs bulk train [-0.45, -0.27]; `sdf_negative_frac` is ~10× lower.
- They cluster exclusively around the 6 restored train cases in 4D-Mahalanobis space.
- **Human issue #803 opened** requesting regeneration of `volume_sdf.npy` for 10 restored cases.
- **Implication for conditioning track**: FiLM/SDF-gate is extracting a bimodal pipeline indicator, not a physics signal. The conditioning experiments (#789, #802) are not invalidated but cannot fix OOD-4 until data is regenerated.
- Diagnostic artefacts merged to tay: `target/sdf_coverage_diagnostic.py`, `target/analysis/sdf_per_case_stats.csv`, `target/analysis/sdf_per_case_hists.npz`.

---

## Schedule-confound finding (2026-05-07)

**The 13-epoch budget is now structurally untestable at 270-min train cap.** Two consecutive PRs (#786 fern Anchor-RoPE, #788 nezuko surface-curvature) ran `--lr-cosine-t-max 13 --epochs 13` but only fit 4 effective epochs before wall-time. LR never cooled. Both produced budget-cut "best" checkpoints at EP4-partial that beat within-cluster controls on test but missed val SOTA bar by 0.18–0.32pp. Going forward: **all new architectural-comparison experiments must use `--epochs 4 --lr-cosine-t-max 4` with 4-ep vol-points curriculum `0:16384:1:32768:2:49152:3:65536` and rescaled kill thresholds.** PR #795 (nezuko) and PR #796 (fern) are the first two budget-aligned re-runs.

---

## Latest Human Researcher Directives

- **Issue #717** (vol_pressure gap): Phase 0 complete (PR #767). Phase 1 conditioning track **updated**: OOD-4 confirmed as pipeline artefacts (PR #797). Data-side fix requested (Issue #803). Conditioning experiments continue but data fix is the highest-leverage intervention.
- **Issue #803** (SDF regeneration): OPENED 2026-05-07 06:38Z. Request to regenerate `volume_sdf.npy` for 10 `REQUIRED_RESTORED_CASE_IDS` (run_44, 133, 158, 184, 203, 226, 249, 310, 416, 484) using standard inside-body sampling. Pending human team action.
- **Issue #618** (STRING/RoPE): PR #786 fern Anchor-STRING RoPE v3 13-ep CLOSED INCONCLUSIVE (schedule confound, val=6.92%); fern reassigned PR #796 same v3 architecture at 4-ep budget-aligned schedule. Thorfinn #779 running σ_max sweep.
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- No new directives pending.

---

## Current Baselines

| Tier | val_abupt | test_abupt | test_vol_pressure | PR | Notes |
|---|---|---|---|---|---|
| **Ensemble SOTA** | **6.1751%** | **7.5347%** | 11.4652% | #612 (nezuko) | K=7 greedy pool-24 |
| **Single-model SOTA** | **6.5985%** | **7.9915%** | 11.933% | #592 (alphonse) | depth-L5, EP4, run `4k25s25e` |
| **Vol-pressure best anchor** | — | — | 11.374% | #681 (dc031qpt) | Issue #717 reference |

**Key finding from #767:** Excluding the 4 OOD cases, test_vol_p = 3.9-4.2% (already below AB-UPT 6.08%). The entire vol_pressure gap is caused by 4 geometrically-OOD test cases. Geometry conditioning of the volume decoder is the highest-priority next intervention.

**Vol-pressure promotion ladder (test_vol_pressure target):**
- Weak: ≤11.0% | Solid: ≤10.0% | Major: ≤8.5% | Target: ≤6.08% (AB-UPT reference)

---

## Current Research Focus: Phase 1 Wave 2 — Volume Decoder Geometry Conditioning + Positional Encoding

8 students running Round 13 PRs. Two themes: (1) geometry conditioning on the volume path (Issue #717), (2) STRING/RoPE follow-ups (Issue #618). τ_z loss upweight experiment queued for next available student.

**Note on student pool:** STUDENT_NAMES = alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn (8 students). All 8 are now active with WIP PRs.

### Active PRs

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| frieren | #802 | AB-UPT geom-branch warmup-fix: geom-conditioning with fixed LR warmup | WIP (assignment picked up 06:23Z, run launching) |
| askeladd | #789 | Vol-decoder SDF-gate v4: decoupled gate LR=5e-5 fixed from backbone cosine | WIP (v4 run `ccnssij7` step 7,351 / EP1 gate ~07:18Z, ~67% to EP1) |
| alphonse | #801 | Anchor-STRING + RoPE stabilized: composing SOTA STRING PE with anchor-RoPE | WIP (run `zcbkv6vx` step 1,356 / 600s _runtime, early warmup) |
| fern | #799 | SOTA 4-ep schedule-aligned control: full SOTA stack `--epochs 4 --lr-cosine-t-max 4` | WIP (run `ddwymxzc` step 6,720, EP1 imminent at step 10,864) |
| thorfinn | #779 | STRING σ_max sweep: Arms A(σ=4.0)✅ B(σ=8.0)✅ C(σ=16.0, run `seuw5fsc`) RUNNING | WIP (Arm C EP2 PASS 8.539% — between A and B; non-monotonic σ_max signal flagged; EP3 gate <9.0% imminent) |
| nezuko | #798 | Surface curvature 4-ep aligned: H,K features on surface path with 4-ep budget | WIP (run `fphcg7b6` step 5,478 / EP1 gate ~07:30Z) |
| edward | #804 | GradNorm α=0.5 4-ep budget-aligned: pure CLI, ports dl24 winner (6.4170%) to short-track | WIP (run `1bkzxf38` launched 06:57Z, step 105 — just past warmup) |
| tanjiro | #793 | vol-w=2.0 + tau-y=2.5 + tau-z=3.0 — rebalance to recover val_abupt while keeping vol_p OOD win | WIP (EP3 7.81% ✅ continue band, run `ss5v4vdx` step 37,356 / 86% to EP4 — slope projects EP4≈5.81% if held) |

**Recently closed PRs:**
- **PR #788** (nezuko, surface curvature H,K on surface path): CLOSED INCONCLUSIVE (budget-limited) — best EMA EP4-partial val_abupt 6.7767% (+0.18pp vs SOTA), test_abupt 8.139% (−0.18pp vs within-cluster control), test_surf_p 4.168% (−0.14pp), test_wall_shear 7.4189% (−0.28pp). Hypothesis-discriminating signals supported on test. Schedule confound. Follow-up PR #795 assigned (4-ep budget-aligned).
- **PR #786** (fern, Anchor-STRING RoPE v3 13-ep): CLOSED INCONCLUSIVE — best EMA EP4-partial val_abupt 6.9197% (+0.32pp vs SOTA), test_abupt 8.1946% (−0.13pp vs control). Schedule confound. v3 architecture diagnostics healthy. Follow-up PR #796 assigned (4-ep budget-aligned).
- **PR #776** (tanjiro, vol-loss-weight sweep {1.5, 2.0}): CLOSED PARTIAL POSITIVE — Arm B (vol-w=2.0) beats SOTA test_vol_p by −0.37pp and shrinks val→test vol_p OOD gap by 0.67pp, but val_abupt regresses 0.62pp. Not merged. Wall-shear regression is the blocker. Follow-up PR #793 assigned.
- **PR #785** (askeladd, SDF-gate v2): CLOSED NEGATIVE (design) — bounded-tanh insufficient; 20× LR jump drove gate to full negative saturation. v3 follow-up assigned as PR #789.
- **PR #777** (alphonse, gc-loss-delayed-EP3): CLOSED NEGATIVE — test_vol_p=12.749% (worse than SOTA 11.933% and anchor 11.374%).
- **PR #781** (askeladd unbounded SDF-gate): CLOSED NEGATIVE (design) — unbounded blow-up; hypothesis intact; PR #785/789 are bounded follow-ups.
- **PR #775** (nezuko learnable affine anchor): CLOSED NEGATIVE — alpha collapsed toward zero.
- **PR #787** (stark τ_z sweep): CLOSED — 'stark' not in student pool; re-assigned to alphonse as PR #790.

### Long-run dl24 track results (2026-05-06 15:22 UTC)

**Three of four dl24 long runs are beating the current single-model SOTA (6.5985%):**

| Student | PR | Hypothesis | Epoch | val_abupt | vs SOTA |
|---|---|---|---|---|---|
| **dl24-fern** | **#740** | **GradNorm α=0.5 adaptive loss balancing** | **EP14/15** | **6.4170%** | **−0.1815pp BEATS SOTA** |
| dl24-nezuko | #741 | Y-axis symmetry augmentation | EP19.5 | 6.5052% | −0.0933pp BEATS SOTA |
| dl24-frieren | #745 | 5L STRING (model-layers 5 on SOTA base) | EP15.4 | 6.5097% | −0.0888pp BEATS SOTA |
| dl24-tanjiro | #749 | Lion lr=9e-5 control (pure CLI) | EP21 | 6.8557% | +0.2572pp worse |

> All three winners are mid-training (50-epoch budget). Need to complete and report final best-val checkpoint for BASELINE update.

### Key Diagnostic Findings

**GradNorm diagnostic (PR #758 tanjiro):**
- **wall_shear_z is the actual training laggard** (r_i=0.01123, weight=1.699, highest among all tasks)
- **vol_pressure is NOT undertrained** (r_i=0.00450, weight near floor) — second-fastest learner
- The vol_pressure val→test gap is a **geometry-OOD generalization problem**, not an undertrained-task problem

**Phase 0 diagnostic (PR #767 askeladd) — DECISIVE:**
- Same 4 test cases (run_133, run_226, run_203, run_158) account for **92%** of squared test_vol_p deviation
- Excluding them: test_vol_p = **3.9-4.2%** (below AB-UPT 6.08% for 46/50 cases)
- Surface encoder generalises fine; volume decoder specifically fails on these geometries

**Unbounded affine blow-up (PRs #770, #781):**
- Both FiLM v1 (frieren) and SDF-gate v1 (askeladd) failed via the same mechanism: outlier training case drives unbounded MLP into extreme regime (~1000× spike in 1 step)
- Root cause: unnormalized inputs spanning different orders of magnitude + no output clamping
- Solution (PRs #778, #785): bounded tanh + input normalization

---

## Completed Experiments This Round

- **PR #781** (askeladd): Unbounded SDF-gate — killed before EP1, unbounded blow-up on rank-7 outlier. CLOSED NEGATIVE (design). Hypothesis intact.
- **PR #783** (fern): Anchor-STRING RoPE v3 branch merged by human into `tay`; v3 code fixes (cosine schedule + Xavier init) now in base; no run completed. Full run assigned as PR #786.
- **PR #773** (edward): Surface curvature v2 (offline) — CLOSED NEGATIVE. H,K features degraded every test channel by 0.1-0.3pp. Hypothesis: SDF+coordinates already encode local geometry sufficiently; curvature is noise.

---

## Potential Next Research Directions

### Geometry conditioning — immediate priority (multiple in flight)

1. **FiLM v2 bounded+delayed** (#778 frieren): Bounded tanh γ∈(0,2) + EP6 onset. Sister to #785.
2. **SDF-gate v2 bounded** (#785 askeladd): Bounded tanh + input normalization. Same insert point as #781 but structurally stable.
3. **gc-loss delayed EP6** (#777 alphonse): Fire ∇p supervision only at V≥49k (EP6+). Temporal gating approach.
4. **SDF-FiLM on vol tokens** (#782 edward): SDF stats condition vol *tokens* (not vol_pred output). Different insertion point from #785.
5. **Learnable affine anchor** (#775 nezuko): Per-case global scalar correction from surface_cp lookup.

**If any geometry conditioning works:** compose with the best architecture (FiLM + SDF-gate, or with vol-head-2L from PR #761's truncated promising result).

### Architecture & positional encoding (in flight)

6. **Anchor-STRING RoPE v3 full run** (#786 fern): Definitive test at full budget with code fixes.
7. **STRING σ_max sweep** (#779 thorfinn): {4.0, 8.0, 16.0} — higher frequency ceiling.

### Loss weighting (in flight)

8. **τ_z targeted upweight** (#790 alphonse): {3.0, 4.0} — directly targets confirmed training laggard. Pure CLI, no code changes.
9. **Vol-loss-weight sweep** (#776 tanjiro): {1.5, 2.0, 2.5} — orthogonal to GradNorm.

### After geometry conditioning confirmed

10. **Compose best vol conditioning with best architecture**: Compound FiLM + 8-octave STRING + curvature features.
11. **Ensemble refresh**: After new single-model candidates emerge, re-run greedy pool selection (pool-25+).

### Other directions not yet tried

12. **Cross-case contrastive loss**: Force model to distinguish the 4 OOD test cases from normal cases at training time.
13. **Surface-only curvature features**: Edward's suggestion — append H,K to surface input dim (not volume). Current infrastructure on disk (`/mnt/new-pvc/Processed/drivaerml_curvature_v2_edward/`).
14. **Surface feature → volume cross-attention**: Direct attention from volume queries to surface keys.

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128 (~15.9M params, SOTA config)
- **Positional encoding:** STRING-separable (rff_num_features=16, sigmas 0.25-4.0, 5-octave)
- **Optimizer:** Lion, lr=9e-5, β2=0.99
- **Weight decay:** 5e-4
- **QK-norm:** enabled
- **Loss weights:** tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0
- **EMA:** 0.999
- **Training budget:** ~270 min (SENPAI_TIMEOUT_MINUTES=360 with 90 min val reserve)
- **Vol-points curriculum:** `0:16384:3:32768:6:49152:9:65536` (16k→65k across epochs 0/3/6/9)
- **VOLUME_X_DIM=4:** (x, y, z, sdf) where channel 4 is precomputed SDF
- **Heads constraint:** heads MUST be power-of-2 for SDPA/Triton fast paths
- **Key finding:** GradNorm shows wall_shear_z is actual training laggard; vol_pressure gap is generalization/distribution-shift, not undertrained-task
- **Key finding (#767):** Val→test gap in vol_pressure is 100% case-dominated: 4 OOD geometries (run_133/226/203/158) account for 92% of squared deviation; excluding them, test_vol_p = 3.9-4.2% (below AB-UPT)
- **Key finding (#781, #770):** Unbounded affine geometry conditioning always fails via single-case outlier blow-up; bounded tanh + input normalization is required for stable training
