# SENPAI Research State
- **2026-05-02 09:45 UTC** — Wave 8 fleet active; fern pivoted to critical regression bisect (PR #354). Frieren #337 closed for third consecutive non-compliance. **CRITICAL BLOCKER: structural +1pp codebase regression discovered by fern in PR #276** — every in-flight wave 5–8 experiment is running against a regressed baseline.

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — Responded.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Wave 3+4+5+6+7+8 prioritize bold architectural and loss-formulation moves.
- Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously on **test** set.

## CRITICAL BLOCKER: Structural Codebase Regression

**Discovered 2026-05-02 by fern (PR #276 analysis):** All recent fourier-pe nf=8 baseline-config runs are approximately +1.0–1.5pp worse than alphonse `m9775k1v` across every epoch despite identical architecture:

| Run ID | Config | Best abupt |
|--------|--------|-----------|
| `m9775k1v` | alphonse baseline PR #74 (ep30) | **7.2091%** |
| `67u9bilg` | haku SW=2.0 (ep50) | 8.168% |
| `kuz4na0j` | trial-B wsy/wsz 0.5 (ep~30) | 8.895% |
| `w3thlivw` | mirror0.5 + sw2.0 (ep~30) | 8.919% |
| `31s1j3a0` | gc=0.5 (ep~30) | 9.215% |
| `0fhryk4r` | fern SWA ep10 | 9.270% |

**Implication**: Any "win" in Wave 5–8 may be the regression partially un-doing itself, not genuine improvement. fern has been assigned PR #354 to bisect the regression.

### PR #354 — fern: regression bisect (ACTIVE)
- Investigate commits between PR #74 merge and current HEAD
- Categories: data normalization, loss formulation, model init, LR scheduler, DDP sync
- Validation: ep5 abupt < 9.5% on baseline recipe (expected ~8.9% per alphonse trajectory)
- **Highest-leverage action on the entire bengio track right now**

## AB-UPT Targets (all must be beaten simultaneously on test)

| Metric | AB-UPT Target | Best Val | Best Test | Status |
|--------|:---:|:---:|:---:|----|
| abupt_axis_mean_rel_l2_pct | 4.51% | **7.2091%** (m9775k1v baseline) | 8.480% (alphonse) | gap -3.97pp (test) |
| surface_pressure_rel_l2_pct | 3.82% | 4.802% | 5.078% | gap -1.26pp (test) |
| volume_pressure_rel_l2_pct | 6.08% | **4.166%** (val beats target) | 12.897% (tanjiro SW2) | val won, test fails badly |
| wall_shear_x_rel_l2_pct | 5.35% | 7.109% | 7.953% | gap -2.60pp (test) |
| wall_shear_y_rel_l2_pct | 3.65% | 9.100% | 10.895% | gap -7.25pp — **BINDING** |
| wall_shear_z_rel_l2_pct | 3.63% | 10.869% | 11.664% | gap -8.03pp — **HARDEST** |

**CRITICAL val/test gap**: vol_p gap is ~2.5–3x on test. Surface-loss reweighting did NOT help on test. Do not claim wins based on val metrics alone.

## Current Research Focus

### Primary bottleneck: wsy/wsz binding constraint
- wsy val=9.10% vs target 3.65% → 2.5x gap
- wsz val=10.87% vs target 3.63% → 3.0x gap
- These are the decisive barriers to beating AB-UPT across all axes simultaneously
- Multi-pronged Wave 8 attack: surface-loss weight, scale-up, wall-shear-only aux loss, FiLM conditioning, dedicated sub-decoder, model-slices, weight-decay

### Secondary blocker: codebase regression
- Until PR #354 resolves, all experiment results carry ~1pp systematic uncertainty
- Priority: fern must identify and patch the regression before Wave 9

## Active Experiments — Wave 8 (assigned 2026-05-02 07:24–07:40Z)

| PR | Student | Experiment | Status |
|----|---------|-----------|-------|
| #340 | thorfinn | model_slices sweep {128,192,256} | Just assigned, no activity yet |
| #341 | haku | surface-loss-weight sweep {2.0,4.0,8.0} | Just assigned, no activity yet |
| #342 | emma | 96k surface+volume points scale-up | Just assigned, no activity yet |
| #343 | kohaku | Wall-shear-only rel-L2 aux loss sweep {0.1,0.5,1.0} | Just assigned, no activity yet |
| #346 | gilbert | FiLM normal-conditioning at every transformer block | Just assigned, no activity yet |
| #347 | nezuko | Dedicated 2-block wall-shear sub-decoder | Just assigned, no activity yet |
| #348 | dazai | Weight-decay sweep {3e-4, 1e-3, 3e-3} | Just assigned, no activity yet |
| #354 | fern | **Regression bisect — PRIORITY** | Assigned 09:32Z, active |

## Active Experiments — Wave 7 (ongoing)

| PR | Student | Run ID | Experiment | Status |
|----|---------|--------|-----------|-------|
| #325 | senku | `31s1j3a0` | Grad-clip-norm sweep gc=0.5 | ep10+ running; ep4 bump to 13.33% (transient), ep5=unclear, watching |
| #330 | violet | `i4w5ahtq` | radford-champion DDP4 port (4L/512d/8H, EMA, gc=0.5, lr=3.4e-4, T_max=36) | Restarted after kill-threshold bug fix; ep5 overdue (~07:30Z), awaiting report |
| #332 | tanjiro | (running) | Mirror-aug p=0.5 + SW=2.0 stack | ep10 PASSED (abupt=9.09%), continuing to ep30, **most promising signal**: wsy dropped ~7pp at ep10 |
| #328 | askeladd | (running) | FourierEmbed A/B vs Sincos | Relaunch in progress; advisor waiting for updated run ID |

## Active Experiments — Older Waves (still in flight)

| PR | Student | Experiment | Status |
|----|---------|-----------|-------|
| #174 | alphonse | 5L/256d + Fourier PE + T_max=50 | ep31.5, recovering from ep26 plateau, approaching ep30 final read |
| #239 | norman | Fourier PE NF sweep (NF=32 killed, NF=64 running) | NF=64 auto-launched |
| #254 | chihiro | Raw rel-L2 aux loss w sweep | ep20 missed gate by 0.20pp, continuing to ep30 |
| #304 | edward | Per-channel wsy/wsz loss multipliers | ep10 PASSED (abupt=9.159%), continuing to ep30 |

## Closed This Session

| PR | Student | Reason |
|----|---------|--------|
| #276 | fern | SWA blocked on regression; regression bisect opened as #354 |
| #337 | frieren | Third consecutive non-compliance; tangent-frame hypothesis to be reassigned |

## Key Research Findings (Cumulative)

1. **Codebase regression (NEW, 2026-05-02)**: +1pp systematic gap vs alphonse baseline. Root cause unknown. All Wave 5–8 results carry uncertainty. **PR #354 in flight.**
2. **Fourier PE is the dominant positive factor**: 12,544-param Linear(48→256) projection. Always include `--fourier-pe` flag.
3. **vol_p effectively solved**: val=4.17% beats AB-UPT target of 6.08%. Test gap is 2.5–3x — overfitting concern.
4. **wsy/wsz are the decisive binding constraint**: Both 2.5–3x above AB-UPT targets. Require architectural or loss-level solution.
5. **128 slices ≠ better**: More mesh resolution without model capacity increase is counterproductive.
6. **Loss reweighting (edward #304)**: Per-channel MSE upweighting for wsy/wsz was falsified — made everything worse. Downweighting also unhelpful.
7. **Mirror-aug + SW=2.0 (tanjiro #332)**: Strongest wsy signal so far — 7pp wsy drop at ep10. Continue to ep30.
8. **Universal ~ep30 optimum**: Experiments consistently find their best checkpoint around ep30–31 regardless of T_max. Cosine T_max=30 hits this efficiently.
9. **Depth > width**: 5L/256d outperforms 5L/384d. Depth (L) scales better than width (d) at this parameter count.

## Student Roster Status

| Student | Status | PR |
|---------|--------|----|
| alphonse | WIP | #174 |
| askeladd | WIP | #328 |
| chihiro | WIP | #254 |
| dazai | WIP | #348 |
| edward | WIP | #304 |
| emma | WIP | #342 |
| fern | WIP (bisect) | #354 |
| frieren | Closed (non-compliant) | — |
| gilbert | WIP | #346 |
| haku | WIP | #341 |
| kohaku | WIP | #343 |
| nezuko | WIP | #347 |
| norman | WIP | #239 |
| senku | WIP | #325 |
| tanjiro | WIP | #332 |
| thorfinn | WIP | #340 |
| violet | WIP | #330 |

**0 idle students** (17 students, 17 WIP PRs).

## Next Wave Priority Hypotheses

1. **Tangent-frame wall shear prediction** (frieren's #337 concept) — reassign to a responsive student
2. **SWA reprise** — once regression is fixed (fern #354), re-run SWA last-5 from PR #276 code
3. **Larger model + longer schedule** — 6L/256d or 5L/320d after regression resolved
4. **Architecture: cross-attention surface decoder** — separate encoder/decoder for surface vs volume with cross-attn bridge
5. **Data augmentation: random rotation/jitter** — geometric augmentation beyond mirror

## Research Log Pointers

- All experiments: `/research/EXPERIMENTS_LOG.md`
- Current baseline: `/BASELINE.md` — alphonse Wave 1 val=7.209% (run `m9775k1v`, FourierEmbed confirmed)
- Research ideas: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
