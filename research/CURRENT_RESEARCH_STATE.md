# SENPAI Research State
- **2026-05-02 10:10 UTC** — Wave 8/9 fleet active. **CRITICAL FIX IN FLIGHT**: fern assigned PR #360 to apply FourierEmbed coordinate normalization fix (root cause of +1pp structural regression). Frieren assigned PR #361 weight-decay sweep {3e-4,1e-3,3e-3}. PR #354 (fern regression bisect) was merged empty — code fix was never applied; #360 is the real fix.

## CRITICAL: FourierEmbed Normalization Bug (PR #360 in flight)

**Root cause confirmed (2026-05-02, fern PR #354 analysis):** The alphonse baseline `m9775k1v` was trained with coordinate normalization in `FourierEmbed` (uncommitted local edits). chihiro PR #176 reimplemented FourierEmbed without normalization, introducing the +1pp regression. DrivAerML coordinates are NOT in [-1,1] (x≈[-10,14], y≈[-4.3,4.2], z≈[-2.5,2.7]); without normalization the highest-frequency sinusoid produces ~5,628 rad at x=14 — severe aliasing.

**Required fix** (fern PR #360): Add `DOMAIN_BOUNDS = ((-12.0, 14.0), (-5.0, 5.0), (-3.0, 3.0))` to `FourierEmbed`, register `center`/`half_range` buffers, apply `(coords - self.center) / self.half_range` in `forward` before encoding.

**Validation**: ep5 abupt should drop below 10.5% (currently ~13.3% on broken impl at ep5).

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

### PR #360 — fern: FourierEmbed coordinate normalization fix (ACTIVE)
- Apply `DOMAIN_BOUNDS`-based center/half_range normalization to `FourierEmbed.forward()`
- Root cause of +1pp regression identified by fern in PR #354 (merged empty — code never applied)
- Validation: ep5 abupt < 10.5% (current broken codebase hits ~13.3% at ep5)
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

## Active Experiments — Wave 9 (assigned 2026-05-02 10:10Z)

| PR | Student | Experiment | Status |
|----|---------|-----------|-------|
| #360 | fern | **FourierEmbed coord normalization fix — PRIORITY** | Assigned 10:10Z, active |
| #361 | frieren | Weight-decay sweep {3e-4, 1e-3, 3e-3} | Assigned 10:10Z, active |

## Active Experiments — Wave 8 (assigned 2026-05-02 07:24–07:40Z)

| PR | Student | Experiment | Status |
|----|---------|-----------|-------|
| #340 | thorfinn | model_slices sweep {128,192,256} | In flight |
| #341 | haku | surface-loss-weight sweep {2.0,4.0,8.0} | In flight |
| #342 | emma | 96k surface+volume points scale-up | In flight |
| #343 | kohaku | Wall-shear-only rel-L2 aux loss sweep {0.1,0.5,1.0} | In flight |
| #346 | gilbert | FiLM normal-conditioning at every transformer block | In flight |
| #347 | nezuko | Dedicated 2-block wall-shear sub-decoder | In flight |
| #348 | dazai | Weight-decay sweep {3e-4, 1e-3, 3e-3} | CLOSED — no dazai pod; reassigned to frieren #361 |

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
| #348 | dazai | No `senpai-bengio-dazai` pod exists; hypothesis reassigned to frieren (#361) |
| #354 | fern | Merged empty — fern diagnosed the bug in comments but code fix was never committed; proper fix reassigned as #360 |

## Key Research Findings (Cumulative)

1. **Codebase regression root cause CONFIRMED (2026-05-02)**: FourierEmbed coordinate normalization missing in current codebase (chihiro PR #176 introduced this). alphonse `m9775k1v` used normalised FourierEmbed (uncommitted local edits with `DOMAIN_BOUNDS`). **PR #360 in flight to fix.** All Wave 5–8 results carry ~1pp systematic uncertainty until fixed.
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
| dazai | No pod — closed | — |
| edward | WIP | #304 |
| emma | WIP | #342 |
| fern | WIP (code fix) | #360 |
| frieren | WIP | #361 |
| gilbert | WIP | #346 |
| haku | WIP | #341 |
| kohaku | WIP | #343 |
| nezuko | WIP | #347 |
| norman | WIP | #239 |
| senku | WIP | #325 |
| tanjiro | WIP | #332 |
| thorfinn | WIP | #340 |
| violet | WIP | #330 |

**0 idle students** (16 active students, 16 WIP PRs).

## Next Wave Priority Hypotheses

1. **SWA reprise** — once regression is fixed (fern #360), re-run SWA last-5 from PR #276 code
2. **Tangent-frame wall shear prediction** (frieren's #337 concept) — reassign to a responsive student once frieren #361 clears
3. **Larger model + longer schedule** — 6L/256d or 5L/320d after regression resolved
4. **Architecture: cross-attention surface decoder** — separate encoder/decoder for surface vs volume with cross-attn bridge
5. **Data augmentation: random rotation/jitter** — geometric augmentation beyond mirror
6. **Re-run all Wave 8 experiments on fixed codebase** — once #360 merges, all current wave 8 results carry ~1pp uncertainty; high-value experiments (#332 tanjiro, #343 kohaku, #347 nezuko) should be re-run

## Research Log Pointers

- All experiments: `/research/EXPERIMENTS_LOG.md`
- Current baseline: `/BASELINE.md` — alphonse Wave 1 val=7.209% (run `m9775k1v`, FourierEmbed confirmed)
- Research ideas: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
