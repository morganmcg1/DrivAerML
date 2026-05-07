# SENPAI Research State
- **Date:** 2026-05-07 ~15:20 UTC (nezuko #816 cross-case contrastive vol_p loss ASSIGNED. All 8 students active. PR #807 thorfinn τ_z×3.0 4-ep CLOSED — hypothesis confirmed; spawned #815 13-ep promotion. New 13-ep promotion PRs in flight: #813 tanjiro vol-w=2.0, #814 alphonse STRING 6-octave, #815 thorfinn τ_z×3.0. frieren #802 agent communication failure — `murzmdxl` healthy at step 30,781 (EP3.2) but no posts since 12:03Z.)
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current Baselines

| Tier | val_abupt | test_abupt | test_vol_pressure | PR | Notes |
|---|---|---|---|---|---|
| **Ensemble SOTA** | **6.1751%** | **7.5347%** | 11.4652% | #612 (nezuko) | K=7 greedy pool-24 |
| **Single-model SOTA** | **6.5985%** | **7.9915%** | 11.933% | #592 (alphonse) | depth-L5, EP4, run `4k25s25e` |
| **Vol-pressure best anchor** | — | — | 11.374% | #681 (dc031qpt) | Issue #717 reference |

**Key finding from #767:** 4 OOD test cases (run_133, run_226, run_203, run_158) account for 92% of squared test_vol_p deviation. Excluding them, test_vol_p = 3.9-4.2% (below AB-UPT 6.08%). Geometry conditioning of the volume decoder is the highest-priority intervention.

**Vol-pressure promotion ladder (test_vol_pressure target):**
- Weak: ≤11.0% | Solid: ≤10.0% | Major: ≤8.5% | Target: ≤6.08% (AB-UPT reference)

---

## Latest Human Researcher Directives

- **Issue #618** (STRING/RoPE): All four experiments must run. Iterate on the two most recoverable STRING/RoPE directions with at least 2 students. Exps 3 (Anchor-STRING stabilized) and 4 (AB-UPT geometry branch v2 warmup fix) are now assigned.
- **Issue #717** (vol_pressure gap): Phase 0 diagnostic COMPLETE (PR #767). Phase 1 continues — geometry conditioning on volume decoder path. SDF-gate v3 (PR #789) closed DESIGN-NEGATIVE; additive LoRA approach now in flight.
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- No new directives pending.

---

## Latest Closeouts

- **PR #808 (nezuko, surface-curvature 4-ep orig-schedule)**: CLOSED DEAD END. 3rd consecutive curvature run (PRs #788, #798, #808) all in 6.78–7.35% range — 0.20–0.21pp above SOTA gate. Only signal: test_vol_p marginally better (~11.81% vs 11.933%). Surface curvature as standalone L=5 augment is exhausted.
- **PR #807 (thorfinn, τ_z×3.0 schedule-aligned 4-ep)**: CLOSED — hypothesis confirmed. τ_z×3.0 beats τ_z×2.0 (bare-SOTA control) at equal 4-ep compute on every channel. Did not beat 13-ep SOTA gate (val_abupt=6.824% > 6.5985%). Spawned PR #815 for 13-ep promotion.
- **PR #809 (askeladd, vol-head LoRA r=4/r=8)**: CLOSED — student idle 90+ minutes. Reassigned as PR #812.
- **PR #793 (tanjiro, vol-w=2.0+tau-y=2.5+tau-z=3.0)**: CLOSED NEGATIVE. Multiple simultaneous channel up-weights degraded all channels.
- **PR #792 (frieren, FiLM v3)**: CLOSED DESIGN-NEGATIVE. γ_max saturates at tanh bound 100% from EP4. FiLM γ∈(0,2) capacity is the bottleneck.
- **PR #790 (alphonse, τ_z upweight sweep)**: Previously CLOSED.

---

## Active PRs (8 WIP)

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| askeladd | #812 | Additive LoRA on vol output head: r=4 (Arm A), r=8 (Arm B) — retry of #809 | WIP — Arm A `6m7vw0tw` EP1=26.60% (passed), EP2 in progress |
| alphonse | #814 | STRING 6-octave extended spectrum (add σ=8.0) — full 13-ep run | WIP — `3efn3v5u` launched 13:49Z, EP1 in progress |
| frieren | #802 | AB-UPT geometry branch v2: backbone freeze 20% warmup + 2× geom LR + vol_p aux weight 2.0 | WIP — v2 relaunch `murzmdxl` ALIVE at step 30,781 (EP3.2), val_abupt 53.44% with −3.17%/1k slope; **AGENT COMMUNICATION FAILURE — silent since 12:03Z despite escalations** |
| fern | #811 | L6 depth scale: does adding a 6th transformer layer reduce OOD vol_p gap? | WIP — `9yrr5j8f` launched 11:52Z, EP1=25.61% (passed), EP2 in progress |
| thorfinn | #815 | τ_z×3.0 promotion to 13-ep full SOTA schedule | WIP — relaunch `y862359i` with relaxed gates (~14:39Z) |
| nezuko | #816 | Cross-case contrastive loss on vol_p embeddings — triplet margin on 4 OOD cases (run_133, run_226, run_203, run_158), λ=0.05, 4-ep diagnostic | WIP — assigned 2026-05-07 |
| edward | #810 | GradNorm α=2.0 4-ep run (dynamic loss weighting) | WIP — `6kv9hzuh` EP1=27.48%, EP2=12.26%; EP3/EP4 pending |
| tanjiro | #813 | vol-w=2.0 full 13-ep schedule-aligned (SOTA gate) | WIP — `4o9wamsr` launched 12:50Z, EP1=29.61% (val_vol_p **better** −1.51pp) |

---

## Current Research Focus

Two parallel themes:

### Theme 1: Geometry Conditioning on Volume Decoder (Issue #717)
- **Additive vol-head LoRA r=4/r=8** (askeladd, #812 retry): Replaces tanh-cap SDF-gate line (v2/v3/v4 all saturated). Additive rank-r correction on `volume_out`, zero-init B, no saturation risk. SDF info already in volume_hidden (VOLUME_X_DIM=4). Arm A r=4, Arm B r=8.
- **AB-UPT geometry branch v2 relaunch** (frieren, #802): v1 was killed by wrong-polarity kill gate, but EP4 before kill showed val_vol_p=10.896% (below anchor 11.374%). v2 relaunch with corrected `<` polarity gates. Budget concern: ~6h timeout may only reach EP5-6 from scratch.
- **FiLM** and **tanh-cap SDF-gate** tracks both closed as design-negative.

### Theme 2: Positional Encoding / Architecture (Issue #618)
- **Anchor-STRING RoPE stabilized** (alphonse, #801): Full 13-epoch run with diff LR (0.1× on rope params), per-module grad clip 1.0, freq init 0.1-10.0. EP1 passed (29.6%). Log-freq diagnostics: barely moved from init (std=1.406≈init).
- **L6 depth scale** (fern, #811): Does L6 continue the L4→L5 trend (−1.90% relative val_abupt)? No launch confirmation yet.
- **Schedule alignment baseline** (fern, #799): Clean 4-ep SOTA control with `--epochs 4 --lr-cosine-t-max 4`.

### Theme 3: Loss / Supervision (secondary)
- **τ_z×3.0 → 13-ep promotion** (thorfinn, #815): In flight. Confirmed 4-ep hypothesis (PR #807) — now testing full 13-ep SOTA schedule. Will it beat single-model gate 6.5985%?
- **GradNorm α=2.0** (edward, #810): Dynamic per-task gradient normalization with α=2.0. Pure diagnostic run to test whether adaptive loss weighting can close the tau_z lag.
- **Vol-w=2.0 13-ep promotion** (tanjiro, #813): Vol-w=2.0 single-variable isolation confirmed at 4-ep (#805, EP3=7.47%). Now testing full 13-ep SOTA schedule. EP1=29.61% (val_vol_p better −1.51pp).
- **Surface curvature (L=5)**: EXHAUSTED after PRs #788, #798, #808. All landed 6.78–7.35%, 0.20–0.21pp above gate. Not pursuing further at this architecture depth.

---

## Key Diagnostic Findings Established

- **Wall shear z is confirmed training laggard** (PR #758): r_i=0.01123, weight=1.699, highest among all tasks. Vol_pressure is NOT undertrained — gap is OOD generalization.
- **4 OOD test cases dominate vol_pressure** (PR #767): 92% of squared deviation. Excluding them, test_vol_p = 3.9-4.2% (below AB-UPT). Geometry conditioning is the right lever.
- **FiLM mechanism works but γ∈(0,2) is a capacity bottleneck** (PR #792): γ_max saturates 100% from EP4. More FiLM-active steps ≠ better metrics in current parameterization.
- **AB-UPT geometry branch compresses OOD gap 3.17×→2.07×** (PR #626): Architecture has real signal for generalization even when aggregate accuracy regresses. Warmup fix + v2 relaunch should resolve training efficiency.
- **EP4 geom-branch v2 val_vol_p=10.896%** (PR #802 frieren, before wrong-polarity kill): Below anchor 11.374%. Warmup architecture is producing the predicted geometry-conditioning benefit.
- **Anchor-STRING RoPE architecture converging at EP4 budget cutoff** (PR #786): val_abupt=6.92% at cutoff with monotone descent. Stabilization fixes (diff LR, grad clip, freq init) are the next lever for full 13-epoch run.
- **Schedule alignment is a confounder** (PR #805 tanjiro, EP3 evidence): vol-w=2.0 regression collapses from +1.79pp at EP1 to +0.035pp at EP3 under 4-ep aligned schedule. High-LR cutoff in 13-ep runs explains most of the apparent regression in prior paired tests.
- **τ_z upweight paradox at EP2** (PR #807 thorfinn): τ_z×3.0 val_tau_z=11.866% vs bare-SOTA 10.96% (+0.91pp) — the upweight isn't helping tau_z at EP2. Need EP3/EP4 to see if effect emerges at lower LR.

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128 (~15.9M params, SOTA config)
- **Positional encoding:** STRING-separable (rff_num_features=16, sigmas 0.25-4.0, 5-octave)
- **Optimizer:** Lion, lr=9e-5, β2=0.99
- **Weight decay:** 5e-4
- **QK-norm:** enabled
- **Loss weights:** tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0
- **EMA:** 0.999
- **Training budget:** ~270 min (SENPAI_TIMEOUT_MINUTES=270)
- **Vol-points curriculum:** `0:16384:3:32768:6:49152:9:65536` (16k→65k across epochs 0/3/6/9)
- **VOLUME_X_DIM=4:** (x, y, z, sdf) where channel 4 is precomputed SDF
- **Heads constraint:** heads MUST be power-of-2 for SDPA/Triton fast paths

---

## Potential Next Research Directions

### Geometry conditioning (highest priority)

1. **Vol-head LoRA r=4/r=8** (#812 askeladd retry): In flight
2. **AB-UPT geometry branch v3** (if v2 #802 relaunch shows gap compression holds at convergence, compose with best SOTA config + 13-ep full run)
3. **Cross-case contrastive loss** (#816 nezuko): Force model to distinguish 4 OOD test cases at training time — triplet margin loss on vol_p embeddings, λ=0.05, 4-ep diagnostic
4. **Surface feature → volume cross-attention**: Direct attention from volume queries to surface keys

### Architecture / positional encoding

5. **L6 depth scale** (#811 fern): In flight — does depth trend continue?
6. **Anchor-STRING stabilized full run** (#801 alphonse): In flight — 13-epoch convergence expected
7. **AB-UPT geometry branch v2 relaunch** (#802 frieren): In flight

### Loss weighting / training dynamics

8. **τ_z targeted upweight escalation to 4.0**: If thorfinn #807 shows positive signal at EP3/EP4, escalate. If paradox persists, close and investigate GradNorm path.
9. **GradNorm α sweep** (edward #810 + follow-up): α=2.0 in flight. If positive, extend to α=1.0 (weaker), α=4.0 (stronger).
10. **Cross-case contrastive loss** (#816 nezuko, ASSIGNED): Triplet margin loss on vol_p embeddings from 4 OOD cases (run_133, run_226, run_203, run_158), λ=0.05, 4-ep diagnostic. No architecture changes (SDF freeze compliant).
11. ~~Surface curvature compose~~: **EXHAUSTED** — 3 runs (PRs #788, #798, #808), all above gate. Closed.

### Ensemble refresh

11. **Ensemble pool-25**: After new single-model candidates emerge from current round, re-run greedy pool selection
