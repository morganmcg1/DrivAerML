# SENPAI Research State

- **Date:** 2026-05-11 — Round 29 active (9 WIP; PR #958 nezuko MERGED)
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current Baselines

| Tier | val_abupt | test_abupt | test_vol_pressure | PR | Notes |
|---|---|---|---|---|---|
| **Ensemble SOTA** | **6.0289%** | **7.3693%** | — | #880 (nezuko) | K=6 greedy pool-32 |
| **Single-model SOTA** | **6.2869%** | — | 12.0063% (NEGATIVE) | #958 Arm A (nezuko) | vol aux decoder head, EP13, run `29nohj67`, best_epoch=13 |
| **Prior single-model SOTA** | 6.4407% | 7.6992% | 11.6704% | #823 (nezuko) | surf→vol xattn, EP13, run `ghh0s4ne` |
| **Vol-pressure best anchor** | — | — | 11.374% | #681 (dc031qpt) | Issue #717 reference |

**Key win from #958 Arm A (nezuko):** vol aux decoder head (`--volume-loss-weight 1.0`) achieves val_abupt=6.2869% at best_epoch=13 (+0.154pp improvement over prior SOTA 6.4407%). test_vol_p=12.0063% is NEGATIVE (worse than 11.6704% baseline) — the dedicated aux head improves val_abupt generally but does NOT fix the 4 OOD test vol_p cases.

**Key win from #823:** surf→vol xattn adds one `nn.MultiheadAttention` (embed_dim=512, num_heads=4) after post-backbone LayerNorm. Q=vol_hidden, K=V=surf_hidden, zero-init out_proj. Improvement is broad-based (~−2.4% relative val, −3.6% test). The OOD test/val ratio is UNCHANGED (3.027× vs 3.025×) — xattn is a general capacity boost, NOT a targeted OOD fix. The 4 outlier test cases remain the dominant vol_pressure failure mode.

**Key finding from #767:** 4 OOD test cases (run_133, run_226, run_203, run_158) account for 92% of squared test_vol_p deviation. Excluding them, test_vol_p = 3.9–4.2% (below AB-UPT 6.08%). Geometry conditioning of the volume decoder is the highest-priority intervention.

**Vol-pressure promotion ladder (test_vol_pressure target):**
- Weak: ≤11.0% | Solid: ≤10.0% | Major: ≤8.5% | Target: ≤6.08% (AB-UPT reference)

---

## Latest Human Researcher Directives

- **Issue #618** (STRING/RoPE): **FULLY CLOSED — all axes exhausted.** STRING positional encoding is confirmed at its optimum: trainable `log_freq` with octave init {0.25,0.5,1.0,2.0,4.0}, trainable `phase`, QK-norm enabled. No new STRING experiments.
- **Issue #717** (vol_pressure gap): Phase 2 active — building on xattn win from #823. Post-xattn capacity fully closed (0-for-3: #891 FFN, #893 k-NN graph, #906 vol-self-attn). Now exploring loss stratification, geometry-conditioned queries, surface loss annealing, vol-pressure aux head, SDF-modulated vol PE, SDF-conditioned attention temperature, bounded SDF vol input encoding, per-case geometry embedding, surface-head-only isolation.
- **Issue #803** (SDF fix): **DISSOLVED.** Edward (#941) CLOSED NEGATIVE — full SOTA retrain with fixed SDF data (W&B: `2ub8dmy7`) completed to EP13: test_vol_p=11.9618% (worse than baseline 11.6704%). The 4 OOD outlier cases remain dominant failure mode even with corrected SDF values.
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- No new directives pending.

---

## Active PRs (Round 29) — 9 WIP

| Student | PR | Hypothesis | Branch | Status |
|---|---|---|---|---|
| nezuko | #958 | Vol aux decoder head: **MERGED** — Arm A val_abupt=6.2868% (new single-model SOTA, run `29nohj67`, EP13). Arm B (`--volume-loss-weight 2.0`) run `6xja19q9` in progress. | `nezuko/vol-pressure-aux-decoder-head` | **MERGED** — Arm B continuing; if Arm B beats 6.2868% it becomes new SOTA |
| tanjiro | #994 | LR-warmup decouple from vol-schedule boundary: complete LR warmup before EP1→EP2 boundary so only vol_points jump happens at that step, eliminating the 18× LR+curriculum co-shock from PR #983. | `tanjiro/lr-warmup-decouple-from-vol-schedule` | **EXCEPTIONAL** — run `p3v6veyl`, step 18,990. val_abupt=7.44%, vol_p=4.83% — BOTH EP3 dual-gate conditions already met. Linear projection ~1.0% abupt at EP3 gate. Likely to beat single-model SOTA 6.2868%. EP3 gate at step ~32,592. |
| askeladd | #986 | Adaptive SDF vol loss weighting: exp(-\|d_sdf\|/sigma) per-token weight; sigma sweep {0.5, 1.0, 2.0} m. Arm A (sigma=0.5). | `askeladd/adaptive-sdf-vol-loss` | WIP — run `i9qlgavj`, step 15,413. val_abupt=29.38%. **EP2 WARNING: needs ≤16% at step ~21,728 (~6,315 steps remaining). Trajectory concerning: needs -13.4pp in 6k steps. Monitoring.** |
| frieren | #995 | Pre-xattn vol LayerNorm ablation: single LN inserted immediately before surf→vol xattn (no MHA); isolates whether EP1 gain in #988 came from normalizing vol_h vs self-attn capacity. Zero FLOP overhead. | `frieren/pre-xattn-vol-ln-only` | WIP — run `sdt3tzq3`, step 14,768. val_abupt=27.65%. **EP2 WARNING: needs ≤16% at step ~21,728 (~6,960 steps remaining). Monitoring.** |
| fern | #996 | Near-surface SDF-stratified vol sampling: correct-direction inverse SDF weighting `exp(-alpha×|sdf|)` concentrates vol points near car surface; Arm A alpha=1.0, Arm B alpha=2.0. | `fern/near-surface-sdf-stratified-sampling` | WIP — Arm A run `4evy7zcx`, step 15,350. val_abupt=28.20%. **EP2 WARNING: needs ≤16% at step ~21,728 (~6,378 steps remaining). Monitoring.** Arm B pending Arm A EP2 verdict. |
| thorfinn | #1002 | Per-head learnable xattn temperature: log-parameterized per-head scale (`log_temp = nn.Parameter(zeros(H))`; `temp = exp(log_temp)`, init=1.0) applied post-W_q before head-split. Manual MHA implementation `SurfToVolCrossAttnPerHeadTemp`. | `thorfinn/per-head-learnable-xattn-temp` | WIP — New run launched 2026-05-11T16:45:17Z. Group=`per-head-xattn-temp`, rank0=`mzghptb0`. EP1 gate monitoring at step ~10,864. (Prior stale run `mgm7o7pb` killed.) |
| edward | #992 | Global surface embedding: learnable aggregation (attention pooling) over all surface tokens → geometry code added to vol queries before surf→vol xattn; richer spatial context than mean-pool. | `edward/global-surface-embedding` | WIP — run `l730ai6d`, step 13,728. val_abupt=27.57%. **EP2 WARNING: needs ≤16% at step ~21,728 (~8,000 steps remaining). Monitoring.** |
| alphonse | #1000 | SDF vol asinh encoding sweep: asinh(sdf/scale) normalization for vol SDF input feature; scale sweep {10.0, 20.0} m. Follow-up to closed tanh (#973) and raw-linear (#966) axes. | `alphonse/sdf-vol-asinh-encoding` | WIP — Arm A (scale=10.0) run `7pjhz629`, launched 2026-05-11T15:38:37Z. Step ~7,600. No val metrics yet (EP1 not fired at ~10,864). |
| nezuko | #1001 | Knowledge distillation: K=6 teacher ensemble → L=5 student via soft-label distillation. `distill_lambda_kd=0.5`. | `nezuko/kd-ensemble-distillation` | WIP — Running smoke tests without KD first: runs `6jz3xwvj` (step 2,066) and `jva4lf5u` (step 3,282), group=`kd-debug-smoke-no-kd`. Advisor suggested leveraging existing SOTA checkpoints (#929, #925-#928) as teacher ensemble members. |

### 9 students active WIP; nezuko #958 MERGED (Arm A new single-model SOTA 6.2868%)

---

## Round 28–29 Closures

| Student | PR | Run | Result |
|---|---|---|---|
| edward | #980 | `nv4g1nhw` | CLOSED NEGATIVE — geo-scalar conditioning: 4-float bbox [L,H,W,A_front] → Linear(4,hidden_dim) additive bias on vol hidden. Run timed out at step 30,609/108,640 (2.82/10 epochs). Forced EP3 val_abupt=7.4350%, test_vol_p=12.154% (+1.55pp vs baseline 10.605%). Widened OOD vol_p gap. Bundled GradNorm α=0.5 confound. |
| frieren | #981 | `xzreqhns` | CLOSED NEGATIVE — SDF-stratified vol sampling alpha=2.0: formula `1+alpha×|sdf|` up-weights freestream tokens 8–162× (WRONG DIRECTION — far-field bias not near-surface). EP1 FAIL val_abupt=39.37%. Correct approach: inverse weighting near surface. |
| fern | #982 | `ua1ohtjb` | CLOSED NEGATIVE — no-SDF-PE-scaling control: EP3 FAIL both gates (val_abupt=8.6996% > 8.0%, vol_p=5.9939% > 5.0%). WITHOUT SDF-PE scaling is WORSE than PR #977 with scaling (+0.44pp both metrics). Fast schedule is the bottleneck; SDF-PE scaling is NOT the root cause of PR #977 failure. |
| thorfinn | #984 | `ojuyombs` | CLOSED NEGATIVE — constant xattn temp scale=0.5 (Q×0.5): EP4 val_abupt=7.4900%, vol_p=5.0086%. Beats MLP baseline from #974 but does NOT beat SOTA 6.2869%. |
| alphonse | #985 | `7e3k06fj` | CLOSED NEGATIVE — stacked geo-xattn double cross-attention (+1.05M params, 18.04M total): EP3 dual gate FAIL (8.274% > 8.0%, vol_p=5.5706% > 5.0%), EP4 terminal val_abupt=8.2145% (+1.93pp vs SOTA), test_vol_p=13.0508%. Architecture trains cleanly but 4-epoch compressed schedule is insufficient for +1.05M param increase. |
| tanjiro | #983 | `hhplwxmk` (Arm A), `xy4yhpm0` (Arm B) | CLOSED NEGATIVE — curriculum warmup ramp (200–500 step vol_points warmup): val_abupt=7.4176% (Arm B EP4), test_abupt=8.7352%. Both arms survived EP1→EP2 transition (fixed divergence from #973) but critical finding: the shock was **LR-driven not curriculum-driven** — 18× LR jump (5e-6→9e-5) completes at EP1 end same step as vol_points jump. EP2→EP3 transition was smooth (LR already in cosine decay). The fix: decouple LR warmup completion from curriculum boundary. |
| frieren | #988 | `mdmkx495` (rank0) | CLOSED NEGATIVE — pre-xattn vol self-attention (O(N²) MHA): TIMEOUT at EP3 partial (step 17445/38030), val_abupt=10.5927%. EP1 showed −1.99pp vs baseline (promising!) but EP2+3 regressed badly. Root cause: O(N²) cost of vanilla MHA over vol_pts grows catastrophically with the vol_pts ramp (16K→32K→49K→65K), EP1 wall time 7103s vs baseline ~3500s. EP2+ too slow to complete within 270-min budget. Pre-xattn LN-only ablation assigned to frieren #995 to test if EP1 gain came from LN vs MHA. |
| fern | #989 | `z6dcbe9g` | CLOSED NEGATIVE — SDF-modulated vol PE octave scaling (MLP: SDF→per-octave scale): EP4 val_abupt=7.4901%, test_abupt=8.7927%. 1.0–1.1pp worse than single-model SOTA. Gap closed each epoch (EP1: −0.35pp, EP2: +3.23pp, EP3: +1.11pp, EP4: +0.68pp) but never recovered baseline. MLP learned physically coherent pattern (near-surface amplification) but far-field attenuation (mean scale=0.157 at sdf=+2.0) degraded far-field vol point discrimination. SDF-modulated PE octave scaling axis CLOSED. |

---

## Round 27–28 Closures

| Student | PR | Result |
|---|---|---|
| dl24-tanjiro | #978 | Per-case bbox coordinate normalization — CLOSED NEGATIVE. EP5 val_abupt=8.4488% (gate ≤7.5%). STRING PE sigmas tuned for raw ~5m coords; bbox normalization to [-1,+1] causes ~5× frequency mismatch. Loader bug fix (empty-view guard) worth cherry-picking separately. |
| askeladd | #975 | Surface-head-only isolation — CLOSED NEGATIVE. EP3 gate failed. Surface head alone cannot match the full model; xattn is load-bearing for surface metrics. |
| alphonse | #976 | Per-case geometry embedding — CLOSED NEGATIVE. EP3=8.073% val_abupt + 5.301% vol_p, dual gate fail. Mean-pool surface hidden states → geometry code does not generalize OOD vol_p. |
| frieren | #971 | Learned distance RPB on surf→vol xattn — CLOSED. SM 12.0 OOM on [B=4, H=4, N_vol=16384, N_surf=65536] attention tensor ~32 GiB bf16. |
| edward | #941 | SDF data fix retrain — CLOSED NEGATIVE. EP13 test_vol_p=11.9618% (worse than 11.6704% baseline). SDF corruption is not the root cause. |
| thorfinn | #974 | SDF-conditioned xattn temperature MLP — CLOSED NEGATIVE. MLP collapsed to global constant scale_mean=0.515, τ_std=0.00177. EP3: val_abupt=8.148%, vol_p=5.350% — narrowly missed dual gate. Follow-up: #991. |
| thorfinn | #967 | SDF-FiLM vol conditioning — CLOSED NEGATIVE. EP3 val_abupt gate FAILED. |
| tanjiro | #973 Arm A | tanh bounded SDF encoding — KILLED. Saturation degeneracy at scale=2.0m (gradient≈1e-8), catastrophic EP2 transition loss=10.6274. |

---

## Round 26 Outcomes (Closed / Superseded)

| Student | PR | Result |
|---|---|---|
| alphonse | #970 | STRING-sep frozen-freq ablation — CLOSED NEGATIVE. EP3=8.3347% (gate ≤8.0%, missed 0.33pp). Trainable `log_freq` confirmed load-bearing. |
| askeladd | #960 | STRING σ-bracket sweep — CLOSED NEGATIVE. Arm A EP3=7.1812%; Arm B v2 EP4=7.4995%. σ-tuning axis exhausted; STRING fully closed. |
| askeladd | #956 | STRING σ-ladder geometry sweep (7-octave) — CLOSED NEGATIVE (EP1=30.27%). |
| thorfinn | #949 | Surface-points curriculum — CLOSED. Results ambiguous; superseded by #953. |
| tanjiro | #966 | SDF scalar vol input feature (raw linear norm) — CLOSED NEGATIVE. EP4 val_abupt=7.6359% (gate ≤6.9%). +20,000σ outliers dominated vol input projection. |
| alphonse | #955 | STRING RoPE slice-centroid — CLOSED NEGATIVE. EP3: abupt=8.1833%, vol_p=5.9055%. |
| fern | #968 | SDF-PE octave scaling (original schedule) — SUPERSEDED by #977 (fast schedule). |

---

## Current Research Focus (Round 29)

**Primary mandate (Issue #717):** Reduce test_vol_pressure below 11.0% → 10.0% → 8.5% → 6.08% (AB-UPT). All channels must not degrade.

### Theme 1: Dedicated Vol Pressure Decoder (nezuko #958) — MERGED
Independent 3-layer MLP branch for vol_p prediction. **Arm A (vol_loss_weight=1.0): MERGED as new single-model SOTA val_abupt=6.2868%** (run `29nohj67`, best_epoch=13). test_vol_p=12.0063% NEGATIVE — dedicated head does NOT fix 4 OOD outlier cases. Arm B (vol_loss_weight=2.0): run `6xja19q9` continuing; if it beats 6.2868% it becomes the new SOTA.

### Theme 2: Pre-xattn Vol LayerNorm Ablation (frieren #995)
Single LN inserted immediately before surf→vol xattn (no MHA) to isolate whether the EP1 −1.99pp gain observed in #988 came from normalizing vol_h vs self-attn capacity. Zero FLOP overhead. This is a causal ablation: if LN alone recovers the EP1 gain, the benefit was normalization not attention capacity. PR #988 CLOSED due to O(N²) MHA cost explosion over 4-epoch vol_pts ramp.

### Theme 3: Near-Surface SDF-Stratified Vol Sampling (fern #996)
Correct-direction inverse SDF weighting `weight = exp(-alpha×|sdf|)` concentrates vol points near car surface during training. Two arms: Arm A alpha=1.0, Arm B alpha=2.0. Directly addresses the finding that far-field bias (frieren #981 wrong direction) catastrophically fails; correct near-surface bias has been untested until now.

### Theme 4: LR-Warmup Decoupling from Vol-Schedule Boundary (tanjiro #994) — EXCEPTIONAL
Complete LR warmup before EP1→EP2 boundary so only the vol_points jump happens at that transition. PR #983 mechanistic finding: the EP1→EP2 shock was driven by an 18× LR jump (5e-6→9e-5 from `--lr-warmup-epochs=1`) completing simultaneously with the vol_points curriculum jump. EP2→EP3 was smooth (LR already in cosine decay). **OUTSTANDING RESULT**: run `p3v6veyl` at step 18,990 shows val_abupt=7.44% AND vol_p=4.83% — both EP3 dual-gate thresholds (≤8.0%, ≤5.0%) already met with ~13,602 steps remaining. abupt slope=-0.476%/1k steps, vol_p slope=-0.268%/1k steps. Linear projection suggests ~1% abupt at EP3 gate (step 32,592). This may beat single-model SOTA 6.2868%.

### Theme 5: Per-Head Learnable xattn Temperature (thorfinn #1002)
Per-head scalar temperature scale using log-parameterization (`log_temp = nn.Parameter(zeros(H))`; `temp = exp(log_temp)`, init=1.0) applied post-W_q before head-split in a manual `SurfToVolCrossAttnPerHeadTemp` MHA implementation. Log-parameterization ensures temperatures stay positive and symmetric around 1.0. Replaces old PR #991 (thorfinn assigned new PR #1002). Follow-up to collapsed MLP (#974) and constant-scale NEGATIVE (#984). New run `mzghptb0` (rank0), group=`per-head-xattn-temp`, launched 2026-05-11T16:45:17Z.

### Theme 6: Global Surface Embedding (edward #992)
Learnable attention pooling over all surface hidden tokens to produce a geometry code, added to vol queries before surf→vol xattn. More expressive than failed mean-pool (#976, #980) because attention pooling preserves spatial selectivity. Zero-init residual.

### Theme 7: SDF Vol asinh Encoding Sweep (alphonse #1000)
asinh(sdf/scale) bounded normalization for the vol SDF input feature, replacing raw linear normalization (closed PR #966 due to +20,000σ outliers) and tanh saturation degeneracy (closed PR #973 Arm A). asinh is sub-linear for large |sdf| but has no hard saturation ceiling. Scale sweep: 10.0 m vs 20.0 m. Not yet started.

*Note: Bbox normalization + corrected STRING PE sigmas (PR #993) was the prior alphonse assignment — reassigned to PR #1000 after #993 review.*

### Theme 8: Adaptive SDF Vol Loss Weighting (askeladd #986)
Weight vol loss per-token by exp(-|d_sdf|/sigma) to focus learning near boundaries where OOD geometry differences are most pronounced. Sigma sweep {0.5, 1.0, 2.0} m. Parameter-free. Running.

### Theme 9: Knowledge Distillation from Ensemble (nezuko #1001)
K=6 teacher ensemble → L=5 student via soft-label distillation. `distill_lambda_kd=0.5`. Student inherits from single-model SOTA architecture (L=5, hidden=512, xattn). Smoke tests running first without KD (no teacher_path) to validate training loop: runs `6jz3xwvj` (step 2,066) and `jva4lf5u` (step 3,282), group=`kd-debug-smoke-no-kd`. Advisor suggested exploring existing SOTA checkpoints from prior PRs (#929, #925-#928) as potential teacher ensemble members to avoid K=6 full new training runs.

### Closed Themes
- **SDF data quality** (edward #941): CLOSED NEGATIVE — SDF corruption is NOT the root cause of vol_p OOD gap.
- **STRING positional encoding** (#960, #970 etc.): FULLY EXHAUSTED — all axes closed. No new STRING experiments.
- **Per-case geometry embedding v1** (alphonse #976): CLOSED NEGATIVE — mean-pool surface code does not generalize OOD vol_p.
- **Surface-head-only isolation** (askeladd #975): CLOSED NEGATIVE — xattn is load-bearing for surface metrics.
- **Geo-scalar conditioning** (edward #980): CLOSED NEGATIVE — 4-float bbox projection widens OOD vol_p gap; too low dimensional to capture case geometry.
- **SDF-stratified sampling (far-field bias)** (frieren #981): CLOSED NEGATIVE — formula `1+alpha×|sdf|` up-weights far-field (wrong direction); EP1 catastrophic failure.
- **Constant xattn temp scale** (thorfinn #984): CLOSED NEGATIVE — Q×0.5 beats MLP baseline but not SOTA; uniform global sharpening insufficient.
- **Stacked geo-xattn (double cross-attention)** (alphonse #985): CLOSED NEGATIVE — +1.05M params, 4-epoch schedule too short to train.

---

## Potential Next Research Directions

### Immediate Follow-ups (high priority)
1. **Attention temperature sweep**: If PR #991 passes EP3, try per-head scales initialized to 0.5, 0.25, 1.5 to explore sharpening spectrum
2. **LR-warmup decouple tuning**: If PR #994 passes EP3, try warmup-epochs=0.5 and warmup-epochs=2 to explore sensitivity
3. **Bbox normalization with corrected PE sigmas**: ASSIGNED — alphonse #993 (sigmas ÷5: {0.05,0.1,0.2,0.4,0.8})
4. **Global surface embedding sweep**: If PR #992 passes EP3, try attention pooling with query learned from vol centroid coordinates
5. **SDF-stratified sampling alpha tuning**: If fern #996 passes EP3, try alpha=0.5 and alpha=4.0 to map the near-surface bias sensitivity curve

### Architecture Directions
7. **Mid-backbone xattn**: Cross-attention at intermediate backbone layer rather than final; tested several variants (PR #917 NEGATIVE), but pre-final-layer variant may differ
8. **Adaptive vol loss weighting by SDF distance**: ASSIGNED — askeladd #986 (sigma sweep {0.5, 1.0, 2.0} m)

### Data / Representation Directions
9. **asinh bounded SDF encoding**: tanjiro #973 Arm B was in progress (asinh replaces tanh); if complete, review results
10. **OOD-4 case upweighting**: Upweight the 4 outlier test cases (run_133, 226, 203, 158) by 2× in the vol aux head loss; direct OOD targeting
11. **Coordinate frame augmentation**: At train time, randomly rotate/scale the canonical coordinate frame; forces the model to learn rotation-equivariant representations

### Bold / Escalation Directions (if plateau continues past Round 29)
12. **GNN vol pathway**: Replace Transolver vol attention with k-NN GNN for vol-vol communication (k=8-16)
13. **Fourier Neural Operator vol decoder**: Replace MLP vol decoder with FNO for better frequency resolution
14. **Implicit neural representation**: Map (surface_context, vol_coord) → (tau, pressure) via coordinate MLP; natural coordinate generalization
15. **Point Transformer V3 vol head**: Replace Transolver vol attention with PTv3 (Hilbert-curve serialized attention)

---

## Closed Axes (Rounds 22–29) — Do Not Revisit

### SDF data quality: CLOSED NEGATIVE
- **#941 edward**: SDF data fix — test_vol_p=11.9618% (worse than 11.6704% baseline). SDF corruption is NOT the root cause.

### Post-xattn capacity: FULLY CLOSED (0-for-3)
- PR #891 (fern post-xattn FFN): EP3=8.50% NEGATIVE
- PR #893 (askeladd vol k-NN graph attn): NEGATIVE
- PR #906 (edward post-xattn vol self-attn): EP3=8.23% NEGATIVE

### Pre-xattn capacity: CLOSED (0-for-2)
- PR #929 (edward pre-xattn vol self-attn): CLOSED NEGATIVE
- PR #985 (alphonse stacked geo-xattn double cross-attn): CLOSED NEGATIVE — 4-epoch schedule too short for +1.05M params

### Learned distance RPB on xattn: CLOSED (OOM)
- PR #971 (frieren xattn-distance-rpe): CLOSED — SM 12.0 OOM; needs memory-efficient reformulation.

### SDF-FiLM hidden-state conditioning: CLOSED NEGATIVE
- PR #967 (thorfinn): FiLM modulation of vol hidden states — EP3 gate FAILED.

### STRING axes: FULLY EXHAUSTED
- PR #956, #960, #970, #955 — all CLOSED NEGATIVE. No further STRING experiments permitted.
- σ=0.25 confirmed load-bearing; trainable `log_freq` confirmed load-bearing.

### Geometry scalar/bbox conditioning: CLOSED NEGATIVE
- **#980 edward** (bbox [L,H,W,A_front] Linear): Widened OOD vol_p gap; 4-float bbox too low-dim. CLOSED NEGATIVE.
- **#976 alphonse** (mean-pool surface code): Mean-pool loses spatial specificity. EP3 dual gate fail. CLOSED NEGATIVE.
- **#961** (geometry-conditioned Q-bias mean-pool): EP1=30.197%. CLOSED NEGATIVE.
- **#926** (vol geo features centroid+bbox): CLOSED.

### Attention temperature scaling: CLOSED (constant/MLP variants)
- **#974 thorfinn** (SDF-conditioned xattn temp MLP): MLP collapsed to global constant scale=0.515. CLOSED NEGATIVE.
- **#984 thorfinn** (constant xattn temp scale=0.5): Beats MLP baseline but not SOTA. CLOSED NEGATIVE.
- Per-head learnable scale (#1002) still in flight.

### SDF-modulated vol PE octave scaling: CLOSED NEGATIVE
- **#989 fern** (MLP: SDF→per-octave scale): EP4 val_abupt=7.4901%, test_abupt=8.7927%. 1.0–1.1pp worse than single-model SOTA. MLP learned physically coherent near-surface amplification but far-field attenuation (mean scale=0.157 at sdf=+2.0m) degraded far-field vol point discrimination. SDF-modulated PE octave scaling axis FULLY CLOSED.

### Pre-xattn vol self-attention (O(N²) MHA): CLOSED (TIMEOUT)
- **#988 frieren** (vanilla MHA over vol_pts): TIMEOUT at EP3 partial (step 17445/38030), val_abupt=10.5927%. EP1 showed −1.99pp (promising) but O(N²) cost of MHA over vol_pts ramp (16K→32K→49K→65K) — EP1 wall time 7103s vs baseline ~3500s — made EP2+ infeasible within 270-min budget. Vanilla O(N²) MHA over vol_pts is not viable. Linear/sparse alternatives remain open.

### SDF-stratified sampling (far-field bias): CLOSED NEGATIVE
- **#981 frieren** (alpha=2.0 formula `1+alpha×|sdf|`): Up-weights far-field 8–162× (wrong direction). EP1 catastrophic failure. Correct inverse weighting assigned as fern #996.

### All other closed axes (do not revisit)
- **Depth scaling**: L=6 CLOSED (both #895 and #811 NEGATIVE)
- **Width scaling**: hidden=640 CLOSED (#886 NEGATIVE)
- **Two-layer xattn**: #884 CLOSED (EP1 kill gate)
- **Detach-K/V xattn**: #890 CLOSED (K/V gradient backflow load-bearing)
- **GQA/MQA xattn**: #893 CLOSED (full MHA n_heads=4 is optimal)
- **MLP ratio=2.0**: #897 CLOSED (ratio=4.0 is optimal)
- **Learned surf pool (Perceiver)**: #894 CLOSED
- **Pos-encoding bias on xattn queries (static)**: #883 CLOSED
- **OOD static loss upweighting**: #888 CLOSED
- **Vol loss upweighting (curriculum)**: #902 CLOSED (vol_w=3.0 degraded all channels)
- **Mid-backbone surf→vol xattn**: #917 CLOSED NEGATIVE
- **Rotation aug (aggressive and mild)**: #925, #937 CLOSED NEGATIVE
- **All augmentation axes** (geometric mixup, train y-mirror, TTA y-mirror, y-flip fixed, y-flip curriculum ramp): ALL CLOSED NEGATIVE
- **Surface cp Laplacian aux loss**: #927 CLOSED
- **Vol-specific RFF sigmas**: #918 CLOSED
- **GradNorm full-mode α=1.5**: #942 CLOSED NEGATIVE — vol weights go DOWN, all weights converge to 0.91–1.11 band
- **GradNorm, EMA, LoRA, FiLM, 2-layer MLP decoder, Huber loss, Lion β2 sweep, BC embedding, dual-tower xattn, spectral-norm attn, stochastic depth, PCGrad, surface curvature features**: ALL CLOSED
- **SDF-zone vol token masking (p_max=0.30 and 0.15)**: #953, #963 CLOSED — high run-to-run variance (±4.2pp), no replicable signal
- **CoordVolHead**: #959 CLOSED NEGATIVE — EP4 val_abupt=7.428% (gate 6.9%); redundant with STRING-sep + xattn
- **Raw SDF linear normalization as vol input feature**: #966 CLOSED NEGATIVE — +20,000σ outliers; closed axis
- **tanh bounded SDF encoding**: #973 Arm A KILLED — saturation degeneracy at scale=2.0m
- **Per-case geometry embedding (mean-pool surface code)**: #976 CLOSED NEGATIVE — EP3 dual gate fail
- **Surface-head-only isolation**: #975 CLOSED NEGATIVE — xattn load-bearing for surface metrics
- **Bbox coordinate normalization (without PE sigma correction)**: #978 CLOSED NEGATIVE — ~5× STRING PE frequency mismatch; fix assigned #993

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128, surf→vol xattn (1 layer, 4 heads, ~1.05M params) → ~16.99M total
- **Depth:** L=5 optimal. FULLY CLOSED.
- **Positional encoding:** STRING-separable (rff=16, sigmas {0.25,0.5,1.0,2.0,4.0}). σ=0.25 load-bearing. Trainable `log_freq` load-bearing.
- **Optimizer:** Lion, lr=9e-5, β₁=0.9, β₂=0.99, wd=5e-4
- **QK-norm:** enabled
- **Loss weights:** tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0. All static loss-weight axes CLOSED.
- **EMA:** 0.999
- **4-ep screen schedule:** `--lr-cosine-t-max 4 --epochs 4` with vol-curriculum `0:16384:1:32768:2:49152:3:65536`
- **`--no-compile-model` required:** torch.compile + DDP NCCL deadlock at step 1
- **`find_unused_parameters=True` required for DDP when using conditional modules**

## Key Diagnostic Findings Established

- **4 OOD test cases dominate vol_pressure** (#767): 92% of squared deviation. Excluding them, test_vol_p = 3.9–4.2% (below AB-UPT). Geometry conditioning is the right lever.
- **Surf→vol xattn is broad-spectrum win, not OOD fix** (#823): OOD test/val ratio unchanged (3.027×). The 4 outlier cases still drive vol_p gap.
- **SDF data quality is NOT the root cause** (#941): test_vol_p=11.9618% with corrected SDF — WORSE than baseline.
- **Post-xattn capacity is a dead end** (0-for-3: #891, #893, #906): all capacity added AFTER surf→vol xattn fails.
- **Pre-xattn vol self-attention also failed** (#929): symmetry does not hold.
- **STRING σ=0.25 is load-bearing** (#819). **Trainable log_freq is load-bearing** (#970).
- **K/V gradient backflow is load-bearing** (#890): detach-K/V failed.
- **Full MHA n_heads=4 is optimal** (#893).
- **GradNorm full-mode CLOSED NEGATIVE** (#942): vol pressure weight goes DOWN under GradNorm (vol_p loss decreases fastest).
- **SDF-zone masking at p_max=0.30 is unstable** (#953): ±4.2pp run-to-run variance.
- **CoordVolHead CLOSED NEGATIVE** (#959): Failed EP4 gate (7.428% vs gate 6.9%). STRING-sep + xattn already encode xyz-awareness.
- **Geometry-conditioned Q-bias CLOSED NEGATIVE** (#961): EP1=30.197% (0.197pp over 30% gate). Mean-pool surf loses spatial specificity.
- **SDF-FiLM CLOSED NEGATIVE** (#967): FiLM modulation of vol hidden states fails EP3 gate.
- **tanh saturation degeneracy** (#973 Arm A): at scale=2.0m, ~99% vol tokens saturate (gradient≈1e-8), catastrophic curriculum-transition loss spike.
- **Vol aux decoder head improves val_abupt but not test_vol_p** (#958 Arm A): new single-model SOTA val_abupt=6.2869%, but test_vol_p=12.0063% WORSE than baseline. OOD vol_p requires geometry-specific conditioning beyond a dedicated decoder head.
- **Flash SDPA required for large xattn tensors**: [B=4, H=4, N_vol=16384, N_surf=65536] bf16 ~32 GiB — OOM without Flash SDPA.
- **Mean-pool geometry embedding fails OOD vol_p** (#976 alphonse): global mean-pool surf hidden states lose spatial specificity needed for OOD geometry differentiation.
- **xattn is load-bearing for surface metrics** (#975 askeladd): surface-head-only ablation fails — xattn contributes to surface metrics not just vol_p.
- **Fast 4-epoch schedule is the bottleneck for SDF-PE experiments** (#982 fern control): SDF-PE scaling is NOT the root cause of PR #977 EP3 failure. Removing scaling makes results WORSE (+0.44pp both gates). The schedule itself prevents convergence of additional capacity.
- **SDF-stratified sampling formula direction matters critically** (#981 frieren): `weight = 1+alpha×|sdf|` up-weights freestream tokens (WRONG). Correct direction = inverse near-surface weighting. Never try far-field-biased SDF sampling again.
- **Stacked double xattn needs longer training** (#985 alphonse): +1.05M params (18.04M total) trains cleanly but 4-epoch EP schedule is insufficient. Architecture may work but requires full 13-epoch run to evaluate properly.
- **Bbox normalization breaks STRING PE without sigma correction** (#978): normalizing to [-1,+1] gives ~5× frequency mismatch. Sigmas must be rescaled ÷5. Both changes bundled as atomic fix in #993.
- **EP1→EP2 shock is LR-driven, not curriculum-driven** (#983 tanjiro): 18× LR jump (5e-6→9e-5 from `--lr-warmup-epochs=1`) completes at EP1 end simultaneously with vol_points curriculum jump. EP2→EP3 transition was smooth (LR already in cosine decay by then). Fix: decouple LR warmup completion from curriculum boundary (PR #994).
- **O(N²) vanilla MHA over vol_pts is prohibitively expensive for 4-epoch schedule** (#988 frieren): EP1 wall time 7103s vs baseline ~3500s; EP2+ infeasible within 270-min budget. vol_pts ramp (16K→32K→49K→65K) makes O(N²) cost grow 16× from EP1 to EP4. Linear/sparse attention is required for vol self-attn to be viable.
- **SDF-PE octave scaling MLP learns coherent physics but far-field attenuation hurts** (#989 fern): Near-surface amplification is physically motivated and learned correctly; however, reducing far-field PE amplitudes (mean scale=0.157 at sdf=+2.0m) degrades far-field vol point discrimination. SDF-modulated PE scaling axis CLOSED.
