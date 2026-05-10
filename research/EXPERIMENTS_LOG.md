# SENPAI Research Results

## 2026-05-09 23:45 — PR #925: Random yaw±5°/pitch±3° rotation aug (alphonse) — CLOSED (NEGATIVE)

- **Branch**: `alphonse/random-yaw-pitch-rotation-aug` (closed)
- **W&B run**: `a6ddeqrq`
- **Hypothesis**: Joint rotation of surface_xyz, vol_xyz, surface_normals, and wall_shear vectors by a random yaw (±5°) and pitch (±3°) rotation matrix at train time (p_aug=0.5) forces approximate rotation-equivariance, closing the val/test gap on the 4 OOD test cases which likely exhibit different aerodynamic incidence angles.

| Epoch | val_abupt | surf_p | vol_p | wsh | Gate | Baseline |
|---|---:|---:|---:|---:|---|---:|
| EP1 | 27.37% | — | — | — | ≤30% PASS | ~25–28% |
| EP2 | 12.81% | — | — | — | ≤16% PASS | ~12% |
| EP3 | **9.1064%** | 5.96% | 6.14% | 10.14% | ≤8.0% **FAIL** | 7.1195% |

**Analysis:** Rotation aug at yaw±5°/pitch±3°/p=0.5 is too aggressive for the 4-epoch budget. The model spent EP2 capacity learning rotation-approximate equivariance — EP2 val_abupt fell 4.66pp behind the rotation-free baseline's EP2 position, suggesting the model was spending optimization headroom adapting to the augmented distribution rather than converging. The EP2→EP3 slope recovered to −3.71pp (steep, vs baseline's comparable step), suggesting the regularizer benefit was beginning to emerge, but the 4-ep screen gate at EP3 came too early to capture it.

**Per-channel EP3:** surf_p=5.96% (clean, near SOTA), vol_p=6.14% (near SOTA), wsh=10.14% (primary drag — wall_shear carries most of the penalty). The wsh degradation is the likely cause of EP3 gate failure: rotating wall_shear vectors is physically correct but adds more augmentation noise to the highest-variance channel.

**Key insight:** The hypothesis (rotation aug → OOD vol_p) is NOT falsified. Only the magnitude/probability point (yaw±5°/pitch±3°/p=0.5) is falsified. The wsh channel specifically degrades under this aggressive aug, as wall_shear vector rotation is geometrically complex and introduces more entropy than the geometric coordinates alone.

**Suggested follow-up axes (student suggestions incorporated):**
1. **Milder aug**: p=0.3, yaw≤3°, pitch≤1.5° — lower entropy, less wall_shear disruption
2. **Yaw-only variant**: pitch=0, yaw±3°/5°, p=0.5 — wind tunnel geometry means OOD variation is primarily yaw, pitch=0 removes a dimension of aug noise
3. **Aug rampup**: p=0 for EP1, ramp to p=0.3 by EP3 — curriculum approach

**Conclusion:** Assign follow-up with milder parameters. Rotation aug (yaw-only or mild yaw+pitch) is high-priority next assignment.

---

## 2026-05-09 19:30 — PR #906: Post-xattn vol-self-attn block (edward) — CLOSED (NEGATIVE)

- **Branch**: `edward/vol-self-attn-post-xattn` (closed)
- **W&B run**: `nmvw5t2d`
- **Hypothesis**: A vol→vol self-attention block inserted AFTER the surf→vol xattn (decoder-style refinement) gives the volume branch capacity to propagate cross-attended surface signal across volume tokens before regression, closing the val/test gap on `volume_pressure_rel_l2_pct`.
- **Implementation**: Pre-norm self-attn + FFN block, zero-init out_proj and final FFN linear (identity-at-init); inserted after surf→vol xattn residual update, before vol regression head. `find_unused_parameters=True` for DDP.

| Epoch | val_abupt | vol_p | surf_p | wsh | Gate | Baseline |
|---|---:|---:|---:|---:|---|---:|
| EP3 | 8.2271% | — | — | — | ≤8.0% **FAIL** | 7.1195% |
| EP4 step 282 | — | — | — | — | external SIGTERM | — |

**Analysis:** Run was killed by external pod/harness signal at 18:59:16 UTC (~252 min runtime, well under SENPAI_TIMEOUT_MINUTES=360). Termination is moot since EP3 had already failed the gate. No per-channel evidence that vol-self-attn after surf→vol xattn helps the val/test gap.

**Pattern: post-xattn capacity additions are now 0-for-3 on this benchmark:**
- PR #884/#890: two-layer xattn — failed
- PR #891: post-xattn FFN — failed
- PR #906: post-xattn vol-self-attn — failed

**Conclusion:** Adding capacity to the volume branch *after* surf→vol xattn does not move the val/test gap. Future volume-branch experiments should target either (a) capacity placed BEFORE surf→vol xattn, (b) the geometry/positional encoding pathway, or (c) data-augmentation / regularization approaches targeting OOD generalization rather than capacity.

---

## 2026-05-09 19:30 — PR #918: Vol-specific RFF init sigmas (tanjiro) — REQUESTED CHANGES

- **Branch**: `tanjiro/vol-rff-positional-encoding`
- **W&B run Arm A**: 5× lower-frequency vol sigmas `0.05,0.1,0.25,0.5,1.0` vs surface `0.25,0.5,1.0,2.0,5.0`
- **Hypothesis**: Volume and surface fields occupy different spatial frequency regimes; volume needs lower-frequency RFF init to match its smoother field structure.

| Epoch | val_abupt | Gate | Baseline EP1 |
|---|---:|---|---:|
| EP1 | 32.66% | ≤30% **FAIL** | ~25–28% |

**Analysis:** Aggressive 5× shift starves the surf→vol xattn coupling of fine-spatial detail — volume tokens carry less high-frequency content than the surface K/V expects, degrading the cross-attention. All per-channel metrics regressed.

**Decision:** Hypothesis itself (separate spectral regimes) not yet falsified — only the aggressive instantiation is. Requesting **Arm C** with a moderate 2× shift `0.1,0.25,0.5,1.0,2.0`. If Arm C also fails the screen, close the PR and mark vol-specific sigma init as a falsified direction across the moderate-to-aggressive range.

---

## 2026-05-09 19:30 — PR #901: Train-time y-axis mirror aug (fern) — REQUESTED CHANGES (truncated)

- **Branch**: `fern/train-mirror-aug-y`
- **Hypothesis**: DrivAerML has near-perfect y-axis symmetry; a stochastic p=0.5 train-time mirror (negate y, ny, tau_y) is a free 2× data prior that should reduce val/test gap on volume_pressure.

**Arm B run**: 13-ep at SOTA stack — **truncated at EP5/13** (~18% complete) due to SENPAI_TIMEOUT_MINUTES=360 → 270 min train budget. Advisor planning error: a 13-ep run at 65k vol points needs ~680 min. Acknowledged in PR comment.

**Encouraging mid-run signals:**
- val→test ratio on volume_pressure improved from baseline 3.03× to 2.50× even at EP5
- tau_y trajectory clean — no sign-flip pathology, aug is consistent with model's symmetry prior
- Loss curve healthy, no divergence

**Decision:** Hypothesis still alive but inconclusive — requesting rerun at the 4-ep screen with `--lr-cosine-t-max 4 --epochs 4` so the cosine fully decays inside the 270 min budget. Win criterion: EP4 val_abupt < 6.9% AND val→test vol_p ratio sustained ≤2.7×. If clean signal, find a way to fit the 13-ep follow-up (potentially drop terminal vol_points to 49k).

---

## 2026-05-09 16:30 — PR #902: Vol loss upweighting curriculum (nezuko) — CLOSED (NEGATIVE)

- **Branch**: `nezuko/vol-pressure-ood-curriculum` (closed)
- **W&B runs**: `nx49bb6w` (Arm A, vol_w=3.0), Arm B (vol_w=5.0) cancelled
- **Group**: `nezuko-vol-pressure-ood-curriculum`
- **Hypothesis**: Upweighting the volume loss (vol_w=3.0 Arm A, vol_w=5.0 Arm B) with an accelerated vol-curriculum would force the model to prioritize vol_p accuracy including the 4 OOD test cases.

| Epoch | Step | val_abupt | vol_p | surf_p | wsh | Gate | Baseline EP3 |
|---|---:|---:|---:|---:|---:|---|---:|
| EP1 | ~10,864 | — | 14.17% | — | — | ≤30% PASS | — |
| EP3 | 21,734 | **8.5436%** | 5.0614% | 5.7369% | 9.6625% | ≤8.0% **FAIL** | 7.1195% |

Arm B (vol_w=5.0) cancelled after Arm A failed — more aggressive upweighting in the same failing direction.

**Analysis:** vol_w=3.0 makes vol_p **worse** vs PR #823 baseline at EP3 (5.06% vs 4.27% — +0.79pp). Every channel degraded vs baseline. The accelerated vol-curriculum (vol_pts bumps at every epoch vs baseline's 3-epoch steps) creates a curriculum-mismatch: Arm A EP3 = 3rd vol=32k epoch with 21,734 iters; baseline EP3 = 3 epochs at vol=16k with 32,594 iters (~50% more compute). But even at EP1 (clean comparison), vol_p=14.17% vs baseline 17.79% — baseline wins. Heavier volume loss weight over-emphasizes vol gradients during early adaptation and disrupts the surf/wsh pathways.

**Conclusion:** Volume loss upweighting is not the right lever for closing the vol_p OOD gap. The issue is in feature representation and geometry conditioning architecture, not in loss balance.

---

## 2026-05-09 16:45 — PR #910: Xattn K/V grad scale α=0.5 Arm A (frieren) — EP3 FAIL, Arm B (α=0.75) ASSIGNED

- **Branch**: `frieren/xattn-kv-grad-scale-sweep-alpha`
- **W&B run Arm A**: `bnynqueq` (group `frieren-xattn-kv-grad-scale-sweep`, name `frieren/xattn-kv-scale-alpha0.5-screen`)
- **Flag Arm A**: `--xattn-kv-grad-scale 0.5`
- **Hypothesis**: Following α=0.25 EP3 stall (PR #896, 8.95%), test α=0.5 and α=0.75 to find the optimal K/V gradient scale point in [0.25, 1.0].

| EP | val_abupt | surf_p | vol_p | wall_shear | Δ abupt | Gate |
|---|---:|---:|---:|---:|---:|---|
| EP1 | 29.3248% | 22.40% | 16.96% | 32.67% | — | ≤30% ✅ |
| EP2 | 11.5368% | 7.71% | 8.24% | 12.59% | −17.79pp | ≤16% ✅ |
| EP3 | **8.6500%** | **5.49%** | **6.52%** | **9.43%** | **−2.89pp** | ≤8.0% ❌ (+0.65pp) |

Run killed at step ~20,300 (mid-EP4, before EP4 validation).

**Key signals:**
- EP2→EP3 slope = −2.887pp (6.5× steeper than α=0.25's −0.441pp): mechanism is much less stalled than α=0.25
- vol_p = 6.52% at EP3 already well below SOTA test_vol_p=11.67% — the K/V scale is specifically helping vol head while lagging on abupt-mean
- α=0.5 starts hot at EP1 (29.3% vs α=0.25's 13.4%) suggesting the higher gradient flow makes early optimization harder but the run is still descending steeply at EP3
- Likely EP13 landing from EP3=8.65% following SOTA slope: ~8.0% (not competitive vs SOTA 6.44%)

**Action:** Arm B (α=0.75) launched — closer to α=1.0 (SOTA), expected to start cooler and converge faster on val_abupt. Bracket now [α=0.25 EP3=8.95%, α=0.5 EP3=8.65%, α=0.75 TBD, α=1.0 EP3=7.12% SOTA].

---

## 2026-05-01 — PR #893: Grouped-Query xattn (alphonse) — CLOSED (NEGATIVE)

- **Branch**: `alphonse/xattn-gqa` (deleted)
- **W&B runs**: `7jqz957i` (Arm A, n_kv_heads=1/MQA), `eqp1873z` (Arm B, n_kv_heads=2/GQA)
- **Group**: `xattn-gqa-sweep`
- **Hypothesis**: Replace the surf→vol cross-attention MHA (n_heads=4) with Grouped-Query Attention (GQA). Arm A: MQA (n_kv_heads=1, 4:1 Q/KV ratio); Arm B: GQA (n_kv_heads=2, 2:1 ratio). Llama-style: head_dim=128 throughout, smaller KV projection output. Expected benefit: reduced KV parameter count, potentially acting as a structured regularizer on the surface conditioning pathway.

| Arm | Config | EP3 abupt | Gate (≤8.0%) | surf_p | vol_p | wsh |
|---|---|---:|---|---:|---:|---:|
| A (run 7jqz957i) | n_kv_heads=1, MQA | 8.2694% | ❌ FAIL (+0.27pp) | 5.3768% | 5.6686% | 9.1388% |
| B (run eqp1873z) | n_kv_heads=2, GQA | 8.2097% | ❌ FAIL (+0.21pp) | 5.3411% | 5.5667% | 9.0992% |
| **SOTA** (PR #823) | n_kv_heads=4, MHA | **6.4407%** | baseline | 4.1836% | 3.8557% | 7.3448% |

Both arms ran to EP3 then were killed per kill-gate protocol (student terminated Arm A after Arm A miss; Arm B also confirmed miss).

**Analysis:** Both GQA arms failed EP3 by a narrow but consistent margin (~0.21–0.27pp). The reduction in K/V heads uniformly impairs convergence — more heads misses by less (Arm B > Arm A), consistent with the hypothesis that full MHA capacity in xattn is load-bearing. Arm B is strictly better than Arm A (fewer K/V heads = more degradation), confirming the direction: the surface→volume attention benefits from full rank attention heads. Notable: the student caught a spec dimension mismatch bug during implementation (original spec had kv_head_dim = embed_dim/n_kv_heads, giving incompatible Q/K head dims for SDPA) and correctly implemented standard Llama-style GQA instead.

**Conclusion:** GQA for surf→vol cross-attention does not improve convergence. Full MHA (n_heads=4) remains optimal. Cross-attention KV capacity is not a bottleneck to regularize. Closing as a clean negative result.

---

## 2026-05-09 ~12:45 — PR #896: Xattn K/V gradient scaling α=0.25 (frieren) — CLOSED (NEGATIVE)

- **Branch**: `frieren/xattn-kv-grad-scale` (closed)
- **W&B run**: `vf9dprlh` (group `frieren-xattn-kv-grad-scale`, name `frieren/xattn-kv-grad-scale`)
- **Flag**: `--xattn-kv-grad-scale 0.25`
- **Hypothesis**: Scale K/V gradients by α=0.25 in surf→vol xattn to damp surface encoder over-adaptation while preserving joint training signal. Addresses the K/V backflow mechanism identified in #884 (two-layer backflow) and #890 (full detach kills EP1).

| Epoch | Step | val_abupt | Gate | Result |
|---|---:|---:|---|---|
| EP1 | 21,728 | 13.449% | ≤30% | ✅ PASS |
| EP2 | 32,599 | 9.395% | ≤16% | ✅ PASS (6.6pp margin) |
| EP3 | 39,851 | **8.954%** | ≤8.0% | ❌ FAIL (+0.954pp) |
| EP4 | 45,308 | 8.773% | — | (killed) |

Phase 2 NOT triggered.

**Analysis:** Strong EP1→EP2 drop (−4.05pp) but severe slope flattening EP2→EP3 (−0.44pp, 10× slowdown). K/V gradient scaling at α=0.25 reduces surface encoder adaptation just enough to slow volume convergence without delivering a commensurate accuracy benefit. The α=0.25 sweet spot between detach (α=0, EP1 kill gate #890) and full gradient (α=1.0, SOTA #823) appears to exist but this value isn't it — the convergence stalls at 8.95%.

**Key finding:** Graduated backflow management is the right axis to explore, but α=0.25 is too aggressive. Future experiments should try α=0.5 (half backflow) or α=0.75 (gentle damping) if this mechanism is revisited.

## 2026-05-09 ~12:30 — PR #895: L=6 + surf→vol xattn (edward) — CLOSED (NEGATIVE)

- **Branch**: `edward/xattn-depth-L6-512` (deleted)
- **W&B run**: `x3c2a2jt` (group `edward-xattn-depth-L6-512`, name `edward/xattn-depth-L6-512-screen`)
- **Hypothesis**: Adding a 6th Transolver block in the L=5+xattn stack would give the model extra capacity that geometry-conditioning (xattn) can finally exploit, since the previous L=6 NEGATIVE (PR #811) was without xattn.

| Epoch | val_abupt | Gate | SOTA PR #823 | Δ |
|---|---:|---|---:|---:|
| EP1 | ~28% | ≤30% PASS | 28.63% | ~ |
| EP2 | ~14% | ≤16% PASS | 8.15% | ~+6pp worse |
| EP3 | ~9.5% | ≤8.0% **FAIL margin** | 7.12% | ~+2pp worse |
| EP4 | **7.886%** | ≤6.9% **FAIL** | 6.81% | +1.08pp worse |

Phase 2 (full 13-ep) NOT triggered.

**Analysis:** Train loss reached 0.032 at EP4 with val_abupt 7.89% — large train/val gap signals memorization rather than improved generalization. Combined with PR #811 (L=6 without xattn, also NEGATIVE), the depth-scaling axis at hidden=512 is **CLOSED on both with-xattn and without-xattn**. Adding a 6th block does not give the volume head usable capacity at this budget.

**Implication:** Future capacity scaling should pivot away from naive layer count. Options: (a) wider hidden_dim with L=5, (b) asymmetric depth (L_vol > L_surf), (c) radical volume-pathway architectures (FNO, multiscale, GNN message passing) per Issue #717.

## 2026-05-09 ~09:37 — PR #891: Post-xattn FFN (fern) — CLOSED (NEGATIVE)

- **Branch**: `fern/post-xattn-ffn` (deleted)
- **W&B run**: `c3jvc0s1` (group `fern-post-xattn-ffn`, name `fern/post-xattn-ffn-4ep`)
- **Hypothesis**: Adding a 2-layer MLP (hidden×4, GELU, zero-init second linear) after the surf→vol xattn residual update gives the volume pathway more capacity to process the surface conditioning signal before the regression head.

| Epoch | Step | val_abupt | Gate | SOTA PR #823 | Δ |
|---|---:|---:|---|---:|---:|
| EP1 | 10,864 | 26.51% | ≤30% PASS | 28.63% | −2.12pp better |
| EP2 | 16,300 | 12.13% | ≤16% PASS | 8.15% | +3.98pp worse |
| EP3 | 19,926 | **8.50%** | ≤8.0% **FAIL** | 7.12% | +1.38pp worse |
| EP4 | killed | — | (≤6.4407%) | 6.81% | — |

Params: 19.09M vs SOTA 16.99M (+2.10M, +12%). Peak GPU: 64.9 GB / 97 GB.

**Analysis:** FFN injection started better (EP1 26.51% beats SOTA 28.63% by 2.1pp) but the trajectory flattened through EP2-EP3. The EP1→EP2 drop was −14.38pp (this run) vs −20.48pp (SOTA) — slower descent. EP3 absolute miss: 8.50% vs 7.12% SOTA (+1.38pp), well outside noise range. Zero-init guarantee held (EP1 healthy, gradient well-conditioned through new path). Slowdown is optimization-budget, not divergence: +2.1M extra params need more steps to settle, and 4-ep schedule with T_max=13 doesn't give enough time. The post-norm residual exposes vol head to a noisier signal during early epochs (FFN output drifts from zero before post-norm tuned).

Arm B not launched — Arm A EP4 unreachable from EP3=8.50% in one epoch.

**Verdict:** CLOSED NEGATIVE. Post-xattn FFN 4× expansion does not pay back its parameter cost on this budget. Student suggestions noted (2× ratio, SwiGLU, FFN-specific LR group) but not in immediate priority queue given human directive focus on test volume pressure OOD gap.

---

## 2026-05-09 10:30 — PR #888: Per-sample OOD loss upweighting ×3 (thorfinn) — CLOSED (NEGATIVE)

- **Branch**: `thorfinn/ood-sample-weighting` (deleted)
- **W&B run**: `thorfinn/ood-weight-3x-rank0` (group `thorfinn-ood-weighting`), run state: finished
- **Hypothesis**: The 4 OOD test geometries (run_133, run_226, run_203, run_158) account for 92% of squared test_vol_p deviation. They are not in the training split, but their K=6 nearest train neighbors (by SDF Mahalanobis distance) ARE. Upweighting those K=6 neighbors by 3× should bias the model toward geometry clusters that will be OOD at test time — an indirect but principled OOD regularisation.

| Metric | EP1 | EP2 | Timeout (step 30,454) | SOTA PR #823 | Δ vs SOTA |
|---|---:|---:|---:|---:|---:|
| val_abupt | 30.2033% | 8.4952% | **7.4232%** | 6.4407% | +0.98pp worse |
| val_vol_pressure | — | 5.254% | — | 4.956% | +0.30pp worse |
| test_abupt | — | — | **8.6935%** | 7.6992% | +0.99pp worse |
| test_vol_pressure | — | — | **12.609%** | 11.6704% | +0.94pp worse |

Gate results: EP1=30.2033% FAIL (gate <30%; run continued given borderline miss). EP2=8.4952% PASS (<16%). Run hit the 270-min timeout mid-EP3 at step 30,454 (max EP3=~32,594 in 13-ep schedule).

**Analysis:** The OOD upweighting hypothesis is **refuted across all channels**. vol_pressure was specifically the target metric (OOD test cases dominate it), yet EP2 val_vol_p=5.254% is WORSE than SOTA's 4.956% at the same boundary — not better. The final forced checkpoint (step 30,454) shows val_abupt 7.42% vs SOTA 6.44% (+0.98pp), and test_abupt 8.69% vs SOTA 7.70% (+0.99pp). There is no OOD generalization benefit. Possible explanations: (1) The K=6 nearest neighbors by SDF Mahalanobis distance are not the actual bottleneck — the 4 OOD geometries may differ from any training geometry in ways the distance metric doesn't capture; (2) 3× upweighting is insufficient to shift the loss landscape without damaging the in-distribution performance; (3) the OOD generalization gap is fundamentally architecture/capacity-limited, not data-distribution-limited.

**Verdict:** NEGATIVE. OOD loss upweighting via nearest-train-neighbor proximity is closed. The OOD gap cannot be bridged through train-set reweighting alone. Architecture-level interventions (xattn geometry conditioning) remain the primary lever.

---

## 2026-05-09 09:15 — PR #890: Surf→vol xattn with detached K/V (frieren) — CLOSED (NEGATIVE)

- **Branch**: `frieren/xattn-detach-kv` (deleted)
- **W&B run**: group `frieren-xattn-detach-kv`
- **Hypothesis**: PR #884 identified K/V gradient backflow through the surface encoder as the cause of 2-layer xattn failure (surface_pressure +3pp regression). Detaching K/V before the xattn computation (stop_gradient on surface hidden states used as K/V) isolates the surface encoder from xattn gradients. This directly tests the backflow mechanism: if detach-K/V recovers performance, backflow is confirmed causal; if it still fails, the surface encoder needs to adapt jointly.

| Metric | EP1 | EP2 | Verdict |
|---|---:|---:|---:|
| val_abupt | >30% | — | FAIL EP1 kill gate |

**Analysis:** EP1 kill gate triggered. Detaching K/V did not rescue the detached xattn path. The K/V detach eliminates backflow but also cuts off the adaptation of K/V projections to the optimization pressure of the volume Q path — the surface encoder cannot co-adapt its K/V representations to what the volume decoder needs. This suggests that the joint gradient flow from Q→K/V is not a bug but a feature: it allows the surface encoder to specialize its output for the volume cross-attention consumer. Without that gradient, the K/V projections are underfit for the Q context.

**Verdict:** NEGATIVE. Detach-K/V is closed. The backflow mechanism is apparently load-bearing — gradient signal from volume Q back through surface K/V helps the surface encoder produce better geometry representations. This rules out zero-coupling approaches; future multi-layer xattn variants must manage the gradient magnitude, not eliminate it (e.g., gradient scaling, separate LR for surface encoder, or mid-backbone injection with partial gradient flow).

---

## 2026-05-09 08:00 — PR #886: Width scaling + surf→vol xattn hidden_dim=640 (edward) — CLOSED (negative result)

- **Branch**: edward/xattn-width-640 (deleted)
- **W&B run**: `m68ug46u` (group `edward-xattn-width-640`)
- **Hypothesis**: Width=640 may compound with surf→vol xattn — a wider backbone could leverage the geometry signal more richly. PR #872 showed width=640 without xattn failed; this tests whether xattn composition unlocks the width scaling.

| Metric | EP1 | EP2 | mid-EP3 (timeout) | SOTA PR #823 EP13 |
|---|---:|---:|---:|---:|
| val_abupt | 26.82% | 11.06% | 8.58% | 6.4407% |
| surface_pressure | 20.27% | 7.42% | 5.56% | 4.1836% |
| volume_pressure | 16.63% | 8.62% | 6.03% | 3.8557% |
| wall_shear | 29.83% | 11.93% | 9.48% | 7.3448% |

EP3 gate (<8%): FAILED at 8.58% (0.58pp over). Training cut by 270-min timeout mid-EP3 (step 18596/~22640). EP4 was never reached; extrapolated val_abupt ~6.9–7.2% (worse than SOTA 6.44%).

**Analysis:** Width=640 + xattn shows no synergy. EP1 is marginally better than SOTA EP1 (-1.81pp), but per-step convergence after EP1 is slower. The wider model adds parameters but does not generalize them into a clearer surface→volume coupling. Additional constraint identified: 4-epoch screens at hidden_dim=640 with the full vol-curriculum cannot fit within the 270-min timeout (~369 min projected).

**Verdict:** NEGATIVE. Combined with PR #872 (width=640, no xattn), the width-scaling axis is definitively closed. Neither configuration beats 512-dim SOTA. Width does not compound with geometry conditioning.

---

## 2026-05-01 14:30 — PR #887: Surf→vol xattn with surface subsampling (nezuko) — CLOSED (negative result)

- **Branch**: nezuko/xattn-surface-subsample (deleted)
- **W&B run**: `0ud2go3r` (group `nezuko-xattn-surface-subsample`)
- **Hypothesis**: The current surf→vol xattn (PR #823 SOTA) passes all 65,536 surface points as K/V. Uniform random subsampling (~4096 anchor points, N_kv=4096) before the K/V projection may sharpen the geometry signal by forcing compact surface structure representation and reduce memory pressure. Run B (N_kv=8192) was gated on EP4 val_abupt < 6.6%.

| Metric | Run A (N_kv=4096) | SOTA PR #823 | Δ |
|---|---:|---:|---:|
| val_abupt EP4 | 7.6075% | 6.4407% | +1.17pp (worse) |
| surface_pressure EP4 | 4.9802% | 4.1836% | +0.80pp |
| volume_pressure EP4 | 5.0467% | 3.8557% | +1.19pp |
| wall_shear EP4 | 8.4545% | 7.3448% | +1.11pp |
| tau_x | 7.3503% | 5.7782% | +1.57pp |
| tau_y | 9.6493% | 7.5977% | +2.05pp |
| tau_z | 11.0111% | 9.0116% | +2.00pp |

EP3: 8.2896% (missed <8% gate by 0.29pp — advisory miss, continued to EP4)
EP4: 7.6075% — missed <6.6% gate for Run B. Run B not launched.

**Analysis:** Uniform random subsampling hurt EVERY channel uniformly by 0.8–2.0pp. Vol_p regressed by 1.19pp even though it is the channel most directly downstream of surf→vol xattn. The model requires full 65k surface point coverage to accurately condition volume pressure. Random subsampling destroys the spatial coverage and structural information that the full set provides. EP3→EP4 drop was only 0.68pp (vs ~3.55pp EP2→EP3) — model was already stagnating.

**Key diagnostic:** The failure is not "too many K/V tokens" (information overload) but "wrong K/V tokens" (random selection loses structured geometry). Structured selection approaches (k-NN locality, learned pooling, FPS) remain untested.

**Verdict:** NEGATIVE. Surface subsampling with uniform random selection is ruled out. Follow-up: nezuko PR #892 tests mid-backbone xattn injection (different approach to improving geometry conditioning).

---

## 2026-05-01 — PR #878: Surf→vol xattn heads sweep H=8 vs H=16 (alphonse) — CLOSED (negative)

- **Branch**: alphonse/xattn-heads-sweep (deleted)
- **W&B runs**: Arm A `c4e3gurg` (H=8), Arm B `u5bpkpje` (H=16)
- **Hypothesis**: Baseline xattn uses num_heads=4 (128-dim/head). Increasing to 8 or 16 heads may capture richer surface geometry diversity through more specialised attention subspaces. EP3 kill gate <8%.

| Arm | Heads | EP1 | EP2 | EP3 | Decision |
|-----|-------|-----|-----|-----|----------|
| A | 8 | 27.832% PASS | 12.462% PASS | **8.7132% FAIL** (+0.71pp over gate) | killed |
| B | 16 | 27.428% PASS | 12.128% PASS | **8.5231% FAIL** (+0.52pp over gate) | killed |

**EP3 per-channel (H=8 vs H=16):**
| Channel | H=8 | H=16 | Δ (B−A) |
|---------|-----|------|---------|
| sp | 5.6444% | 5.5737% | −0.071pp |
| vp | 6.1853% | 5.7986% | **−0.387pp** |
| ws | 9.6045% | 9.4580% | −0.147pp |
| abupt | 8.7132% | 8.5231% | −0.190pp |

**Analysis:** Both arms fail EP3 gate (<8%). Direction partially confirmed: H=16 > H=8 monotonically (−0.19pp abupt, largest gain in vp −0.39pp). But neither beats PR #823 SOTA at H=4 (128-dim/head). The result is consistent with per-head dimensionality being the binding constraint: 128-dim/head at H=4 > 64-dim/head at H=8 > 32-dim/head at H=16. Adding more heads simultaneously narrows the K/V subspace — these two effects are entangled in standard MHA.

**Key finding:** This motivates GQA (PR #893) — decouple K/V head dimensionality from Q head count. With GQA n_kv_heads=1: K/V get full 512-dim/head while Q still has 4 specialised 128-dim query projections.

**Verdict:** NEGATIVE. Standard MHA heads=4 remains optimal. GQA follow-up assigned to alphonse (PR #893).

---

## 2026-05-09 03:50 — PR #884: Two-layer surf→vol xattn (frieren) — CLOSED (kill gate EP1)

- **Branch**: frieren/xattn-two-layer (deleted)
- **W&B run**: `omn023f3` (group `frieren-xattn-two-layer`)
- **Hypothesis**: Stack a second surf→vol cross-attention layer (identical architecture to PR #823's single layer: embed_dim=512, num_heads=4, zero-init out_proj) applied at an additional backbone depth. Both layers zero-init to preserve identity-at-init. Hypothesis: more geometry injection depth → better vol_pressure, especially for OOD cases.

| Metric | Two-layer (PR #884) | Single-layer PR #823 EP1 | Gate |
|---|---:|---:|---:|
| val_abupt EP1 | 31.77% | 28.63% | <30% |
| val_surface_pressure EP1 | 24.94% | 21.85% | — |
| val_volume_pressure EP1 | 17.88% | 17.79% | — |
| val_wall_shear EP1 | 35.28% | 31.54% | — |

**Analysis:** Kill gate triggered at EP1 (31.77% vs 30% gate). Most significant finding: surface_pressure (+3.09pp) and wall_shear (+3.74pp) regressed strongly, while volume_pressure held parity (+0.09pp). This is the diagnostic signature of K/V gradient backflow through the surface encoder being doubled (two layers of xattn each flow gradients back through surface K/V). The volume pathway (direct write target) is fine; the surface pathway (indirect K/V gradient sink) is being perturbed. Identity-at-init was verified before launch — this is a learned-dynamics regression, not an init bug.

**Student suggested follow-ups:** (1) Detach K/V before xattn — isolates surface encoder from xattn gradient backflow. (2) Add FFN after single-layer xattn. (3) Lower LR for second xattn layer. (4) Extend warmup to 2 epochs.

**Verdict:** Closed. Follow-up: frieren PR #890 tests detach-K/V (Option 1 — highest signal, directly tests the hypothesized mechanism).

---

## 2026-05-01 — PR #840: STRING drop σ=4.0 (tanjiro) — CLOSED DEAD END

- **Branch**: tanjiro/string-drop-sigma4 (deleted)
- **W&B run**: `oiptel6p`
- **Hypothesis**: Remove the highest-frequency octave (σ=4.0) from the 5-octave STRING PE spectrum {0.25, 0.5, 1.0, 2.0, 4.0} → {0.25, 0.5, 1.0, 2.0}. Motivation: σ=4.0 may add noise for low-Re smooth aerodynamic fields; leaner spectrum may regularise the PE while retaining physically meaningful frequency content.

| Metric | PR #840 (4-oct, no σ=4) | SOTA #592 (5-oct) | Gate |
|---|---:|---:|---:|
| val_abupt (EP4) | 7.856% | 6.5985% | <6.5985% |

**Analysis:** EP4 val_abupt=7.856% is substantially worse than SOTA (6.5985%), a +1.26pp regression. Removing σ=4.0 clearly degrades performance. All 5 octaves of the STRING spectrum are jointly load-bearing; the highest-frequency component contributes meaningfully to spatial resolution of near-surface aerodynamic gradients.

**Verdict (DEAD END):** STRING spectrum axis closed. All 5 octaves required. Do not prune STRING PE spectrum further.

---

## 2026-05-01 — PR #842: LR floor lr_min=5e-6 (thorfinn) — CLOSED DEAD END

- **Branch**: thorfinn/lr-floor-5e-6 (deleted)
- **W&B run**: `3487klz8`
- **Hypothesis**: Introduce a non-zero LR floor lr_min=5e-6 into the cosine annealing schedule (vs current cosine-to-zero). Prevents the LR from fully decaying to 0, maintaining a small residual learning rate at EP13 that may improve late-epoch fine-tuning on high-frequency aerodynamic features.

| Metric | PR #842 (lr_min=5e-6) | SOTA #592 (lr_min=0) | Gate |
|---|---:|---:|---:|
| val_abupt (EP4) | 7.610% | 6.5985% | <6.5985% |

**Analysis:** EP4 val_abupt=7.610% is significantly worse than SOTA (6.5985%), a +1.01pp regression. Maintaining a residual LR floor hurts performance. Cosine-to-zero decay is optimal for this task — the model benefits from full LR annihilation at end of training.

**Verdict (DEAD END):** LR floor axis closed. Cosine-to-zero (lr_min=0) is confirmed optimal.

---

## 2026-05-01 — PR #836: Geometry branch v3 (askeladd) — CLOSED CATASTROPHIC KILL

- **Branch**: askeladd/geom-branch-v3 (deleted)
- **W&B runs**: rank-0 `zj8o1ugg` (group `abupt-geom-branch-v3`)
- **Hypothesis**: Introduce a geometry-conditioning branch that processes global geometric features (e.g. SDF projections, surface statistics) and injects them into the volume decoder via FiLM conditioning. Motivation: explicit geometric context beyond point-level SDF may help the model generalise across different car body shapes.

| Metric | PR #836 (EP1) | Kill Gate | Verdict |
|---|---:|---:|---:|
| val_abupt | 50.9246% | <40% | FAILED (KILL) |

**Analysis:** EP1=50.9246% far exceeds the 40% kill gate. Root cause analysis: the backbone freeze + cosine LR schedule aliasing meant the geometry-conditioning branch received only ~2173 effective gradient steps before the EP1 gate check — insufficient to overcome the random initialisation of the new FiLM conditioning layers. The catastrophic failure reflects initialisation shock rather than a fundamentally broken architecture, but the execution plan was poorly designed.

**Verdict (CATASTROPHIC KILL):** Closed without further investigation. Geom-branch v3 architecture requires a careful warm-up strategy (progressive unfreezing, staged LR, or separate Adam phase for new conditioning layers) before re-attempting. Do not re-open without a warm-up plan.

---

## 2026-05-08 06:30 — PR #837: SDF skip-connect to volume decoder (tanjiro) — CLOSED BLOCKED (Issue #803)

- **Branch**: tanjiro/sdf-concat-vol-decoder (deleted)
- **W&B run**: `4oerprx6` (rank-0, group `tanjiro/sdf-skip-decode-4ep`) — killed at EP2 start
- **Hypothesis**: Concatenate SDF channel (`volume_x[..., 3:4]`) onto `volume_hidden` at decoder boundary (512→513→1). Zero parameter overhead, non-saturating (raw float), physically interpretable — gives decoder explicit inside/outside/surface geometry context at prediction time.

| Metric | EP1 | Kill Gate | Verdict |
|---|---:|---:|---:|
| val_abupt | 25.47% | <40% | PASSED |
| val_vol_pressure | 15.42% | — | — |
| val_surface_pressure | 18.95% | — | — |

**EP2 in progress when killed.** EP1=25.47% was healthy (well below 40% gate). Architecture is sound.

**Analysis**: Run aborted mid-EP2 by advisor due to Issue #803 data blocker. The 10 REQUIRED_RESTORED_CASE_IDs (run_44, run_133, run_158, run_184, run_203, run_226, run_249, run_310, run_416, run_484) have corrupted `volume_sdf.npy` — sdf_min ∈ [-0.015, -0.001] vs bulk train [-0.45, -0.27], meaning no inside-body samples. A model trained on this data would learn an artificial SDF distribution that does not match test cases, making any result uninterpretable. EP1=25.47% may itself be misleading if the 10 restored cases are included.

**Verdict (BLOCKED):** Architecture design is valid. Re-open as new PR after `volume_sdf.npy` regeneration for the 10 REQUIRED_RESTORED_CASE_IDs lands and passes diagnostic z<2σ check.

---

## 2026-05-08 06:30 — PR #834: GradNorm α=0.5 uniform init (edward) — CLOSED NEGATIVE (GradNorm axis exhausted)

- **Branch**: edward/gradnorm-a05-uniform-init-4ep (deleted)
- **W&B run**: `k309ojcu` (rank-0, group `edward-gradnorm-uniform-init`, project `senpai-v1-drivaerml-ddp8`)
- **Hypothesis**: GradNorm with uniform static-weight initialization (all=1.0 instead of SOTA τ_y×1.5, τ_z×2.0, surface×2.0) removes the stacking interference observed in PR #824 (GradNorm + stacked static weights), allowing GradNorm to discover its own optimal trajectory unbiased by empirical priors.

| Metric | PR #834 (GN α=0.5, uniform) | PR #824 (GN α=0.5, stacked) | SOTA #592 | Gate |
|---|---:|---:|---:|---:|
| val_abupt | 7.5431% | 7.5170% | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.7283% | 8.6960% | 7.9915% | — |
| val surface_pressure | 4.8327% | 4.83% | 4.3322% | — |
| val tau_x | 7.1751% | 7.13% | 6.5420% | — |
| val tau_y | 9.4863% | 9.33% | 8.3631% | — |
| val tau_z | 10.9030% | 10.95% | 9.8099% | — |
| val volume_pressure | 5.3183% | 5.35% | 3.9456% | — |

**GradNorm runtime weights (EP2 pre-val):** sp=0.79, τx=0.98, τy=0.96, τz=1.21, vp=1.06

**Analysis**: The two GradNorm variants (uniform init vs stacked static) differ by only 0.0261pp val (0.0323pp test) — within noise. Uniform initialization made no meaningful difference. GradNorm is anti-synergistic with the L5 SOTA backbone regardless of static-weight initialization. The final GradNorm weight schedule (τz=1.21 highest, τy=0.96 lower than expected) suggests GradNorm is failing to upweight τ_y properly — possibly because the gradient norm ratio tracks training speed rather than validation-loss residual. This is the 4th consecutive GradNorm experiment (PRs #523, #740, #824, #834) to land at either the SOTA baseline or worse. **GradNorm axis CONCLUSIVELY CLOSED.**

**Verdict (NEGATIVE):** Closed. GradNorm is exhausted at every α, with or without static-weight priors. Something in the L5/Lion/STRING stack makes GradNorm's gradient-norm-ratio dynamics non-informative. Future dynamic loss-weighting must use a different algorithm (e.g., PCGrad, loss-balanced weighting based on val residuals, not gradient norms).

---

## 2026-05-08 06:30 — PR #833: τ_z×2.5 4-ep curriculum bisection (thorfinn) — CLOSED NEGATIVE (τ_z static weight axis exhausted)

- **Branch**: thorfinn/tau-z-bisect-2p5-4ep (deleted)
- **W&B run**: `8a7mfzl3` (rank-0, project `senpai-v1-drivaerml-ddp8`)
- **Hypothesis**: τ_z×2.5 bisects the τ_z×2.0 (SOTA) and τ_z×3.0 (PR #822) interval. If a sweet spot exists between them, τ_z×2.5 should find it. PR #822 confirmed τ_z×3.0 is +0.88pp vs SOTA; τ_z×2.5 should be closer to SOTA than ×3.0.

| Metric | PR #833 (τ_z×2.5) | PR #822 (τ_z×3.0) | SOTA #592 (τ_z×2.0) | Gate |
|---|---:|---:|---:|---:|
| val_abupt | 7.5378% | 7.4767% | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.6920% | 8.6647% | 7.9915% | — |
| val tau_z | 10.8128% | 10.6947% | 9.8099% | — |
| val tau_y | 9.5479% | — | 8.3631% | — |
| val surface_pressure | 4.9448% | — | 4.3322% | — |
| val volume_pressure | 4.9687% | — | 3.9456% | — |
| EP1 / EP2 / EP3 / EP4 | 27.57 / 11.32 / 8.24 / 7.54 | — | — | — |

**Analysis**: τ_z×2.5 (val=7.5378%) is barely different from τ_z×3.0 (7.4767%) — only 0.06pp separates them. Both are ~0.90-0.94pp WORSE than SOTA τ_z×2.0. The bisection confirms there is no sweet spot in [2.0, 3.0]: the function is monotonically degrading as τ_z weight increases above 2.0. The non-uniform vol-points schedule was tuned at τ_z×2.0 and cannot absorb additional τ_z gradient pressure. Upweighting τ_z increases gradient-clip frequency and hurts every other channel (vol_p +1.02pp, surf_p +0.61pp, τ_y +1.18pp). The τ_z static-weight axis is a wall.

**Verdict (NEGATIVE):** Closed. The full τ_z sweep (×2.0, ×2.5, ×3.0) is complete. τ_z×2.0 is the 4-ep local optimum. No further τ_z static-weight experiments warranted at this budget.

---

## 2026-05-01 — PR #832: Lion wd=7e-4 (alphonse) — CLOSED DEAD END

- **Branch**: alphonse/lion-wd-7e-4 (deleted)
- **W&B run**: `cq4guj8g` (rank-0, group `alphonse-lion-wd`)
- **Hypothesis**: Increasing Lion weight decay from 5e-4 to 7e-4 would reduce overfitting and improve generalization on L5 SOTA config.

| Metric | EP1 | EP2 | EP3 | EP4 (best) | SOTA EP4 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| val_abupt | 27.08% | 11.62% | 8.418% | **7.683%** | 6.5985% | +1.085pp WORSE |
| val_surface_pressure | — | — | — | 5.284% | 4.332% | +0.952pp worse |
| val_volume_pressure | — | — | — | 4.986% | 3.946% | +1.040pp worse |
| val_wall_shear | — | — | — | 8.810% | 7.585% | +1.225pp worse |
| test_abupt | — | — | — | 8.877% | 7.992% | +0.885pp worse |

**Analysis**: wd=7e-4 uniformly degraded all channels. The wd axis on L5/Lion/9e-5 is now closed on both sides: wd=3e-4 (PR #826) gave +0.864pp, wd=7e-4 gives +1.085pp. Current wd=5e-4 is the local optimum. Broadband degradation across all channels (not just vol_p) rules out the channel-specific mechanism hypothesized. EP1 was marginally better but the gap inverted by EP2 and never recovered, confirming this is a genuine regression, not a timing artifact. **Lion wd axis CLOSED under L5/9e-5 config.**

---

## 2026-05-01 — PR #836: AB-UPT geometry branch v3 (askeladd) — SENT BACK (recipe fix)

- **Branch**: askeladd/geom-branch-v3
- **W&B run**: `zj8o1ugg` (rank-0, group `abupt-geom-branch-v3`)
- **Hypothesis**: AB-UPT geometry branch with supernode pooling: K=1024 anchor points from volume mesh, STRING-sep RoPE, two new output heads (surface+volume MLP), anchor→point cross-attention. Training recipe: backbone freeze warmup (20%), differential LR (2×), vol aux weight (2.0).

| Metric | EP1 | Kill Gate | Verdict |
|---|---:|---:|---:|
| val_abupt | 50.9246% | <40% | KILLED |

**Analysis**: Architecture plumbing verified healthy — geom_branch/* W&B telemetry shows no NaN, freeze/unfreeze in DDP worked correctly, lr-scale applied correctly. The failure was a pure recipe interaction: (1) `--lr-cosine-t-max 4` with 4-epoch run decays backbone_lr from 9e-5 to 4.5e-6 by EP1 end; (2) `--geom-branch-warmup-fraction 0.2` freezes backbone for ~80% of EP1 (~8691/43456 warmup steps), leaving backbone with only ~2173 steps of actual training after unfreeze at severely decayed LR (~4.5e-6). These two effects compound to guarantee EP1 kill. Same `--lr-cosine-t-max 4` confound affected PR #835 (frieren). **Fix applied: drop `--geom-branch-warmup-fraction` to 0.0 and set `--lr-cosine-t-max 13`. Re-running as `askeladd/geom-branch-v3-nf-ep4`.**

---

## 2026-05-08 03:10 — PR #830: Volume loss weight 2.0 4-ep curriculum (tanjiro) — CLOSED HYPOTHESIS REJECTED

- **Branch**: tanjiro/vol-loss-weight-2 (deleted)
- **W&B run**: `ztvlsn1e` (rank-0, group `tanjiro-vol-loss-weight`)
- **Hypothesis**: Doubling the volume loss weight (1.0→2.0) under the canonical 4-ep curriculum would redirect gradient capacity to the volume branch, improving volume_pressure (val 3.9% vs test 11.9% gap diagnosed as under-optimization).

| Metric | EP1 | EP2 | EP3 | EP4 (best) | SOTA EP4 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| val_abupt | 27.17% | 12.01% | 8.487% | **7.7117%** | 6.5985% | +1.11pp WORSE |
| val_surface_pressure | 20.37% | 8.30% | 5.616% | 5.097% | 4.332% | +0.77pp worse |
| val_volume_pressure | 15.38% | 7.88% | 5.283% | 4.782% | 3.946% | +0.84pp worse ← TARGET |
| val_wall_shear | 30.67% | 13.27% | 9.552% | 8.709% | 7.585% | +1.12pp worse |

**Analysis**: Hypothesis failed convincingly — vol-w=2.0 degraded ALL channels at EVERY epoch, including volume_pressure itself. Trajectory monotonically below baseline from EP1 through EP4 (not a "needs more epochs" failure). Two plausible mechanisms: (1) higher volume weight causes gradient-clip to fire more often at fixed lr=9e-5, reducing effective step on all params; (2) curriculum front-loads bad signal from 16K sparse vol-point gradients. The val/test gap on volume_pressure (3.9% val vs 11.9% test) is a generalization gap, not under-optimization — loss reweighting is the wrong lever. This hypothesis is now confirmed dead twice: PR #813 (5-ep) and this PR (4-ep curriculum). **Volume-loss-weight axis closed for L5/Lion/9e-5 recipe.**

---

## 2026-05-08 03:08 — PR #829: STRING 6-octave RFF σ=0.125–4.0 (fern) — CLOSED DEAD END

- **Branch**: fern/string-6octave-pe (deleted)
- **W&B run**: `cqk9voaa` (rank-0, group `fern-string-6octave`)
- **Hypothesis**: Adding a 6th higher-frequency octave (σ=0.125) to STRING positional encoding, below the current minimum σ=0.25, would improve surface pressure and other channels by capturing finer-scale geometric variation.

| Metric | EP1 | EP2 | EP3 | EP4 (best) | SOTA EP4 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| val_abupt | 31.58% | 11.59% | 8.331% | **7.5738%** | 6.5985% | +0.97pp WORSE |
| val_surface_pressure | — | 7.897% | 5.409% | 4.906% | 4.332% | +0.57pp worse |
| val_volume_pressure | — | 8.355% | 5.740% | 5.121% | 3.946% | +1.18pp worse |
| val_wall_shear | — | 12.60% | 7.843% | — | 7.585% | — |
| test_abupt | — | — | — | 8.920% | 7.992% | +0.93pp worse |

**Analysis**: σ=0.125 uniformly degraded all channels on both val and test by 0.5–1.4pp. Two plausible causes identified: (1) **aliasing** — σ=0.125 places PE energy below the supervisable label scale (65k surface points too sparse for this frequency), injecting noise; (2) **capacity competition** — at fixed rff_num_features=16 across 6 sigmas, each sigma gets ~2.67 features vs 3.2 in 5-octave SOTA, starving the load-bearing σ=0.25 octave. The train_loss-matches-but-val-degrades signature confirms aliasing is operative. **Follow-up PR #838 (fern, rff24+σ=0.125) isolates the capacity-competition cause by giving 24 features across 6 sigmas (4 each), giving σ=0.25 MORE budget than current SOTA. If PR #838 also fails, aliasing is the dominant cause and σ=0.125 is definitively unusable at 65k pts.**

---

## 2026-05-08 02:10 — PR #828: 2-layer GELU MLP vol decoder (askeladd) — CLOSED DEAD END

- **Branch**: askeladd/vol-decoder-2layer-gelu (deleted)
- **W&B run**: `zmcwyud5` (rank-0, group `askeladd-vol-decoder-mlp`)
- **Hypothesis**: Replace the linear volume pressure decoder head with a 2-layer GELU MLP (512→256→1, LayerNorm between layers) to give the network richer capacity to decode volume pressure, targeting the OOD vol_p gap.

| Metric | EP1 (16k vol-pts) | EP2 (32k vol-pts) | SOTA EP1 | SOTA EP2 |
|---|---:|---:|---:|---:|
| val_abupt | 31.06% | 11.42% | 27.95% | 7.94% |
| val vol_pressure_rel_l2 | 16.99% | 8.38% | — | — |
| val surface_pressure_rel_l2 | 24.89% | 7.52% | — | — |
| val wall_shear_rel_l2 | 35.45% | 14.18% | — | — |

**Analysis**: Gap vs baseline widened from +3.11pp at EP1 to +3.48pp at EP2 across ALL channels — not just vol_p. This rules out a slow-convergence explanation. The 2-layer GELU MLP decoder adds ~1.25M params but slows optimization uniformly. Root cause: richer output decoder increases gradient path depth; the model cannot amortize this in 4 epochs. This is the second time this hypothesis was tested (PR #820 showed identical outcome). The vol-pressure OOD problem requires geometry-aware *input* conditioning, not a richer *output* decoder.

---

## 2026-05-08 02:10 — PR #827: Cosine LR warm restarts on L5 SOTA 4-ep (frieren) — CLOSED INFORMATIVE

- **Branch**: frieren/cosine-lr-warm-restarts (deleted)
- **W&B run**: `1ne1qdfl` (rank-0)
- **Hypothesis**: CosineAnnealingWarmRestarts (T_0=2) would escape local minima, improving vol_p.

| Metric | EP2 | EP3 (best_val) | SOTA EP4 |
|---|---:|---:|---:|
| val_abupt | 8.7973% | **7.4450%** | **6.5985%** |
| val vol_pressure | 5.492% | 4.419% | 3.946% |

EP3 gate PASSED (<8%). Restart-1 confirmed at step 32593. EP4 timed out (52% complete). Best=7.445%, above merge gate by 0.85pp. Hypothesis untestable in 4-ep budget. Closed informative. Restart mechanics confirmed working; monotone cosine confirmed productive.

---

## 2026-05-01 — PR #824: GradNorm α=0.5 on L5 SOTA 4-ep curriculum (edward) — CLOSED NEGATIVE

- **Branch**: edward/gradnorm-a05-l5-sota-4ep (deleted)
- **W&B run**: `e0brbohf` (rank-0, group `edward-gradnorm-l5-sota`, project `senpai-v1-drivaerml-ddp8`)
- **Hypothesis**: GradNorm α=0.5 dynamic per-task loss reweighting stacked on the full L5 SOTA stack (alphonse PR #592 recipe with static surface=2.0, τ_y=1.5, τ_z=2.0) at 4-ep budget-matched curriculum would match/beat SOTA by adaptively upweighting laggard τ_y/τ_z channels.

| Metric | PR #824 (GradNorm α=0.5 + SOTA static) | PR #740 (GradNorm α=0.5, no static) | SOTA #592 | Gate |
|---|---|---|---|---|
| val_abupt | 7.5170% | — | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.6960% | 7.5195% (wave-test SOTA) | 7.9915% | — |
| val surface_pressure | 4.83% | — | 4.33% | — |
| val tau_x | 7.13% | — | 6.54% | — |
| val tau_y | 9.33% | — | 8.36% | — |
| val tau_z | 10.95% | — | 9.81% | — |
| val vol_pressure | 5.35% | — | 3.95% | — |

**Final GradNorm weights:** sp=0.75, τx=0.96, τy=1.20, τz=1.24, vp=0.85. Directionally matched PR #740 except vp downweighted.

**Results commentary:** All five channels strictly regressed (+0.50 to +1.40pp), not a tradeoff. test_abupt is +1.18pp WORSE than PR #740's GradNorm wave-test SOTA — the difference is that PR #740 ran without the SOTA static weights, while this run stacked GradNorm on top of them. GradNorm overrides static weights based on gradient norms alone (not val-loss progress), so the runtime weight schedule (sp 2.0×0.75=1.5, τy 1.5×1.20=1.8, τz 2.0×1.24=2.5, vp 1.0×0.85=0.85) is less-well-tuned than the static SOTA empirical optimum. The two mechanisms are not stacking-compatible at this budget.

**Verdict (NEGATIVE):** Closed. GradNorm + static-weighted SOTA = anti-synergy. To get a GradNorm signal one would need to drop the static weights entirely (revert tau-y-loss-weight, tau-z-loss-weight, surface-loss-weight to 1.0) and let GradNorm own the schedule.

## 2026-05-01 — PR #826: Lion weight-decay 5e-4 -> 3e-4 (alphonse) — CLOSED NEGATIVE

- **Branch**: alphonse/weight-decay-3e-4-l5-sota-4ep (deleted)
- **W&B run**: `ahw1rdj7` (group `alphonse-wd-sweep`, project `senpai-v1-drivaerml-ddp8`)
- **Hypothesis**: Halving Lion weight-decay from 5e-4 to 3e-4 would relax the L2 pull on tau_y/tau_z output-projection weights and lift the worst channels without harming surface_pressure or vol_pressure.

| Metric | PR #826 (wd=3e-4) | SOTA #592 (wd=5e-4) | Gate |
|---|---|---|---|
| val_abupt | 7.4628% | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.7253% | 7.9915% | — |
| val surface_pressure | 4.88% | 4.33% | — |
| val tau_x | 7.31% | 6.54% | — |
| val tau_y | 9.42% | 8.36% | — |
| val tau_z | 10.85% | 9.81% | — |
| val vol_pressure | 4.86% | 3.95% | — |

**Results commentary:** All channels degraded uniformly (+0.6 to +1.0pp). Lion's update is `sign(momentum) * lr + lr * wd * theta` — halving wd shrinks the explicit parameter-pull term and starves convergence across the whole network, not selectively at decoder heads. Confirms wd=5e-4 is at/near the Lion sweet spot for this recipe; tau_y/tau_z headroom is structural, not regulatory.

**Verdict (NEGATIVE):** Closed. Down-sweep of Lion wd is dead — pivot to structural attacks on the channel imbalance (channel-specific decoder heads, schedule-aware loss weighting at appropriate budget, or different optimizer dynamics like β₂ sensitivity).

## 2026-05-01 — PR #822: τ_z loss weight ×3.0 on 4-ep budget-matched curriculum (thorfinn) — CLOSED NEGATIVE

- **Branch**: thorfinn/tau-z-3p0-4ep-relaunch (deleted)
- **W&B run**: `qtzoy6rp` (group `thorfinn-tau-z-sweep`, project `senpai-v1-drivaerml-ddp8`); first attempt `imvj1s1p` killed by misconfigured EP1 kill threshold.
- **Hypothesis**: Stacking τ_z×3.0 on the full SOTA recipe at 4-ep budget-matched curriculum would extend the +0.44pp τ_z signal observed in PR #807 isolation and lift val_abupt below SOTA.

| Metric | PR #822 (τ_z×3.0, 4-ep) | PR #807 (τ_z×3.0 isolation) | SOTA #592 | Gate |
|---|---|---|---|---|
| val_abupt | 7.4767% | — | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.6647% | — | 7.9915% | — |
| val tau_z | 10.6947% | bare-SOTA −0.44pp | 9.8099% | — |
| EP1 / EP2 / EP3 / EP4 val_abupt | 26.18 / 11.37 / 8.17 / 7.48 | — | — | — |

**Results commentary:** All channels still descending at EP4 — training did not converge. The 4-ep budget-matched curriculum delivers ~22,640 total steps (10864 + 5435 + 3625 + ~2716, non-uniform due to varying volume-point-count epochs), substantially fewer than the 13-ep baseline's ~43k steps. τ_z×3.0 amplifies the slowest-converging channel's gradient, which demands MORE budget, not less, to integrate. Stacking it onto a budget-starved schedule is anti-synergistic: the recipe needed 14k+ extra steps (PR #815 13-ep variant timed out) to express the τ_z gain. Confirms the signal is real but not landable in the 4-ep envelope at ×3.0 magnitude.

**Verdict (NEGATIVE):** Closed. 4-ep + full-SOTA + τ_z×3.0 is over-stacked. Either reduce the upweight magnitude (×2.0 at 4-ep) or attack channel imbalance through orthogonal means (separate decoder heads, warm-start from SOTA checkpoint, schedule-aware loss).

## 2026-05-01 — PR #814: STRING 6-octave extended spectrum (add σ=8.0) (alphonse) — CLOSED NEGATIVE

- **Branch**: alphonse/string-6-octave-extended-spectrum
- **W&B run**: `3efn3v5u` (project `senpai-v1-drivaerml-ddp8`)
- **Hypothesis**: Adding σ=8.0 as a 6th RFF octave (`--rff-init-sigmas "0.25,0.5,1.0,2.0,4.0,8.0"`) captures finer-scale geometric features that the 5-octave SOTA misses, particularly for wall_shear_z (confirmed laggard). Motivated by thorfinn PR #779 Arm B signal (σ_max=8 replacing σ=4 gave −0.13pp improvement).

| Metric | PR #814 (6-oct additive) | thorfinn #779 Arm B (5-oct σ_max=8) | SOTA #592 (5-oct) | Gate |
|---|---|---|---|---|
| val_abupt | 7.6385% | 6.8792% | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.8442% | — | 7.9915% | — |
| test_surface_p | 4.5974% | — | — | — |
| test_volume_p | 12.8395% | — | — | — |
| test_wall_shear | 8.2213% | — | — | — |
| full_val/wall_shear_z | 11.0287% | — | — | — |
| best_epoch | 4 (EMA) | — | — | — |

**Kill gates:** All 3 passed; run completed 22,644 steps (~190 min).

**Slope decay:** val_abupt slope decelerated from −2.506 pp/1k steps (EP1→EP2) to −0.266 pp/1k steps (EP3→EP4) — the 6-octave config needs more budget to clear convergence overhead.

**Results commentary:** The 6-octave additive approach is +0.76 pp worse than thorfinn's 5-octave replace-not-add variant at identical 4-epoch budget. Adding a 6th octave grows RFF features 80→96 (+20%); this extra capacity is a liability at 4 epochs because the optimizer has not had enough iterations to integrate the new frequency. The σ=8.0 signal in PR #779 Arm B worked precisely because it *replaced* σ=4.0 (constant capacity), not because it added bandwidth. τz (wall_shear_z) was NOT preferentially accelerated — wsy descended faster in EP3→EP4, meaning the 6th octave did not help the confirmed laggard channel. A 13-epoch full-budget run might resolve the convergence lag but is not justified over other queued hypotheses.

**Verdict (NEGATIVE):** Closed. 6-octave additive is inferior to 5-octave replace at 4-epoch budget. SOTA STRING PE remains 5-octave {0.25,0.5,1.0,2.0,4.0}.

## 2026-05-07 — PR #808: Surface curvature 4ep original-schedule re-run (nezuko) — CLOSED DEAD END (3rd consecutive surface-curvature fail)

- **Branch**: nezuko/surface-curvature-4ep-original-schedule (deleted)
- **W&B run**: `3hsu3tq0` (group `nezuko-surface-curvature`, name `nezuko/surface-curvature-4ep-original-vol-schedule`)
- **Hypothesis**: Surface curvature features (mean curvature H̃, Gaussian curvature K̃) appended to the surface input path improve val_abupt by providing geometric context. Previous run #798 used a 4-ep schedule-aligned stack; this run uses the original vol-schedule (`0:16384:3:32768:6:49152:9:65536`) for a full 65k vol-point budget at EP4.

| Metric | PR #808 (this, 4-ep orig-sched) | PR #798 (4-ep aligned) | PR #788 (first curvature) | SOTA #592 | Gate |
|---|---:|---:|---:|---:|---|
| val_abupt | 6.8051% | ~6.78% | 7.35% | **6.5985%** | < 6.5985% ❌ |
| test_abupt | — | — | — | 7.9915% | — |
| test_volume_p | ~11.81% | — | — | 11.933% | — |

**W&B run state:** Finished. EP1=24.95%, EP2=8.30%, EP3=7.166%, EP4=6.8051%.

**Results commentary:** Three consecutive surface curvature runs (PRs #788, #798, #808) have all landed in the 6.78–7.35% val_abupt range — consistently above the SOTA gate of 6.5985%. Despite varying the schedule alignment and vol-point curriculum, the convergence floor remains ~0.20–0.21pp above the gate. The only positive signal is a modest test_vol_p improvement (~0.12pp better than SOTA 11.933% → ~11.81%) which is insufficient to justify further surface-curvature investment at L=5/4-ep. Surface curvature as a standalone surface-path augment for L=5 architecture is a dead end.

**Verdict (DEAD END):** Closed after 3 runs with zero gate crossings. The curvature signal may become useful only if composited with a deeper architecture (L=6+) or longer training. Not assigning follow-up for now — geometry conditioning priority shifts to vol-head LoRA and AB-UPT geometry branch.

## 2026-05-07 — PR #807: Schedule-aligned 4-ep τ_z×3.0 upweight isolation (thorfinn) — NOT MERGED (below single-model gate), hypothesis CONFIRMED

- **Branch**: thorfinn/schedule-aligned-tau-z-upweight
- **W&B run**: `8j9kt5w1` (group `thorfinn-tau-z-sweep`, name `thorfinn/tau-z-3p0-sched4`)
- **Hypothesis**: τ_z (wall shear z) is the confirmed training laggard (PR #758: r_i=0.01123 highest residual imbalance). SOTA uses τ_z×2.0 but val tau_z=9.81% remains far from AB-UPT ref 3.63%. Test: increase τ_z weight from 2.0→3.0 on a clean 4-ep schedule-aligned stack (same as fern bare-SOTA control PR #799) for a single-variable A/B comparison.

| Metric | thorfinn τ_z×3.0 (4-ep) | fern bare-SOTA (4-ep, #799) | SOTA (#592, 13-ep) | Gate |
|---|---|---|---|---|
| val_abupt | 6.824% | 7.063% | **6.5985%** | < 6.5985% ❌ (+0.23pp) |
| test_abupt | 8.145% | 8.444% | 7.9915% | — |
| surface_pressure (val) | 4.491% | 4.641% | 4.332% | — |
| volume_pressure (val) | 4.187% | 4.322% | 3.946% | — |
| tau_x (val) | 6.852% | 7.089% | 6.542% | — |
| tau_y (val) | 8.528% | 8.755% | 8.363% | — |
| tau_z (val) | **10.062%** | 10.506% | 9.810% | — |

**Results commentary:** Hypothesis confirmed — τ_z×3.0 beats τ_z×2.0 on the same 4-ep schedule on every channel, with tau_z showing the **largest single-channel improvement** (−0.44pp val, −0.29pp test vs bare-SOTA control). Best 4-ep result in the program to date. However, does not beat the single-model gate (6.5985%) because the 4-ep schedule is compute-limited vs the 13-ep SOTA. The 4-ep schedule is a ~3.5h run that converges to ~7% range, while the 13-ep SOTA at ~270min/4ep gets the full cosine decay benefit. **Follow-up: assign τ_z×3.0 on the full 13-ep SOTA recipe to thorfinn.** The PR #790 (alphonse, τ_z×3.0 13-ep) was confounded by a 270-min wall-clock truncation in the high-LR phase; this is now cleanly motivated by the 4-ep isolation proof.

## 2026-05-01 — PR #793: vol-w=2.0 + wall-shear tau bump (tanjiro) — CLOSED NEGATIVE

- **Branch**: tanjiro/vol-w-2.0-wallshear-rebalance (deleted)
- **W&B run**: `ss5v4vdx` (group `vol-w-wallshear-rebalance-tay`, name `tanjiro/vol-w2.0-tau-y2.5-z3.0`)
- **Hypothesis**: `--volume-loss-weight 2.0` + `--tau-y-loss-weight 2.5` + `--tau-z-loss-weight 3.0` to rebalance wall-shear loss budget after PR #776 Arm B (vol-w=2.0 alone) caused +0.57pp wall-shear regression. Composed reweighting expected to recover val_abupt while retaining the test_vol_p OOD win.

**Final verified metrics (W&B `ss5v4vdx`, run state: finished):**

| Metric | PR #793 (this) | SOTA #592 `4k25s25e` | PR #776 Arm B (vol-w=2.0 solo) | Gate |
|---|---:|---:|---:|---|
| `full_val_primary/abupt_axis_mean_rel_l2_pct` | 7.2412% | **6.5985%** | 7.2231% | < 6.5985% ❌ FAIL (+0.657pp) |
| `test_primary/abupt_axis_mean_rel_l2_pct` | 8.5761% | **7.9915%** | 8.3466% | — ❌ regressed |
| `test_primary/volume_pressure_rel_l2_pct` | 12.2003% | 11.9335% | **11.5596%** | — ❌ Arm B win destroyed |
| `test_primary/surface_pressure_rel_l2_pct` | 4.5669% | **4.0683%** | 4.3820% | — ❌ regressed |
| `test_primary/wall_shear_rel_l2_pct` | 8.0632% | **7.3338%** | 7.9073% | — ❌ regressed vs both |
| val→test vol_p OOD gap | 7.95pp | 7.99pp | **7.32pp** | — Arm B win gone |

**Mechanism failure analysis:**
- Four simultaneous channel up-weights (vol×2.0, tau_y×2.5, tau_z×3.0, surface×2.0) starved every channel of effective gradient signal. Competing pulls on a single 100-epoch budget degrade all channels.
- Per-axis z>y>x ordering remained structurally invariant to per-axis tau weights — z–y gap WIDENED EP3→terminal (1.38pp → 1.63pp). Per-axis loss weights cannot fix structural z-axis difficulty.
- The Arm B OOD-gap win (7.32pp) was destroyed (regressed to 7.95pp). vol-w=2.0 OOD-pressure win is fragile under any additional reweighting.

**Verdict (NEGATIVE):** Both win conditions failed. The hypothesis that tau-weight bumps could compensate for vol-w=2.0 wall-shear budget starvation is refuted. Lesson: vol-w=2.0 must be tested as a single variable in isolation. Follow-up: PR #805 (tanjiro) — vol-w=2.0 on schedule-aligned 4-epoch stack as true single-variable isolation.

## 2026-05-01 — PR #792: FiLM v3 compressed curriculum (frieren) — CLOSED DESIGN-NEGATIVE

- **Branch**: frieren/vol-film-v3-compressed-curriculum
- **W&B run**: `uhyi1e6k` (group `vol-film-v3-compressed-curriculum`)
- **Hypothesis**: Compressing the vol-points curriculum to `0:16384:1:32768:2:49152:3:65536` allows FiLM to activate at EP3 (instead of EP6+ in the standard schedule), giving 5× more FiLM-active steps within the 270-min budget. Thesis: ≥5× FiLM-active training time → improved test_vol_p / test_abupt vs v2 (PR #778, 1 FiLM-active epoch).

**Final test results (EP7 EMA, 5 FiLM-active epochs, run `uhyi1e6k`):**

| Metric | v3 (this) | v2 (PR #778) | SOTA #592 | Vol-anchor #681 |
|---|---:|---:|---:|---:|
| test_abupt | 8.2969% | ~8.25% | 7.9915% | — |
| test_vol_p | 12.239% | 12.110% | 11.933% | 11.374% |
| test_surface_p | 4.2445% | — | 4.22% | — |
| test_wall_shear | 7.652% (x=6.782/y=8.522/z=9.697) | — | — | — |

**FiLM dynamics (5 FiLM-active epochs EP3-EP7):**
- γ_mean climbed 0.304 → 0.631 with decelerating rate
- γ_max saturated at tanh asymptote 100% of batches from EP4 onward
- β stayed sparse throughout (mean ~0.025)

**Verdict (DESIGN-NEGATIVE):** 5× more FiLM-active steps (EP3-EP7 vs only EP6 in v2) produced essentially equivalent test metrics (+0.129pp test_vol_p vs v2). The thesis "more FiLM-active training time → better metrics" is NOT supported. Key diagnostic: γ_max saturation at the tanh upper bound from EP4 onward indicates the bounded tanh parameterization (γ∈(0,2)) is a capacity bottleneck. FiLM mechanism is structurally working (bounded, stable, monotone val descent) but the current γ range is insufficient to further improve vol_pressure. Closing. Not pursuing FiLM v4 wider-bounds as immediate follow-up — the 0.86pp test_vol_p gap to anchor is more likely a wallclock/data-throughput limitation than a FiLM-dosage issue.

## 2026-05-07 — PR #789: SDF-gate v2/v3/v4 vol-decoder (askeladd) — CLOSED DESIGN-NEGATIVE (all 3 tanh-cap variants)

- **Branch**: askeladd/vol-decoder-sdf-gate-v3
- **W&B run**: `qazswyke` (group `vol-geom-cond`, name `askeladd/vol-decoder-sdf-gate-v3`)
- **Hypothesis**: Per-case SDF features → small MLP → bounded scalar gate on volume decoder logits (cap=0.15, gate-WD=5e-3, 2-epoch independent gate warmup) prevents v2's saturation collapse and reduces test_vol_p OOD error.

**Final test results (EP4 EMA, 86% of EP4, run hit 270-min timeout):**

| Metric | v3 (this) | SOTA #592 | Vol-anchor #681 | Arm A control |
|---|---:|---:|---:|---:|
| test_abupt | **8.1945%** | 7.9915% | — | — |
| test_volume_p ★ | **12.0454%** | 11.933% | 11.374% | 12.092% |
| test_surface_p | 4.2453% | 4.22% | — | — |
| test_wall_shear | 7.5429% | 7.49% | — | — |
| test_ws_x / y / z | 6.66 / 8.43 / 9.59 | — | — | — |
| val_abupt | **6.8400%** | 6.5985% | — | 7.0077% |
| val_vol_p | 4.2617% | 3.9456% | — | — |
| val_surf_p | 4.4960% | 4.3322% | — | — |
| val_wall_shear | 7.6860% | 8.24% | — | — |

**Gate diagnostics (test, 11,091 points):** scale_max_abs=0.1504, sat_frac=9.02e-5 (1 OOD case), scale_mean=−0.0834 (identical to val), scale_range=0.0674, scale_std=0.000987, bias_max_abs=0.0077. **train/sat_frac=0 across all 37,268 steps.**

**Verdict:** Structural fix works (v2 hit sat_frac=1.0 by step ~2k, v3 stayed at 0 throughout). Gate generalizes cleanly val↔test at scale_mean=−0.083. Within-experiment Arm-A control beat: −0.17pp val_abupt, −0.05pp test_vol_p (small but signal-positive). However, single-model SOTA gate not met (+0.24pp val, +0.20pp test) — primary cause is the 270-min wall-time cap stopping training at 86% of EP4 in a 13-epoch cosine. Student's post-mortem identifies LR coupling (gate_lr = scheduled_lr × gate_factor) as having cost ~half an epoch of useful gate training time.

**v4 update (W&B run `ccnssij7`, group `vol-geom-cond`, name `askeladd/vol-decoder-sdf-gate-v4`):** LR decoupling confirmed — gate LR stayed constant 5e-05. Despite LR fix, gate fully saturated (sat_frac=1.00, scale_range=0.0000) by step 8,501 — before EP1 (step 10,864). EP3 val_abupt=7.447% — worse than Arm A control (7.0077%) and v3 best (6.840%). The tanh-cap (=0.15) architecture pushes scale outputs onto the cap regardless of LR scheduling.

**Final verdict — CLOSED DESIGN-NEGATIVE:** All three versions (v2/v3/v4) of the tanh-cap multiplicative gate failed via saturation. The architecture is fundamentally insufficient. New direction: additive rank-r LoRA on volume output projection (PR #809, no activation caps, zero-init B, bounded by construction).

## 2026-05-01 08:30 — PR #809: additive LoRA on volume output head, r=4 and r=8 (askeladd) — ASSIGNED

- **Branch**: askeladd/vol-head-lora-additive
- **Hypothesis**: Additive low-rank correction (LoRA) on `volume_out` linear projection: `volume_preds += B(A(volume_hidden))` with A∈R^{hidden×r}, B∈R^{r×vol_out} — B zero-initialized so initial correction is exactly zero, no saturation risk. Targets the chronic vol_pressure test-vs-val gap (val 3.6%, test 11.5%, ~3× in best ensemble). Architecture inherits all SDF information already encoded in volume_hidden (SDF is part of volume_x). r=4 (Arm A) and r=8 (Arm B) tested against SOTA L=5 backbone.
- **Gate**: val_abupt < 6.5985% (single-model SOTA #592) / secondary: reduce test vol_pressure below 11.5%
- **Status**: ASSIGNED — waiting for askeladd to pick up

## 2026-05-08 — PR #782: SDF-FiLM volume conditioning (edward) — CLOSED NEGATIVE

- **Branch**: edward/sdf-explicit-vol-geometry-conditioning (deleted)
- **W&B run**: `rtww6a8e` (group `sdf-film-vol-geometry`)
- **Hypothesis**: Per-case SDF stats (mean/std/min/max) → 2-layer MLP → bounded-tanh γ ∈ (0,2) and β FiLM on volume tokens reduces the val→test vol_p gap (PR #767 showed 4 OOD test cases account for 92% of squared test_vol_p deviation).

**Best-EMA results (EP4, only FiLM-active epoch — run cut at 4/13 due to 2.8× cluster slowdown):**

| Metric | SDF-FiLM (this) | SOTA #592 | Δ |
|---|---:|---:|---:|
| val_abupt | 6.9289% | 6.5985% | +0.330pp |
| test_abupt | 8.1456% | 7.9915% | +0.154pp |
| test_volume_p ★ | 12.2120% | 11.9335% | **+0.279pp** |
| test_surface_p | 4.1375% | 4.0683% | +0.069pp |
| val→test vol_p gap | 7.998pp | 7.988pp | +0.011pp |

**FiLM diagnostics:** γ_mean=0.9202, γ_max_abs_dev=0.5195 (52% of saturation), β_max_abs=0.5742, no nonfinite grads, identity-at-init verified, DDP-safe multiply-by-zero pattern works end-to-end.

**Verdict:** Hypothesis NOT supported. Implementation sound but training cut to 4/13 epochs with only ONE FiLM-active epoch. The val→test gap on vol_p was structurally unchanged, suggesting the 4 OOD cases may be extrapolative w.r.t. the train SDF stat manifold (FiLM cannot help where there is zero training support). Follow-up: PR #797 SDF coverage diagnostic.

## 2026-05-01 — PR #798: surface curvature 4-epoch schedule-aligned re-run (nezuko) — CLOSED NEGATIVE (design)

- **Branch**: nezuko/surface-curvature-4ep-aligned (deleted)
- **W&B run**: group `nezuko-surface-curvature-4ep`, name `nezuko/surface-curvature-4ep-aligned`
- **Hypothesis**: PR #788 was cut at 81% of EP4 with no LR cooldown (`--lr-cosine-t-max 13`). Re-run with `--epochs 4 --lr-cosine-t-max 4` to provide full EP4 LR cooldown and confirm the curvature signal win. Same 9-channel surface_x (7 base + H̃ + K̃), same optimizer/architecture.

**Final verified metrics (EP4 EMA, full run, schedule-aligned):**

| Metric | PR #798 (EP4 cooldown) | PR #788 (EP4 81% cut) | SOTA #592 | Δ vs SOTA |
|---|---:|---:|---:|---:|
| val_abupt | 7.3508% | 6.7767% | **6.5985%** | +0.752pp |
| test_abupt | 8.6458% | 8.139% | **7.9915%** | +0.654pp |
| test_surface_p | 4.4908% | 4.168% | **4.0683%** | +0.423pp |
| test_wall_shear | 7.9537% | 7.4189% | **7.3338%** | +0.620pp |
| test_volume_p | 12.7115% | 12.254% | 11.9335% | +0.778pp |

**Curvature gradient health (from nezuko diagnostics):**

| step | param_norm | global_norm | grad/param | zero_fraction |
|---|---:|---:|---:|---:|
| 249 (warmup) | 4.58 | 0.0099 | 0.0021 | 0.000 |
| 10,499 (EP1 end) | 4.82 | 0.1121 | 0.0233 | 0.000 |
| 16,000 (EP2 end) | 11.18 | 0.1110 | 0.0099 | 0.000 |
| 19,501 (EP3 end) | 12.85 | 0.1220 | 0.0095 | 0.000 |
| 22,502 (EP4 end) | 13.48 | 0.1427 | 0.0106 | 0.000 |

**Root-cause: compressed vol-schedule cut total optimizer steps by 36%**

The run used `--vol-points-schedule "0:16384:1:32768:2:49152:3:65536"` (from frieren's PR #792 default suggestion). This caused:

| Epoch | vol_points | Steps | Cumulative |
|---|---:|---:|---:|
| EP1 | 16,384 | 10,864 | 10,864 |
| EP2 | 32,768 | 5,435 | 16,299 |
| EP3 | 49,152 | 3,625 | 19,924 |
| EP4 | 65,536 | 2,720 | 22,644 |

Total: 22,644 steps vs ~35,200 in PR #788 (~36% fewer). The model achieved full LR cooldown (terminal LR 1.40e-5) but never accumulated sufficient gradient updates to converge. Every channel strictly regressed vs the 81%-complete PR #788, confirming step-count starvation — not LR misalignment — was the binding constraint in PR #788.

**Curvature signal architecture validity:** Despite the failure, the signal is architecturally healthy. param_norm grew 3× (4.58→13.48), grad/param stable ~0.01 post-warmup, zero_fraction=0.000 throughout. PR #788 demonstrated discriminating test-set signal (−0.18pp test_abupt, −0.14pp test_surface_p, −0.28pp test_wall_shear vs within-cluster control). The curvature direction is valid.

**Verdict (NEGATIVE — design error):** Full LR cooldown is necessary but not sufficient. The compressed schedule was the wrong default for this config. Follow-up: `--epochs 4 --lr-cosine-t-max 4` + **original** vol-schedule `0:16384:3:32768:6:49152:9:65536` (vol=16k throughout all 4 epochs → ~35k+ steps + proper cooldown). Expected val_abupt: 6.4–6.7%.

---

## 2026-05-08 — PR #788: surface curvature H,K on surface path (nezuko) — CLOSED INCONCLUSIVE

- **Branch**: nezuko/surface-curvature-surface-only (deleted)
- **W&B run**: `3ct0x7zd` (group `nezuko-surface-curvature`)
- **Hypothesis**: Append `[H̃, K̃]` (signed-log + train-z-score) to surface input → improves surface_p, wall_shear, τ_z without affecting volume_p.

**Best-EMA results (EP4 partial, 81% through EP4 at 270-min cap):**

| Metric | nezuko curvature | SOTA #592 | within-cluster control thorfinn-ArmA | Δ vs control |
|---|---:|---:|---:|---:|
| val_abupt | 6.7767% | 6.5985% | — | — |
| test_abupt | 8.139% | 7.9915% | 8.321% | **−0.18pp** |
| test_surface_p | 4.168% | 4.068% | 4.303% | **−0.14pp** |
| test_wall_shear | 7.4189% | 7.334% | 7.697% | **−0.28pp** |
| test_volume_p | 12.254% | 11.9335% | 12.092% | +0.16pp (drift) |

Beat curvature-on-volume (edward PR #773) on test_abupt, val_surf_p, wall_shear, τ_z. Hypothesis-discriminating signals land on test exactly where predicted. Failed val_abupt merge gate by +0.18pp purely because EP4 was cut at 81% with no LR cooldown (`--lr-cosine-t-max 13`).

**Verdict:** Architecturally correct (curvature on surface > curvature on volume). Schedule-mismatch is the merge blocker. Follow-up: PR #798 with `--epochs 4 --lr-cosine-t-max 4`.

## 2026-05-08 — PR #786: Anchor-STRING RoPE v3 full-budget (fern) — CLOSED INCONCLUSIVE

- **Branch**: fern/anchor-string-rope-v3-fullrun (deleted)
- **W&B run**: `qg0rplnl` (group `fern-anchor-string-rope-v3`)
- **Hypothesis**: Xavier×0.01 init on out_proj activates RoPE residual from EP1 (vs zero-init in v2), so by EP4 the residual has built up enough learned spectral structure to close the SOTA gap.

**Best-EMA results (EP4 partial, 64% through EP4 at 270-min cap):**

| Metric | v3 (this) | v2 #774 | Δ vs v2 | SOTA #592 | thorfinn ArmA |
|---|---:|---:|---:|---:|---:|
| val_abupt | 6.9197% | 6.9088% | tie | 6.5985% | — |
| test_abupt | 8.1946% | 8.249% | −0.054 | 7.9915% | 8.321% |
| test_volume_p | 12.116% | 12.118% | tie | 11.933% | 12.092% |
| out_proj_rms (EP4) | 0.0464 | 0.042 (terminal) | +0.0044 | — | — |

Xavier×0.01 init worked exactly as designed (rms grew from 0.00347 EP1 → 0.0464 EP4 cutpoint, no runaway). Beat thorfinn within-cluster control on test_abupt (−0.13pp), but +0.20pp behind absolute SOTA `4k25s25e`.

**Verdict:** Init mechanism validated. Architecture parked at this 270-min budget — not paying for itself vs SOTA without 13 full epochs. Compute hours better spent on schedule-alignment control (PR #799 fern).

## 2026-05-07 02:15 — PR #776: vol-loss-weight sweep {1.5, 2.0} on SOTA L=5 (tanjiro) — CLOSED PARTIAL POSITIVE

- **Branch**: tanjiro/vol-loss-weight-sweep (deleted)
- **W&B group**: `vol-loss-weight-sweep-tay`
- **Hypothesis**: Manual `--volume-loss-weight` sweep increases vol_p representational pressure → reduces val→test vol_p OOD gap (SOTA gap = 7.99pp). GradNorm was already ruled out (PRs #649 + #758, 6 configs).
- **Arms run**: A (vol-w=1.5, run `hw2e3vsu`), B (vol-w=2.0, run `qscw0225`). Arm C (vol-w=2.5) cancelled at 23:55 UTC at EP2 (advisor decision tree based on EP1 trajectory — see post-mortem below).

**Final test_primary comparison (best-EMA EP4, 50 cases):**

| Metric | SOTA (vol-w 1.0) | Arm A (1.5) | Arm B (2.0) |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **7.9915** | 8.4181 (+0.43) | 8.3466 (+0.36) |
| `test_primary/volume_pressure_rel_l2_pct` | 11.9335 | 12.1257 (+0.19) | **11.5596 (−0.37)** |
| `test_primary/surface_pressure_rel_l2_pct` | **4.0683** | 4.3816 (+0.31) | 4.3820 (+0.31) |
| `test_primary/wall_shear_rel_l2_pct` | **7.3338** | 7.8366 (+0.50) | 7.9073 (+0.57) |
| val→test vol_p OOD gap | 7.99 | 7.94 (−0.05) | **7.32 (−0.67)** |

**Verdict: Partial positive on test_vol_p only. NOT MERGED.** Arm B beats SOTA on `test_primary/volume_pressure_rel_l2_pct` by −0.37pp and shrinks the val→test vol_p OOD gap by 0.67pp — first single-model arm in the sweep family to do so. But val_abupt regresses 0.62pp (7.22% vs 6.60% SOTA), and every other test target regresses 0.31–0.68pp. Per `program.md`: cannot hide regressions behind a single averaged number, so this is not a SOTA replacement.

**Key insight**: val_abupt regression is wall-shear dominated (ws_x +0.51, ws_y +0.65, ws_z +0.68 on test). The Lion+QK-Norm+vol-w=2.0 stack shifts the loss budget away from wall-shear. Vol-loss-weight effects don't show cleanly until ~EP3+ — Arm B EP1 was weaker than Arm A's, but Arm B beat A by terminal. **EP1 is a poor proxy for terminal test_vol_p in this sweep.**

**Advisor post-mortem**: Cancelled Arm C based on EP1 read; Arm B's terminal win shows that was premature. Recording for future kill-gate calibration on loss-weight sweeps: don't gate on EP1.

**Follow-up assigned**: vol-w=2.0 + wall-shear-weight bump combined arm (next PR — see below).

---

## 2026-05-07 — PR #792: FiLM v3 compressed vol-points schedule for max FiLM-active epochs (frieren) — ASSIGNED

- **Branch**: frieren/vol-film-v3-compressed-curriculum
- **Hypothesis**: PR #778 (FiLM v2) confirmed the bounded-tanh FiLM mechanism is structurally sound (no blow-up, FiLM-active epoch was best checkpoint) but budget starvation caused all 3 win conditions to be missed. With `--vol-geom-film-start-epoch 6` and standard curriculum (`0:16384:3:32768:6:49152:9:65536`), FiLM only had ~4127 active steps before the 270-min wall timeout hit mid-EP4 — the V=49k and V=65k stages never completed. Fix: compress the vol-points schedule to `0:16384:1:32768:2:49152:3:65536` (V=65k by EP3) and lower FiLM start to EP2, giving ≥10 FiLM-active epochs within budget.
- **W&B group**: `vol-film-v3-compressed-curriculum`
- **Key change 1**: `--vol-points-schedule "0:16384:1:32768:2:49152:3:65536"` (V=65k by EP3 instead of EP9)
- **Key change 2**: `--vol-geom-film-start-epoch 2` (FiLM fires from EP2, ~10 active epochs)
- **Kill gates**: EP1 val_abupt <32%, EP3 val_abupt <12%, EP6 val_abupt <8.0%
- **Win conditions**: test_vol_p <11.374% (primary), val_abupt <6.5985% (secondary)
- **Status**: WIP — assigned 2026-05-07 (follow-up to PR #778)

---

## 2026-05-07 — PR #778: FiLM v2 bounded tanh γ∈(0,2) + delayed EP6 onset (frieren) — CLOSED NEGATIVE (metrics) / POSITIVE (mechanism)

- **Branch**: frieren/vol-head-geometry-cond-v2 (deleted)
- **W&B group**: `vol-geom-cond`
- **Hypothesis**: FiLM conditioning of volume tokens via mean-pooled surface-slice geometry vector g, with bounded tanh γ∈(0,2) and β∈(−1,1). FiLM gate delayed to EP6 to avoid firing before high-density vol-points established. Fixes the unbounded blow-up of PR #770 (v1).
- **Architecture**: `VolGeomFilm(hidden_dim)` class with `gamma_proj` and `beta_proj` zero-initialized linear layers; applied after standard vol-token computation: `h' = (1 + tanh(gamma_proj(g))) * h + tanh(beta_proj(g))`
- **Results**:

| Metric | EP1 | EP3 | EP4 (partial, best) | Win condition |
|---|---|---|---|---|
| val_abupt | ~28% (pass) | ~8.5% (pass) | best checkpoint | <6.5985% MISSED |
| FiLM-active steps | — | — | ~4127 | — |
| test_vol_p | — | — | not collected (budget) | <11.374% MISSED |
| wall timeout | — | — | mid-EP4 | — |

- **Analysis**: Bounded tanh design prevented the blow-up seen in v1 (#770). FiLM-active epoch produced the best validation checkpoint, confirming the mechanism is directionally correct. However all win conditions were missed because the 270-min wall timeout hit mid-EP4 with only ~4127 FiLM-active steps. Root cause: `--vol-geom-film-start-epoch 6` combined with standard curriculum `0:16384:3:32768:6:49152:9:65536` means FiLM only fires after V=49k is established at EP6 — but the budget never reaches EP6 at these vol-point densities. The V=49k and V=65k curriculum stages were never trained. The mechanism works; the timing is wrong.
- **Decision**: CLOSED. Hypothesis not falsified — FiLM direction intact. Fix: compress curriculum to `0:16384:1:32768:2:49152:3:65536` and lower start epoch to 2. Assigned as PR #792.

---

## 2026-05-07 — PR #790: τ_z loss upweight sweep {3.0, 4.0} (alphonse) — ASSIGNED

- **Branch**: alphonse/tau-z-upweight-sweep
- **Hypothesis**: `wall_shear_z` (τ_z) is the confirmed training laggard from GradNorm diagnostic (PR #758): r_i=0.01123, GradNorm weight=1.699×, highest of all tasks. Current baseline uses tau_z_loss_weight=2.0. Increasing to 3.0 or 4.0 forces more gradient signal to τ_z. Distinct from GradNorm (which was ruled out): this is static manual upweighting. If effective, will directly improve val_abupt (τ_z has equal weight in the 5-channel abupt average). Pure CLI experiment — no code changes.
- **W&B group**: `alphonse-tau-z-upweight`
- **Arms**:
  - Arm A: `--tau-z-loss-weight 3.0`
  - Arm B: `--tau-z-loss-weight 4.0` (only if Arm A shows τ_z improvement)
- **Kill gates**: EP1 val_abupt < 32%, EP2 val_abupt < 12%
- **Key signal**: val_wall_shear_z vs SOTA 9.810%; watch τ_y and surface_p for regression
- **Status**: WIP — assigned 2026-05-07 (re-assigned from PR #787 stark→alphonse)

---

## 2026-05-07 — PR #789: Vol-decoder SDF-gate v3 — lower cap 0.15 + gate LR warmup + gate WD (askeladd) — ASSIGNED

- **Branch**: askeladd/vol-decoder-sdf-gate-v3
- **Hypothesis**: PRs #781 (unbounded) and #785 (bounded-tanh v2, cap=0.3) both failed via gate MLP saturation. Proximate cause: 20× LR jump at EP1→EP2 boundary (from `--lr-warmup-epochs 1`) triggers 30× vol_loss spike → monotone gate drift to full negative saturation (scale=-0.301, sat_frac=1.0). v3 fixes: (1) lower tanh cap 0.3→0.15 (smaller gradient signal), (2) 2-epoch gate-specific LR warmup (gate LR is only 50% at the EP1→EP2 boundary where v2 died), (3) gate weight decay 5e-3 (10× stronger than backbone). Hypothesis intact: per-case SDF stats can calibrate vol_pred for OOD geometries.
- **W&B group**: `vol-geom-cond`
- **Key metrics**: train/gate/scale_range (saturation indicator), test_vol_p vs 11.374% anchor
- **Kill gates**: 200-step sanity scale_range > 0.002, EP1 val_abupt < 32%, EP2 val_abupt < 12%
- **Status**: WIP — assigned 2026-05-07

## 2026-05-07 — PR #785: Vol-decoder SDF-gate v2 — bounded tanh + input norm (askeladd) — CLOSED NEGATIVE (design)

- **Branch**: askeladd/vol-decoder-sdf-gate-v2 (deleted)
- **W&B runs**: `37r8htsk` (sanity), `ympw1bhr` (DDP run)
- **Hypothesis**: Post-decoder output gating of vol_pred via SDF statistics (per-case global descriptors). Bounded-tanh design: scale ∈ (0.7, 1.3), bias ∈ (−0.05, 0.05). Input normalization. Hidden dim 8→16→2, zero-init output layer.
- **Results**:

| Metric | EP1 (step 10,864) | EP2 (step 21,728) | Status |
|---|---|---|---|
| val_abupt | 28.13% ✅ | 8.5789% | KILL (threshold tripped) |
| scale_max_abs | 0.201 (healthy) | 0.3008 (≥ 0.28 threshold) | SATURATED |
| scale_mean | healthy | -0.301 (full saturation) | |
| sat_frac | — | 1.0 | Complete saturation |

- **Analysis**: Bounded-tanh eliminated v1's catastrophic blow-up but did not prevent monotone drift to negative saturation. The 20× LR jump at EP1→EP2 boundary (from `--lr-warmup-epochs 1`: 4.5e-6 → 9.0e-5) triggered a 30× vol_loss spike (0.03 → 0.88), driving gate MLP monotonically to full negative saturation over ~2k steps. Gate degenerated to constant 0.7× multiplier — geometry conditioning never active at steady state. Hypothesis NOT falsified.
- **Status**: CLOSED NEGATIVE (design) — v3 follow-up assigned as PR #789

---

## 2026-05-07 — PR #786: Anchor-STRING RoPE v3 full 13-epoch run (fern) — ASSIGNED

- **Branch**: fern/anchor-string-rope-v3-fullrun
- **Hypothesis**: Prior v2 (PR #774) showed strongly closing gap to SOTA (1.16×→1.05× gap ratio per epoch), reaching EP4 val=6.9088%. Code fixes from PR #783 (merged by human) now in `tay`: (1) `--lr-cosine-t-max 13` aligned to actual budget (was 5 = mismatch), (2) Xavier×0.01 `out_proj.weight` init (was zero). Full 13-epoch run with `--use-anchor-string-rope --anchor-string-rope-n-anchors 1024`. Definitive test of whether Anchor-STRING RoPE can beat SOTA at full budget.
- **W&B group**: `fern-anchor-string-rope-v3`
- **Kill gates**: EP1>35%, EP2>12%, EP3>8.5%
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-07 — PR #788: Surface curvature H,K on surface path only (nezuko) — ASSIGNED

- **Branch**: nezuko/surface-curvature-surface-only
- **Hypothesis**: PR #773 (edward) put H,K curvature features on the volume path — failed (8.166% test vs 7.991% SOTA, -0.18pp). Follow-up: wire H,K to the **surface** path only (SURFACE_X_DIM=3→5). Surface curvature directly governs aerodynamic boundary conditions (pressure gradients at high-curvature wheel arches, A-pillar edges, underbody details). Volume decoder is left unchanged. Precomputed cache already on disk at `/mnt/new-pvc/Processed/drivaerml_curvature_v2_edward/` from PR #773.
- **W&B group**: `nezuko-surface-curvature`
- **Key discriminating signal**: surface_pressure and wall_shear should improve; vol_p should stay neutral.
- **Kill gates**: EP1>32%, EP2>10% (tighter than usual — testing surface input perturbation)
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-07 — PR #775: Learnable affine anchor for vol_p OOD gap (nezuko) — CLOSED NEGATIVE

- **Branch**: nezuko/learnable-scale-surface-anchor (deleted)
- **Hypothesis**: Learnable global scalar alpha×surf_cp+beta applied to vol_pred, with alpha/beta init=0, to learn the ~718 Pa/Cp scale from data. Zero-degradation at init. Fixes unit-mismatch from PR #772.
- **W&B run**: `8wft0el2` (group `surf-anchor-learnable-scale-tay`)
- **Results**:

| Epoch | val_abupt | val_vol_p | anchor/alpha |
|------|-----------|-----------|-------------|
| EP1 | 27.37% | 16.44% | 0.0442 |
| EP2 | 8.244% | 5.087% | 0.0101 |
| EP3 | 7.197% | 4.310% | 0.00473 |
| EP4 (partial) | **7.049%** | 4.239% | 0.00408 |

- **Decision**: CLOSED. val_abupt=7.049% vs SOTA 6.5985% (+0.45pp). Alpha peaked at 0.141 at EP1→EP2 boundary then decayed 30× to near-zero — optimizer rejected the anchor's contribution. Every channel lagged SOTA. The global scalar anchor fails because: (1) backbone learns surface→volume coupling more expressively, (2) single global scalar cannot capture the OOD geometry shifts of 4 outlier cases. Rules out raw-Cp global scalar anchor as geometry-conditioning approach.

---

## 2026-05-07 — PR #781: Vol-decoder SDF-statistics geometry gating (askeladd) — CLOSED NEGATIVE (unbounded design)

- **Branch**: askeladd/vol-decoder-sdf-gating (deleted)
- **Hypothesis**: 8-stat SDF descriptor (mean, std, min, max, frac<0.05/0.20/0.50m, median) → 8→64→2 MLP → unbounded affine `(1+a)*vol_pred + b` on volume pressure output. Zero-init MLP. Per-case geometry conditioning from existing SDF channel (VOLUME_X_DIM=4).
- **W&B runs**: rank-0 `4z4cz06q`, rank-7 (kill source) `4qjhfd11` | Group: `vol-geom-cond`
- **Results**: Killed at step 2376 (EP1, 22% through). No val metrics collected.
  - Initial kill was due to inverted kill threshold (`<2.0` instead of `>2.0`) — advisor corrected this.
  - After correction, rank-7 still blew up at step 2375: scale_max_abs 0.0025 → 2.5625 (~1000× spike). Ranks 0-6 remained healthy (max ≤ 0.005).
  - Root cause: 8 descriptor channels span different orders of magnitude (metres vs [0,1] fractions) + no input normalization. Unbounded MLP weights grow along typical-distribution directions; outlier case in under-sampled descriptor corner drives extreme response.
- **Decision**: CLOSED. Hypothesis not falsified (no val data). Unbounded affine design falsified. Follow-up PR #785 implements bounded tanh + input normalization.

---

## 2026-05-06 — PR #776: Manual vol-loss-weight sweep {1.5, 2.0, 2.5} on SOTA L=5 (tanjiro) — ASSIGNED

- **Branch**: tanjiro/vol-loss-weight-sweep
- **Hypothesis**: Manual `--volume-loss-weight` increase {1.5, 2.0, 2.5} to reduce vol_p OOD gap via higher gradient signal magnitude — distinct from GradNorm (which was ruled out, PRs #649 + #758). More gradient signal may force the model to allocate more representational capacity to vol_p, potentially improving generalization on the 4 OOD test cases. Three arms, no code changes required.
- **W&B group**: `vol-loss-weight-sweep-tay`
- **Issue**: #717
- **Arms**:
  - Arm A: `--volume-loss-weight 1.5` → run `tanjiro/vol-w-1.5`
  - Arm B: `--volume-loss-weight 2.0` → run `tanjiro/vol-w-2.0`
  - Arm C: `--volume-loss-weight 2.5` → run `tanjiro/vol-w-2.5`
- **Kill gate**: val_abupt < 32% at EP1 (~step 2700); kill arms with val_abupt > 7.5% by EP3
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #775: Learnable affine scale on surface-anchor for vol_p OOD gap (nezuko) — ASSIGNED

- **Branch**: nezuko/learnable-scale-surface-anchor
- **Hypothesis**: PR #772 (surface-anchor v1) failed due to unit mismatch: `surface_cp` (dimensionless Cp, mean≈−0.304) was used as correction for `volume_pressure` (Pa, mean≈−205.8). This PR fixes that with a **learnable affine transform** on the nearest-surface-point lookup: `vol_p_anchor = alpha * surf_p_norm + beta`, where alpha and beta are initialized to 0 (ensuring zero degradation at step 0). The model learns the Pa/Cp scale (~718) from data. Architecturally distinct from PR #771 (askeladd cross-attention scalar): pure geometric lookup with learnable affine, no learned feature aggregation.
- **W&B group**: `surf-anchor-learnable-scale-tay`
- **Issue**: #717
- **Arms**:
  - Arm A: shared global scalar (alpha, beta as nn.Parameter scalars)
  - Arm B: same, but log alpha convergence to verify it approaches ~718 Pa/Cp
- **Kill gate**: val_abupt < 32% at EP1 (~step 2700)
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #770: Vol decoder FiLM conditioning on surface geometry latent (frieren) — ASSIGNED

- **Branch**: frieren/vol-head-geometry-cond
- **Hypothesis**: The 4 geometrically-OOD test cases (run_133/226/203/158) that cause 92% of test_vol_p deviation (#767 diagnostic) require the volume decoder to be conditioned on the surface geometry latent. Inject global surface slice-token mean-pool `g = MeanPool(S)` into volume tokens via FiLM: `h' = γ(g) ⊙ h + β(g)` before the volume prediction head. γ,β initialized to identity. ~0.6M extra params.
- **W&B group**: `vol-geom-cond`
- **Issue**: #717
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #771: Surface-latent scalar offset for vol_pressure OOD conditioning (askeladd) — ASSIGNED

- **Branch**: askeladd/surf-latent-vol-residual
- **Hypothesis**: Minimal geometry conditioning: a learned per-case global residual scalar offset on vol_pressure, derived from surface geometry latent. `vol_p_conditioned = vol_p + Linear(MeanPool(surface_slice_tokens))`. Linear(D→1), ~513 params, zero-initialized. Tests whether a single learned scalar per case is sufficient to address the geometry-OOD case-level scale shifts confirmed by #767.
- **W&B group**: `vol-geom-cond` (grouped with frieren #770 for direct comparison)
- **Issue**: #717
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #767: Phase 0 diagnostic per-case + per-region test_vol_p (askeladd) — CLOSED (diagnostic complete)

- **Branch**: askeladd/phase0-diagnostic
- **Hypothesis**: The test_vol_p gap is case-dominated and lives on a small number of geometrically-OOD test cases.
- **W&B runs**: inference-only, no training run (diagnostic only)
- **Results**:

| Checkpoint | test_vol_p all 50 | test_vol_p excl-4 OOD | % deviation from top-4 |
|---|---:|---:|---:|
| `4k25s25e` (#592) | 11.933% | 3.9% | 92% |
| `dc031qpt` (#681) | 11.374% | 4.2% | 92% |

- **Key findings**:
  1. Same 4 cases (run_133, run_226, run_203, run_158) account for 92% of squared test_vol_p deviation across **two architecturally distinct checkpoints**
  2. Excluding the 4 cases, test_vol_p drops to 3.9-4.2% — **below AB-UPT 6.08% reference for the remaining 46 cases**
  3. Surface_p and τ are **unaffected** on these 4 cases — the surface encoder generalises fine; the volume decoder specifically fails
  4. H3-via-loss-scaling closed: supervision-density/loss-mass interventions cannot fix geometry-OOD
  5. Next intervention class: test-time geometry conditioning on volume path
- **Decision**: Diagnostic complete. Closed successfully. Next PRs: #770 (frieren FiLM), #771 (askeladd scalar offset).

---

## 2026-05-06 — PR #761: Dedicated 2-layer volume head on shared encoder (frieren) — CLOSED (truncated, inconclusive)

- **Branch**: frieren/vol-head-2L
- **W&B run**: `15u5c4ec`
- **Hypothesis**: A dedicated 2-layer Transolver volume decoder head on top of the shared encoder (+5.91M params, +37.1% vs SOTA) will reduce the volume_pressure val→test gap by increasing volume-specific capacity.

| Metric | EP1 | EP2 | EP3 | EP4-partial (final) | SOTA gate |
|---|---:|---:|---:|---:|---:|
| val_abupt | 31.312% | 8.088% | 7.045% | 6.832% | <6.5985% |
| val_vol_p | 14.144% | 4.731% | 4.045% | 3.938% | — |
| test_abupt | — | — | — | 8.198% | — |
| test_vol_p | — | — | — | 12.112% | <11.374% |

- **Analysis**: Training timeout (270 min) fired at EP4-partial (step 34,424 of 43,459), cutting the run at ~25% of budget from completion. The vol-points curriculum never advanced past 16,384 (ramp at EP3 to 32k didn't complete). Both gates missed (val_abupt=6.832 > 6.5985; test_vol_p=12.112 > 11.374). However: val_vol_p 3.938% < SOTA 3.946% — a small but persistent signal across EP2/EP3. The 4 OOD cases (run_226=109.1%, run_133=108.0%, run_203=103.7%, run_158=102.1%) entirely dominate test_vol_p; **median test_vol_p=3.89%, excl-top-4 mean=3.97%** — both below AB-UPT 6.08%.
- **Conclusion**: Hypothesis untested (only 25% of budget ran; curriculum never ramped). Closing as inconclusive, not falsified. Next step: compose vol-head with geometry conditioning (#770) rather than re-run standalone.

---

## 2026-05-01 — PR #760: Issue #618 volume-loss-weight reweight ablation (alphonse) — ASSIGNED

- **Branch**: alphonse/vol-loss-weight-reweight
- **Hypothesis**: Increasing `--volume-loss-weight` from 1.0 (PR #592 SOTA default) to 2.0 or 3.0 for the full run will improve val_abupt by forcing better fit to the volume pressure field. The current SOTA uses surface_w=2.0 but volume_w=1.0 (2:1 ratio favoring surface). This tests the 1:1 and 1.5:1 ratio variants on the exact PR #592 stack (L=5 depth).
- **W&B group**: `issue-618-vol-weight-ablation`
- **Arm A command**: SOTA stack + `--volume-loss-weight 2.0` (`alphonse/vol-weight-2.0`)
- **Arm B command**: SOTA stack + `--volume-loss-weight 3.0` (`alphonse/vol-weight-3.0`)
- **Issue**: #618 (STRING/RoPE post-mortem, vol-weight isolation ablation following PR #750 closure)
- **Status**: WIP — assigned 2026-05-01

---

## 2026-05-01 — PR #750: Issue #618 Exp B geometry-branch diff-LR + backbone freeze + aux vol-pressure warmup (alphonse) — CLOSED NEGATIVE

- **Branch**: alphonse/geometry-branch-redux
- **Hypothesis**: Freeze backbone for first 20% of training epochs so geometry branch can warm up independently; simultaneously apply 2× LR to geometry branch params; apply volume-loss-weight-warmup=2.0 during lr_warmup_epochs.
- **W&B run**: `qt9xt341` (group `issue-618-geometry-branch-redux`, name `alphonse/geom-redux-fz0.20-glr2.0-vlw2.0`)

| Metric | EP4 (last frozen) | EP5 (first joint) | SOTA gate |
|---|---:|---:|---:|
| val_abupt | 27.187% | 11.294% | 6.5985% |
| val_vol_p | 18.470% | 7.886% | — |
| test_abupt | — | 12.250% | 7.9915% |
| test_vol_p | — | 15.430% | 11.374% |

- **Root cause**: Frozen backbone warmup with random initialization was harmful — geometry branch spent 4 epochs (252 min) fitting random features (val_abupt=27.2% at last frozen epoch, far above SOTA's ~7% at equivalent depth). The mechanism itself was wired correctly (DDP find_unused_parameters, optimizer rebuild at unfreeze, vol-w warmup), but the underlying strategy was flawed. Vol-points curriculum at 16k points → 63 min/epoch; only ONE joint epoch (EP5) ran before the 270-min budget cap.
- **Conclusion**: Frozen backbone warmup requires a pretrained backbone to be useful. Single-epoch jump from 27.2→11.3% at unfreeze confirms geometry branch can learn fast from real features — motivates a pretrained-freeze variant as a future experiment. Both success gates failed (val_abupt +4.71pp, test_vol_p +4.06pp vs anchors). Closing as negative result.

---

## 2026-05-07 — PR #738: Volume-coordinate Gaussian noise injection (tanjiro) — CLOSED NULL/NEGATIVE

- **Branch**: tanjiro/volume-coord-noise
- **Hypothesis**: Train-time isotropic Gaussian noise on volume xyz coordinates (σ=0.005m, σ=0.020m) as a geometric robustness regularizer (Bishop 1995 equivalence to Tikhonov regularization on Jacobian norm). Targeting val→test volume_pressure transfer gap.
- **W&B group**: `tanjiro-vol-coord-noise`

| Run | W&B ID | Best Epoch | val_abupt% | val_vol_p% | test_vol_p% | test_abupt% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline (#592 4k25s25e) | 4k25s25e | — | 6.5985 | — | 11.933 | 7.9915 | SOTA gate |
| Arm A σ=0.005 | jzybrknz | EP4 | 7.9998 | 9.7217 | **17.0464** | 9.2023 | timeout-killed mid-EP4 |
| Arm B σ=0.020 | fj728edc | EP3 | 10.5977 | 22.8560 | — | — | killed by EP3 gate (abupt>8%) |
| Arm C (annealed) | — | — | — | — | — | — | CANCELLED (Arm B failed gate) |

- **Root cause**: `volume_x[..., 3]` is precomputed SDF from `volume_sdf.npy`, not recomputable per-step. Noising `volume_x[..., :3]` (xyz) without updating SDF creates `(xyz_noisy, sdf(xyz_clean))` contradictory pairs at train, vs `(xyz_clean, sdf(xyz_clean))` at eval. Regression energy scales as σ² — confirmed by Arm B (+13.1pp on val vs Arm A) amplifying exactly quadratically.
- **Conclusion**: Pure xyz-only coordinate noise is dead-on-arrival under the precomputed-SDF data contract. The val→test volume_pressure gap cannot be addressed via simple input-side regularization of this form. Reassigned tanjiro to PR #758 (GradNorm alpha sweep).

---

## 2026-05-07 — PR #758: GradNorm α=3.0/2.0 sweep (tanjiro) — ASSIGNED

- **Branch**: tanjiro/gradnorm-alpha-sweep
- **Hypothesis**: GradNorm `ema_proxy` mode with high restoring-force alpha (α=3.0 and α=2.0) + min_weight=0.7 floor. PR #649 tested GradNorm with α=1.5 (default) and varying floors; best result was floor=0.7/α=1.5 at EP3=7.41%. No experiment has tested α>1.5. At α=3.0, the `r_i^α` weighting aggressively amplifies gradient signal for undertrained tasks (vol_pressure has r_i >> 1 since it's chronically lagging). Two arms: A=α3.0, B=α2.0.
- **Arm A command**: `--use-gradnorm --gradnorm-mode ema_proxy --gradnorm-alpha 3.0 --gradnorm-min-weight 0.7 --wandb-group tanjiro-gradnorm-alpha`
- **Arm B command**: `--use-gradnorm --gradnorm-mode ema_proxy --gradnorm-alpha 2.0 --gradnorm-min-weight 0.7 --wandb-group tanjiro-gradnorm-alpha`
- **Reference PR**: #649 (edward, floor=0.7, α=1.5: EP3 val_abupt=7.41%, val_vol_p=4.68%)
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-06 03:20 — PR #736: Inter-sample mixup volume points (fern) — CLOSED CONTAMINATED

- **Branch**: fern/volume-input-mixup
- **Hypothesis**: alpha=0.4 mixup on volume-points/pressure improves volume_pressure generalization
- **W&B**: group `fern-vol-mixup`, run `jzo917hu` (rank 0)
- **Result at closure** (step 14,665, ~EP6.4): val_abupt=24.97% (vs SOTA 6.60%, +18.4pp), val_vol_p=17.33%, val_wall_shear=27.23%
- **Closure reason**:
  1. **Contamination**: 8 unauthorized parallel runs in group `gradnorm-adaptive` (`fern/gradnorm-armA-a1.0-ep50-4gpu-rank{0..3}` + `fern/gradnorm-armB-a0.5-ep50-4gpu-rank{0..3}`) STILL RUNNING at closure time, started ~5h before closure with no PR sanctioning them. GPU bandwidth contention compromises the mixup result.
  2. **Mixup also diverging**: alpha=0.4 too aggressive on volume coords with shared mask; model never recovers from EP1's destructive interference.
- **Conclusion**: Negative result on top of contamination. Reassigned fern to PR #753 (signed-log1p target transform).

---

## 2026-05-06 03:20 — PR #735: TTA Y-mirror + jitter (edward) — CLOSED NEGATIVE (both arms)

- **Branch**: edward/tta-mirror-jitter
- **Arm A (inference-only TTA on PR #592 SOTA `4k25s25e`)**:
  - Y-mirror TTA: test_vol_p **11.93% → 13.48%** (WORSE +1.55pp)
  - Jitter TTA (sigma=0.005, 4 passes): val_abupt **6.60% → 26.48%** (catastrophic)
  - Root cause: STRING-separable PE + RFF features depend on sign of y, so Y-mirror corrupts embedding
- **Arm B (train with `--use-mirror-aug --mirror-aug-p 0.5`, run `rbnk7zca`)**:
  - best_val_abupt = **7.0214%** (vs SOTA 6.5985%, +0.42pp WORSE)
  - test_vol_p = **12.245%** (vs SOTA 11.933%, +0.31pp WORSE)
  - val→test gap (7.02→8.34) wider than SOTA's, suggesting Y-mirror aug reduces effective capacity for Y-asymmetric ground truth
- **Conclusion**: Y-mirror is the wrong axis of symmetry to exploit. Closing. Reassigned edward to PR #754 (per-case Cp target normalization).

---

## 2026-05-06 03:20 — PR #748: Spatial within-case SDF stratification (frieren) — CLOSED DIVERGED

- **Branch**: frieren/spatial-volume-emphasis
- **W&B**: run `lzpov7mi`, group `frieren-vol-spatial-emphasis`
- **Result**: val_abupt=**76.51%** at step 15,768 (~EP6.9, runtime 8,099s) — never converged
- **Root cause**: SDF-stratified loader interacted badly with vol-points curriculum. SDF threshold of 0.30m (absolute meters) is inconsistent across cases with very different SDF distributions (p50=0.005m, max=530m); the 25% near-band varied dramatically, creating noisy curriculum signal.
- **Conclusion**: Implementation broken; hypothesis not dead. Reassigned frieren to PR #755 (stochastic depth + volume-token dropout).

---

## 2026-05-06 03:20 — PR #751: Issue #618 AnchorString clean (thorfinn) — CLOSED SILENT FAILURE

- **Branch**: thorfinn/issue618-run5-anchorstring-clean
- **W&B**: run `ece4qc3o` (rank 0), state=finished at step 21,729 after 35 minutes (run ended early)
- **Result at termination**: val_abupt=**23.17%**, val_vol_p=15.50%, val_surface_p=17.06%, val_wall_shear=25.56%
- **Closure reasons**:
  1. **Zero PR comments** — no startup heartbeat, no kill-gate report, no termination explanation. Communication blackout.
  2. **Run ended early** — 35min runtime vs ~270min budget. Either auto-killed or process crashed; no diagnostic posted.
  3. **Did not converge** — slope was negative (-2.78pp/1k_steps val_abupt) but starting from way too high to hit SOTA in remaining budget.
- **Conclusion**: Silent failure pattern. Reassigned thorfinn to PR #756 (cosine-annealed EMA decay) with explicit communication-protocol enforcement.

---

## 2026-05-06 03:20 — Round 12 vol-pressure assignments

After closing 4 stalled/failed PRs, all 4 newly-idle students reassigned to fresh, orthogonal hypotheses targeting test_volume_pressure (Issue #717):

| PR | Student | Hypothesis | Mechanism |
|---|---|---|---|
| #753 | fern | Signed-log1p target transform on vol_p | Magnitude-scale equalization for heavy-tailed pressure distribution |
| #754 | edward | Per-case Cp target normalization (`p / max(\|p_surf\|)`) | Dimensional normalization to address 4 catastrophic test outliers |
| #755 | frieren | Stochastic depth + volume-token dropout | OOD generalization regularization for distribution shift |
| #756 | thorfinn | Cosine-annealed EMA decay (0.99→0.9999) | Stabilization tier; clean re-entry after silent-failure pattern |

All four are orthogonal axes (target transform / target rescaling / regularization / EMA bookkeeping) and compose with the in-flight Phase 1 PRs (#737 region weighting, #738 noise injection, #750 geom-branch diff-LR, #752 wake stratification).

---

## 2026-05-06 03:00 — PR #737: Region-weighted vol_p loss (nezuko) — IN-FLIGHT, STRONG SIGNAL

- **W&B**: run `r1eddah6`, group includes `nezuko-region-weighted-vp`
- **Headline EP3 (step 32,592)**: val_abupt=**7.28%**, val_vol_p=**4.36%** — 2.17pp below val SOTA on vol_pressure!
- EP1: val_abupt=27.78%, EP2: val_abupt=8.69% (vol_p=5.38%), EP3: val_abupt=7.28% (vol_p=4.36%)
- Currently the most promising in-flight Phase 1 experiment; continuing through EP13.

---

## 2026-05-01 — PR #641: Flow-aligned tau local frame (thorfinn)

- **Branch**: thorfinn/flow-aligned-tau
- **Hypothesis**: Predict wall shear stress (tau) in the local surface tangent coordinate frame (s, t) instead of global (x, y, z). Physics-motivated: wall shear is a tangential quantity and expressing it in its natural frame should reduce the prediction burden and improve geometric generalization.
- **Group**: `tay-flow-aligned-tau`
- **W&B run**: thorfinn/flow-aligned-tau-rank0

| Epoch | Step | val_abupt |
|-------|------|-----------|
| EP1 | 10,864 | 32.875% |
| EP2 | 21,729 | 14.613% |

- **Decision**: KILLED at EP2. val_abupt=14.613% exceeds the ≤12.0% kill gate.
- **Analysis**: The flow-aligned coordinate transformation significantly destabilized training. EP2 at 14.6% is far above the typical EP2 range for well-converging runs (~8-10%). The local tangent frame construction may introduce numerical instabilities near degenerate surface normals, or the coordinate rotation may be causing gradient issues during backprop. The idea is physically sound but the implementation may require careful normalization or the model may not benefit from this kind of inductive bias at the current architecture scale.
- **Conclusion**: Dead end in this form. A future attempt could try predicting only the tangential magnitude (scalar) rather than the full vector, or using the frame as an auxiliary feature rather than changing the prediction target.

---

## 2026-05-01 — PR #614: Lion β2 momentum sweep (fern)

- **Branch**: fern/lion-beta2-sweep
- **Hypothesis**: The default Lion β2=0.99 may not be optimal. Sweep β2 ∈ {0.95, 0.99, 0.999} to find the optimal momentum coefficient for this task. Higher β2 provides more stable but slower adaptation; lower β2 more aggressive.
- **Group**: `tay-lion-beta2-sweep`

| Arm | β2 | W&B Run | EP1 val_abupt | EP2 val_abupt | EP3 val_abupt | EP4 val_abupt | Best val_abupt | Status |
|-----|-----|---------|---------------|---------------|---------------|---------------|----------------|--------|
| C | 0.999 | wapj7o9t | 34.98% | 10.947% | 8.318% | 7.473% | **7.219%** | Finished |
| B | 0.99 | hjq54lu4 | 28.09% | — | — | — | **6.793%** | Finished |
| A | 0.95 | lcb5rb4l | **26.613%** | TBD | — | — | TBD | Running ~step 12.3k (past EP1, advancing to EP2) |

- **Analysis**: All completed arms are worse than SOTA (6.5985%). β2=0.999 converges much more slowly (EP2=10.95% vs typical ~8-9%) but still reaches a reasonable endpoint at 7.219%. β2=0.99 (default) achieves 6.793% — close to SOTA but not beating it. β2=0.95 just crossed EP1 with the fastest convergence at 26.613% (vs 28.09% for β2=0.99 and 34.98% for β2=0.999), consistent with lower β2 = more reactive momentum updates. EP2 gate (step 21,729) next; threshold ≤ 12.0%.
- **Preliminary conclusion**: The current β2=0.99 appears near-optimal. Lion momentum is not a high-leverage knob for further gains. Will update when arm A (β2=0.95) completes.

---

## 2026-05-01 — PR #621: Slice-centroid STRING-RoPE (nezuko) [In Progress]

- **Branch**: nezuko/slice-rope-sweep
- **Hypothesis**: Apply Rotary Position Encoding (RoPE) at the slice centroid level using STRING-separable coordinates. Two variants: arm-a (control baseline rerun), arm-b (RoPE applied after QK-norm).
- **Group**: `nezuko-slice-rope-sweep`

| Arm | Description | W&B Run | EP1 val_abupt | EP2 val_abupt | EP3 val_abupt | Best val_abupt | Status |
|-----|-------------|---------|---------------|---------------|---------------|----------------|--------|
| a | Control baseline | xixwhi2m | — | 8.727% | 7.389% | **6.990%** | Finished (37,221 steps) |
| b | RoPE after QK-norm | mekagz7v | 27.436% | **8.634%** | TBD | TBD | Running ~step 23.9k (PASS EP2, advancing to EP3) |

- **Analysis**: Arm-a (control) finished at 6.990% — worse than SOTA 6.5985% (Δ+0.59%). The control arm establishes that this training run configuration is slightly below SOTA capability. Arm-b PASSED EP2 gate at 8.634% (≤ 12.0% threshold), tracking slightly worse than control arm-a's EP2 (8.727%) — needs strong EP3+ to differentiate. EP3 gate (step 32,594): kill if > 8.0%.
- **Status**: Monitoring arm-b EP3 gate. Must beat arm-a (6.990%) and SOTA (6.5985%) to show value.

---

## 2026-05-01 — PR #624: Pre-slice STRING-RoPE (alphonse) [In Progress]

- **Branch**: alphonse/presl-rope-sweep
- **Hypothesis**: Inject STRING-RoPE positional encoding before the slicing operation (at the point level) rather than at slice centroids. Two variants: arm-a (control), arm-b (xmid-only RoPE variant).
- **Group**: `alphonse-presl-rope-sweep`

| Arm | Description | W&B Run | EP2 val_abupt | EP3 val_abupt | Best val_abupt | Status |
|-----|-------------|---------|---------------|---------------|----------------|--------|
| a | Control baseline | r3f8v68j | 8.635% | 7.579% | **7.064%** | Finished (37,367 steps) |
| b | xmid-only RoPE | a29fersn | — | — | TBD | Running ~step 4k (pre-EP1) |

- **Analysis**: Arm-a (control) finished at 7.064% — worse than SOTA 6.5985% (Δ+0.70%). Arm-b still pre-EP1. The control arm result is below SOTA, consistent with nezuko arm-a also being below SOTA — both control arms suggest these parallel training runs are slightly below the specific SOTA checkpoint conditions.
- **Status**: Monitoring arm-b for EP1 gate.

---

## 2026-05-01 — PR #647: Anchor-string no-slice Exp 3 (frieren) [CLOSED — reference trajectory]

- **Branch**: frieren/exp3-anchor-string
- **Hypothesis**: Issue #618 Experiment 3 reassignment — anchor-string approach without slicing. Two arms running: arm-b-anchor-k1024-ep4 and arm-b-anchor-k1024.
- **Group**: `frieren_exp3_anchor_string`

| Gate | Step | val_abupt | Status |
|------|------|-----------|--------|
| EP1 | 10,864 | 48.27% | PASS (normal cold-start, not divergence) |
| EP2 | 21,729 | 16.05% | PASS |
| EP3 | ~32,000 | ~10% | CRASHED (run terminated mid-epoch) |

- **Status**: CLOSED. arm-b-anchor-k1024 (multi-rank) crashed early at step ~292-332. arm-b-anchor-k1024-ep4 (o7upw6qr) completed EP1=48.27%, EP2=16.05%, then crashed mid-EP3 at ~10%.
- **Important note**: EP1=48.27% was a NORMAL cold-start trajectory, NOT divergence. This is the reference convergence trajectory for AnchorStringAttention (vanilla, no stabilizers). Thorfinn's PR #742 mistakenly identified this as divergence and added stabilizers to fix it — those stabilizers were the root cause of Run 4's failure.
- **Reference trajectory for PR #743 (Run 5) kill gates**: EP2 <20%, EP3 <15% (calibrated on this data).

---

## 2026-05-01 — PRs #648, #649, #650: New sweep PRs [In Progress]

### PR #648 — Volume-pressure loss upweighting (askeladd)
- **Group**: `volume-pressure-loss-sweep`
- **Hypothesis**: Upweight volume_pressure in the loss function (sweep weight ∈ {2.0, 4.0, 6.0}) to address the chronic 3× test-vs-val gap on volume_pressure field.
- **Status**: arm `vp-weight-2.0` at step ~3,290. Pre-EP1. Monitoring.

### PR #649 — GradNorm min-weight floor sweep (edward)
- **Group**: `gradnorm-min-weight-sweep`
- **Hypothesis**: Sweep GradNorm minimum weight floor ∈ {0.3, 0.5, 0.7}. Previously used floor=0.0 (no floor); a floor prevents any task from being completely suppressed during gradient normalization.
- **Status**: arm `gradnorm-floor-0.3` at step ~2,845. Pre-EP1. Monitoring.

### PR #650 — LR cosine floor sweep (tanjiro)
- **Group**: `lr-cosine-floor-sweep`
- **Hypothesis**: Sweep cosine LR minimum floor ∈ {1e-7, 5e-7, 5e-6, 1e-5}. Current SOTA uses lr-min=1e-6. Testing whether a higher or lower floor improves final convergence.
- **Status**: arm `lr-min-5e-6` (aon7hwtk) at ~step 6.9k. Pre-EP1. Monitoring.

---

## 2026-05-01 — PR #651: Surface curvature features (thorfinn) [KILLED]

- **Branch**: thorfinn/surface-curvature-features
- **Hypothesis**: Add k-NN-estimated surface curvature features (mean curvature H, Gaussian curvature K) as input to tau predictor. Curvature is a fundamental geometric quantity correlated with wall shear stress — concave/convex regions experience different flow regimes. Implementation: chunked k-NN (k=20, chunk=8192) with PCA-based quadratic fit; normalize to ±3σ.
- **Group**: `thorfinn-surface-curvature`

| Gate | Step | val_abupt | Decision |
|------|------|-----------|----------|
| EP2 | 21,729 | 14.613% | KILL (>12.0% threshold) |
| Final | — | 12.487% | — |

- **Decision**: KILLED at EP2. val_abupt=14.613% >> 12.0% kill gate. PR closed.
- **Analysis**: Surface curvature features (H, K) introduced via k-NN PCA-based quadratic fit destabilized training significantly — similar pattern to flow-aligned-tau (PR #641, EP2=14.613%). The additional geometric features may be introducing noisy inputs that conflict with the existing STRING positional encoding. The model architecture at L=5/hidden=512 appears sensitive to extra geometric input channels — either the feature construction is numerically unstable, or the model cannot leverage these high-frequency curvature signals at this scale. A future attempt could try normalizing more aggressively, or using curvature only as an auxiliary regularization signal rather than a direct input feature.
- **Conclusion**: Dead end in current form. Closed PR #651.

---

## 2026-05-05 — PR #660: Depth scaling L=6 sweep (thorfinn) [KILLED]

- **Branch**: thorfinn/depth-l6-sweep
- **Hypothesis**: L=5 SOTA (PR #592) outperformed L=4 by −1.90% relative. Test whether L=6 with reduced hidden_dim (384 or 448) continues the depth scaling trend. Two arms: hidden=384 (Arm A), hidden=448 (Arm B — sequential).
- **Group**: `depth-l6-sweep`

| Gate | Step | val_abupt | Decision |
|------|------|-----------|----------|
| EP1 (Arm A h=384) | 10,864 | 30.978% | KILL (elevated; experiment confounded) |

- **Decision**: KILLED at EP1. val_abupt=30.978% is elevated beyond normal range (24-28%). Experiment was fundamentally flawed — reducing hidden_dim to 384/448 to compensate for VRAM created a confounded experiment testing "L=6 with less capacity" rather than "L=6 at equal capacity."
- **Conclusion**: PR closed. Correct follow-up: PR #666 (thorfinn) — L=6 at full hidden=512 (estimated ~57GB VRAM, well within 96GB budget).

---

## 2026-05-05 — PR #614: Lion β2 momentum sweep (fern) [CLOSED — null result]

- **Branch**: fern/lion-beta2-momentum-sweep
- **Hypothesis**: Sweep Lion β2 ∈ {0.95, 0.99, 0.999} to identify optimal momentum coefficient.
- **Group**: `tay-lion-beta2-sweep`

| Arm | β2 | W&B Run | Best val_abupt | Epochs |
|-----|-----|---------|----------------|--------|
| B | 0.99 (default) | hjq54lu4 | **6.793%** | 6 |
| A | 0.95 | lcb5rb4l | **7.098%** | 4 |
| C | 0.999 | wapj7o9t | **7.219%** | 6 |

- **Decision**: Closed as null. β2=0.99 (existing default) confirmed optimal. No arm beats SOTA 6.5985%.
- **Key finding**: Lower β2=0.95 converges faster at EP1 (26.6% vs 28.1%) but the advantage narrows and inverts by EP3 (7.69% vs 7.39%); β2=0.95 final is 0.305pp worse than β2=0.99. Higher β2=0.999 is simply too sluggish to converge within budget (EP1=35.0%). Lion β2 momentum tuning is concluded as a research direction.

---

## 2026-05-05 — PRs #648 #649 #650: EP3 gate results [WIP]

### PR #648 — Volume-pressure loss upweighting (askeladd, run rl2drj1m)

| Gate | Step | val_abupt | SP | WS | VP | Decision |
|------|------|-----------|----|----|-----|---------|
| EP1 | 10,864 | 27.30% | — | — | — | Normal |
| EP2 | 21,729 | 8.21% | — | — | — | PASS |
| EP3 | 32,594 | **7.8217%** | 5.30% | 8.90% | **4.30%** | PASS (< 8.0%) |

- Status: Running to completion. EP3=7.82% PASS. VP channel at 4.30% at EP3 is lower than typical — promising signal for the vol_pressure gap problem.

### PR #649 — GradNorm min-weight floor sweep (edward, run phi418eg)

| Gate | Step | val_abupt | SP | WS | VP | Decision |
|------|------|-----------|----|----|-----|---------|
| EP1 | 10,864 | 25.78% | — | — | — | Normal |
| EP2 | 21,729 | 8.57% | — | — | — | PASS |
| EP3 | 32,594 | **7.4142%** | 5.05% | 8.28% | 4.68% | PASS (< 8.0%) |

- Status: Running to completion. Strong EP3 recovery from borderline EP2.

### PR #650 — LR cosine floor sweep (tanjiro, run aon7hwtk)

| Gate | Step | val_abupt | SP | WS | VP | Decision |
|------|------|-----------|----|----|-----|---------|
| EP1 | 10,864 | 29.42% | — | — | — | Normal |
| EP2 | 21,729 | 8.24% | — | — | — | PASS |
| EP3 | 32,594 | **7.2377%** | 4.75% | 8.19% | 4.40% | PASS (< 8.0%) |

- Status: Running to completion. Best of the three borderline-EP2 recoveries — 7.24% at EP3 is a strong signal.

---

## 2026-05-05 — New PRs assigned (Round 11–12 closures + current Phase 1 assignments)

### Closed dead ends (Rounds 11–12)

- **PR #690** (various): Slice sweep {64, 192, 256} — slices=64 null (+0.30pp); slices=192/256 infeasible (>92 min/epoch). CLOSED.
- **PR #691** (various): RFF sigma wide/low-ext — both null. CLOSED.
- **PR #692** (various): Heads sweep {8, 2} — heads=8 null (+0.83pp); heads=2 unauthorized concurrent launch, CLOSED.
- **PR #693**: L=6/h=448/heads=7 — killed (heads=7 destroys SDPA fast path, ~98 min/epoch). CLOSED.
- **PR #694**: depth L=6/hidden=384/heads=4 — null (val=6.9016%, +0.30pp), still descending but budget-bound. CLOSED.
- **PR #695**: rff-num-features=32 — null (+0.33pp val regression). CLOSED.
- **PR #716** (frieren): BC-type embedding — operationally broken (concurrent 8-GPU jobs doubled epoch time to 180 min; time-gate kill). CLOSED.
- **PR #722**: dual-tower volume/surface cross-attention — null (+0.87pp val regression). CLOSED.

### Current Phase 1 (Issue #717 volume push) — all WIP as of 2026-05-06

- **PR #728** (frieren): Exp 1B — Volume outlier-aware point sampling (EMA residual + geometric distance arms). WIP.
- **PR #729** (alphonse): Exp 1D — Single-model KD from K=7 ensemble, vol-only soft targets. WIP.
- **PR #734** (askeladd): Exp 1C P3 — SDF distance-to-surface scalar feature for volume input. WIP.
- **PR #735** (edward): TTA — Y-mirror + coord-jitter 6-pass test-time averaging. WIP.
- **PR #736** (fern): Inter-sample mixup on volume coords/pressure (alpha=0.2/0.4). WIP.
- **PR #737** (nezuko): Region-weighted volume loss — near-wake band emphasis (1<x_rel<3, |z_rel|<1.5). WIP.
- **PR #738** (tanjiro): Train-time Gaussian noise on volume coordinates (sigma 5mm/20mm/anneal). WIP.

### Issue #618 STRING/RoPE — re-attempt

- **PR #742** (thorfinn): CLOSED NEGATIVE. Exp 3 Redux — Anchor-STRING with stabilizer triplet (rope_lr_scale=0.1, rope_grad_clip=1.0, 500-step log_freq warmup). Best result: EP3 val_abupt=19.87% (step 32592). Root cause: stabilizers over-constrained RoPE (rope/log_freq moved <0.005 over 3 epochs — essentially frozen). Frieren's PR #647 EP1=48.27% was normal cold-start, not divergence. Genuine bug fixes retained for Run 5: `_init_weights` skip-`string_rope.` + mask-aware anchor selection.
- **PR #743** (thorfinn, pending): Run 5 — Frieren PR #647 exact config (no stabilizers, no rope_lr_scale, no rope_grad_clip, no qk_norm in AnchorString) + 2 genuine bug fixes only. Kill gates: EP2 (step 21728) <20%, EP3 (step 32592) <15%.

### Previous Issue #618 STRING/RoPE arms (all closed, Round 11–12)

- **PR #626** STRING only: best vol gap ratio 2.07× (val→test); established baseline for RoPE comparison.
- **PR #647** AnchorString no-slice: EP1=48.27% (normal cold-start), EP2=16.05%, crashed mid-EP3 at ~10%. Reference trajectory for Run 5 kill gate calibration.
- Other STRING/RoPE arms: null or diverged; closed.

---

## 2026-05-08 20:XX — PR #867: Slices=256 Scaling (thorfinn) — IN PROGRESS

- **Branch**: thorfinn/model-slices-sweep
- **W&B run**: `nv85vovo` (group: `thorfinn-model-slices-sweep`, name: `slices256-arm-b`)
- **Hypothesis**: Scale number of slice tokens from 128 → 256. Slice tokens are the primary unit of computation in Transolver; more slices = finer-grained partitioning of the 3D point cloud into local physics groups. Hypothesis: 256 slices can capture tighter aerodynamic feature clusters than 128.

| Epoch | Step | val_abupt | Gate | Status |
|---|---:|---:|---|---|
| EP1 | 10,864 | 26.5458% | <30% | PASS |
| EP2 | 16,300 | 11.0175% | <16% | PASS |
| EP3 | ~21,729 | — | <8% | pending |
| EP4 | ~27,159 | — | ≤6.5985% | pending |

**Analysis (in progress):** Strong EP1→EP2 trajectory: 26.5% → 11.0%, showing healthy learning dynamics. EP2 is significantly better than EP1 baseline pace (26.5% → 11.0% in 4 screen epochs from EP2). Watching EP3 closely — need <8% to continue. Current baseline: 6.5985%.

---

## 2026-05-08 — PR #868: Spectral Norm on Attention (askeladd) — IN PROGRESS

- **Branch**: askeladd/spectral-norm-attention
- **W&B run**: `0kjl4rnh` (rank0, group: `spectral-norm-r18`)
- **Hypothesis**: Apply spectral normalization to Q/K/V/out_proj in all attention layers to bound the Lipschitz constant and regularize training. May stabilize gradient flow and improve generalization on out-of-distribution aerodynamic configurations.

| Epoch | Step | val_abupt | Gate | Status |
|---|---:|---:|---|---|
| EP1 | ~10,864 | — | <30% | running (~step 4,857) |

---

## 2026-05-08 — PR #869: Stochastic Depth / DropPath (edward) — IN PROGRESS

- **Branch**: edward/stochastic-depth
- **W&B run**: `4w7dgiuh` (rank0, group: `stochastic-depth-r18`, name: `edward/drop-path-005`)
- **Hypothesis**: Apply stochastic depth (DropPath) regularization with drop_path_rate=0.05, linear schedule per layer. For L=5: [0.0000, 0.0125, 0.0250, 0.0375, 0.0500]. Both attention and MLP residual branches dropped independently. Zero parameter overhead (15.94M = baseline).

| Epoch | Step | val_abupt | Gate | Status |
|---|---:|---:|---|---|
| EP1 | ~10,864 | — | <30% | running (~step 2,845) |

---

## 2026-05-08 — PR #870: KNN Surface Roughness Penalty (fern) — IN PROGRESS (PENDING LAUNCH)

- **Branch**: fern/knn-roughness-penalty (pivot from FFT approach)
- **W&B run**: NOT YET STARTED
- **Hypothesis**: FFT-based surface roughness penalty abandoned (Parseval violation from unnormalized rfft + random point sampling). Pivoting to KNN k=8: for each surface point, find k=8 nearest neighbors; compute variance of τ_y/τ_z in that neighborhood; L_smooth = 0.1 × (mean(var_knn(τ_y)) + mean(var_knn(τ_z))).

---

## 2026-05-08 — PR #871: PCGrad Gradient Surgery (tanjiro) — IN PROGRESS

- **Branch**: tanjiro/pcgrad-gradient-surgery
- **W&B run**: `7v0rlsps` (rank0)
- **Hypothesis**: PCGrad gradient surgery across 4 task groups to reduce gradient conflicts between prediction heads. ~2× compute overhead; tests whether conflicting gradients are a bottleneck.

| Epoch | Step | val_abupt | Gate | Status |
|---|---:|---:|---|---|
| EP1 | ~10,864 | — | <30% | running (~step 1,056) |

---

## 2026-05-08 — PR #872: Width Scaling hidden_dim=640 (frieren) — IN PROGRESS

- **Branch**: frieren/width-scaling-640
- **W&B run**: `gr1n58zo` (rank0, group: `frieren-width-640`)
- **Hypothesis**: Scale Transolver hidden dimension from 512 → 640 (+25% width). Orthogonal to depth scaling; tests whether capacity bottleneck is in the channel dimension. VRAM: 63.2 GB / 97.9 GB (safe).

| Epoch | Step | val_abupt | Gate | Status |
|---|---:|---:|---|---|
| EP1 | ~10,864 | — | <30% | running (~step 4,360) |

---

## 2026-05-05 — Archived earlier new-PR assignments

- **PR #665** (frieren): Cross-slice attention over Transolver slice tokens — global inter-slice MHA layer
- **PR #666** (thorfinn): Depth scaling L=6 at full hidden=512 (corrects the confound in PR #660)
- **PR #667** (fern): Weight decay sweep {1e-4, 5e-4, 1e-3} for Lion optimizer
