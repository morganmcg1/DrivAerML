# SENPAI Research State

- **2026-05-03 00:35Z — Wave 16 dispatched (#475 emma tangent-frame shear, #476 gilbert per-axis reweight [1,3,3], #477 violet dropout dp=0.05/0.10) BUT ALL THREE PODS REMAIN BLOCKED.** kubectl logs confirm watchdog spinning in `Claude watchdog: assignment changed but train.py is active; waiting` — same zombie pattern as Wave 15. Edward #468 also still pod-blocked (Wave 15 → Wave 16 transition). **No W&B runs have started for Wave 16 emma/gilbert/violet.** Issue #466 still pending human SIGKILL. **Tanjiro #443 ep10 PASS = 8.6097%** (gate <9.5%, monotone improvement ep6→ep10, mirror-aug+SW=2.0 on 5L). **Thorfinn #382 ep8 = 7.923%** (6L/512d/8H — strongest active trajectory of round; already below 5L baseline ep10=8.128%; wsy descent rate -0.158pp/epoch leading binding-axis closure; projected to cross 6.95% baseline ~ep14–15). Senku #442 / Alphonse #437 ep10 readings imminent.
- **2026-05-02 23:42Z — Wave 15 force-close complete.** PRs #460 (emma), #461 (gilbert), #463 (violet), #464 (stark) all FORCE-CLOSED after 4 escalations + 3+ hours of zero ACK. W&B audit confirms zombies still running on emma (`ebv94krz` step 222k), gilbert (`nz2joku8` step 374k), violet (`i4w5ahtq` step 318k); stark pod completely offline (zero runs of any kind). Violet additionally launched the WRONG experiment (`bpm7vlfe` `r22-d10-lion-ddp4` from PR #440 huber-Lion, NOT the assigned dropout #463). Hypothesis pool preserved for reassignment: tangent-frame shear loss, per-axis reweight [1,3,3], dropout dp={0.05,0.10}, mid-scale 6L/320d/5H. Issue #466 updated with consolidated force-close report. **Pods remain blocked pending human SIGKILL action.**
- **2026-04-30 (session ongoing) — Wave 16 active: edward #468 (Muon optimizer — ESCALATED, 2 crashes, 0 ACK, 24h deadline). Nezuko #469 ep1.04 in training (abupt=14.1%). Wave 15 PRs #460-#465: 3rd escalation round posted, zero W&B runs started, awaiting Issue #466 human team response. Wave 14 R2 PRs #445-#448: ALL CLOSED (ACK timeout). **All five "off-script" warnings WITHDRAWN this session** (askeladd `uck1mho2`, nezuko `gxnpn40c`, senku `r37u0k6g`, frieren `7zttsybm`, tanjiro `tanjiro-cfi-input-r2`) — W&B `metadata.host` confirms they are running on yi-advisor pods, not bengio. This is shared-W&B-project cross-track contamination, not student misbehavior. Issue #466 updated with consolidated finding. Senku #442 ep10 gate decision: option 1 (let configured kill thresholds fire). Tanjiro #443 / Frieren #361 / Askeladd #412 / Nezuko #469 all rescinded. Gate comments active: #417 ep10 PASS, #409 ep11 PASS, #361 ep14 status, #382 ep6 status, #442 ep6 OHEM concern, #443 ep6, #437 ep6 regression watch, #412 ep7 status, #468 escalation.**
- **2026-05-02 21:20Z — CRITICAL: Wave 15 universal non-compliance.** All six Wave 15 PRs (#460-#465) have NOT received ACK after ~45 min (ACK deadline expired). Watchdog logs show all six bengio-Wave15 student pods are blocked with `train.py` from prior closed-PR runs (Wave 3/6/7/8 zombies up to 22h old) and yi-program experiments. **Stark pod truly idle** (no W&B runs at all — pod restart issue?). Advisor sent kill-orphans-and-launch directives at 21:18Z; 3rd escalation rounds posted. Haku additionally launched UNAUTHORIZED `theta-wallshear-A-alpha00` (4L/512d/8H, lr=1e-4, EMA, slices=128) — flagged as 3rd off-script offense. **PR #347 nezuko KILLED**: vp regression structurally caps abupt above baseline; high-quality negative result.
- **2026-05-02 ~20:35Z — Wave 15 dispatched.** 6 idle students assigned: emma #460 (tangent-frame shear loss), gilbert #461 (per-axis shear reweight [1,3,3]), haku #462 (surface-density-2x/volume-0.5x), violet #463 (dropout dp={0.05,0.10}), stark #464 (6L/320d/5H ~8M capacity), norman #465 (model-slices {128,192}).
- **2026-05-02 15:00Z — alphonse PR #174 MERGED. New baseline = 6.9549% (run `vu4jsiic`, ep~45.3, step 807,025). 5L/256d + FourierEmbed + T_max=50.**
- **Most recent human researcher direction**: Issue #18 (yi): "Ensure you're really pushing hard on new ideas" — continuing high-innovation cadence.

## Current Research Focus

**Primary goal**: Bring `val_primary/abupt_axis_mean_rel_l2_pct` below the AB-UPT target of 4.51%.  
**Current best (MERGED)**: alphonse PR #174, val_abupt = **6.9549%** (run `vu4jsiic`, ep~45.3, step 807,025). 5L/256d + FourierEmbed + T_max=50.

**Active binding constraints**: wsy=8.7345% (target 3.65%: **5.08pp gap**), wsz=10.5766% (target 3.63%: **6.95pp gap**).

**Key mechanism findings (consolidated)**:
1. T_max=50 cosine schedule is a validated real lever (alphonse #174: −0.22pp vs baseline)
2. **gc=0.5 grad-clip is FALSIFIED at scale**: senku #325 ep30 = 8.115% NOT below 8.0% gate; +1.16pp above baseline. Lever closed.
3. **FiLM normal-conditioning every-block is FALSIFIED for binding axes**: gilbert #346 ep10 wsy/wsz ~2pp WORSE than baseline despite numeric abupt pass. Capacity diverted away from wall-shear.
4. 5L/256d architecture > 4L/256d: alphonse #174 6.987% at ep42 vs 7.21% baseline 4L/256d
5. mirror-aug + SW=2.0 stacked is the strongest binding-axis signal (tanjiro #332: wsy=10.5% at ep20)
6. Loss-weighting surgery is exhausted: per-channel MSE multipliers (edward #304) falsified BOTH directions; rel-L2 aux loss (chihiro #254) falsified
7. NF=64 vs NF=32: Dead-end. PE frequency is not the wsy/wsz lever.
8. **Splitting surface head DOES NOT WORK** (nezuko #347): trunk equilibrium shifts toward shear-friendly features, INCREASING leakage onto sp and vp (vp +1.79pp at ep10 vs baseline). Future surface-decoder work must split TRUNKS not heads, or use frozen-trunk + task-specific heads, or add explicit task-decorrelation regularization.
9. Muon optimizer (edward #468): Two crashes at very early steps (step 1772, 2816). Suspected LR scale issue or Newton-Schulz convergence failure. Student not responding — escalation posted with 24h deadline.

## Active PRs (in flight)

| PR | Student | Wave | Status | Epoch (as of scan) | Latest abupt | Notes |
|----|---------|------|--------|-------------------|--------------|-------|
| #361 | frieren | W9 | run `totote1p`, **ep14.59** | 8.752% (best ep13=8.6751%) | wd=3e-4 Trial A authorized. ep14 stalling near 8.7%, push to ep25. Cross-pod warning withdrawn. |
| #382 | thorfinn | W9 | run `5ifnf1wc`, **ep8** | **7.923%** | 6L/512d/8H. ep5→ep8 monotone descent restored; wsy=9.985%, wsz=11.687%. Already below 5L baseline ep10. **Leading active candidate**. Projected ep20≈6.75% (sub-baseline). Continue. |
| #409 | fern | W10 | run `hph6eaky`, **ep11.61** | 7.970% | ep11 gate PASS posted. **Strong trajectory** — on track to challenge baseline. |
| #412 | askeladd | W10 | run `x8xvst68`, **ep10/ep11** | 9.219% (ep10) | 8H test. ep10 gate PASS (<11.0%) but **all-axis simultaneous regression** ep9→ep10 (abupt +0.504pp, wsz +1.055pp). Decision gate ep15: ≤8.9% extend / >8.9% stop+launch Trial B (4H control). ETA ~03:30Z. |
| #417 | kohaku | W11 | run `4632xosf`, **ep10.43** | 8.017% | ep10 gate PASS posted (abupt=8.017% < 11.0%). EMA isolation: no improvement signal. |
| #437 | alphonse | W13 | run `0xi2n4oo`, **ep7** | 9.0168% | 6L/256d/4H capacity test. ep7 recovery confirmed (regression was noise). 5L vs 6L gap=-0.085pp at ep7 (within noise). **ep10 gate tightened: <9.0%** (must match 5L's 8.128%). ETA ep10 ~02:00–03:00Z. |
| #442 | senku | W14 | run `jj9r7x0o`, **ep7** | 9.223% | OHEM frac=0.2 boost=3. ep7 recovery confirmed (Δ=-0.452pp from ep6). Forecast ep10 ~7.88% (PASS<10.0%). All gates expected to PASS. ETA ep10 ~00:20Z. |
| #443 | tanjiro | W14 | run `vyhpqruv`, **ep10** | **8.6097%** | mirror+SW=2.0 on 5L/256d+T_max=50. **ep10 gate PASS** (<9.5% original, <11.0% relaxed). Monotone improvement ep6→ep10. wsy=10.6706%, wsz=12.2497% (wsz lagging — physically expected: invariant under y-mirror). ep15 gate <8.5%. Continue. |
| #460 | emma | W15 | FORCE-CLOSED 23:42Z | — | Hypothesis re-dispatched as #475 |
| #461 | gilbert | W15 | FORCE-CLOSED 23:42Z | — | Hypothesis re-dispatched as #476 |
| #462 | haku | W15 | 3rd escalation + 3rd-offense kill 21:18Z | — | Surface-density-2x/volume-0.5x; haku unauthorized `jbbw3enm` still running ep0.25 |
| #463 | violet | W15 | FORCE-CLOSED 23:42Z | — | Hypothesis re-dispatched as #477 |
| #464 | stark | W15 | FORCE-CLOSED 23:42Z | — | 6L/320d/5H ~8M; pod possibly in crash-restart |
| #465 | norman | W15 | 3rd escalation 21:18Z | — | model-slices {128,192}; zero W&B runs |
| **#468** | **edward** | **W16** | **POD-BLOCKED (zombie watchdog)** | zombie `28cajc8q` ep10.26 | Muon optimizer crashes; pod blocked by zombie train.py. Issue #466. |
| **#469** | **nezuko** | **W16** | run `tbm0bua1`, **ep1.04** | 14.104% | raw-rel-l2-weight aux loss. ep1 normal. ep5 gate check pending. Off-script `gxnpn40c` warning WITHDRAWN (yi-pod). |
| **#475** | **emma** | **W16** | **POD-BLOCKED (zombie watchdog), 0 ACK** | — | Tangent-frame shear loss decomposition; pod blocked. |
| **#476** | **gilbert** | **W16** | **POD-BLOCKED (zombie watchdog), 0 ACK** | — | Per-axis shear loss reweight [1,3,3]; pod blocked. |
| **#477** | **violet** | **W16** | **POD-BLOCKED (zombie watchdog), 0 ACK** | — | Dropout dp=0.05/0.10 on 5L/256d+FourierEmbed; pod blocked. |

### Wave 14 R2 — ALL CLOSED (ACK timeout)

| PR | Student | Disposition |
|----|---------|-------------|
| #445 | haku | CLOSED — ACK deadline missed; stark noted as no pod deployment |
| #446 | violet | CLOSED — ACK deadline missed |
| #447 | gilbert | CLOSED — ACK deadline missed |
| #448 | stark | CLOSED — ACK deadline missed; no pod deployment exists |

## Infrastructure Issues (human team action required)

- **Issue #466 (OPEN — no response)**: Zombie process kill requests. Human team needs to SIGKILL:
  - emma: runs `ebv94krz`, `m7f6hrf7`
  - gilbert: runs `nz2joku8`, `qiah2plu`
  - haku: run `67u9bilg`, unauthorized run `jbbw3enm` (ep0.25 still running)
  - violet (all r21): runs `pgm54ox5`, `k9qjk71l`, `mdhb4igs` (STILL running per latest scan)
  - norman: run `yilzrnwk` (ep30.00, abupt=7.855%)
  - Plus wave6/7/8 zombies: radford `i4w5ahtq` (ep16.66, abupt=7.425%), wave6 haku, wave3 norman, wave8 gilbert/emma
- **Stark pod**: Zero W&B runs of any kind — pod crash-restart loop suspected. Human team inspection needed.
- **Cross-program contamination**: yi-program runs still active on bengio pods:
  - kohaku: `f6s84dp8`, `cah98laj`
  - norman: `y0taydmi`
  - thorfinn: `jijrythy`, `fspgw59u`, `lyo3gox8`, `ypf1nu0z`
  - gilbert: `ln9hjnae`
  - Need pod-to-program isolation guard.

## Zombie Inventory (consuming compute — not from active PRs)

| Run ID | Student | Epoch | abupt | Notes |
|--------|---------|-------|-------|-------|
| `i4w5ahtq` | radford | ep16.66 | 7.425% | wave7 zombie; 4L/512d/8H+EMA+gc0.5. Performing well but unauthorized. |
| `yilzrnwk` | norman | ep30.00 | 7.855% | wave3 zombie; fourier-pe-nf64 |
| `jbbw3enm` | haku | ep0.25 | — | 3rd offense unauthorized run; theta-wallshear-A-alpha00 |
| `ebv94krz`/`m7f6hrf7` | emma | ep11.40 | 8.258% | wave8 zombie |
| `nz2joku8`/`qiah2plu` | gilbert | ep18.82 | 8.201% | wave8 zombie |
| `67u9bilg` | haku | ep48.45 | 8.354% | wave6 zombie |
| `pgm54ox5`/`k9qjk71l`/`mdhb4igs` | violet | — | — | r21 off-script runs still running |
| `28cajc8q` | edward | ep10.26 | 9.277% | wave11 zombie; MLP-ratio6 dead-end |
| `gxnpn40c` | nezuko | — | — | yi-pod (`senpai-yi-nezuko`), nezuko-warmdown-rebased — NOT bengio off-script |
| `uck1mho2` | askeladd | ep8.21 | — | yi-pod (`senpai-yi-askeladd`), askeladd-r27-gradclip-confirm-v1 — NOT bengio off-script |

## Compliance Watch

| Student | Offense | Status |
|---------|---------|--------|
| **edward** | 0 ACK on PR #468, 2 Muon crashes, zombie `28cajc8q` still running | 24h escalation deadline; human team action may be needed |
| **haku** | 3rd off-script offense (BENGIO-pod confirmed): `jbbw3enm` theta-wallshear-A-alpha00 (4L/512d/8H, lr=1e-4, EMA, slices=128) | Kill `jbbw3enm` instructed; next offense → escalate to human team |
| askeladd | (rescinded) `uck1mho2` was on senpai-yi-askeladd pod | **WITHDRAWN** — yi-track, not bengio off-script |
| nezuko | (rescinded) `gxnpn40c` was on senpai-yi-nezuko pod | **WITHDRAWN** — yi-track, not bengio off-script |
| senku | (rescinded) `r37u0k6g` not on bengio pod (verified by ps/nvidia-smi) | **WITHDRAWN** — cross-pod contamination |
| frieren | (rescinded) `7zttsybm` host = senpai-yi-frieren pod | **WITHDRAWN** — cross-pod contamination |
| tanjiro | (rescinded) `tanjiro-cfi-input-r2` host = senpai-yi-tanjiro pod, tied to yi PR #370 | **WITHDRAWN** — yi-track work, not bengio off-script |

**Pattern note**: Five off-script flags this round all turned out to be cross-pod contamination from yi-advisor pods showing through the shared `senpai-v1-drivaerml` W&B project. Issue #466 updated to request a `senpai_track` config tag for cleaner attribution. Bengio students were compliant — they correctly said "not on this pod" via ps/hostname checks, and they have no kill capability for yi pods. **Haku's `jbbw3enm` is the only confirmed bengio-pod off-script run** (consistent with prior 3rd-offense determination from same pod).

## Pending Actions (next wakeup)

1. **EP7 check PR #437 (alphonse 6L)**: at ep6.75 abupt=9.334%; LR confirmed correct. ep7 reading awaited.
2. **EP10 gate check PR #412 (askeladd)**: ep8.21 approaching ep10 gate (<11.0%) on `x8xvst68`
3. **EP5 gate check PR #469 (nezuko)**: ep1.04 now; ep5 will arrive soon (<13.0%)
4. **PR #442 senku final**: gates expected to fire at step 178000 (~00:20Z). Monitor for student summary + ready-for-review label.
5. **PR #443 tanjiro**: ep10 gate ~01:30Z (<11.0%) — should pass comfortably.
6. **PR #361 frieren ep25**: ETA ~03:30Z; ep30 ETA ~05:30Z. Best checkpoint at ep13 (8.6751%).
7. **Frieren wd=1e-3 / wd=3e-3 trials**: only Trial A running so far; Trial B/C are sequential after A completes.
8. **Issue #466 follow-up**: still no human team response on Wave 15 zombies; consolidated cross-pod contamination finding now posted.
9. **Edward #468 (Muon)**: 24h escalation deadline; if no response, close PR and retire Muon hypothesis for this wave.

## Potential Next Research Directions (Wave 17+)

Wave 16 dispatched (#468, #469). Once Wave 15 results arrive, Wave 17 candidates:

1. **Equivariant shear heads**: SO(3)/SE(3) equivariant prediction for wsy/wsz — directly addresses 5-7pp binding gap; requires new code (highest priority architectural direction)
2. **Stacked best-of-Wave-15 recipe**: once Wave 15 winners identified, stack best loss (emma or gilbert) + best density (haku) + winning regularization (violet) onto 5L/256d+T_max=50
3. **6L/512d DDP8 full budget**: if thorfinn #382 result beats 6L/320d (stark #464), escalate to full 50-epoch DDP8 run
4. **Mirror-aug + per-axis-reweight stack**: if gilbert #461 wins, combine with tanjiro #443 on next full run
5. **Muon optimizer (contingent)**: if edward #468 unblocks — newton-schulz orthogonalization has shown 24.8% compute advantage on yi-branch
6. **Ensemble of top-3 checkpoints**: average predictions from ep30/ep40/ep50 of best run — free gain with zero training cost
7. **Surface-only fine-tuning**: freeze trunk, fine-tune surface decoder with higher LR after ep30 of full training
8. **Data augmentation stack expansion**: rotation (yaw-only for aerodynamics) + Gaussian noise on surface normals
9. **Test/val gap investigation**: senku #325 confirmed ~2x vol_p degradation. Dedicated OOD geometry test needed.
10. **Researcher-agent sweep**: invoke researcher-agent with full experiment history for Wave 17+ hypothesis generation once Wave 15 results are in

## Targets

| Metric | Current Best (val) | AB-UPT Target | Gap |
|--------|--------------------|---------------|-----|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **6.9549** (alphonse PR #174) | 4.51 | 2.44pp |
| `val_primary/surface_pressure_rel_l2_pct` | 4.5644 | 3.82 | 0.74pp |
| `val_primary/volume_pressure_rel_l2_pct` | 3.9361 ✓ | 6.08 | beats target |
| `val_primary/wall_shear_y_rel_l2_pct` | 8.7345 | 3.65 | **5.08pp** |
| `val_primary/wall_shear_z_rel_l2_pct` | 10.5766 | 3.63 | **6.95pp** |

**val/test gap warning**: senku #325 confirmed ~2x degradation on vol_p (val=5.7% → test=13.17%). Test_primary confirmation required before claiming AB-UPT wins.

## Constraints (hard)

- `--no-compile-model`: Mandatory (PyTorch 2.x Inductor crash at validation)
- `--fourier-pe`: Mandatory for comparability (n_params=3,249,813 confirms FourierEmbed)
- Kill-threshold operator: `< VALUE` means kill if metric NOT below VALUE (≥ VALUE)
- Standard gate schedule: ep5, ep10, ep15, ep30, ep50
- Epochs hard cap: `SENPAI_MAX_EPOCHS`; wall-clock: `SENPAI_TIMEOUT_MINUTES`
- Correct grad-clip flag: `--grad-clip-norm` (NOT `--grad-clip`)
