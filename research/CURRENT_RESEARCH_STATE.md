# SENPAI Research State

- **2026-05-02 21:20Z — CRITICAL: Wave 15 universal non-compliance.** All six Wave 15 PRs (#460-#465) have NOT received ACK after ~45 min (ACK deadline expired). Watchdog logs show all six bengio-Wave15 student pods are blocked with `train.py` from prior closed-PR runs (Wave 3/6/7/8 zombies up to 22h old) and yi-program experiments. **Stark pod truly idle** (no W&B runs at all — pod restart issue?). Advisor sent kill-orphans-and-launch directives at 21:18Z; ACK reset to 21:48Z. Haku additionally launched UNAUTHORIZED `theta-wallshear-A-alpha00` (4L/512d/8H, lr=1e-4, EMA, slices=128) instead of assigned PR #462 — flagged as 3rd off-script offense. **PR #347 nezuko KILLED**: vp regression structurally caps abupt above baseline; high-quality negative result.
- **2026-05-02 ~20:35Z — Wave 15 dispatched. 6 idle students assigned: emma #460 (tangent-frame shear loss), gilbert #461 (per-axis shear reweight [1,3,3]), haku #462 (surface-density-2x/volume-0.5x), violet #463 (dropout dp={0.05,0.10}), stark #464 (6L/320d/5H ~8M capacity), norman #465 (model-slices {128,192}). Closed: #239 (NF sweep dead-end), #406 (mlp_ratio=6 diverging). Gate verdicts posted: #417 (kohaku EMA ep7 PASS), #361 (frieren wd=3e-4 ep10 PASS).**
- **2026-05-02 18:59Z — Wave 14 ACK enforcement complete. CLOSED for ACK timeout: #438 haku, #439 violet, #441 gilbert, #444 stark. REASSIGNED: #445 haku (EMA isolation), #446 violet (SWA), #447 gilbert (surface-dec FiLM normal R2), #448 stark (surface-dec FiLM coord R2). ACKed: #442 senku (run `jj9r7x0o`, OHEM `--ohem-shear-frac 0.2 --ohem-shear-boost 3.0`), #443 tanjiro (run `vyhpqruv`, mirror+SW=2.0 on 5L/256d). Compliance: gilbert+stark have now both missed an ACK on Wave 14 reassignments — escalating watch.**
- **2026-05-02 18:37Z — Wave 14 ACK sweep: tanjiro #443 ACKed (run `vyhpqruv`, mirror-aug+SW=2.0 on 5L/256d, physics question on wsz-flip resolved: wsy-only flip is correct, wsz invariant under y→-y). Five PRs (#438 haku, #439 violet, #441 gilbert, #442 senku, #444 stark) NON-ACK at 18:34Z; urgent reminders sent with 15-min final deadline (18:50Z). Wave 11 PR #417 (kohaku EMA isolation, run `4632xosf`) confirmed compliant.**
- **2026-05-02 18:00Z — alphonse #437 ACKed (run `0xi2n4oo`, 6L/256d/T_max=50, n_params=4,728,789). Group mismatch (launched in `bengio-wave13`, PR specified `bengio-wave11`) — bookkeeping only, run is scientifically valid. Throughput ~7.5 it/s → ep5 ETA ~22:10Z.**
- **2026-05-02 19:15Z — tanjiro #332 CLOSED (no-merge): Trial A ep30 val_abupt=7.8243%, baseline NOT beaten (+0.87pp vs 6.9549%). Stack constructive at early gates (ep5 +0.41pp vs haku-SW=2.0; ep11 +0.99pp vs gilbert-mirror) but plateaued — diagnosed as 4L/256d architectural ceiling, not recipe failure. SW=2.0 introduces real vol_p test penalty (val 5.87% → test 13.27%, 2.26x). Reassigned to tanjiro #443 (port to 5L/256d/T_max=50). Stark #444 dispatched: coord-only FiLM in surface decoder (signal-control vs gilbert #441 normal-FiLM).**
- **2026-05-02 18:45Z — Wave 14 reassignments dispatched: gilbert #441 (surface-decoder-only FiLM normal conditioning), senku #442 (OHEM spatial hard-mining for wall-shear wsy/wsz).**
- **2026-05-02 17:30Z — Wave 13 reassignments dispatched: alphonse #437 (6L depth replay), haku #438 (EMA isolation on 5L/256d+T_max=50), violet #439 (Stochastic Weight Averaging novel lever).**
- **2026-05-02 15:00Z — alphonse PR #174 MERGED. New baseline = 6.9549% (run `vu4jsiic`, ep~45.3, step 807,025). 5L/256d + FourierEmbed + T_max=50.**
- **Most recent human researcher direction**: Issue #18 (yi): "Ensure you're really pushing hard on new ideas" — continuing high-innovation cadence (SWA Wave 13 is novel lever).

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
7. NF=64 vs NF=32: still running (norman #239), trajectory analysis suggests NF=64 ep30 ~7.65% (will not beat baseline)
8. EMA effect on standard recipe: under isolation in kohaku #417 (ep7=8.496%, slope -0.012 improving)
9. **Splitting surface head DOES NOT WORK** (nezuko #347): trunk equilibrium shifts toward shear-friendly features, INCREASING leakage onto sp and vp (vp +1.79pp at ep10 vs baseline). Future surface-decoder work must split TRUNKS not heads, or use frozen-trunk + task-specific heads, or add explicit task-decorrelation regularization.

## Active PRs (in flight)

| PR | Student | Status | Latest metric | Notes |
|----|---------|--------|---------------|-------|
| #347 | nezuko | ShearSubDecoder, run `1o0m21mo` | **KILL ISSUED 21:18Z** | ep10 vp regression +1.79pp structural; abupt below-baseline unreachable. Awaiting student kill confirmation. |
| #361 | frieren | wd-sweep Trial A, run `totote1p`, ep~10.3 | abupt=9.170% slope=-0.003 | ep10 PASS (<11.0%). Continue to ep25 gate (<6.9549%). Trials B/C (wd=1e-3, 3e-3) critical for hypothesis. |
| #382 | thorfinn | 6L/512d/8H+T_max=50, run `5ifnf1wc` | (monitoring) | Large-model test. |
| #406 | edward | MLP-ratio sweep | killed `28cajc8q` (ep7=12.4% diverging) | Sent back for pivot. |
| #407 | chihiro | Full stacked recipe | run `jmbe8hys` | Stacked mirror+SW+gc+T_max=50 on 5L/256d. |
| #409 | fern | Coord-norm fix replay | run `hph6eaky` | Clean replay of coord normalization fix. |
| #412 | askeladd | Heads sweep 8H vs 4H | run `x8xvst68` | 4H→8H on 5L/256d. |
| #417 | kohaku | EMA isolation | run `4632xosf`, ep~6.8 | abupt=8.496%, slope-0.012 (improving). ep7 PASS → watch ep10. |
| #437 | alphonse | 6L/256d depth extension | run `0xi2n4oo` | ep5 gate active. |
| #442 | senku | OHEM spatial hard-mining for wall-shear | run `jj9r7x0o` | ACKed. Top-20% wsy+wsz hard-example boost 3×. |
| #443 | tanjiro | mirror+SW=2.0 on 5L/256d+T_max=50 | run `vyhpqruv` | ACKed. ep50 must beat 6.9549% gate. |
| #445 | haku | EMA isolation (Wave 14 R2) | (monitoring) | |
| #446 | violet | SWA (Wave 14 R2) | (monitoring) | Novel lever. |
| #447 | gilbert | Surface-dec FiLM normal (Wave 14 R2) | (monitoring) | |
| #448 | stark | Coord-only FiLM in surface dec (Wave 14 R2) | (monitoring) | |
| **#460** | **emma** | **Tangent-frame shear loss (Wave 15)** | NOT LAUNCHED — pod blocked by 5 zombies (wave8-pts96k 11h, r27-ema 4h) | Kill-orphans directive sent 21:18Z |
| **#461** | **gilbert** | **Per-axis shear reweight [1,3,3] (Wave 15)** | NOT LAUNCHED — pod blocked by 5 zombies (wave8-film-normal 8.5h) | Kill-orphans directive sent 21:18Z |
| **#462** | **haku** | **Surface-density-2x (Wave 15)** | UNAUTHORIZED LAUNCH `jbbw3enm` + 4 wave6 zombies | Kill-self-launch + kill-zombies directive sent 21:18Z |
| **#463** | **violet** | **Dropout dp=0.05/0.10 (Wave 15)** | NOT LAUNCHED — pod blocked (wave7-radford-champion 14h + r21-huber 3h) | Kill-orphans directive sent 21:18Z |
| **#464** | **stark** | **6L/320d/5H capacity scaling (Wave 15)** | NOT LAUNCHED — pod truly idle (no W&B runs) | Possible pod restart issue; check pod log 21:18Z |
| **#465** | **norman** | **model-slices {128, 192} (Wave 15)** | NOT LAUNCHED — pod blocked (wave3-pe-bands 13h + r27-log-x 1.8h) | Kill-orphans directive sent 21:18Z |

## Recently Closed (this round)

| PR | Student | Reason |
|----|---------|--------|
| **#239** | **norman** | **NF sweep: NF=64 ep26=8.008% positive slope (+0.0012pp/1k). Dead-end. PE frequency is not the wsy/wsz lever.** |
| **#406** | **edward** | **mlp_ratio=6 diverging ep7=12.405% slope +0.177pp/1k. Instability at 5L/256d scale. Sent back for pivot.** |
| #332 | tanjiro | mirror+SW=2.0 on 4L/256d plateaued at 4L ceiling. Ported to 5L/256d (#443). |
| #325 | senku | gc=0.5 falsified at scale |
| #346 | gilbert | FiLM every-block dilutes shear capacity (wsy/wsz +2pp worse) |
| #438, #439, #441, #444 | Waves 13-14 | ACK timeout; reassigned |
| #432, #433, #434 | Wave 12 | Zero ACK; reassigned in Wave 13 |

## Compliance Watch

- **Wave 15 universal non-compliance (21:20Z)**: All six students failed initial ACK by ~45 min. Five blocked by zombie runs from prior closed PRs and yi-program experiments. One (haku) self-launched off-script work. One (stark) appears truly idle. Reset deadline 21:48Z.
- **Multi-program zombie runs**: emma, gilbert, violet, norman are running yi-program experiments alongside or instead of bengio assignments. The yi advisor's experiments continue to consume bengio pod GPU time. **This is a deployment/orchestration issue requiring human intervention to clarify pod-to-program isolation.** The bengio-pod claude watchdog cannot distinguish a zombie wave3 run from a current Wave 15 run — it just sees `train.py` and waits indefinitely.
- **Haku 3rd off-script offense**: theta-wallshear-A-alpha00 launched in lieu of PR #462. If 4th offense, escalate to human team.
- **Stark idle**: no W&B runs at all — different failure mode (likely pod restart issue). Investigate watchdog log if no ACK by 21:48Z.
- **Frieren**: #361 Trial A compliant; sequential B/C plan documented.
- **Repeat-defiance unauth runs still in environment**: alphonse-string-multiscale, norman-surface-only-mask, thorfinn-asinh-r25, nezuko-warmdown-3ep. Address case-by-case as PRs come in for review.

## Potential Next Research Directions (Wave 16+)

Wave 15 dispatched (#460-#465). Remaining untested levers after Wave 15 results come in:

1. **Equivariant shear heads**: SO(3)/SE(3) equivariant prediction for wsy/wsz — directly addresses 5-7pp binding gap; requires new code
2. **Muon optimizer on bengio**: edward's yi-branch Muon result (24.8% compute advantage) warrants bengio port on 5L/256d+T_max=50
3. **Stacked best-of-Wave-15 recipe**: once Wave 15 winners identified, stack best loss (emma or gilbert) + best density (haku) + winning regularization (violet) onto 5L/256d+T_max=50
4. **6L/512d DDP8 full budget**: thorfinn #382 result pending; if 6L/512d beats 6L/320d (stark #464), escalate to full 50-epoch DDP8 run
5. **Test/val gap investigation**: senku #325 confirmed ~2x vol_p degradation. Dedicated OOD geometry test needed.
6. **Mirror-aug + per-axis-reweight stack**: if gilbert #461 wins, combine with tanjiro #443 mirror-aug on next full run
7. **Data augmentation stack expansion**: rotation (yaw-only for aerodynamics) + Gaussian noise on surface normals
8. **Ensemble of top-3 checkpoints**: average predictions from ep30/ep40/ep50 of best run — free gain
9. **Surface-only fine-tuning**: freeze trunk, fine-tune surface decoder with higher LR after ep30 of full training

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
