# Research Ideas — 2026-05-04 10:22 UTC

Wave: DrivAerML long-run single-model DDP8 validation (advisor branch `drivaerml-long-20260504`).

This document captures the four hypotheses dispatched in round 1 and the next-tier candidates queued for round 2+. The selection is driven by the human launch directive, the May 4 research map, and the constraint that this branch has no merged experiments yet (so we test mechanism families individually before composing).

## Round-1 hypotheses (one per idle student, dispatched 2026-05-04)

### H1 (dl24-fern) — Mild static channel-aware tau weighting + lr 9e-5

- **Reference:** `9mm3sz7x` (test 8.1229 single-model SOTA; tau_y 8.326, tau_z 9.543).
- **Mechanism:** apply per-axis loss multipliers w_p=1.0, w_tau_x=1.0, w_tau_y=1.2, w_tau_z=1.3 on the 4-channel surface output, train with lr=9e-5 (rest of stack matches `main`).
- **Why it's high value:** strongest single-model validation in the May 4 map; the per-module gradient sample showed surface_out and volume_out remaining bounded (medians ~0.18 and ~0.20), no nonfinite gradients. The 4.7 h reference run was still descending at cutoff — long DDP8 should let the lever fully resolve.
- **Risk:** small-margin baseline-beating depends on lr and channel weights interacting cleanly; the volume guard must verify test volume does not regress vs. unweighted control.

### H2 (dl24-frieren) — Multi-sigma STRING log-frequency init

- **Reference:** `ki2q9ko9` (test 8.4791; volume-pressure best 11.503; aggregate beat `m9775k1v` by 0.0008).
- **Mechanism:** replace `ContinuousSincosEmbed` with multi-sigma STRING positional encoder using log-frequency initialization across multiple bandwidths. Source code from `alphonse/multi-sigma-string-init` or the corresponding state on `yi`/`tay`/`bengio` after explicit fetch.
- **Why it's high value:** unique mechanism — moves the volume-pressure component decisively (11.503 vs 12.051 baseline frontier). The map's matched-control conclusion is that representation wins are the cleanest broad lever.
- **Risk:** PR #488 had documented provenance drift — the merged code did not match the run config until #511/#516. The student must verify the code path matches `ki2q9ko9` before launching the long run.

### H3 (dl24-nezuko) — Volume-point curriculum 16k → 32k → 49k → 65k

- **Reference:** `r5rw40rn` (censored at 4.7 h; best validation 7.179, surface 4.363, tau_x 6.897, tau_z 10.077; map says it timed out before reaching the 49k/65k stages).
- **Mechanism:** schedule `--train-volume-points` to ramp through 16k → 32k → 49k → 65k across the run, with stage boundaries set so the largest stage runs for at least 30 % of total steps. Baseline trunk and Fourier PE; no other levers.
- **Why it's high value:** the reference run already produced new best validation, surface, tau_x, and tau_z. With 24h DDP8 we can finish the curriculum and see whether the volume-pressure test gap closes (val volume 4.x vs test volume 12.2 in the reference).
- **Risk:** the curriculum schedule can be sensitive to LR and step budget; smoke must confirm the scheduler advances stages and W&B logs the active point count.

### H4 (dl24-tanjiro) — EMA-proxy GradNorm α=0.5 with volume guards

- **Reference:** `wyz68o8r` (test 8.2355; tau_y 8.466, volume 12.213, no nonfinite gradients).
- **Mechanism:** dynamic per-channel loss weighting based on EMA of per-channel gradient norms with α=0.5; explicit volume-loss guard so test volume regression > X % triggers a fail/early-stop check. Implemented as a callback that updates loss weights between optimizer steps; do not use exact multi-backward GradNorm (map says it is too expensive in this loop).
- **Why it's high value:** complementary to H1 — H1 tests static weighting; H4 tests bounded dynamic weighting. The map shows alpha 0.75-1.0 (`341czkol`) achieves the best single-model tau_y but regresses volume; α=0.5 is the safer point. Long DDP8 lets the EMA stabilize.
- **Risk:** dynamic schemes can drift over long horizons; the volume guard must be explicit, and the smoke must confirm channel weights are logged each step.

## Why these four and not others

- **Coverage of mechanism families:** static weighting (H1), representation init (H2), data curriculum (H3), dynamic weighting (H4). No two arms share the dominant mechanism, so the round-1 results are interpretable individually.
- **Single-model only:** the directive explicitly excludes ensembles and model soups in this wave. None of the four uses output averaging.
- **Test-metric reporting:** all four are designed to produce clean terminal test metrics under 24h DDP8.
- **Provenance discipline:** H2 carries the strongest provenance risk (STRING code drift) and is therefore assigned a single student rather than as a composition layer; H1, H3, H4 only require small additions to the `main` baseline trunk.
- **Avoided composition this round:** the directive permits composition tests but only "when they remain single-model and controlled." None of the round-1 arms is yet a confirmed mechanism on this branch, so composing two unconfirmed levers would dilute the signal.

## Round-2 candidate queue (dispatch after at least two round-1 terminal results)

These are queued in rough priority order; final ordering will depend on which round-1 mechanism produces the largest delta and which physical bucket (surface, volume, tau_y, tau_z) gains.

1. **Compose round-1 winner + second-best lever.** Examples: mild tau weighting (H1) + multi-sigma STRING (H2); mild tau weighting + volume curriculum (H3); STRING + curriculum.
2. **STRING follow-ups (single arms):** STRING + QK norm (`tkiigfmc`), STRING + volume MLP (`8x7c537j`), 5L STRING (`70lnb3dt`), original STRING/GRAPE (`gcwx9yaa`).
3. **Stronger static tau weights bounded:** `nh96x7m4` (y=1.5/z=2.0) under long DDP8.
4. **Surface-loss weight 2.0** (`qqtdnlwq`) at long DDP8 — clean single-arm replication of map evidence that surface weighting is not dead on the modern stack.
5. **Extended cosine T_max=13 with controlled validation budget** (`5o7jc7wi`) — the censored reference run was still improving at cutoff; long DDP8 plus a tuned validation budget should let the late-tau gains land.
6. **Per-axis output scaling** (`wgvvevb9`) — channel-calibration lever orthogonal to most others.
7. **GradNorm α=1.0 with mandatory volume guard** (`341czkol`) — only after H4 (α=0.5) result is in.
8. **Soft cross-flow geometry features** on top of the round-1 winner: cross-flow exposure index, k-NN PCA curvature, local tangent basis as input features (not output projection). Map evidence says these are under-harvested and target the tau_y/tau_z bottleneck physically.
9. **Channel-specific surface_out gradient instrumentation** so future reweighting waves can read allocation, not endpoint metrics. This is a measurement PR, not a metric-target PR — assign only if a slot would otherwise be idle.

## Notes on metric reporting

- The wave's primary lookup is `test_primary/abupt_axis_mean_rel_l2_pct`. Every PR must additionally report `test_primary/surface_pressure_rel_l2_pct`, `test_primary/wall_shear_rel_l2_pct`, `test_primary/volume_pressure_rel_l2_pct`, `test_primary/wall_shear_x_rel_l2_pct`, `test_primary/wall_shear_y_rel_l2_pct`, `test_primary/wall_shear_z_rel_l2_pct`.
- The terminal `SENPAI-RESULT` JSON marker must include `pending_arms=false` and `terminal=true` once the long run finishes.
- Validation may be cited only as context, not as the headline.
