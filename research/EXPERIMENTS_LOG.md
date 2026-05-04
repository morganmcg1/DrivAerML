# SENPAI Research Results — `drivaerml-long-20260504`

Single-model long DDP8 validation wave; started 2026-05-04.

This log is appended in reverse-chronological order as PRs are reviewed. Each entry should include: PR number/title, student branch, hypothesis, results table (with W&B run IDs and test metrics), and brief commentary.

The wave's evidence contract: test metrics from `test_primary/*` only; validation is for steering and checkpoint selection.

## 2026-05-04 22:30 — PR #643: Bug-fix: flip train.py defaults (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/train-defaults-fix`
- **Type:** Code fix (not an experiment — no SENPAI-RESULT marker)
- **Fix:** Three `Config` defaults in `train.py` were silently diverging from every healthy long DDP8 reference run on this branch:

| Field | Old default | New default | Evidence |
|---|---|---|---|
| `train_surface_points` | 40,000 | 65,536 | All 4 reference runs (`nh96x7m4`, `9mm3sz7x`, `341czkol`, `ug6c3nks`) |
| `train_volume_points` | 40,000 | 16,384 | Same 4 reference runs |
| `compile_model` | True | False | Same 4 reference runs; True triggered `torch._inductor.exc.InductorError` |

- **Failure modes caught:** (1) Run `syl1zx3r` (40k/40k defaults) inverted the volume:surface gradient ratio under a surface-loss hypothesis; (2) run `xw6sp0rt` (compile_model=True with corrected sampling) hit `torch._inductor` tiling assertion at end-of-EP1.
- **Risk:** Low — all existing long DDP8 commands already explicitly override these defaults. The fix only changes behavior for new commands that omit these flags.
- **Merged to advisor branch 2026-05-04 via direct squash-merge (code fix, no experiment SENPAI-RESULT).**

## (Pending round-1 results)

Round-1 long DDP8 assignments are actively running:
- PR #599 (dl24-frieren) — multi-sigma STRING log-freq init, run `sogus8sx`, EP~15.7/50 as of 2026-05-05 00:30 UTC. Wave SOTA: val_abupt=**6.5281%** at step 153,831.
- PR #608 (dl24-nezuko) — volume-loss ×2.0, run `y301z78k`, EP~20.4/50 as of 2026-05-05 00:30 UTC. val=14.46%, vol=6.99%.
- PR #611 (dl24-fern) — mild tau_y=1.2/tau_z=1.3 AdamW, run `ug6c3nks`, EP~15.7/50 as of 2026-05-05 00:30 UTC. val=12.370%, vol=6.42%.
- PR #623 (dl24-tanjiro) — strong tau_y=1.5/tau_z=2.0, smoke run `b7pbsdx7` at EP1 PASS (step 10,865, val=25.996%). Awaiting EP2/EP3 to authorize full long run.

Terminal results will be appended here as students post SENPAI-RESULT markers.
