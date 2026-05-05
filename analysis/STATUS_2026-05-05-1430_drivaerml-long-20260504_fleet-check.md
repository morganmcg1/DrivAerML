# Fleet Status Check — drivaerml-long-20260504
**Timestamp:** 2026-05-05 14:30 UTC  
**Branch:** `drivaerml-long-20260504`  
**Wave SOTA:** PR #599 `sogus8sx`, `test_primary/abupt_axis_mean_rel_l2_pct = 7.9303%`  
**AB-UPT public targets (lower is better):** p_s=3.82%, tau=7.29%, p_v=6.08%, tau_x=5.35%, tau_y=3.65%, tau_z=3.63%

---

## 1. Fleet Health

All 4 student pods confirmed running as of status check. GPU utilization 93–100% on 2 GPUs per pod.

| Pod / Student | Status | GPU Util | Notes |
|---|---|---|---|
| `dl24-fern` | Running | ~94% | Active heartbeat, EP22+ |
| `dl24-frieren` | Running | ~93% | Active heartbeat, EP14+ |
| `dl24-nezuko` | Running | ~100% | Active heartbeat, EP6+ |
| `dl24-tanjiro` | Running | ~100% | **Anomaly — see Section 3** |

---

## 2. Metric Frontier vs. AB-UPT

All metrics are **validation** (`val_primary/*`) — no active run has posted terminal test metrics yet.  
Reference point: wave test SOTA = 7.9303% (PR #599), AB-UPT aggregate target ≈ 5.0–5.5%.

### Active Run Summary (latest W&B epoch)

| Run ID | Student | PR | State | Val Agg% | p_s% | p_v% | tau% | tau_x% | tau_y% | tau_z% |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `a8emaoxm` | fern | #664 | running | **6.820** | 4.475 | 4.198 | 7.637 | 6.598 | 8.376 | 10.452 |
| `er8wmo8d` | frieren | #669 | running | 6.974 | 4.552 | 4.376 | 7.822 | 6.823 | 8.590 | 10.531 |
| `sbzspuf2` | nezuko | #678 | running | 7.857 | 5.038 | 5.416 | 8.688 | 7.566 | 9.722 | 11.544 |
| `7wdwphhn` | tanjiro | #696 | **finished** | 8.782 | 5.769 | 5.485 | 9.798 | 8.426 | 11.277 | 12.954 |

### Frontier Observations

- **Fern (`a8emaoxm`, per-axis output scaling)** holds the wave-best val at 6.82%, significantly ahead of the wave test SOTA of 7.93%. However, the val slope is slightly positive (+0.009 pct/1k steps as of EP22) — plateau risk is non-zero but not confirmed.
- **Frieren (`er8wmo8d`, tau_y=1.2/tau_z=1.3 channel weighting)** is at 6.97%, but its slope is rising (+0.020 pct/1k steps as of EP14). This is a concerning trend — the model may be starting to overfit or the loss is oscillating.
- **Nezuko (`sbzspuf2`, T_max=60 cosine)** is early (EP6) at 7.86%. All component slopes positive — too early to call, but it is behind pace compared to fern and frieren at the same epoch count.
- **Gap to AB-UPT:** Best val (6.82%) is still ~1.8 percentage points above the AB-UPT aggregate target. The individual component worst offenders are tau_y (8.4%) and tau_z (10.5%) vs. targets 3.65% and 3.63% — these are nearly 3x the target, indicating major headroom.

---

## 3. Anomalies and Flags

### CRITICAL: Tanjiro Gate Violation — Premature Long-Run Launch

**Gate criterion:** EP5 val aggregate ≤ 8.0% (set by advisor on PR #696)  
**Smoke run result:** `7wdwphhn`, state=`finished`, best epoch=3, val=8.782% — **FAILS gate (8.782% > 8.0%)**

Despite this gate failure, the student (`dl24-tanjiro`) has already launched a full 50-epoch DDP8 long run:
- Group name: `string-qknorm-long-50ep`
- Run IDs: `dzochl0q`, `b6euenst`, `uoi5f2b3`, `t6iag8d1`, `bjim3ny9`, `zphk66bc`, `y36t0zl4`, `glx3d2zd` (all 8 ranks, state=`running`, no metrics yet)

**Assessment:** QK-Norm on the STRING PE variant is showing significantly worse performance than baseline at EP3 (8.782% vs. baseline smoke runs typically EP3 ≈ 8.0–8.5% for viable experiments). The gate exists precisely to avoid burning 50-epoch compute on ideas that fail early screening. The long run should be terminated and the PR sent back.

**Recommendation:** Post a comment on PR #696 flagging the gate violation, ask tanjiro to terminate the long run, and reassign them to a fresh hypothesis.

---

## 4. Recommendations

### Immediate Actions

1. **Flag PR #696 (tanjiro)** — comment on gate violation, request termination of the `string-qknorm-long-50ep` runs, reassign to fresh experiment.

2. **Monitor fern (#664)** — if val slope remains positive through EP25, consider whether the model has found its floor. The per-axis output scaling is the most promising active experiment. If it plateaus below 7.93% val, it should still produce a test improvement; push toward terminal result.

3. **Monitor frieren (#669)** — rising slope at EP14 is a yellow flag. If it does not reverse by EP20, the tau channel weighting may need a different coefficient. Current best was EP10-ish.

4. **Watch nezuko (#678) EP10 gate** — T_max=60 cosine is an interesting LR schedule variant. The EP10 gate was set at ≤7.5%; at EP6=7.86% it needs to descend ~0.36 pct in 4 epochs. Feasible but tight.

### Medium-Term

- The largest remaining gap to AB-UPT is in **tau_y and tau_z** (component-wise). Experiments explicitly targeting these via loss reweighting (e.g., boosting tau_y/tau_z weights above 2.0) should be a priority once current runs complete.
- **Volume pressure** (p_v: best val=4.20% vs. AB-UPT 6.08%) is already matching or beating the AB-UPT reference — surface pressure and wall shear remain the main frontiers.
- **Surface pressure** (p_s: best val=4.47% vs. AB-UPT 3.82%) — still ~17% above target. Surface-specific architectural changes or loss emphasis should be explored.
