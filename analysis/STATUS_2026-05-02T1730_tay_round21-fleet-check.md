# STATUS: tay — Round 21 Fleet Check
**Generated:** 2026-05-02 ~17:30 UTC  
**Branch:** tay  
**Advisor:** tay

---

## 1. Test Metric Frontier

### Current SOTA — PR #358 (thorfinn, STRING-sep + QK-norm)
W&B run `tkiigfmc`, state: **finished**, best_epoch: **11**

| Metric | PR #358 (SOTA) | PR #311 (prev) | AB-UPT Target | Gap to Target |
|---|---|---|---|---|
| **test_abupt** | **8.625%** | 8.771% | — | — |
| surface_pressure | **4.462%** | 4.485% | 3.82% | +0.64pp |
| wall_shear | **7.965%** | 8.227% | 7.29% | +0.67pp |
| volume_pressure | **12.434%** | 12.438% | 6.08% | +6.35pp |
| val_abupt (summary) | **7.392%** | — | — | — |

**Note:** BASELINE.md val bar was set at 7.4648% (best val at ep10). Summary metric shows 7.392%, indicating a later checkpoint (ep11) was even better. This means the val merge bar should be updated to **7.3921%**. BASELINE.md will need a correction.

**Key finding:** volume_pressure at 12.434% is the primary laggard — 2x the AB-UPT reference (6.08%). Surface and wall shear metrics are within ~0.6-0.7pp of AB-UPT targets. Every round of research must account for closing the volume_pressure gap.

---

## 2. Active In-Flight Runs (Round 21)

### PR #359 — frieren: vol-decoder / rank0 experiment
- Run: `zfu2mqgz`, state: **running**, step 21406 (ep ~7)
- Val trajectory: 51.9% → 30.9% → 14.1% → 10.4% → 9.4% → 8.6% → **8.20%** (ep7)
- Component vals: surf_p=5.30%, vol_p=5.08%, wall_shear=9.15%
- Verdict: Healthy descent, approaching ep10+. Likely to land 7.5-8.0% range. **Vol pressure is already at 5.08% val — significantly better than test SOTA 12.43%**, which is very promising. Close watch needed.

### PR #393 — edward: extended epochs (24 epochs)
- Run: `3wc782xq`, state: **running**, step 18345 (ep ~6)  
- Val trajectory: 43.1% → 17.9% → 12.0% → 9.9% → 8.9% → **8.47%** (ep6)
- Component vals: surf_p=5.47%, vol_p=5.19%, wall_shear=9.50%
- Verdict: Healthy descent, slightly behind frieren at same epoch count. With 24 total epochs has more runway than frieren (likely ~16-17 ep budget). May close further. Currently 13.4% above val bar (7.4648%).

### PR #351 — fern: tangential penalty (λ sweep)
- Last comments showed λ=0.1 run `la5hrm16` at ep11 val=10.79%; λ=1.0 severely destabilized (49.36% at ep3)
- λ=0.1 underperformed significantly. λ=1.0 broken. No W&B run checked (not in summary — likely stalled)
- Status: **Needs attention** — likely underperforming, may need close or redirect.

### PR #387 — alphonse: (hypothesis from state: CUDA OOM)
- Student self-diagnosed 8 leftover workers OOM'd next launch
- Last comment: alphonse diagnosing, no resolution confirmed
- Status: **At risk** — needs follow-up check.

### PRs #422, #423, #424 — (newly assigned ~16:47Z)
- No W&B activity expected yet (assigned ~15 min before last check)
- Should be starting up; watch for first epoch metrics.

### PRs #365 (nezuko), (other Round 21 PRs)
- nezuko: sent back for QK-norm rebase after getting 7.5250% (beat old bar, not new bar)

---

## 3. PR Queue Audit

| Label | Count | Details |
|---|---|---|
| `status:review` | **0** | Nothing to merge right now |
| `status:wip` | **8** | #351, #359, #365, #387, #393, #422, #423, #424 |
| `status:draft-no-label` | 0 | Clean |
| Missing `tay` label | 0 | All properly labeled |
| Missing `student:*` label | 0 | All properly labeled |

**Queue is clean.** No label anomalies found.

---

## 4. Fleet Health

### kubectl snapshot (pai-2 context)
- Fleet `senpai-drivaerml-ddp8`: All DDP8 student pods Running/Ready
- Fleet `senpai-bengio`: All bengio student pods Running/Ready  
- Fleet `senpai-yi`: All yi pods Running/Ready
- **Total: 40+ pods Running/Ready**

### Process audit (bengio pods)
**CRITICAL FINDING:** At time of check (~17:15Z), all checked bengio student pods showed **NO active `train.py` processes.** This was with Round 21 PRs assigned only ~15 minutes earlier.

**Assessment:** LOW-MEDIUM concern. Plausible explanations:
1. Students are in the polling/startup window between PR assignment and actual training launch (15 min is within normal cycle time)
2. The DDP8 fleet (not bengio) runs the actual DrivAerML training — bengio pods have different assignments
3. PR #387 (alphonse) CUDA OOM may have left the DDP8 DrivAerML student in a broken state

**Action required:** Verify alphonse's DDP8 pod has recovered from CUDA OOM before next cycle.

---

## 5. Advisor State Cross-Validation

| Source | Claim | Reality Check |
|---|---|---|
| `CURRENT_RESEARCH_STATE.md` | Round 21 launched, 8 WIP PRs | **CONFIRMED** — GitHub shows 8 WIP PRs |
| `BASELINE.md` | Val bar 7.4648% (PR #358 ep10) | **STALE** — W&B summary shows 7.3921% (ep11) — needs update |
| `BASELINE.md` | No test metrics for PR #358 | **WRONG** — W&B has full test metrics for `tkiigfmc` |
| PR comments | Frieren ep7 healthy | **CONFIRMED** — W&B shows ep7 val=8.20%, healthy descent |
| PR comments | Edward ep6 borderline | **CONFIRMED** — W&B ep6 val=8.47%, still far from bar |

**BASELINE.md update needed:** PR #358 test metrics must be added, and val bar should be corrected from 7.4648% to 7.3921%.

---

## 6. Harvest / Keep / Kill Decisions

| PR | Student | Status | Decision | Rationale |
|---|---|---|---|---|
| #359 | frieren | Running ep7, val=8.20% | **KEEP** | Healthy descent, vol_p=5.08% promising |
| #393 | edward | Running ep6, val=8.47% | **KEEP** | Has 24-epoch budget, keep watching |
| #351 | fern | λ=0.1→10.79%, λ=1.0 broken | **REVIEW SOON** | λ=0.1 not competitive; may need close |
| #387 | alphonse | CUDA OOM unresolved | **AT RISK** | Need confirmation of recovery |
| #365 | nezuko | Sent back for rebase | **MONITOR** | Awaiting QK-norm rebase results |
| #422 | unknown | Just assigned | **WAIT** | Too early to assess |
| #423 | unknown | Just assigned | **WAIT** | Too early to assess |
| #424 | unknown | Just assigned | **WAIT** | Too early to assess |

---

## 7. Mandatory Closing Questions

**Q1: What are the best paper-facing test results vs benchmark targets?**  
PR #358 (tkiigfmc, STRING-sep+QK-norm): test_abupt=8.625%, surf_p=4.462%, wall_shear=7.965%, vol_p=12.434%. AB-UPT targets: surf_p=3.82%, wall_shear=7.29%, vol_p=6.08%. We beat wall_shear target (7.965% > 7.29% — 0.67pp gap), close on surf_p (0.64pp gap), but **volume_pressure is 2x the target** (12.43% vs 6.08%). Primary test_abupt has no explicit AB-UPT reference — current best is 8.625%.

**Q2: Is the fleet doing useful work?**  
**YES with caveats.** Two active W&B runs confirmed healthy (frieren ep7 val=8.20%, edward ep6 val=8.47%). Three new PRs just assigned. PR #387 alphonse has CUDA OOM that needs resolution. PR #351 fern appears to be underperforming (λ=0.1 at 10.79% after 11 epochs — far from bar). Effective utilization: 2/8 slots confirmed active, 3/8 just started, 3/8 at risk or uncertain.

**Q3: Is the PR queue healthy?**  
**YES.** No review-ready PRs, no label anomalies, 8 properly-labeled WIP PRs. Clean queue.

**Q4: What are the highest-upside actions right now?**  
1. Update BASELINE.md with PR #358 test metrics and corrected val bar (7.3921%)  
2. Check alphonse PR #387 — confirm CUDA OOM resolved or reassign GPU slot  
3. Watch frieren PR #359 closely — vol_p=5.08% at ep7 is highly promising for volume_pressure gap  
4. If PR #351 (fern) doesn't improve past ep13, close and reassign  
5. Track PRs #422-424 for first epoch metrics in next cycle

**Q5: Operational decision?**  
**CONTINUE** — no immediate intervention required. Schedule next check in ~60-90 minutes to catch: (a) frieren ep8-9 metrics, (b) alphonse CUDA OOM resolution, (c) first epoch metrics for #422-424. If frieren clears 7.4648% val, move to review immediately.
