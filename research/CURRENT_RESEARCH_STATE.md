# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-05 09:15 UTC (Round 14 mid-cycle — 8 active PRs, 0 idle students)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Fleet:** 0 idle students; all 8 students active
- **Tay-deployed students:** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn

---

## ENSEMBLE SOTA — PR #612 nezuko greedy K=7 (pool 24) — val_abupt **6.1751%** / test_abupt **7.5347%**

W&B run `5veexq8r` (group `nezuko-ensemble-greedy-v3`). Pool 24 members: `d777epep, nh96x7m4, 5o7jc7wi, wyz68o8r, 9mm3sz7x, 49aimdiz, 19qf6di1`.

**Chronic issue:** volume_pressure test/val gap ~3.2× (val=3.61%, test=11.47%) — primary systematic pathology.

**Ensemble gate:** new ensemble must beat val_abupt < **6.1751%**.

## SINGLE-MODEL SOTA — PR #592 alphonse depth-L5 — val_abupt **6.5985%** / test_abupt **7.9915%**

W&B run `4k25s25e` (group `model-depth-sweep`). L=5, hidden=512, Lion lr=9e-5, wd=5e-4, tau_y×1.5/tau_z×2.0, slw=2.0, rff16 multi-sigma, STRING-sep+QK-norm, EMA 0.999, grad-clip 0.5.

**Single-model gate:** val_abupt < **6.5985%**

**Noise floor:** run-to-run variation ~±0.05pp. Genuine win threshold: **< 6.5485%**; borderline: 6.5485–6.5985%.

---

## Latest research direction from human researcher team

- No new issues from human team as of 09:15 UTC 2026-05-05.
- All previously issued directives (Issue #618 STRING-RoPE 4-experiment sweep) are now complete: #621 nezuko CLOSED NEGATIVE (slice-centroid RoPE), #624 alphonse in-flight (point-level pre-slice, Arm C running).

---

## Currently in-flight (8 active WIP PRs, ZERO idle)

| PR | Student | Lever | Status |
|---|---|---|---|
| #666 | thorfinn | Depth L=6 at full hidden=512 (corrected after #660 confound) | EP2=8.47% PASSED; EP3 gate pending (~09:45 UTC) — **strongest SOTA candidate, projected 6.3–6.5%** |
| #624 | alphonse | Point-level pre-slice STRING-RoPE 3-arm sweep (Arm A control, B xmid, C xmid+fxmid) | Arm C EP1=24.82% PASSED; EP2 result expected ~09:30 UTC; timeout ~09:55 UTC |
| #665 | frieren | Cross-slice attention over Transolver slice tokens (global inter-slice MHA) | Arm A (control) EP2=8.70% PASSED; EP3 pending; Arm B cross-slice-attn to follow |
| #667 | fern | Weight decay sweep {1e-4, 5e-4, 1e-3} for Lion optimizer | Arm A (wd=5e-4 control) EP2=8.58% PASSED; EP3 pending; Arms B/C to follow |
| #650 | tanjiro | LR cosine floor sweep — Arm D (lr-min=1e-5) skipped; Arm C (5e-6) finished 6.9535% | Arm B (lr-min=5e-7) running; Arm A (1e-7) queued |
| #649 | edward | GradNorm min-weight floor sweep {0.3, 0.5, 0.7} | Arm A (floor=0.3, no-op) done: 6.9999%; Arms B+C running sequentially |
| #648 | askeladd | Volume-pressure loss upweighting sweep {2.0, 4.0} | Arm A (vp-w=2.0): 7.1107% worse; Arm B (vp-w=4.0) running; strict EP4 gate: val_vp < 3.9456% AND val_abupt < 8.0% |
| #676 | nezuko | Ensemble K=7 distillation teacher → L=5 student (kd-alpha=0.7 and 0.5 sweep) | **Newly assigned** — requires adding --kd-teacher-runs + --kd-alpha flags to train.py |

---

## Active research themes

### 1. Single-model depth scaling (primary near-term opportunity)
- L=5 → L=6 is the most credible near-term single-model SOTA attack (PR #666 thorfinn)
- L=6 is tracking ~0.3pp below L=5 SOTA at every gate checkpoint
- Projected final: 6.3–6.5% range — would be new single-model SOTA
- If L=6 wins, L=7 is the natural follow-up

### 2. Optimizer tuning (hyperparameter sweep, composable)
- Weight decay for Lion (PR #667 fern): baseline wd=5e-4 from L=5 SOTA; testing {1e-4, 1e-3} as adjacent values
- LR cosine floor (PR #650 tanjiro): testing lower lr-min values (5e-7, 1e-7) after 5e-6 overshot

### 3. Architecture: attention / loss / geometry variants
- Cross-slice attention (PR #665 frieren): tests whether global inter-slice MHA improves on local slice-attention  
- GradNorm adaptive weighting (PR #649 edward): floor values that recover sp-priority domain knowledge
- Volume-pressure upweighting (PR #648 askeladd): higher vp-weight to close test/val ratio — Arm A failed; Arm B with 4.0 underway
- Pre-slice STRING-RoPE (PR #624 alphonse): final arm testing xmid+fxmid combined rotation

### 4. Knowledge distillation (new direction)
- PR #676 nezuko: K=7 ensemble teacher → L=5 student via soft-label distillation
- Primary motivation: reduce the volume_pressure test/val gap (~3.2×) by teaching the student the ensemble's implicit noise-smoothing behavior
- Requires new `--kd-teacher-runs` and `--kd-alpha` CLI flags added to train.py
- kd-alpha=0.7 and kd-alpha=0.5 sweep (2 arms)

### 5. Ensemble expansion (pool 25, ongoing)
- Each single-model winner from the current round auto-feeds the greedy selector
- If L=6 (PR #666) wins, run pool 25 greedy expansion immediately
- Volume_pressure test/val ratio is the target diagnostic for pool 25

---

## Negative results catalog (do not retry on current stack)

| Lever | Outcome |
|---|---|
| Local tangent-frame INPUT features | NEGATIVE (#423) |
| Channel-selective Huber on tau | NEGATIVE (#353) |
| Volume-loss-weight scalar rebalancing | NEGATIVE (#451) |
| Separate volume decoder | NEGATIVE val→test overfit (#452) |
| Muon optimizer | NEGATIVE (#299) +4.09pp |
| Sandwich-norm | NEGATIVE diverged |
| U-net skips | NEGATIVE (+0.555pp) |
| 256d / 768d hidden | NEGATIVE |
| Per-axis output head scaling (#467) | NEGATIVE |
| TTA mirror-y inference (#499 old stack) | NEGATIVE +1.18pp |
| Y-mirror training aug (#536) | NEGATIVE — gap is structural |
| 2× surface point density (#506) | NEGATIVE |
| mlp_ratio=6/8 wider FFN (#458) | NEGATIVE |
| Signed-log target transform (#471) | NEGATIVE |
| log1p target transform (#481) | NEGATIVE |
| AdamW vs Lion (#532) | NEGATIVE — Lion optimal |
| Full GradNorm (5× autograd overhead) | NEGATIVE operationally |
| Unit-vector cosine direction loss on tau (#531) | NEGATIVE |
| Coord jitter regularization (#553) | NEGATIVE +38% |
| slw=2.0 13-epoch full (#537) | NEGATIVE — within noise |
| NorMuon optimizer (#619 alphonse, #640 edward) | NEGATIVE on L=5 stack |
| Slice-centroid STRING-RoPE (centroid-based) | NEGATIVE (#621) — QK-norm washes out RoPE; STRING-sep already covers this inductive bias |
| VP loss upweight arm A (vp-weight=2.0) | NEGATIVE (#648) — absolute val_vp worse |

---

## Potential next research directions

1. **L=7 depth** — if L=6 wins, natural follow-up; need to confirm VRAM budget (~64GB at L=7)
2. **Cross-dataset ensemble diversity** — add any winning L=6 checkpoint to pool 25; run greedy re-selection
3. **β-NLL heteroscedastic loss on tay stack** — yi-branch showed wins; direct benefit for vp test/val ratio via explicit variance estimation
4. **Loss-aware query sampling** — boosting-style oversampling in regions of historically high tau_y/tau_z error
5. **Curriculum on RFF sigmas** — start narrow-band, expand during training to complement rff16 multi-sigma win
6. **Point-level RoPE before slice aggregation** (nezuko suggested follow-up from #621) — would compose with STRING-sep at finest granularity
7. **z-only / axis-restricted RoPE** (nezuko suggestion) — tests whether failure is high-dim rotation vs. position-encoding redundancy
8. **Greedy ensemble with TTA members** — if any TTA wins, double pool at zero training cost via mirror-y pairs
9. **Two-stage training: warmup-only on volume, then unfreeze surface heads** — addresses joint-loss imbalance
10. **Vol-point density beyond 65536** — VRAM headroom probe at 98304 or 131072

---

## Procedural notes

- **Never commit research state to a student branch** — only to `tay`.
- **Multi-arm WIP pattern**: students run sweep arms sequentially within a single PR. Use `student_poll_for_work` to identify live assignment.
- **EP2 rate for standard L=5 arms:** ~75–82 min/epoch. Pre-slice RoPE arms: ~99–123 min/epoch.
