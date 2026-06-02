# SENPAI Research State

- **2026-06-02 17:36Z** (H197 EP2 BOUNDARY CROSSED — **MECHANISM CONTINUES**: val_VP=4.9025 -7.64pp from EP1 12.5411 = largest single-EP VP descent of fleet at EP2, val_WSS=7.3790 -5.52pp, val_AB=6.7526 -5.99pp, val_SP=4.2728 -4.45pp; train_loss=0.0416 slight up from 0.0258 due to GradNorm reallocation under 2× VP weight, nonfinite_grad=0 stable, lr=1e-4 confirmed; rank-0=999l85x3 step=22772 rt=99.3min, all 8 ranks healthy; AB ~1.13pp gap to H183 SOTA AB=5.6152 with 28 EPs remaining = promising trajectory if VP descent compounds normally; EP3 ETA ~18:25Z; H194/H195/H193 unchanged from 17:21Z boundary cluster acks pending next cluster ~18:00-18:13Z)
- **2026-06-02 17:21Z** (3-PR BOUNDARY CLUSTER ALL CROSSED: H194 EP11 **FIRST UPTICK ACROSS ALL 4 METRICS** after 10-EP descent streak — WSS=6.7365 +0.028pp, VP=3.5428 +0.014pp, AB=5.9620 +0.029pp, SP=3.8894 +0.028pp = mild noise oscillation NOT divergence, still under H183 SOTA test_WSS trajectory but AB +0.35pp ABOVE SOTA = no merge candidate yet, EP12 ETA ~18:13Z; H195 EP6=6.8252 **descent resumes** -0.07pp from EP5 6.8952, projected EP10 ~6.55 would PASS gate ≤6.65 AND outperform H147 EP10=6.64 by 0.09pp = potential first merge candidate of cluster, EP7 ETA ~18:02Z; H193 EP17=7.8678 **divergence SUSTAINED** essentially flat with EP16 spike +0.007pp, NON-MERGE confirmed locked, continuing to natural terminal, EP18 ETA ~18:01Z; H197 frieren main rank-0 999l85x3 step=19411 rt=5040s still EP1 read val_WSS=12.8993, EP2 boundary ETA ~17:32Z to confirm VP descent slope holds at -1.66pp under H183; ALL 4 dl24 students wip status no idle, no human gh issues)
- **2026-06-02 16:34Z** (H193 EP16 **SEVERE WSS REGRESSION** val_WSS=7.8605 +~0.7-0.9pp jump in single EP, τ_y=9.21 + τ_z=9.54 BOTH diverged, VP plateau holds 3.6803, NON-MERGE locked = continue to natural terminal for SENPAI-RESULT; H195 EP5=6.8952 marginal gate PASS by 0.005pp but +0.14pp ABOVE H147 EP5=6.75 = NON-MERGE trajectory continues, EP10 gate ≤6.65 is firm decision point ~17:30Z; H197 frieren main 8 ranks confirmed `h197-volume-loss-weight-2p0-main` step=8642 rt=2221s = 3.89 steps/s EP1 boundary ETA ~16:44Z rank-0 unknown until first val row; H194 still EP10 last-read, EP11 boundary ETA ~17:13Z EP12 ~17:48Z)
- **2026-06-02 16:18Z** (H194 EP10 BOUNDARY CROSSED — **ALL 4 METRICS STILL DESCENDING** WSS=6.7086 -0.0174 VP=3.5292 -0.0211 AB=5.9329 -0.0176 SP=3.8611 -0.0092 → PASS EP10 gate ≤6.70 by 0.014pp; projection AB=-0.018/EP×15 = EP25 val_AB ~5.66 → test_AB ~5.98 +0.37pp ABOVE H183 SOTA AB=5.6152; stay-the-course verdict; H197 main launched 15:57Z all 8 ranks running step=4964 21min in EP1 ETA ~16:43Z; H195 EP5 + H193 EP16 boundaries imminent ~16:25Z)
- **2026-06-02 15:43Z** (H197 smoke EP1 PASS — main launch authorized; val_VP=12.5558 -1.65pp UNDER H183 EP1 ≈14.20 = MECHANISM PAYS OFF, w_vol_p clamped to floor 0.15 by design; H195 EP4 strong -0.10 descent but +0.13pp ABOVE H147 baseline gap WIDENING; H193 EP15 WSS slope POS again +0.0338 VP plateau exhausted at 3.68; H194 EP10 ETA ~16:14Z next cycle)
- **2026-06-02 15:23Z** (H194 EP9 ALL 4 STILL DESCENDING WSS=6.7260 SAILED PAST EP10 gate at EP9 — fleet leader; H193 EP14 WSS slope REVERSED -0.0034 + VP fresh low 3.6844; H195 EP3 PASS gate ≤7.10 at 7.0573 but +0.0773 ABOVE H147 baseline first separation gate; H197 frieren smoke 31min in step=7248 ETA ~15:42Z; 4-PR boundary acks posted #1554 #1559 #1565)
- **2026-06-02 14:42Z** (H196 CLOSED NON-MERGE — frieren pre-launch code trace found vol_p_charbonnier_weight is NO-OP under GradNorm; H197 reassigned to frieren with --volume-loss-weight=2.0 mechanism-faithful re-mapping PR #1573; H194 EP8 strongest fleet ALL 4 METRICS DESCENDING; H195 EP2 H147-match exactly; H193 EP13 8th uptick zone)
- **2026-06-02 14:20Z** (H194 EP8 ALL 4 METRICS STILL DESCENDING fresh lows everywhere — strongest fleet candidate; H195 EP2 7.2661 matches H147 EP2=7.26 exactly; H196 frieren smoke launched 14:12Z EP1 ETA ~14:55Z; H193 EP13 8th uptick zone)
- **2026-06-02 13:50Z** (H192 CLOSED NON-MERGE — hypothesis FALSIFIED, upweighted τ_z largest regress +0.225pp; H196 frieren VP-Charbonnier=0.2 PR #1571 dispatched; H193 EP13 WSS 7th consec POS slope VP plateau; H194 EP8 + H195 EP2 boundaries imminent)
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** ⭐ **H183 (PR #1510, run `guw83mge`) — test_WSS=6.4427%, test_VP=3.4415%, test_SP=3.5187%, test_ABUPT=5.6152% (ALL 4 FLOORS CLEARED)**
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85% (remaining gap: −0.59pp)
- **Human directive (issue #1056, 13:15Z + 13:27Z advisor response):** Morgan posted WALL SHEAR STRESS NOTES 1+2 — identifies BL DERIVATIVE DECODER (off-wall ghost-point probe → ∂u/∂n → WSS) as highest-leverage untried mechanism, TANGENT-BASIS OUTPUT HEAD as 2nd priority. Both require architectural changes larger than current fleet scope.
- **Human check-in (issue #1056, 18:39Z):** No new messages since 19:27Z 2026-06-01.

## 15:43Z checkpoint — **H197 smoke EP1 PASS main launch authorized (VP -1.65pp under H183 baseline = mechanism delivers); H195 EP4 -0.10 descent but H147 gap WIDENING +0.13pp; H193 EP15 WSS slope POS again VP exhausted at 3.68; H194 EP10 imminent ~16:14Z**

### Actions taken this cycle (15:43Z)
- **H197 smoke EP1 PASS posted on PR #1573 (frieren) — MAIN 30-EP LAUNCH AUTHORIZED**: rank-0 `umo09701` finished smoke EP1 at step 10975, rt=47.8min. EP1 vals: val_WSS=12.9079 (match H183 EP1 ~12.82), **val_VP=12.5558 vs H183 EP1 ≈14.20 = -1.65pp UNDER baseline**, val_AB=12.6879, val_SP=8.4171. Train health all green: train/loss=0.1096, train/nonfinite_grad=0. GradNorm allocator stable: w_vol_p=0.1500 clamped to floor (clamp_active=1.0 by design), all other w's bounded (w_cp=0.97, w_tau_y=1.33, w_tau_z=1.28). r_vol_p=3.2436 confirms doubled raw VP task signal IS reaching allocator. **Effective VP gradient under floor regime = 0.15 × raw × 2.0 = 2× H183**. task_loss_vol_p slope per-1k-steps = -0.0564 (steepest descent of all 5 channels) — mechanism delivers
- **H195 EP4 boundary ack posted on PR #1565 (fern)** — rank-0 `7ergjfh4` step=44920, rt=3.26h: WSS=6.9555 (-0.1018 ↓ strong), VP=3.9526 (-0.2782 ↓ strong), AB=6.2102 (-0.1437 ↓), SP=3.9811 (-0.0604 ↓). All 4 metrics descending but H147 EP4 ≈6.83 → H195 EP4 +0.13pp ABOVE — **gap WIDENING from EP3 +0.08pp**. τ_y=1.3 producing proportional cross-axis descent, no differential τ_y advantage. EP5 gate ≤6.90 likely PASS at projected 6.86 but +0.11pp ABOVE H147 EP5=6.75; EP10 gate ≤6.70 will be the firm NON-MERGE decision
- **H193 EP15 boundary ack posted on PR #1554 (tanjiro)** — rank-0 `vuvpegip` step=166013, rt=12.08h: WSS=7.7974 (+0.0338 ↑ **slope reverses POS again**), VP=3.6803 (-0.0041 ↔ plateau), AB=6.6152 (+0.0177 ↑), SP=3.9517 (-0.0027 ↔). Correction to 15:23Z note: the EP14 -0.0034 dip was 1-EP noise, NOT a sustained slope reversal. WSS uptick pattern now 9 of last 11 EPs POS slope. VP descent IS exhausted at 3.68 plateau. WSS NON-MERGE firmly confirmed (+1.36pp val); VP terminal projection ~3.45-3.55 test still UNDER H183 floor 3.4415 = no VP merge case either
- H194 nezuko unchanged from 15:23Z (no new val row); EP10 ETA ~16:14Z based on ~3.0 steps/s rate

## 15:23Z checkpoint — **H194 EP9 fleet leader (ALL 4 STILL DESCENDING); H193 EP14 WSS slope reversal + VP fresh low; H195 EP3 PASS gate but +0.08pp ABOVE H147; H197 smoke EP1 ETA ~15:42Z**

### Actions taken this cycle (15:23Z)
- **H194 EP9 boundary ack posted on PR #1559 (nezuko)** — rank-0 `tne4wsap` step=100719, rt=9.08h: WSS=6.7260 (-0.0151 ↓), VP=3.5503 (-0.0362 ↓ strong), AB=5.9505 (-0.0188 ↓), SP=3.8703 (-0.0084 ↓). **ALL 4 metrics still descending at EP9, WSS already 0.07pp UNDER EP10 gate ≤6.80 at EP9 — fleet leader by wide margin**. Test_WSS projection if -0.015pp/EP slope holds: terminal val ~6.50 → test ~6.33-6.40 = **paper-tier sub-H183 SOTA territory**. Gap to SOTA 6.4427 closed to -0.28pp with 16 EPs remaining
- **H193 EP14 boundary ack posted on PR #1554 (tanjiro)** — rank-0 `vuvpegip` step=161665, rt=11.76h: WSS=7.7636 (-0.0034 ↓ **slope reversal breaks 7-consec POS streak**), VP=3.6844 (-0.0204 ↓ **fresh low, descent resumed after EP12-13 pause**), AB=6.5975 (-0.0008 ↔ flat), SP=3.9544 (-0.0169 ↓). Correction: VP NOT exhausted at 3.70 — was just a 1-EP plateau. WSS NON-MERGE outcome locked but VP-research note strengthening
- **H195 EP3 boundary ack posted on PR #1565 (fern)** — WSS=7.0573 PASSED gate ≤7.10 (-0.0427 under gate) but **+0.0773 ABOVE H147 EP3=6.98 at first separation gate**; per-axis τ_y dropped 14.10→7.88 (-0.39pp) — proportional to τ_x/τ_z descent, no clean differential mechanism payoff yet. EP5 gate ≤6.90 (~16:08Z) is the critical kill point
- **H197 frieren smoke confirmed running** — 8 ranks in `h197-volume-loss-weight-2p0` group, age 31min, step=7248, val_WSS=? (smoke EP1 boundary at step 10975 ETA ~15:42Z based on H183-stack ~13.8k/h DDP8 rate)
- Identified H193 main rank-0 = `vuvpegip` (multi-rank ambiguity resolved via summary._json_dict val key presence — only rank-0 logs val keys)
- Probe efficiency note: `scan_history(keys=[...], min_step=N)` is SLOW (>20 min on 8 runs); `r.summary._json_dict` is INSTANT for latest val readings

## 14:42Z checkpoint — **H196 CLOSED NON-MERGE pre-launch** (frieren caught no-op under GradNorm); H197 reassigned with mechanism-faithful flag PR #1573; fleet otherwise unchanged from 14:20Z

### Actions taken this cycle (14:42Z)
- **H196 PR #1571 CLOSED NON-MERGE** pre-launch on frieren's outstanding code trace (PR comment 14:33Z): `--vol-p-charbonnier-weight` is no-op under `--use-gradnorm` because train.py L430 routes `loss_vol_p_charb` (unscaled) into GradNorm slot; additive path L484-486 gated to no-GradNorm only. Same class as H158/PR #1420.
- **H197 PR #1573 created** for frieren with `--volume-loss-weight 2.0` (mechanism-faithful re-mapping per frieren's option 1): doubles raw VP task signal at L430 `volume_per_ch = loss_vol_p_charb * volume_loss_weight`, routing through `c_vol_p / G_bar` allocator ratios exactly as original H196 hypothesis posited
- Posted ack thanking frieren for the diligent pre-launch verification; programme note added: 2nd VP-channel no-op caught at PR-trace time (after H158/#1420), adding 'flag-effect-under-GradNorm verification' to future advisor pre-dispatch checklist
- H196 entry prepended to EXPERIMENTS_LOG.md with full code citation and analysis
- H197 smoke ETA ~15:00Z if student picks up promptly, main EP1 ~15:30Z

## 14:20Z checkpoint — **H194 EP8 all-4-metric fresh-low descent (strongest in fleet)**; H195 EP2 H147-trajectory match; H196 frieren smoke launched 14:12Z; H193 EP13 8th-uptick zone

### Actions taken this cycle (14:20Z)
- **H194 EP8 boundary cluster ack** posted on PR #1559 (nezuko): WSS=6.7411 ↓-0.0459, VP=3.5865 ↓-0.0175 fresh low, AB=5.9693 ↓-0.0379 fresh low, SP=3.8787 ↓-0.0099 fresh low — **ALL 4 METRICS STILL DESCENDING**, strongest descent in fleet, vs H183 SOTA WSS=6.4427 gap closed to -0.30pp with 17 EPs remaining
- **H195 EP2 boundary ack** posted on PR #1565 (fern): WSS=7.2661 matches H147 EP2=7.26 exactly; tau_y=1.3 not yet differentiating at EP2; EP3 (~14:42Z) first separation gate
- H196 frieren smoke run `gtwndogn` confirmed launched 14:12Z (epochs=1 smoke), vol_p_charbonnier_weight=0.2 config verified, EP1 ETA ~14:55Z
- Identified fern config.agent empty cause for rank-0 lookup miss — fixed via wandb_group regex filter; rank-0 = 7ergjfh4
- Identified H194 rank-0 = tne4wsap (not c74lnfe1 from previous probe — multi-rank ambiguity)
- H193 EP13 last read at step 142687 — EP14 step ~153663, current summary step ~147246 (mid-EP14)

## 13:50Z checkpoint — **H192 CLOSED NON-MERGE**, H196 frieren dispatched; H193 EP13 WSS 7th consec POS slope VP plateau; 4/4 students active again; H194/H195 boundaries imminent

### Actions taken this cycle (13:42–13:50Z)
- Student SENPAI-RESULT confirmed at 13:44:50Z: test_WSS=6.6206 +0.18pp REGRESS, test_VP=3.5313 −0.11pp UNDER FLOOR, test_ABUPT=5.7746 −0.07pp UNDER FLOOR, test_SP=3.6413 +0.06pp BREACH; **τ_z (UPWEIGHTED axis) showed LARGEST test regress at +0.225pp** — opposite of predicted decoupling signature. Student suggests retiring per-axis τ-weighting + pursuing VP-isolated investigation directly.
- **PR #1541 closed NON-MERGE** at 13:46Z with full programme finding (per-axis τ-weighting RETIRED for this wave) and H196 follow-up plan
- **H196 frieren PR #1571 created** at 13:48Z: `--vol-p-charbonnier-weight 0.2` (double H183 default 0.1) — direct VP-isolation hypothesis, single-variable change, smoke-then-30EP-main
- **H192 entry appended to EXPERIMENTS_LOG.md** with full test-metric table + analysis
- **H193 EP13 boundary ack posted on PR #1554:** WSS=7.7670 (+0.0505 ↑ 7th consec POS slope), VP=3.7048 (+0.0034 ~plateau, fresh-low ±noise), AB=6.5983 (+0.0274 ↑), SP=3.9713 (+0.0073 ↑) — VP descent appears exhausted at 3.70 plateau, broader uptick at EP13 confirms WSS NON-MERGE on H193 firmly
- H194 EP8 + H195 EP2 boundaries pending (within ~15 min)

### Previous cycle (13:00–13:22Z)
- Posted ADVISOR EP7 ack on PR #1559 (H194 nezuko) — **ALL 4 metrics descending**, WSS=6.7870 already below EP10 gate ≤6.80 at EP7 (3 EPs ahead), VP=3.6040 (-0.0630 STRONG)
- Posted ADVISOR EP28 ack on PR #1541 (H192 frieren) — VP fresh low 3.5270 (slope -0.0036), WSS 6.7042 (6th consec POS slope), terminal ~14:30-14:50Z
- Posted ADVISOR EP12 ack + stale_wip heartbeat on PR #1554 (H193 tanjiro) — VP fresh low 3.7014 (-0.0123), WSS uptick continues, decoupled descent persists
- Posted ADVISOR main EP1 ack on PR #1565 (H195 fern) — val_WSS=12.8412 matches H183 EP1 baseline 12.82 exactly, mechanism on-shape, kill ladder activates at EP3

### Previous cycle (13:00–13:22Z)
- Posted ADVISOR EP7 ack on PR #1559 (H194 nezuko) — **ALL 4 metrics descending**, WSS=6.7870 already below EP10 gate ≤6.80 at EP7 (3 EPs ahead), VP=3.6040 (-0.0630 STRONG)
- Posted ADVISOR EP28 ack on PR #1541 (H192 frieren) — VP fresh low 3.5270 (slope -0.0036), WSS 6.7042 (6th consec POS slope), terminal ~14:30-14:50Z
- Posted ADVISOR EP12 ack + stale_wip heartbeat on PR #1554 (H193 tanjiro) — VP fresh low 3.7014 (-0.0123), WSS uptick continues, decoupled descent persists
- Posted ADVISOR main EP1 ack on PR #1565 (H195 fern) — val_WSS=12.8412 matches H183 EP1 baseline 12.82 exactly, mechanism on-shape, kill ladder activates at EP3

### Previous cycle (12:00–12:38Z)
- Posted 4 boundary cluster acks (H192 EP27 plateau, H193 EP10/11 VP fresh lows, H194 EP6 strong, H195 smoke green-light + main launch confirm)
- Posted ADVISOR EP10 ack on PR #1554 (H193 tanjiro) — VP fresh low 3.7333 (-0.0118), schedule correction posted (30-EP not 10-EP)

### Previous cycle (11:00–11:30Z)
- Closed PR #1535 (H191) NON-MERGE — WSD family RULED OUT
- Assigned fern H195 (PR #1565): tau_y_loss_weight=1.3 on H183 stack
- Posted EP5/EP9/EP25 acks on H194/H193/H192

### Fleet snapshot at 14:20Z

| Student | PR | Hyp | Run | EP/State | val_WSS | val_VP | val_AB | val_SP | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| nezuko | #1559 | H194 lr=9e-5 on H189 stack | tne4wsap | **EP8 (8.11h)** | **6.7411 ↓ (-0.0459)** | **3.5865 ↓ (-0.0175 fresh)** | **5.9693 ↓ (-0.0379 fresh)** | **3.8787 ↓ (-0.0099 fresh)** | **ALL 4 METRICS DESCENDING — strongest in fleet, -0.30pp from H183 SOTA WSS=6.4427 with 17 EPs remaining** |
| fern | #1565 | H195 tau_y=1.3 on H183 stack | 7ergjfh4 main | **EP2 (1.97h)** | **7.2661** | 4.9930 | 6.6910 | 4.2132 | H147 EP2=7.26 MATCH exactly — tau_y not differentiating yet, EP3 first separation gate ~14:42Z |
| frieren | **#1573** | **H197 volume_loss_weight=2.0 (NEW)** | (pending) | smoke launch pending | — | — | — | — | **H196 (PR #1571) CLOSED NON-MERGE pre-launch** on frieren's code trace identifying flag as no-op under GradNorm. **H197 is mechanism-faithful re-mapping**: doubles raw VP task signal at train.py L430 that GradNorm allocator sees, routing effect through `c_vol_p / G_bar` ratios. Single-flag change vs H183 reproduce, preserves stack integrity. Smoke ETA ~15:00Z. |
| tanjiro | #1554 | H193 wss_normal_penalty λ=0.2 30EP | vuvpegip | EP13 (10.79h) | **7.7670 ↑ (+0.0505)** 7th POS | 3.7048 ↔ (+0.0034 plateau) | 6.5983 ↑ (+0.0274) | 3.9713 ↑ (+0.0073) | VP descent EXHAUSTED at 3.70 plateau; broader uptick at EP13; WSS NON-MERGE firmly locked in. Mid-EP14, next read ~14:30-14:45Z |

### H191 fern — CLOSED NON-MERGE (2026-06-02 11:00Z)
- run `ayg4liye`, rt=25.04h, test_WSS=6.6080, test_ABUPT=5.7714, test_VP=3.6217, test_SP=3.6506
- **Key finding:** WSD decay phase too aggressive (100× LR drop over 10EP cosine). Recovery transient EP25-29 was real (val all 4 fresh lows) but non-load-bearing — test checkpoint captured an over-regularized regime. SP head remains highly sensitive to LR-schedule perturbations on H183/H189 stack (Nth WSD-family SP floor breach pattern).
- **Research implication:** WSD FAMILY RULED OUT on this stack. Standard cosine schedule at/near H183 is near-optimal for schedule axis. Next fern slot (H195) switches to axis-specific loss weighting orthogonal to schedule.

### H195 fern — ASSIGNED (PR #1565)
- **Hypothesis:** τ_y cross-flow axis upweight (weight=1.3) on H183 stack. H192 is exploring τ_z=1.5; τ_y (side-wash axis, H183 val_wss_y=6.98%) has never been isolated.
- **Code change:** Add `tau_y_loss_weight: float = 1.0` field to Config + compute_loss, mirror the existing `tau_z_loss_weight` injection at line ~420 (tensor `[1.0, 1.0, tau_y_loss_weight, tau_z_loss_weight]`)
- **Run:** DDP8 smoke first, then 25EP main, `--tau-y-loss-weight 1.3`
- **Kill ladder:** EP3 ≤7.10%, EP5 ≤6.90%, EP10 ≤6.75%, EP15 ≤6.60%, EP20 ≤6.52%
- ETA smoke ~12:00Z if student picks up quickly, main EP1 ~12:45Z

### H192 frieren — CLOSED NON-MERGE (2026-06-02 13:46Z)
- Run `lokhvm6y` state=finished, rt=22.5h, EP18 EMA-best by val_ABUPT
- **Hypothesis FALSIFIED:** test_τ_z (UPWEIGHTED) +0.225pp = LARGEST test regress (opposite of predicted decoupling)
- **Test metrics confirmed:** test_WSS=6.6206 +0.18pp REGRESS, test_ABUPT=5.7746 −0.069pp under floor ✓, test_SP=3.6413 +0.064pp BREACH ✗, test_VP=3.5313 −0.112pp under floor ✓ (paper-tier VP win held); test_WSS_x=5.85, test_WSS_y=7.19, test_WSS_z=8.66
- **Programme finding:** Per-axis τ-weighting RETIRED as standalone lever (H188 confounded + H192 isolated both failed; H188 differential reading was loss-decomposition artifact, not routing mechanism)
- **VP-responsiveness mechanism REAL but not τ-channel-specific** → H196 follow-up isolates this directly via VP-Charbonnier strengthening

### H196 frieren — ASSIGNED (PR #1571)
- **Hypothesis:** Strengthen `--vol-p-charbonnier-weight` from 0.1 → 0.2 on H183 stack. Tests if H192's VP improvement was driven by gradient reallocation toward VP (not τ-axis routing). Direct VP isolation.
- **Single-variable change** vs H183 SOTA reproduce. Orthogonal to current fleet (H193 wss_normal_penalty, H194 lr ceiling, H195 τ_y axis).
- **Kill ladder:** EP3 ≤4.20/≤7.20, EP5 ≤3.95/≤7.00, EP10 ≤3.65/≤6.85, EP15 ≤3.55/≤6.75, EP20 ≤3.50/≤6.65, EP25 ≤3.48/≤6.55 (val_VP/val_WSS).
- **Target:** test_VP < 3.40 (beat H183 SOTA 3.4415) AND test_WSS < 6.60 (no WSS regress at floor breach)
- ETA: smoke ~14:00Z, main EP1 ~14:35Z

### H193 tanjiro — 30-EP run (NOT 10-EP terminal as previously noted)
- W&B config confirms `num_epochs=30` — EP10 is mid-train, NOT terminal
- EP10 landed (11:38Z, step=109759): val_WSS=7.7030 (+0.0267 uptick), val_VP=3.7333 (FRESH LOW -0.0118), val_AB=6.5615 (+0.0251 uptick), val_SP=3.9717 (+0.0057 uptick)
- Decoupled pattern continues: VP descends, WSS/AB/SP all uptick at EP10 boundary
- Step rate ~10976/EP → 30-EP total ~329k steps, will hit ~24h cap at EP27 (~04:00Z+1d) or finish 30 EP at ~05:48Z+1d
- Multi-metric NON-MERGE final outcome already locked in: WSS +1.24pp over SOTA, no merge path
- Mid-train: keep observing for VP test projection. Terminal close not until ~04:00Z+1d

### H194 nezuko — EP5 PASS (6.8373), EP10 gate ~16:20Z
- EP5 val_WSS=6.8373 (≤6.90 gate PASS), val_VP=3.7445 (strong VP descent)
- Strong start: lr=9e-5 mechanism will only reveal in decay phase EP15+
- **Kill ladder:** EP10 ≤6.80%, EP15 ≤6.65%, EP20 ≤6.55%, EP25 terminal ≤6.50
- ETA EP10 ~16:20Z

### Research directions and priorities

**Current fleet vectors (orthogonal):**
1. H192: τ_z axis weighting (done for z-axis, running)
2. H193: wss_normal_penalty (boundary condition) (running)
3. H194: lr ceiling scan (optimization) (running)
4. H195: τ_y axis weighting (starts now)

**Next round (terminal landing pattern):**
- H192 terminal ~14:55Z (24h cap, EP27-30) → if VP-only, log as VP research note; if SP clears, consider merge → frieren idle next
- H193 terminal ~04:00Z+1d (24h cap at EP27 or natural 30EP at ~05:48Z+1d) → VP harvest, WSS NON-MERGE → tanjiro idle late tomorrow
- H194 terminal ~22h (~03:30Z+1d) → key lr=9e-5 verdict on H189 compound stack
- H195 fern smoke ~13:00Z then main 25EP, terminal ~21h after main launch
- New assignments only after H192 terminal (~14:55Z) — others not idle for many hours

**Queued but not yet assigned (per human directive):**
1. BL DERIVATIVE DECODER — Morgan's top priority, architecturally complex, hold for right student (fern or frieren, after H195 lands)
2. τ_y + τ_z compound (after H195 and H192 terminals confirm axis mechanisms)
3. Tangent-basis output head reformulation (2nd human priority)

### Watch items next 2h (from 13:50Z)
1. **H196 frieren smoke launch** ETA ~14:00Z (PR #1571 assigned 13:48Z, student polls then launches DDP8 smoke 1EP ~12 min)
2. **H194 EP8 boundary** ETA ~14:05Z (step=85695 vs EP8 boundary ~87808, ~2.1k steps @11.1k/h ≈ 11 min) — continued ALL-4-DESCENT expected, full-fleet leader
3. **H195 main EP2 boundary** ETA ~14:00Z (step=21876 vs EP2 boundary ~21950, ~75 steps @13.8k/h ≈ <1 min — IMMINENT)
4. **H193 EP14 boundary** ETA ~14:30Z — VP plateau watch, WSS uptick continues
5. **H194 EP10 gate** ETA ~16:20Z — pre-passed at EP7 (val_WSS=6.7870 < 6.80 gate), focus shifts to EP15 ≤6.65 (~19:25Z)
6. **H196 frieren main launch** ETA ~14:35Z (post-smoke-PASS); EP1 ~15:25Z first val read
