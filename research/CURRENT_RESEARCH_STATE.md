# SENPAI Research State

- **2026-06-02 13:50Z** (H192 CLOSED NON-MERGE — hypothesis FALSIFIED, upweighted τ_z largest regress +0.225pp; H196 frieren VP-Charbonnier=0.2 PR #1571 dispatched; H193 EP13 WSS 7th consec POS slope VP plateau; H194 EP8 + H195 EP2 boundaries imminent)
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** ⭐ **H183 (PR #1510, run `guw83mge`) — test_WSS=6.4427%, test_VP=3.4415%, test_SP=3.5187%, test_ABUPT=5.6152% (ALL 4 FLOORS CLEARED)**
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85% (remaining gap: −0.59pp)
- **Human directive (issue #1056, 13:15Z + 13:27Z advisor response):** Morgan posted WALL SHEAR STRESS NOTES 1+2 — identifies BL DERIVATIVE DECODER (off-wall ghost-point probe → ∂u/∂n → WSS) as highest-leverage untried mechanism, TANGENT-BASIS OUTPUT HEAD as 2nd priority. Both require architectural changes larger than current fleet scope.
- **Human check-in (issue #1056, 18:39Z):** No new messages since 19:27Z 2026-06-01.

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

### Fleet snapshot at 13:22Z

| Student | PR | Hyp | Run | EP/State | val_WSS | val_VP | val_AB | val_SP | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| nezuko | #1559 | H194 lr=9e-5 on H189 stack | tne4wsap | EP7 (7.06h) | **6.7870 ↓ (-0.0150)** | **3.6040 ↓ (-0.0630 STRONG)** | **6.0072 ↓ (-0.0258)** | **3.8886 ↓ (-0.0136)** | **ALL 4 DESCENDING — already BELOW EP10 gate ≤6.80 at EP7, strongest in fleet** |
| fern | #1565 | H195 tau_y=1.3 on H183 stack | 7ergjfh4 main | EP1 (0.91h) | **12.8412** | 14.2048 | 13.0318 | 8.7329 | ON-SHAPE H183 EP1=12.82 match, mechanism non-disruptive at init, kill ladder activates EP3 |
| frieren | #1571 | **H196 vol_p_charbonnier=0.2 (NEW)** | (pending) | smoke launch (post-13:50Z) | — | — | — | — | Direct VP-isolation hypothesis (double H183 default 0.1); single-var change; smoke-then-30EP-main. Replaces H192 (PR #1541 closed NON-MERGE 13:46Z). |
| tanjiro | #1554 | H193 wss_normal_penalty λ=0.2 30EP | vuvpegip | EP13 (10.40h) | **7.7670 ↑ (+0.0505)** 7th POS | 3.7048 ↔ (+0.0034 plateau) | 6.5983 ↑ (+0.0274) | 3.9713 ↑ (+0.0073) | VP descent EXHAUSTED at 3.70 plateau; broader uptick at EP13; WSS NON-MERGE firmly locked in. Continue mid-train. |

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
