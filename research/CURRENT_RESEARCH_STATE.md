# SENPAI Research State

- **2026-06-02 11:42Z** (H193 schedule corrected — 30-EP not 10-EP terminal)
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** ⭐ **H183 (PR #1510, run `guw83mge`) — test_WSS=6.4427%, test_VP=3.4415%, test_SP=3.5187%, test_ABUPT=5.6152% (ALL 4 FLOORS CLEARED)**
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85% (remaining gap: −0.59pp)
- **Human directive (issue #1056, 13:15Z + 13:27Z advisor response):** Morgan posted WALL SHEAR STRESS NOTES 1+2 — identifies BL DERIVATIVE DECODER (off-wall ghost-point probe → ∂u/∂n → WSS) as highest-leverage untried mechanism, TANGENT-BASIS OUTPUT HEAD as 2nd priority. Both require architectural changes larger than current fleet scope.
- **Human check-in (issue #1056, 18:39Z):** No new messages since 19:27Z 2026-06-01.

## 11:30Z checkpoint — **4/4 students active**; H192 EP25 VP FRESH LOW; H193 EP9 decoupled; H194 EP5 PASS; H195 assigned awaiting smoke

### Actions taken this cycle (11:00–11:30Z)
- Closed PR #1535 (H191) NON-MERGE — W&B confirmed all 4 components regress, SP floor breach +0.074pp
- Appended H191 terminal entry to EXPERIMENTS_LOG.md (WSD family RULED OUT on H183 stack)
- Assigned fern H195 (PR #1565): tau_y_loss_weight=1.3 on H183 stack — isolated τ_y axis upweighting, orthogonal to running fleet
- Posted ADVISOR EP5 ack on PR #1559 (H194 nezuko) — EP5 PASS confirmed, EP10 gate ~16:20Z
- Posted ADVISOR EP9 ack on PR #1554 (H193 tanjiro) — VP decoupled confirmed, terminal harvest EP10 ~11:55Z
- Posted ADVISOR EP25 ack on PR #1541 (H192 frieren) — VP fresh low 3.5304 confirmed, EP30 terminal ~14:00Z

### Fleet snapshot at 11:30Z

| Student | PR | Hyp | Run | EP/State | val_WSS | val_VP | val_AB | val_SP | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| nezuko | #1559 | H194 lr=9e-5 on H189 stack | tne4wsap | EP5 PASS (5.45h est) | **6.8373 ✓ gate PASS** | 3.7445 | 6.0712 | 3.8997 | EP6 running, next gate EP10 ≤6.80 ~16:20Z |
| fern | #1565 | H195 tau_y=1.3 on H183 stack | TBD | ASSIGNED, awaiting smoke | — | — | — | — | pod active, no comments yet |
| frieren | #1541 | H192 τ_z=1.5 only 30EP | lokhvm6y | EP25 (20.76h) | 6.7015 (uptick) | **3.5304 ✅ FRESH LOW -0.113pp** | ~5.94 | ~3.93 | decoupled descent, VP paper-tier, WSS rising; terminal EP30 ~14:00Z |
| tanjiro | #1554 | H193 wss_normal_penalty λ=0.2 30EP | vuvpegip | EP9 (~8.05h est) | ~7.676 (rising) | **~3.72 ↓ descent** | ~6.54 | ~3.97 | WSS NON-MERGE, VP harvest at EP10 terminal ~11:55Z |

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

### H192 frieren — decoupled descent confirmed (VP paper-tier, WSS 5th consecutive uptick)
- EP25 VP=3.5304 (FRESH LOW, -0.113pp under floor 3.643) — deepening decoupled VP descent
- EP25 WSS=6.7015 (+0.0012) — 5th consecutive POS slope, diverging from SOTA
- **Test projection:** val_VP~3.530 → test_VP~3.35-3.44 (paper-tier VP win over H183 baseline 3.4415)
- Multi-metric NON-MERGE confirmed: WSS rising +0.26pp over SOTA, SP val ~3.93 (floor=3.577)
- EP30 terminal ETA ~14:00Z; will close as VP-research note

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

### Watch items next 3h (from 11:42Z) — CORRECTED H193 schedule
1. **H192 EP27-30 terminal** ETA ~14:55Z (24h cap) — VP fresh-low series, NON-MERGE close (or merge if SP clears), frieren idle next
2. **H193 EP10 mid-train landed 11:38Z** (NOT terminal — 30 EP run) — VP fresh low 3.7333, WSS uptick 7.7030; observe for VP test projection
3. **H195 fern smoke EP1** ETA ~13:00Z — DDP8 8 ranks running, step=2744/0.20h
4. **H194 EP10 gate** ETA ~16:20Z — key gate ≤6.80%, currently on-shape at EP5=6.8373
5. **NO IDLE GPUs** — 4/4 active. First idle: frieren@H192 terminal ~14:55Z → assignment needed at that boundary
