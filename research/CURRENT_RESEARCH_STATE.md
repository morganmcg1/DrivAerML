# SENPAI Research State

- **2026-06-02 11:00Z**
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** ⭐ **H183 (PR #1510, run `guw83mge`) — test_WSS=6.4427%, test_VP=3.4415%, test_SP=3.5187%, test_ABUPT=5.6152% (ALL 4 FLOORS CLEARED)**
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85% (remaining gap: −0.59pp)
- **Human directive (issue #1056, 13:15Z + 13:27Z advisor response):** Morgan posted WALL SHEAR STRESS NOTES 1+2 — identifies BL DERIVATIVE DECODER (off-wall ghost-point probe → ∂u/∂n → WSS) as highest-leverage untried mechanism, TANGENT-BASIS OUTPUT HEAD as 2nd priority. Both require architectural changes larger than current fleet scope.
- **Human check-in (issue #1056, 18:39Z):** No new messages since 19:27Z 2026-06-01.

## 11:00Z checkpoint — **H191 CLOSED NON-MERGE** (test_WSS=6.6080 +0.165pp REGRESS, SP floor breach +0.074pp); **H195 assigned to fern** (PR #1565, tau_y_loss_weight=1.3); H192 EP24 still latest; H193 EP8 still latest; H194 EP4 still latest

### Actions taken this cycle
- Closed PR #1535 (H191) NON-MERGE — W&B confirmed all 4 components regress, SP floor breach +0.074pp
- Appended H191 terminal entry to EXPERIMENTS_LOG.md (WSD family RULED OUT on H183 stack)
- Assigned fern H195 (PR #1565): tau_y_loss_weight=1.3 on H183 stack — isolated τ_y axis upweighting, orthogonal to running fleet
- Fleet probe: H192/H193/H194 same EP boundaries as last cycle (mid-EP25/EP9/EP5) — no new boundaries to ACK

### Fleet snapshot at 11:00Z

| Student | PR | Hyp | Run | EP/State | val_WSS | val_VP | val_AB | val_SP | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| nezuko | #1559 | H194 lr=9e-5 on H189 stack | tne4wsap | EP4 (4.43h) | **6.8713 ✓ gate PASS** | 3.8449 | 6.1218 | 3.9264 | mid-EP5, next gate EP5 ≤6.90 ~11:30Z |
| fern | #1565 | H195 tau_y=1.3 on H183 stack | TBD | ASSIGNED, awaiting smoke | — | — | — | — | ✨ NEW ASSIGNMENT |
| frieren | #1541 | H192 τ_z=1.5 only 30EP | lokhvm6y | EP24 (19.89h) | 6.7003 (uptick) | **3.5331 ✅ FRESH LOW -0.110pp** | 5.9434 | 3.9257 | decoupled descent confirmed; VP paper-tier, WSS rising |
| tanjiro | #1554 | H193 wss_normal_penalty λ=0.2 30EP | vuvpegip | EP8 (7.10h) | 7.6782 (rising) | **3.7680 ↓ descent sustained** | 6.5431 | 3.9700 | WSS exhausted, VP descent decoupled; multi-metric NON-MERGE likely |

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

### H192 frieren — decoupled descent confirmed (VP paper-tier, WSS 4th consecutive uptick)
- EP24 VP=3.5331 (+0.0011 from EP23 fresh low 3.5320) — noise-level bounce, still -0.110pp under floor
- EP24 WSS=6.7003 (+0.0036) — 4th consecutive POS slope
- **Test projection:** val_VP~3.533 → test_VP~3.35-3.45 (paper-tier VP win over H183 baseline 3.4415)
- Multi-metric NON-MERGE still expected (SP val 3.93, SP test likely +0.15-0.27pp over floor 3.577)
- EP25 ETA ~11:15Z

### H193 tanjiro — WSS exhausted, VP descent decoupled
- EP8 WSS=7.6782 (+0.0155 vs EP7), VP=3.7680 (-0.0273 vs EP7)
- Pattern: wss_normal_penalty now over-constraining WSS (pulling UP), while VP gradient descends independently
- VP at current rate (-0.0273/EP): EP10 projects val_VP~3.71 → test_VP~3.51-3.59 (under floor 3.643)
- Multi-metric NON-MERGE: WSS at 7.68 is +1.24pp over SOTA (regress, no path to merge)
- EP9 ETA ~11:15Z; EP10 terminal ETA ~11:50Z

### H194 nezuko — on-shape EP4 (6.8713), EP5 gate ~11:30Z
- EP4 val_WSS=6.8713 (Δ+0.016pp vs H183 interpolated EP4 ~6.855) — second on-shape confirm
- Strong start: lr=9e-5 mechanism will only reveal in decay phase EP15+
- **Kill ladder:** EP5 ≤6.90% (imminent, will pass), EP10 ≤6.80%, EP15 ≤6.65%, EP20 ≤6.55%, EP25 terminal ≤6.50
- ETA EP5 ~11:30Z

### Research directions and priorities

**Current fleet vectors (orthogonal):**
1. H192: τ_z axis weighting (done for z-axis, running)
2. H193: wss_normal_penalty (boundary condition) (running)
3. H194: lr ceiling scan (optimization) (running)
4. H195: τ_y axis weighting (starts now)

**Next round (when runs terminate ~13:00-14:00Z):**
- H192 terminal → if VP-only, log as VP research note; if SP clears, consider merge
- H193 terminal EP10 → VP harvest, WSS NON-MERGE
- H194 terminal EP25 → key lr=9e-5 verdict on H189 compound stack
- New assignments after terminal cycles

**Queued but not yet assigned (per human directive):**
1. BL DERIVATIVE DECODER — Morgan's top priority, architecturally complex, hold for right student (fern or frieren, after H195 lands)
2. τ_y + τ_z compound (after H195 and H192 terminals confirm axis mechanisms)
3. Tangent-basis output head reformulation (2nd human priority)

### Watch items next 3h
1. **H195 fern smoke** ETA ~12:00Z — report EP1 val_WSS
2. **H194 EP5** ETA ~11:30Z — gate ≤6.90 (expected PASS)
3. **H193 EP9/EP10** ETA ~11:15/11:50Z — VP harvest, terminal close
4. **H192 EP25** ETA ~11:15Z — decoupled descent continued
5. **NO IDLE GPUs** — 4/4 students active now (fern just assigned H195)
