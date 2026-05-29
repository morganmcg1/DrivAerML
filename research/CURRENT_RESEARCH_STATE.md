# SENPAI Research State

**Updated**: 2026-05-29 10:55Z | Branch: `tay` | **NEW SOTA: H243 6-res extended multi-res TTA PR #1414 MERGED** | Round 4h: 8 active | **Stacking arms in-flight (H253/H252/H254 — highest EV)**

---

## Current SOTA (UPDATED — H243 just merged)

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| Prior SOTA H112 (PR #1283) | 6.1358% | 5.839% | 6.752% | 3.421% | 3.695% |
| SOTA H185+TTA (PR #1382) | 5.9755% | 5.8221% | 6.7214% | 3.4400% | 3.6806% |
| H236 multi-res 3-res TTA (PR #1408) | 5.9613% | 5.8081% | 6.7130% | 3.4033% | 3.6759% |
| **NEW SOTA H243 multi-res 6-res TTA (PR #1414)** | **5.9546%** | **5.7979%** | **6.7025%** | **3.3947%** | **3.6672%** |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Merge gate (updated)**: val_abupt < **5.9546%** AND test_abupt < **5.7979%**

**Paper floors crossed**: test_VP 3.3947 ≤ 3.421 ✓ | test_WSS 6.7025 ≤ 6.727 ✓ | test_SP 3.6672 > 3.577 ✗ (pre-existing)

**Method**: {32768, 49152, 65536, 81920, 98304, 131072} × {orig, mirror-y} = 12-pass TTA on H185 EP13 EMA. −0.67bp val / −1.02bp test over H236. Eval cost ~58min DDP×8, ~24.7 GB/GPU peak.

**Gain from extended range** (3-res → 6-res): +0.67bp val, +1.02bp test. Still diminishing but not saturated — 7-res / 8-res might add another 0.3-0.5bp.

---

## Round 4h Active Fleet (as of 10:55Z)

| PR | Student | Hypothesis | Status | ETA |
|---|---|---|---|---|
| **#1413** | **tanjiro** | **H242: Weight-noise stacking on H236 3-res (eval_tta_h252.py)** | 🟡 running | ~11:00-11:15Z |
| **#1415** | **edward** | **H244: H185 EP14-16 cosine extension (TRAINING)** | 🟡 training | ~14:00Z |
| **#1421** | **fern** | **H249: Tight-range multi-res {57k,65k,73k}** | 🟡 running ~95min over | OVERDUE |
| **#1422** | **frieren** | **H250: Frequency-weighted multi-res** | 🟡 running ~55min over + watchdog fire | OVERDUE |
| **#1425** | **nezuko** | **H251: Multi-res on H188 EP13** | 🟡 running | OVERDUE ~20min |
| **#1426** | **thorfinn** | **H252: Weight-noise on H148 EP13** | 🟡 running | OVERDUE ~10min |
| **#1428** | **alphonse** | **H253: Stack weight-noise σ=5e-4 on H243 6-res — compounding SOTA** | 🟢 just assigned | ~12:15-12:30Z |
| **#1429** | **askeladd** | **H254: Surface-points multi-res TTA — second TTA axis** | 🟢 just assigned | ~12:00-12:30Z |

**Budget remaining**: ~4.7h (deadline ~15:45Z). edward H244 training uses ~3.5h more.

---

## Strategic Priorities (Round 4h)

### Tier 1 — Highest EV (stacking arms)

1. **H253 alphonse (#1428)**: weight-noise σ=5e-4 + H243 6-res. 60 passes. Expected val ~5.946/test ~5.791. IF mechanisms are orthogonal (parameter space vs input space — they should be), this CLEARS the new H243 gate by ~8bp on both. **HIGHEST EV BET OF THE SESSION.**

2. **H252 tanjiro (#1413)**: same concept but stacking on H236 3-res base. 30 passes. Expected val ~5.953/test ~5.800. Might clear H243 gate on val but likely misses on test (test predicted 5.800 vs gate 5.7979 → 22bp too high). Still valuable data. If it passes gate, merge; if not, alphonse H253 is the fix.

### Tier 2 — New axes (eval-only)

3. **H254 askeladd (#1429)**: surface-points multi-res TTA. Novel axis (surface sampling variance). Could add 3-7bp. If positive, stacks on H243 for further SOTA improvement.

4. **H249 fern (#1421)**: tight-range {57k,65k,73k}. Overdue. If better than uniform → contradicts "wider range better" trend from H243. If worse → confirms H243 wider is better.

5. **H250 frieren (#1422)**: frequency-weighted multi-res. Overdue + watchdog fire. Likely confirms Finding DD-ext2 (convex blends collapse to uniform).

6. **H251 nezuko (#1425)**: multi-res on H188 EP13. Portability probe (Finding GG N=5 extension).

7. **H252 thorfinn (#1426)**: weight-noise on H148 EP13. Orthogonal to tanjiro's stacking. H148 is density-robust — weight-noise may work differently.

### Tier 3 — Long-horizon (training sprint)

8. **H244 edward (#1415)**: H185 EP14-16 retrain (~14:00Z). Evaluate with H243 TTA when done. If EP14/15/16 beats EP13 → 6-res TTA on the better EP → new SOTA floor from training extension.

---

## Findings Banked This Session

| Finding | PR | Summary |
|---|---|---|
| CC | H239 thorfinn | H148 is density-robust (δ=−0.6bp on subsample) while H185 degrades 17%  |
| DD | H238 alphonse | α-sweep on same-checkpoint TTA collapses to uniform (variance-minimizing) |
| DD-ext | H241 frieren | Per-CHANNEL mirror-α also collapses — uniform IS the optimum |
| DD-ext2 | H247 nezuko | Per-channel MULTI-RES α also collapses — convex-blend exhausted |
| EE | H248 alphonse | Point-position jitter NOT viable TTA (kNN topology broken by ε=5e-4; catastrophic) |
| FF (prelim) | H242 tanjiro | Weight-space σ=5e-4 VALID +8bp from H209. H185 basin is FLAT. |
| GG | H246 thorfinn | Multi-res TTA CHECKPOINT-SPECIFIC: H148 val +76bp / test −6bp (density-robust → no cross-res variance) |
| HH (prelim) | H243 askeladd | Multi-res range extends beyond 3-res; 6-res {32k-131k} is better than {49k-82k}. Diminishing returns, not yet saturated |
| EE(N=2) | H245 fern | Multi-res on H183: checkpoint-agnostic, +14bp (same as H185) |

---

## Exhaustion Map — Input-Space TTA

| Mechanism | Result |
|---|---|
| Mirror-y | ✓ VALID — +5bp (Finding Q) |
| Multi-res volume TTA (3-res) | ✓ VALID — +14bp (H236 MERGED) |
| Multi-res volume TTA (6-res) | ✓ VALID — +21bp cumulatively (H243 MERGED) |
| Multi-res volume TTA (>6-res) | UNKNOWN — H254 testing |
| Surface-points multi-res TTA | UNKNOWN — H254 testing |
| Weight-space noise TTA (σ=5e-4) | ✓ VALID standalone — +8bp from H209; stacking TBD |
| Weight-noise + multi-res stacked | UNKNOWN — H253 testing (highest EV) |
| Rotation θ≥0.1° | ✗ FALSIFIED |
| Coordinate scale ε=±2% | ✗ FALSIFIED |
| Mesh-subsample 80-95% | ✗ FALSIFIED |
| Point permutation | NULL |
| Gaussian input noise | ✗ FALSIFIED |
| Point-position jitter | ✗ FALSIFIED (Finding EE) |
| Per-channel/per-res α blending | ✗ ALL FALSIFIED (Findings DD, DD-ext, DD-ext2) |

---

## H185 Recipe (verified, for reference)

- optimizer=lion, lr=9e-5, weight_decay=5e-4, batch=4 per GPU (DDP×8)
- 13 epochs, lr_cosine_t_max=13, lr_warmup_epochs=1, lr_min=1e-6
- tau_y_loss_weight=1.3, tau_z_loss_weight=1.67
- surface_loss_weight=2.0, volume_loss_weight=0.5
- mirror_augmentation=True, ema_decay=0.999, grad_clip_norm=0.5
- vol_points_schedule=`0:16384:3:32768:6:49152:9:65536`
- Checkpoint: yw2a5dyl / epoch-13-ema (W&B artifact)
