# SENPAI Research State

**Updated**: 2026-05-29 14:20Z | Branch: `tay` | **SOTA: H244-pending (H252 still gate until merge)** | Round 4i: 8 active | **H244 EP15+6res clears gate — awaiting terminal merge**

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| SOTA H185+TTA (PR #1382) | 5.9755% | 5.8221% | 6.7214% | 3.4400% | 3.6806% |
| H236 3-res mirror TTA (PR #1408) | 5.9613% | 5.8081% | 6.7130% | 3.4033% | 3.6759% |
| H243 6-res mirror TTA (PR #1414) | 5.9546% | 5.7979% | 6.7025% | 3.3947% | 3.6672% |
| H252 stacked TTA: noise×3-res×mirror (PR #1413) | 5.9492% | 5.7975% | 6.7030% | 3.3996% | 3.6662% |
| **H244 EP15+6-res mirror TTA (PR #1415, PENDING MERGE)** | **5.9452%** | **5.7896%** | **6.6947%** | **3.3882%** | **3.6595%** |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Current merge gate**: val_abupt < **5.9492%** AND test_abupt < **5.7975%** (H252, until H244 merges)
**After H244 merge gate**: val_abupt < **5.9452%** AND test_abupt < **5.7896%**
**Paper floors**: test_VP 3.3882 ≤ 3.421 ✓ | test_WSS 6.6947 ≤ 6.727 ✓ | test_SP 3.6595 > 3.577 ✗

**H244 note**: EP15 single-res baseline is 6.0079 (vs EP13 6.0172, −9.3bp). EP15+6-res TTA matches additive prediction within 0.03bp val — TTA gains compose nearly linearly in the late-cosine flat basin. All paper floor metrics improved except test_SP which still has the H185-family binding constraint.

---

## Round 4i Active Fleet (as of 14:20Z)

| PR | Student | Hypothesis | Status | W&B | ETA |
|---|---|---|---|---|---|
| **#1415** | **edward** | **H244 EP15+6res mirror TTA (SOTA CANDIDATE, Arm 1 done)** | 🟡 awaiting terminal marker | bh7we7p6 (done), h1ae7x1j (running) | marker: soon; Arm2: ~16:39Z |
| **#1428** | **alphonse** | **H253: noise σ=5e-4 + H243 6-res (60 passes, HIGHEST EV)** | 🟢 running | qytjlv97 partial 5.95 | ~15:00Z |
| **#1437** | **askeladd** | **H261: H188 EP13 + 6-res mirror TTA (Finding HH N=4 full res)** | 🟢 running | jr1nn07o partial 6.13 | ~14:35Z |
| **#1431** | **fern** | **H255: 8-res {16k-164k} (7-res done: val 5.9566/test 5.7948)** | 🟢 8-res running | s8silyb9 partial 5.9541 | ~14:30Z |
| **#1432** | **nezuko** | **H256: H183 + 6-res mirror + weight-noise stack** | 🟢 running | pzbwsouc (stacked arm) | ~14:30Z |
| **#1433** | **thorfinn** | **H257: σ-sweep on H243 6-res stack {1e-4,2e-4,1e-3}** | ⚠ crashes, restarting | sanity ✓, sigma arms crashing | ~14:45Z |
| **#TBD** | **tanjiro** | **H262: K-noise saturation sweep (K=5/10/20 at σ=5e-4)** | 🟡 being assigned | — | ~14:45-15:00Z |
| **#TBD** | **frieren** | **H263: Multi-EP EMA avg (avg(EP14,EP15)+6-res mirror)** | 🟡 being assigned | — | ~15:30Z |

**Budget remaining**: ~1.5h (deadline ~15:45Z).

---

## Strategic Priorities (Round 4i — updated after H244 EP15 win)

### Critical insight from H244 EP15

The late-cosine training extension (EP13→EP15) gave −9.3bp val_orig gain. Crucially, this gain STACKS with 6-res TTA: the compound prediction was val ~5.945, actual val 5.9452 (0.03bp error). This confirms TTA and checkpoint-quality gains are additive. The next big questions:

1. **Does EP15 + FULL STACK beat EP13 + FULL STACK?** Edward's Arm 2 (h1ae7x1j) answers this, but runs past deadline (ETA 16:39Z).
2. **Does multi-EP weight averaging (H263 frieren) beat single EP15?** If avg(EP14,EP15) generalizes better, another push is possible.
3. **Does H253 alphonse's 6-res+noise stack itself beat H244 EP15 alone?** Very likely not (H252 val 5.9492 vs H244 5.9452), but H253 uses the STRONGER 6-res grid vs H252's 3-res, so could surprise.

### Tier 1 — SOTA candidates still in flight

1. **H263 frieren** (being assigned): Multi-EP EMA average — novel weight-space axis. Predicted val ~5.940-5.945 if EMAs diverse. Cheap, ~70 min.
2. **H253 alphonse** (HIGHEST EV on original grid): 6-res+noise on EP13. If stacking orthogonality holds, predicted val ~5.940-5.945/test ~5.790-5.795 — may match H244 EP15+6res, or exceed it.
3. **H256 nezuko**: H183 + stack portability. Lower floor than H185-family.
4. **H261 askeladd**: H188+6-res. If scaling holds: predicted val ~5.945-5.95.
5. **H255 fern 8-res**: Adding 16k resolution point. 7-res was marginal (+0bp test), 8-res may be flat or better.

### Tier 2 — Findings / informative

6. **H262 tanjiro**: K-saturation probe (30 min) — informs all future stacking K choices.
7. **H257 thorfinn**: σ-sweep on stack. H259 result (LL-noise) predicts σ=5e-4 is still optimal for the stack, thorfinn confirms/denies.

---

## Findings Banked This Round

| Finding | Source | Summary |
|---|---|---|
| LL | H249 fern | Tight-range multi-res WORSE; wider is better |
| HH N=4 | H251 nezuko | Multi-res +12-15bp portable to H185/H183/H188 |
| FF generalized | H252 thorfinn | H148 flat basin; noise_only σ=5e-4 gives −39bp test |
| DD-ext3 | H250 frieren | Frequency-weighted multi-res MONOTONICALLY WORSE |
| Test-floor convergence | H252 thorfinn | H148 weight-noise ≈ H243 multi-res (≈5.7978) |
| **Stacking orthogonality** | **H252 tanjiro** | **Weight-space + input-space TTA orthogonal; super-additive (+4bp excess)** |
| GG-decomp | H258 frieren | H148 multi-res gain 14× smaller than H185: mirror −4.5bp works (symmetry), multi-res −1.1bp fails (pre-absorbed by density robustness) |
| **KK** | **H254 askeladd** | **Surface multi-res null on H185 EP13: surface var-reduction (+3-8bp/channel) cancelled by VP degradation (+18-20bp) via surf→vol cross-attention. Axes coupled, not independent.** |
| **LL-noise** | **H259 tanjiro** | **σ=5e-4 is optimal for H185 EP13 weight-noise TTA. Basin edge between 5e-4 and 1e-3 (+3bp degradation at 1e-3, catastrophic ≥5e-3).** |
| **EE-volume** | **H260 frieren** | **Volume-point coordinate jitter catastrophic at all scales (σ_v=1e-4 → val 14.85%). Closes entire point-position-jitter TTA axis.** |

---

## Exhaustion Map — TTA Mechanisms

| Mechanism | Result |
|---|---|
| Mirror-y | ✓ VALID — +5bp |
| Multi-res vol 3-res | ✓ VALID — +14bp (H236) |
| Multi-res vol 6-res | ✓ VALID — +21bp (H243) |
| Multi-res vol 7-res | ✗ MARGINAL — +0bp test (H255 fern 7-res val 5.9566/test 5.7948) |
| Multi-res vol 8-res | UNKNOWN — H255 fern 8-res arm still running |
| Surface-points multi-res | ✗ NULL — Finding KK: surf→vol cross-attn coupling |
| H188 + 6-res mirror portability | UNKNOWN — H261 askeladd testing |
| Weight-space noise σ=5e-4 | ✓ VALID — +8bp standalone |
| Noise + 3-res×mirror stacked | ✓ VALID — H252 NEW SOTA (+30bp vs H209; super-additive) |
| Noise + 6-res×mirror stacked | UNKNOWN — H253 alphonse testing (HIGHEST EV) |
| Noise + H183 6-res×mirror stacked | UNKNOWN — H256 nezuko testing |
| σ optimal for noise (σ basin edge) | ✓ CONFIRMED σ=5e-4 — Finding LL-noise |
| Volume-point jitter | ✗ FALSIFIED ALL SCALES — Finding EE-volume |
| Multi-EP EMA weight averaging | UNKNOWN — H263 frieren testing |
| K-noise saturation (K>5) | UNKNOWN — H262 tanjiro testing |
| EP-extension (EP15 > EP13) | ✓ CONFIRMED −9.3bp single-res; +additional −4bp with 6-res TTA compound |
| Tight-range multi-res | ✗ WORSE than medium (Finding LL) |
| Frequency-weighted multi-res | ✗ MONOTONICALLY WORSE (Finding DD-ext3) |
| Per-channel/per-res α blending | ✗ ALL FALSIFIED (Findings DD, DD-ext, DD-ext2) |
| Point-position jitter (surface) | ✗ FALSIFIED (Finding EE) |
| Rotation θ≥0.1° | ✗ FALSIFIED |
| Coordinate scale ε=±2% | ✗ FALSIFIED |
| Mesh-subsample 80-95% | ✗ FALSIFIED |
| Gaussian input noise | ✗ FALSIFIED |
| Permutation | NULL |
