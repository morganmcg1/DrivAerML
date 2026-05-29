# SENPAI Research State

**Updated**: 2026-05-29 13:35Z | Branch: `tay` | **SOTA: H252 (PR #1413)** | Round 4i: 8 active | **Finding KK: surface multi-res null; H261 askeladd H188 portability**

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| SOTA H185+TTA (PR #1382) | 5.9755% | 5.8221% | 6.7214% | 3.4400% | 3.6806% |
| H236 3-res mirror TTA (PR #1408) | 5.9613% | 5.8081% | 6.7130% | 3.4033% | 3.6759% |
| H243 6-res mirror TTA (PR #1414) | 5.9546% | 5.7979% | 6.7025% | 3.3947% | 3.6672% |
| **H252 stacked TTA: noise×3-res×mirror (PR #1413)** | **5.9492%** | **5.7975%** | **6.7030%** | **3.3996%** | **3.6662%** |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Merge gate**: val_abupt < **5.9492%** AND test_abupt < **5.7975%**
**Paper floors**: test_VP 3.3996 ≤ 3.421 ✓ | test_WSS 6.7030 ≤ 6.727 ✓ | test_SP 3.6662 > 3.577 ✗

**Note**: H252 regresses slightly on test_VP (+0.5bp) and test_WSS (+0.05bp) vs H243 because it uses 3-res {49k,65k,82k} vs H243's 6-res {32k-131k}. The expected next step (H253 alphonse) stacks noise on H243's full 6-res grid and should recover/extend those channel improvements.

---

## Round 4i Active Fleet (as of 13:35Z)

| PR | Student | Hypothesis | Status | ETA |
|---|---|---|---|---|
| **#1415** | **edward** | **H244 follow-on: EP15 + 6-res mirror TTA (predicted SOTA)** | 🟡 sent back | ~13:30Z (60min eval) |
| **#1428** | **alphonse** | **H253: noise σ=5e-4 + H243 6-res (60 passes, HIGHEST EV)** | 🟢 running | ~12:30-13:00Z |
| **#1437** | **askeladd** | **H261: H188 EP13 + 6-res mirror TTA (Finding HH N=4 full res)** | 🟢 just assigned | ~14:35Z |
| **#1431** | **fern** | **H255: 8-res {16k-164k} AUTONOMOUS follow-up** | 🟢 running | ~14:00-14:30Z |
| **#1432** | **nezuko** | **H256: H183 + 6-res mirror + weight-noise stack** | 🟢 running | ~13:00-13:30Z |
| **#1433** | **thorfinn** | **H257: σ-sweep on H243 6-res stack {1e-4,2e-4,1e-3}** | 🟢 running | ~13:30-14:30Z |
| **#1436** | **frieren** | **H260: Volume-point coordinate jitter TTA (novel axis)** | 🟢 just assigned | ~13:45-14:00Z |
| **#1435** | **tanjiro** | **H259: σ basin-edge probe {5e-4,1e-3,5e-3,1e-2} (FINDING)** | 🟢 running | ~12:10-12:30Z |

**Budget remaining**: ~2.2h (deadline ~15:45Z).

### H244 edward result (submitted 12:14Z, sent back)

- EP14/15/16 trained successfully via cosine extension (t_max=16 schedule, EP13 LR=8.5e-6).
- EP15 is the sweet spot: single-res val 6.0079 vs H185 EP13 6.0172 (−9.3bp).
- With 3-res mirror TTA: val 5.9516 / test 5.7998 — beats H236 gate but **misses H252 SOTA gate** by 2.3bp/2.3bp.
- **HOWEVER**: by additive composition, EP15 + 6-res TTA → val ~5.945 / test ~5.788 — strong SOTA candidate. Sent back to run that eval immediately.
- Edward also landed key infrastructure: `--resume-from-wandb`, `--epochs-already-done`, `--mirror-augmentation`, `--save-every-epoch`, and `eval_multi_res.py --local-checkpoint-path`. These unblock all future EP-extension experiments.

---

## Strategic Priorities (Round 4i — updated after H252 merge)

### Critical insight from H252

H252 used 3-res {49k,65k,82k} × K=5 noise × mirror = 30 passes. It beats H243 on abupt overall but slightly regresses on VP and WSS (which need the wider res range). **alphonse H253 stacks noise on the full H243 6-res grid** — this should combine the best of both: full resolution diversity PLUS weight-space averaging. Predicted val ~5.940/test ~5.793 if mechanisms stack the same way.

### Tier 1 — Potential SOTA beats

1. **H253 alphonse** (HIGHEST EV): noise σ=5e-4 × H243 6-res = 60 passes. Predicted val ~5.940-5.945/test ~5.790-5.795. Should recover VP+WSS and improve overall abupt further.
2. **H259 tanjiro** (FINDING, HIGH-VALUE): σ basin-edge probe. If σ=1e-3 still improves, ALL stacking runs should be redone with σ=1e-3. ~25min, very high information.
3. **H258 frieren**: H148 + 6-res multi-res. If Finding HH N=5 positive, H148 base test 5.835 − 15bp → test ~5.72-5.80 → potential large SOTA push.
4. **H254 askeladd**: surface-points multi-res (novel axis). Unknown. Could add 3-7bp.
5. **H255 fern**: 7-res {32k-164k}. Predicted +0.2-0.4bp tiny SOTA push.
6. **H257 thorfinn**: σ-sweep on H243 6-res stack. If σ=1e-3 still improves (aligned with H259), opens better stacking.

### Tier 2 — Informative / bank

7. **H256 nezuko** (H183 full stack): portability across checkpoints. Might hit test < 5.797 on H183.

### Tier 3 — Long horizon

8. **H244 edward** (~14:00Z): H185 EP14-16. If EP14/15 beats EP13 baseline: 6-res TTA + stacking → potential large SOTA push. Highest-ceiling bet.

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

---

## Exhaustion Map — TTA Mechanisms

| Mechanism | Result |
|---|---|
| Mirror-y | ✓ VALID — +5bp |
| Multi-res vol 3-res | ✓ VALID — +14bp (H236) |
| Multi-res vol 6-res | ✓ VALID — +21bp (H243) |
| Multi-res vol 7-res | UNKNOWN — H255 fern testing |
| Surface-points multi-res | ✗ NULL — Finding KK: surface var-reduction cancelled by surf→vol cross-attn VP cost (+18bp) |
| H188 + 6-res mirror portability | UNKNOWN — H261 askeladd testing |
| Weight-space noise σ=5e-4 | ✓ VALID — +8bp standalone |
| Noise + 3-res×mirror stacked | ✓ VALID — H252 NEW SOTA (+30bp vs H209; super-additive) |
| Noise + 6-res×mirror stacked | UNKNOWN — H253 alphonse testing (HIGHEST EV) |
| Noise + H183 6-res×mirror stacked | UNKNOWN — H256 nezuko testing |
| σ optimal for noise (σ basin edge) | UNKNOWN — H259 tanjiro probing (5e-4 still on slope) |
| Multi-res portability H148 | ✗ PARTIAL — mirror −4.5bp works; multi-res −1.1bp only (density-robustness pre-absorbs, Finding GG-decomp) |
| Volume-point jitter | UNKNOWN — H260 frieren testing |
| Tight-range multi-res | ✗ WORSE than medium (Finding LL) |
| Frequency-weighted multi-res | ✗ MONOTONICALLY WORSE (Finding DD-ext3) |
| Per-channel/per-res α blending | ✗ ALL FALSIFIED (Findings DD, DD-ext, DD-ext2) |
| Point-position jitter | ✗ FALSIFIED (Finding EE) |
| Rotation θ≥0.1° | ✗ FALSIFIED |
| Coordinate scale ε=±2% | ✗ FALSIFIED |
| Mesh-subsample 80-95% | ✗ FALSIFIED |
| Gaussian input noise | ✗ FALSIFIED |
| Permutation | NULL |
