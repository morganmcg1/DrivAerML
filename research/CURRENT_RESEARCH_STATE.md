# SENPAI Research State

**Updated**: 2026-05-29 15:20Z | Branch: `tay` | **SOTA: H244 EP15+6-res (PR #1415 merged 14:57Z)** | Round 4j: 8 active | **H253 val clears gate** ✓

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| SOTA H185+TTA (PR #1382) | 5.9755% | 5.8221% | 6.7214% | 3.4400% | 3.6806% |
| H236 3-res mirror TTA (PR #1408) | 5.9613% | 5.8081% | 6.7130% | 3.4033% | 3.6759% |
| H243 6-res mirror TTA (PR #1414) | 5.9546% | 5.7979% | 6.7025% | 3.3947% | 3.6672% |
| H252 stacked TTA: noise×3-res×mirror (PR #1413) | 5.9492% | 5.7975% | 6.7030% | 3.3996% | 3.6662% |
| **H244 EP15+6-res mirror TTA (PR #1415, merged 14:57Z)** | **5.9452%** | **5.7896%** | **6.6947%** | **3.3882%** | **3.6595%** |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Merge gate**: val_abupt < **5.9452%** AND test_abupt < **5.7896%**
**Paper floors**: test_VP 3.3882 ≤ 3.421 ✓ | test_WSS 6.6947 ≤ 6.727 ✓ | test_SP 3.6595 > 3.577 ✗

---

## Active Fleet (Round 4j, as of 15:20Z)

| PR | Student | Hypothesis | Status | ETA |
|---|---|---|---|---|
| **#1428** | **alphonse** | **H253: noise σ=5e-4 + 6-res stack EP13 (val 5.9418 ✓ clears gate)** | 🟢 test arm pending | ~17:00Z |
| **#1432** | **nezuko** | **H256: H183 + 6-res + noise stack (portability)** | 🟢 running (crash fixed) | ~16:00Z |
| **#1433** | **thorfinn** | **H257: σ-sweep on 6-res+noise stack** | 🟢 running (σ=1e-3 arm) | ~16:30Z |
| **#1438** | **tanjiro** | **H262: K-noise saturation K=10/20 noise_only** (run dev5yglv) | 🟢 running, step=0 only | ~16:00Z |
| **#1439** | **frieren** | **H263: avg(EP14,EP15) EMA + 6-res mirror TTA** | 🟡 BLOCKED on edward W&B artifact upload | TBD |
| **#1440** | **fern** | **H264: EP16 + 6-res mirror TTA** | 🟡 BLOCKED on edward W&B artifact upload | TBD |
| **#1441** | **edward** | **H265: EP14 + 6-res mirror TTA + UPLOAD ARTIFACTS** | 🟡 instructed to upload EP14/15/16 first | ~16:30Z+ |
| **#1442** | **askeladd** | **H268: Anti-thetic noise pairs ±δ** (run n2j8u2lo) | 🟢 running, no metrics yet | ~16:00Z |
| (background) | edward | Arm 2: EP15+full stack h1ae7x1j | 🟢 running | ~16:39Z |

---

## Strategic Focus (Round 4j)

### Critical insight after H244 merge

The EP15 checkpoint is the key differentiator. EP15 single-res val_orig 6.0079 (vs EP13 6.0172, −9.3bp) — this gain stacks additively with TTA. The research direction should now systematically explore:

1. **EP-extension chain**: Is EP16 > EP15? EP17? Or is EP15 the cosine floor?
2. **Multi-EP averaging**: Does avg(EP14,EP15) generalize better than either alone?
3. **Full stack on EP15**: EP15 + K=5 noise + 6-res + mirror (edward Arm 2, h1ae7x1j) — predicted val ~5.938-5.942
4. **K-optimum for noise**: Is K=5 the saturation point? (tanjiro H262, fast probe)

### Next SOTA candidates (ranked by EV)

1. **Alphonse H253** (`qytjlv97`): val_stacked **5.9418** ✓ clears val gate. EP13 + noise × 6-res × mirror. Test arm pending ~17:00Z. **If test_stacked < 5.7896 → IMMEDIATE SOTA merge.**
2. **Edward Arm 2 (h1ae7x1j)**: EP15 + full H252 stack = EP15 × noise × 6-res × mirror. Predicted val ~5.935-5.940 (compounds H253 stack with EP15 advantage). ETA ~16:39Z.
3. **Frieren H263** (BLOCKED): avg(EP14,EP15) + 6-res mirror. Pending edward W&B artifact upload.
4. **Fern H264** (BLOCKED): EP16 + 6-res mirror. Pending edward W&B artifact upload.

### Checkpoint accessibility blocker (15:15Z)

- **Issue**: EP14/EP15/EP16 EMA checkpoints from edward's H244 run `0gjfv45i` exist ONLY on his pod local disk (`outputs/drivaerml/run-0gjfv45i/`). The W&B run only logged the final `model-edward-h244-h185-ep16-cosine-ext-0gjfv45i:v0` artifact (single `checkpoint.pt` = EP15 EMA).
- **Resolution**: Instructed edward (PR #1441) to log EP14, EP15, EP16 as W&B artifacts before running H265. Frieren and fern will download via W&B API and proceed.
- **Lesson for next round**: Add a routine artifact-upload step to checkpoint-saving training PRs so downstream eval-time hypotheses don't get blocked on pod-local files.

---

## Findings Banked This Round (11 total)

| Finding | Source | Summary |
|---|---|---|
| LL | H249 fern | Tight-range multi-res WORSE; wider is better |
| LL-noise | H259 tanjiro | σ=5e-4 optimal; basin edge 5e-4→1e-3 |
| LL-extend | H255 fern | Resolution saturates at 6-res for H185 EP13 EMA |
| HH N=4 | H251 nezuko | Multi-res +12-15bp portable H185/H183/H188 |
| **HH-H188** | **H261 askeladd** | **H188 EP13 not competitive with H185 EP13 (14-16bp worse absolute baseline)** |
| FF generalized | H252 thorfinn | H148 flat basin; noise_only σ=5e-4 gives −39bp test |
| DD-ext3 | H250 frieren | Frequency-weighted multi-res monotonically worse |
| Stacking orthogonality | H252 tanjiro | Weight-space + input-space TTA super-additive (+4bp excess) |
| GG-decomp | H258 frieren | H148 multi-res gain 14× smaller than H185 |
| KK | H254 askeladd | Surface multi-res null: surf→vol cross-attn coupling cancels all gain |
| EE-volume | H260 frieren | Vol-point jitter catastrophic all scales — closes point-position-jitter axis |

---

## Exhaustion Map — TTA Mechanisms

| Mechanism | Result |
|---|---|
| Mirror-y | ✓ VALID — +5bp |
| Multi-res vol 6-res {32k-131k} | ✓ VALID — +21bp (H243), OPTIMAL range |
| Multi-res vol 7-res / 8-res | ✗ MARGINAL (LL-extend): 7-res val regresses; 8-res test only |
| EP-extension (EP15) | ✓ VALID — −9.3bp single-res → additive with 6-res TTA |
| EP14 extension | UNKNOWN — H265 edward testing |
| EP16 extension | UNKNOWN — H264 fern testing (sanity gate first) |
| Multi-EP EMA averaging | UNKNOWN — H263 frieren testing |
| Weight-space noise σ=5e-4 K=5 | ✓ VALID — +8bp standalone |
| Noise + 3-res×mirror stacked | ✓ VALID — H252 +30bp super-additive |
| Noise + 6-res×mirror stacked | UNKNOWN — H253 alphonse testing (HIGHEST EV) |
| Noise + H183 6-res×mirror stacked | UNKNOWN — H256 nezuko testing |
| EP15 + 6-res×mirror+noise stacked | UNKNOWN — edward Arm 2 h1ae7x1j testing |
| K-noise saturation (K>5) | UNKNOWN — H262 tanjiro testing |
| Anti-thetic noise pairs ±δ | UNKNOWN — H268 askeladd testing |
| σ optimal for noise | ✓ CONFIRMED σ=5e-4 (Finding LL-noise) |
| Surface multi-res | ✗ NULL — Finding KK |
| Vol-point jitter | ✗ FALSIFIED all scales — Finding EE-volume |
| H188 family TTA | ✗ NOT VIABLE — H188 baseline 14-16bp worse than H185 |
| Tight-range multi-res | ✗ WORSE (Finding LL) |
| Frequency-weighted multi-res | ✗ MONOTONICALLY WORSE (Finding DD-ext3) |
| Per-channel/per-res α blending | ✗ ALL FALSIFIED |
| Point-position jitter (surface) | ✗ FALSIFIED (Finding EE) |
| Rotation, coord-scale, mesh-subsample, Gaussian noise, permutation | ✗ ALL FALSIFIED |
