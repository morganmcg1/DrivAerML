# SENPAI Research State

_Last updated: 2026-05-28 18:30Z (β-grid disentanglement COMPLETE; H152 aborted EP5.4 (β2↓ falsified); H150 plateau at EP9.9=6.737%; H154 primary launched 17:28Z @ EP1.5; H155 (lr=9e-5) assigned to fern PR #1368)_

---

## Current research focus: Wave 41 — closing β-grid + tier-shift to tau_z and lr-axis

**Primary target:** Beat test_WSS < 5.85% (Transolver-3 SOTA) without degrading test_VP ≤ 3.643% or test_SP ≤ 3.577%.  
**Current single-model SOTA:** PR #1344 (H147) — test_WSS = **6.5409%** — gap to target = −0.69pp.  
**Hard constraint (Issue #1056):** WSS is the main focus. NO ENSEMBLES. ~1 day until shutdown (2026-05-29).

---

## Most recent human research directive (Issue #1056)

- "test_wss is the main focus now!!" (comment #92)
- "NO MORE ENSEMBLES! Its the lazy route to better results" (comment #17)
- "there are only 2 days left before we shut down this experiment" (~2026-05-29)
- Test VP and SP floors from PR #972 must NOT be regressed: test_VP ≤ 3.643%, test_SP ≤ 3.577%

---

## Wave 41: β-grid disentanglement — COMPLETE (2026-05-28 18:30Z)

### Final β-grid verdict

```
        β1=0.93    β1=0.95    β1=0.97
β2=0.97  H149⛔    H152⛔     —
β2=0.98            H147⭐     H153⛔
β2=0.985           —          H150⏳
```

| Direction | Outcome | Evidence |
|-----------|---------|----------|
| β1↓ + β2↓ | ⛔ falsifying | H149 aborted EP3 val_WSS=7.083% |
| β2↓ pure | ⛔ falsifying | H152 aborted EP5.4 val_WSS=7.113% (β1=0.95, β2=0.97) |
| β1↑ pure | ⛔ destabilizes EP1 | H153 aborted EP1 val_WSS=13.86% (β1=0.97, β2=0.98) |
| β1↑ + β2↑ joint | ⏳ delayed converger | H150 (β1=0.97, β2=0.985) — see below |
| **β-grid optimum** | **⭐ H147 (0.95, 0.98)** | SOTA at 6.5409% test_WSS |

**β-space is fully exhausted.** Only H150 remains as an active β-grid arm and it is a delayed converger.

### Fleet status (2026-05-28 18:30Z)

| Run ID | Student | H# | Epoch | val_WSS | val_VP | val_SP | val_ABUPT | State | Δ vs H147 EP10=6.64% |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 5bgp2ryq | tanjiro | H150 (β1=0.97/β2=0.985) | 9.92 | **6.737%** | 3.637% | 3.947% | 5.994% | running | **+0.097pp** behind (plateau confirmed at EP9.9) |
| d20sf8th | nezuko | H151 (extended canonical) | 10.19 | 6.870% | 3.753% | 4.040% | 6.131% | running | +0.230pp behind (RNG drift) |
| 8w7qtm5e | frieren | H154 (tau_z=1.3, primary) | 1.51 | 15.54% | 8.175% | 9.466% | 13.90% | running | EP1 warmup phase (kill at EP3>8.0%) |
| u3vbwwhd | (frieren old) | H149 (β1=0.93/β2=0.97) | 3.1 | 7.083% | — | — | — | **crashed** (abort) | β1↓+β2↓ falsifying |
| 2h8cddnz | (fern old) | H152 (β1=0.95/β2=0.97) | 5.4 | 7.113% | — | — | — | **crashed** (abort) | β2↓ falsifying |

### H147 SOTA trajectory reference

| EP | 1 | 2 | 3 | 5 | 10 | final test |
|---:|---:|---:|---:|---:|---:|---:|
| H147 val_WSS | 12.82% | 7.26% | 6.98% | 6.75% | 6.64% | **6.5409%** |

### H150 EP9.9 plateau analysis

H150 (β1=0.97, β2=0.985) has flattened at val_WSS=6.737% at EP9.9 (step 108867). At EP10 H147 was 6.64%, so H150 is +0.097pp behind at the same epoch. The strong −1.63pp EP1 lead eroded by EP2, recovered toward parity at EP7-8, but the descent has stalled into a plateau. **Verdict: H150 is unlikely to beat H147 by EP10.** H150 may still recover in EP11-30 if cosine LR decay drives further improvement; will continue observing through EP15 before deciding.

### Tier-shift arms (active)

**H154 (frieren, PR #1367)** — Primary 30-EP launched 17:28Z. tau_z_loss_weight=1.3 directly targets H147's worst WSS component (test_WSS_z=8.49%). EP1.5 val_WSS=15.54% (expected hot from tau_z=1.3 weighting; smoke EP1=15.3% recovered to EP3=7.79%, so same trajectory expected). Kill ladder: EP3>8.0%, EP6>7.20%, EP10>6.80%, EP30 must beat 6.5409% by ≥0.05pp.

**H155 (fern, PR #1368)** — lr=9e-5 on canonical H147 β-config (β1=0.95, β2=0.98). Single-knob ablation: the May 4 research map (run 9mm3sz7x) showed lr=9e-5 + tau_y + tau_z produced test_ABUPT=8.12% on a different stack — extracting just the lr=9e-5 axis tests whether lower lr enables longer cosine descent. **Orthogonal to H154** — if both win independently, future compound test possible. PR created 18:25Z; awaiting student pickup for smoke.

---

## Decisive epochs ahead (active runs)

- **H154 primary EP3** (~19:24Z) — kill if val_WSS > 8.0%
- **H150 EP10-11** (~18:30-19:15Z) — definitive plateau or recovery
- **H151 EP10-15** — continue tracking RNG drift; expect ±0.2pp single-trial noise envelope
- **H155 smoke EP1** (after fern picks up PR #1368, ~18:35-19:00Z) — kill if val_WSS > 16% (H147 EP1 was 12.82%)

---

## Potential next research directions (post β-sweep)

### If H150 plateau holds (likely)
1. **Capacity tier**: hidden_dim 512→768 or layers 6→8 on H147 stack (orthogonal axis)
2. **Surface point density**: 65k→130k surface points on H147 stack
3. **Per-axis WSS Charbonnier**: τ_x/τ_y/τ_z loss reformulation
4. **Architectural tier**: cross-attention surface-volume coupling

### If H154 (tau_z=1.3) wins
1. **tau_z grid**: 1.2, 1.4, 1.5 on H147 stack
2. **Compound**: H154 tau_z + H155 lr (if both win)
3. **tau_y_loss_weight** code add: parallel τ_y axis lever (would need train.py extension)

### If H155 (lr=9e-5) wins
1. **lr grid**: 7e-5, 8e-5, 1.1e-4 on H147 stack
2. **lr × cosine T_max**: longer T_max with lower lr
3. **Compound**: lr=9e-5 + tau_z=1.3 if both win independently

### Bold tier (if all current arms plateau)
1. **Flash attention 2 + RoPE PE** replacing STRING PE
2. **Multi-resolution FPS hierarchical processing**
3. **Physics-guided pretraining** on synthetic RANS solutions

---

## Key technical constraints (canonical H147 config)

```bash
torchrun --nproc_per_node=8 target/train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --batch-size 1 --weight-decay 0.005 \
  --train-surface-points 65000 --train-volume-points 65000 \
  --eval-surface-points 65536 --eval-volume-points 65536 \
  --epochs 30 --lr 1e-4 --lr-cosine-t-max 30 --lr-warmup-epochs 1 \
  --optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
  --ema-decay 0.999 --ema-start-step 500 \
  --model-hidden-dim 512 --model-layers 6 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" --pe-num-features 16 \
  --surface-out-width-factor 2.0
```

**Dataset:** `rawcanon_20260511` (corrected split — old split had +7-8pp artifact)  
**EMA:** eval EMA with decay=0.999, start_step=500 (already wired in train.py)  
**Optimizer:** Lion (NOT Adam) — β changes have outsized impact on early-epoch trajectory  
**DDP8:** 8-GPU distributed training, effective batch=8  

---

## Historical context: How we arrived at H147 SOTA

| Wave | Key finding | Best test_WSS |
|------|-------------|--------------|
| Pre-wave | H39 wider surface_out MLP (factor=2.0) | 6.6506% (PR #1284) |
| Wave 36 (H138-H148) | Disentanglement: β-drift is single driver; wd-drift null; curvature null | 6.5409% (H147) |
| Wave 41 (H149-H155) | β-grid exhausted; tier-shift to tau_z (H154) + lr-axis (H155) | TBD |

The gap from current SOTA (6.5409%) to Transolver-3 SOTA (5.85%) is **0.69pp**. β-sweep closed 0pp (H147 stays optimum). Remaining headroom must come from tier-shift: tau_z weighting (H154), lr-axis (H155), capacity/density, or architecture/loss reformulation.

---

## References

- Issue #1056 — active research directive and hard constraints
- BASELINE.md — locked scoreboard with corrected-split results
- PR #1344 — H147 merge commit with full β-attribution analysis
- PR #1367 — H154 tau_z=1.3 active arm
- PR #1368 — H155 lr=9e-5 active arm
- `RESEARCH_IDEAS_*.md` — researcher-agent design briefs from prior waves
