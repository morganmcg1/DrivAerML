# SENPAI Research State

_Last updated: 2026-05-28 14:10Z (EP4 trajectory snapshot + H152 smoke verdict)_

---

## Current research focus: Wave 41 — Lion β-sweep disentanglement for WSS optimisation

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

## Wave 41: β-sweep summary (2026-05-28)

### What we know

H147 (PR #1344, merged 2026-05-28) proved that **Lion β1=0.95/β2=0.98** (vs canonical 0.90/0.99) is the single confirmed driver of the WSS improvement from the H138 complex. EP1 val exactly matched H138 (12.83%), confirming mechanism replication. All other candidate drivers (wd-drift H146, curvature-Charb H145) are confirmed null.

H148 (PR #1345, closed) — compound z-coord + curvature spatial reweighting on clean H39 stack — produced test_WSS=6.7638% (null, +0.225pp behind SOTA). Spatial reweighting mechanisms are not worth pursuing in isolation on this stack.

### Current β-grid

```
        β1=0.93    β1=0.95    β1=0.97
β2=0.97  H149⛔    H152⌛     —
β2=0.98            H147⭐     H153⌛(new)
β2=0.985           —          H150🔥(leading)
```

| H# | PR | Config | Status | EP4 val_WSS vs H147 EP4 (6.842%) | Notes |
|----|-----|--------|--------|----------------------------------|-------|
| H147 | #1344 ⭐MERGED | β1=0.95, β2=0.98 | test_WSS=**6.5409%** | reference | CURRENT SOTA; confirmed single driver |
| H149 | #1358 ⛔CLOSED | β1=0.93, β2=0.97 | Aborted EP3 val_WSS=7.4046% | — | β1↓ direction ruled out |
| H150 | #1359 🔥ACTIVE | β1=0.97, β2=0.985 | EP4.2 val_WSS=**6.872%** (+0.030pp) | TIED — strong EP1 lead (-1.63pp) but converged by EP2 |
| H151 | #1360 🔄ACTIVE | canonical β=0.95/0.98 | EP4.5 val_WSS=**6.997%** (+0.188pp) | BEHIND — RNG drift on extended 45-ep replication |
| H152 | #1361 🔄ACTIVE | β1=0.95, β2=0.97 | smoke EP3=7.33% (+0.35pp) | β2↓ direction looks unproductive; greenlit primary 30-EP with EP3>7.40% kill |
| H153 | #1366 🆕RUNNING | β1=0.97, β2=0.98 | EP0.89 (just started) | Pure β1↑ isolation; first val ~30min |

### Interpretation logic (updated 14:10Z)

**Emerging picture from EP4 readings:**
- H150 (β1↑/β2↑) shows fastest EP1 convergence (-1.63pp lead) but converges to TIE H147 by EP2-4. The early lead suggests β1↑ accelerates early-phase learning rather than improving asymptote.
- H151 (canonical) drifting +0.19pp behind H147 — this is RNG noise on a different seed (H147 was a single trial); confirms ~0.1-0.2pp variance band on this stack.
- H152 smoke (β2↓) showing β2=0.97 destabilizes EP1 (15.96% vs 12.82%); recovers by EP3 but consistently +0.35pp behind. Direction looks unproductive.
- H153 (pure β1↑) too early to call.

**Decisive epochs ahead:**
- H150 EP5 (target <6.75% to BEAT H147 trajectory) — next 30min
- H150 EP10 (target <6.64% to confirm SOTA contender) — next 90min
- H151 EP10 (target <6.64% to confirm canonical reproducibility)
- H152 primary EP3 — if >7.40% confirms β2↓ dead, abort
- H153 EP1-3 — if matches H150 trajectory, β1↑ is the dominant driver

---

## Active fleet (2026-05-28)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| dl24-fern | #1361 | H152: β1=0.95/β2=0.97 (pure β2↓) | WIP |
| dl24-frieren | #1366 | H153: β1=0.97/β2=0.98 (pure β1↑) | WIP (just dispatched) |
| dl24-nezuko | #1360 | H151: Extended training 45ep canonical | WIP |
| dl24-tanjiro | #1359 | H150: β1=0.97/β2=0.985 (leading) | WIP |

---

## Potential next research directions (post β-sweep)

### If β-sweep confirms optimum at (0.97, 0.985) [H150 wins]
1. **Fine-grained β1 scan around 0.97**: try β1=0.96, 0.98 with fixed β2=0.985
2. **Fine-grained β2 scan around 0.985**: try β2=0.990, 0.975 with fixed β1=0.97
3. **Capacity increase on canonical config**: hidden_dim 512→768 or layers 6→8

### If β-sweep plateaus near 6.50% (no clear winner beyond H147)
1. **Architecture tier**: Transolver variant with attention pooling over surface-volume cross-attention
2. **Loss reformulation**: Per-axis WSS Charbonnier on τ_x/τ_y/τ_z on the canonical β config (H23-style)
3. **Surface point density**: 65k→130k surface points on canonical β config
4. **Learning rate sensitivity**: lr=1e-4 vs 5e-5 on β-calibrated canonical config

### Bold tier (if plateau persists)
1. **Flash attention 2 + RoPE PE** replacing STRING PE — geometric positional encoding
2. **Multi-resolution feature hierarchy**: hierarchical point cloud processing with FPS sampling
3. **Physics-guided pretraining**: pretrain decoder on synthetic RANS solutions, fine-tune on DrivAerML

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
| Wave 41 (H149-H153) | β-space exploration; H150 leading in early epochs | TBD |

The gap from current SOTA (6.5409%) to Transolver-3 SOTA (5.85%) is **0.69pp**. β-sweep may close 0.1-0.2pp; the remainder likely requires a tier-shift experiment (architecture or fundamentally different loss approach).

---

## References

- Issue #1056 — active research directive and hard constraints
- BASELINE.md — locked scoreboard with corrected-split results
- PR #1344 — H147 merge commit with full β-attribution analysis
- `RESEARCH_IDEAS_*.md` — researcher-agent design briefs from prior waves
