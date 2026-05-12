# SENPAI Research State

- **Date:** 2026-05-12 (latest invocation: 2026-05-12 ~22:00Z)
- **Branch:** tay
- **W&B project:** wandb-applied-ai-team/senpai-v1-drivaerml-ddp8

---

## CORRECTED DATASET IN EFFECT (since 2026-05-11)

Issue #1053 (deployed 2026-05-11) fixed a case-split/indexing bug that was producing an artificial ~3× vol_p OOD gap (val_vol_p ≈ 3.9% vs test_vol_p ≈ 12%). On the corrected split, top models achieve test_vol_p ≈ 3.6–4.0%, close to val values.

**Corrected dataset path:** `/mnt/pvc/Processed/drivaerml_processed_rawcanon_20260511` (also `/mnt/new-pvc/Processed/...`)
**Split:** val=34 cases/7295 views, test=50 cases/11091 views

All new runs MUST use the corrected dataset path.

---

## Most Recent Research Direction from Human Researcher Team

From Issue #717 (2026-05-09): **test-volume pressure L2 reduction is the single target.** Radical architecture and data preprocessing ideas welcome.

**Status post-corrected-split:** The vol_p OOD gap was a data artifact. On the corrected split the gap is eliminated. Current research focus is now pushing val_abupt below 6.0% (single-model) on the corrected dataset.

---

## Current Best Results (Corrected Split)

### Single-Model SOTA (PR #972 — SDF-stratified vol importance sampling)
- val_abupt = 6.126% | test_abupt = 5.844%
- val_vol_p = 3.798% | test_vol_p = 3.643%
- W&B: source=`56bcqp3m`, eval=`zxnhtagj`

### Rank 2 Single-Model (PR #968 — Stochastic vol subsampling)
- val_abupt = 6.278% | test_abupt = 5.986%
- W&B: source=`a0yoxy85`, eval=`qbg9pkmx`

### Best Ensemble (PR #880 — K=6 greedy ensemble)
- val_abupt = 6.031% | test_abupt = 6.010%
- W&B: source=`zst3y2mp`, eval=`x78xbsfn`
- Note: test_WSS=6.708% is best WSS across all configurations

### Gate to Beat (corrected split)
- **PASS:** val_abupt ≤ 6.2% AND val_vol_p ≤ 4.5%
- **MARGINAL:** val_abupt ≤ 6.5% AND val_vol_p ≤ 5.0%
- **KILL:** otherwise

---

## Current Research Focus and Themes

### Key Insight from Corrected Split
Training-time volume point sampling changes (PR #972 SDF-stratified, PR #968 stochastic) generalize better than loss-weighting, SDF-conditioning, or deeper-stack experiments. The vol_p OOD gap was a data artifact, not a model-capacity problem. New research direction: architecture and training innovations to push below val_abupt 6.0%.

### Active WIP PRs (all transitioning to corrected dataset)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #1042 | alphonse | INR coord-conditioned vol decoder | Running `kbkj2lko` to EP13 on OLD dataset; propose corrected-split retrain if PASS/MARGINAL |
| #1043 | nezuko | (details TBD — awaiting student update) | WIP |
| #1047 | frieren | (details TBD — awaiting student update) | WIP |
| #1048 | fern | Log-space vol_p loss (sign·log1p) | Continue `skozsn41` EP2 as directional signal; follow-up PR on corrected dataset if positive |
| #1049 | askeladd | EMA weight averaging | Running; EP6 gate = val_abupt ≤8.0% AND val_vol_p ≤5.5%; raw model eval requested |
| #1050 | edward | Corrected-split baseline rerun | Launched `nc7lpobi` on corrected dataset — standard retrain |
| #1051 | tanjiro | Coarse-to-fine vol curriculum (16K→65K) | EP3 gate check pending (~22:35Z); NOT killed at EP2 (val_abupt=8.03% is close call) |
| #1052 | thorfinn | (details TBD — awaiting student update) | WIP |

---

## Potential Next Research Directions

### High Priority (push below val_abupt 6.0% on corrected split)
1. **Extend SDF-stratified sampling (PR #972)** — it's the SOTA on corrected split; explore stronger SDF biasing, dynamic curriculum over training
2. **Combine SDF-stratified sampling + architecture improvements** — stack the best sampling strategy with deeper/wider backbone changes
3. **Ensemble diversity on corrected split** — rebuild greedy ensemble pool from corrected-split trained models; PR #880 K=6 was built on old dataset
4. **Vol point density curriculum** — progressive increase in vol sampling density during training (related to PR #1051 coarse-to-fine)
5. **Geometry-aware surface→vol cross-attention** — condition cross-attention keys on SDF values to focus on far-field vol pressure
6. **Per-run instance normalization (RevIN-style)** — normalize vol_p predictions per-case at test time to handle distribution shift within corrected split

### Architecture Directions
7. **L=7 depth** — deeper backbone for corrected-split training; PR #1032 was run on old dataset
8. **Wider MLP heads** — increase decoder capacity for vol prediction given corrected-split has larger test set
9. **Multi-scale vol encoding** — hierarchical vol point aggregation at coarse/fine scales

### Learning / Optimization
10. **Longer training (more epochs)** — baseline training regimen was tuned for old dataset; corrected split has 34 val cases vs 50 test; may benefit from more epochs
11. **Learning rate schedule tuning** — OneCycleLR or cosine warmup variants tuned for corrected-split scale
12. **SAM (Sharpness-Aware Minimization)** — better generalization across geometry distribution in corrected split

---

## Key Findings to Date

- **Corrected dataset (2026-05-11)** eliminated the artificial ~3× vol_p OOD gap — this was the biggest single insight of the research program so far
- **SDF-stratified vol importance sampling** (PR #972) is the new SOTA: val_abupt=6.126%, test_abupt=5.844%
- **Training-time vol sampling strategy** matters more than loss weighting, SDF conditioning, or architecture depth for vol_p generalization
- **L=5 depth** remains the backbone sweet spot
- **Surf→vol cross-attention** (PR #823) gave +2.4% val improvement — biggest single architecture win historically
- **Vol_p aux head** (PR #958) improved val further but adds complexity; still Rank 4 on corrected split
- **K=6 ensemble** (PR #880) provides best ensemble result on corrected split but was built before corrected dataset; needs rebuild with corrected-split models
- **Infra note:** `--data-root` (not `--data-path`); mount point is `/mnt/new-pvc/` (not `/mnt/pvc/`); students must use corrected flag names
- **EMA lag warning:** eval_raw_vs_ema=False means only EMA weights evaluated — early-epoch EMA metrics appear worse than raw model due to trailing-window lag
