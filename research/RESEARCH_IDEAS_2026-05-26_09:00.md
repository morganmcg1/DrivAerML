# Wave 40 Hypotheses — 2026-05-26 09:00Z

**Context**: H144 terminal result pending (~09:20Z). Three contingency branches prepared.
**Constraint invariants**: Zero capacity addition, no ensembles, DDP 8 GPUs, ~13 epochs.
**SOTA baseline**: H112 val_abupt 6.1358% / test_WSS 6.752% / test_WSS_z 8.720%
**Merge gate**: val_abupt < 6.1358%, test_WSS ≤ 6.727%, test_VP ≤ 3.421%, test_SP ≤ 3.577%
**Target**: test_WSS < 5.85% (−0.90pp from SOTA)

---

## Branch A — H144 A WIN (test_WSS ≤ 6.727%)

H144 becomes new SOTA. Compound and extend the ESCALATE class while simultaneously testing orthogonal mechanisms that cannot interact adversarially with loss-weight escalation.

---

### A1 — H151: Joint tau_z + tau_y Escalation Compound

**Mechanism class**: ESCALATE (multi-axis extension)

**Rationale**: H144 (tau_z=6.0) wins on WSS_z. H145 (tau_y=3.0) is simultaneously testing axis extension. If both channels share the heavy-tail / hard-gradient property, joint escalation should compound rather than cancel — the monotone-accelerating response curve on tau_z (2.0→4.0→6.0: −0.045pp, −0.114pp, 2.5× acceleration) suggests the mechanism is real gradient-pressure relief, not channel-specific memorization. Cross-channel bleed under Lion is the primary risk, but H145's result will quantify the tau_y safe operating range before this runs.

**Implementation summary**:
- No code changes needed, CLI flags only
- `--tau-z-loss-weight 6.0 --tau-y-loss-weight 3.0`
- Contingent: wait for H145 result to confirm tau_y axis is alive; if H145 alive, run joint; if H145 dead, try H144×tau_y=2.0 (half-step)

**Expected impact**: −0.12 to −0.20pp test_WSS (additive of independent gains; conservative given bleed)

**Risk**: Cross-channel bleed under Lion amplified at joint escalation. H139 (tau_z SOFTEN) produced +0.356pp tau_y regression — symmetric bleed in ESCALATE direction not yet quantified. Monitor val_WSS_y and val_WSS_x closely at EP1 gate.

**Kill threshold**: At step 10864, if val_abupt > 6.60% (>+0.46pp over H144 EP1 proxy), abort.

---

### A2 — H152: H144 × GradNorm Full Compound

**Mechanism class**: ESCALATE × Dynamic reweighting

**Rationale**: H147 GradNorm (alpha=1.5) autonomously converged to tau_z weight ~2.9× (between H112 2.0 and H143 4.0). H144 is static at 6.0. A compound of static tau_z=6.0 as the base weight with GradNorm dynamically adjusting ON TOP of that base would let the optimizer reclaim gradient budget for tau_y, tau_x, sp, vp when they fall behind — the static escalation's cross-channel risk is precisely what GradNorm was built to manage. GradNorm cannot be pre-seeded from 6.0 (always inits uniform), so the static base weight is the correct way to embed the ESCALATE prior.

**Implementation summary**:
- `--tau-z-loss-weight 6.0 --gradnorm-mode full --gradnorm-lr 1e-3 --no-compile-model`
- No code changes; requires `--no-compile-model` for `full` mode
- Static base weight 6.0 acts as GradNorm's anchor; dynamic adjustment handles cross-channel balance

**Expected impact**: −0.10 to −0.18pp test_WSS; reduced cross-channel regression vs. pure H151 joint compound

**Risk**: `full` mode is slower (no torch.compile); GradNorm may override escalation early if tau_z gradient norm initially dominates. If GradNorm collapses tau_z weight back toward 1.0 at EP1, the run is reproducing H147 with overhead noise — kill.

**Kill threshold**: At step 10864, val_abupt > 6.55%; or if W&B gradnorm/weight_tau_z < 2.0 at step 5000 (GradNorm not holding the ESCALATE prior).

---

### A3 — H153: tau_z=8.0 Magnitude Continuation

**Mechanism class**: ESCALATE (next point on response curve)

**Rationale**: The three-point curve (2.0→4.0→6.0) is monotone-accelerating: −0.045pp, −0.114pp (2.5× increment growth). If the curve continues accelerating, 8.0 should yield −0.20pp to −0.25pp over H144. This is the cheapest possible experiment — one CLI flag change, no architecture or data modification, directly falsifies whether the curve saturates or continues.

**Implementation summary**:
- `--tau-z-loss-weight 8.0`
- Single flag change from H144; all other config identical to H144 recipe

**Expected impact**: −0.15 to −0.25pp test_WSS (if curve continues accelerating); saturation or regression if curve inflects

**Risk**: Cross-channel bleed collapse. At weight 10+ (next step) the Lion sign-only updates may flood tau_y decoder weights. Symptom: val_WSS_y sharply degrades at EP1 gate with tau_z improving. If val_WSS_y regresses more than 0.3pp vs H144 at EP1, the bleed floor has been hit.

**Kill threshold**: Step 10864: val_abupt > 6.50% (regression from H144 projected ~6.05% EP1 equivalent); or val_WSS_y >9.2% (cross-channel collapse proxy).

---

### A4 — H154: H144 × Mirror Augmentation Compound

**Mechanism class**: ESCALATE × Data invariance

**Rationale**: Mirror augmentation (H148) tests whether slope catastrophe is a data-memorisation artifact — if alive, it provides orthogonal regularization that does not add params. Combining with H144's tau_z=6.0 base should stack: ESCALATE increases gradient pressure on hard channels while mirror augmentation reduces memorization of left-right asymmetric training geometries. The two mechanisms operate at different levels (loss gradient vs. data distribution) and should not interfere.

**Implementation summary**:
- Contingent on H148 mirror augmentation code landing in `data/loader.py` (WIP)
- `--tau-z-loss-weight 6.0 --mirror-aug-prob 0.5` (or equivalent H148 flag once implemented)
- Zero param overhead; train-only; axes per spec: surface_x[...,1] (y), surface_x[...,4] (normal_y), surface_y[...,2] (tau_y), volume_x[...,1] (vol y); volume_y invariant

**Expected impact**: −0.05 to −0.15pp test_WSS over H144 alone; primarily on WSS generalization

**Risk**: Dependent on H148 code quality. If H148 is closed (mirror aug dead standalone), this compound may degrade rather than help — ESCALATE cross-channel bleed observed in H148 EP2 (+1.98pp tau_z from tau_y mirror) suggests yaw-flip augmentation perturbs WSS_z more than expected.

---

### A5 — H155: tau_z=10.0 Upper Bound Probe

**Mechanism class**: ESCALATE (saturation/collapse boundary)

**Rationale**: Deliberately maps the ceiling of the ESCALATE curve. The research program needs to know whether the curve inflects before or after 10.0 — this is a diagnostic more than an A-WIN candidate, but if it wins outright it extends the frontier significantly. Schedule after H153 (8.0) result confirms direction; if H153 shows saturation, skip; if H153 still accelerating, run 10.0 to find inflection.

**Implementation summary**:
- `--tau-z-loss-weight 10.0`
- Lower priority than H153 — do not run in parallel with it

**Expected impact**: Could produce −0.30pp if curve continues; more likely saturation or +regression marking curve ceiling

**Risk**: High cross-channel bleed probability. Historical: H139 SOFTEN at narrow tau_z scope produced +0.356pp tau_y regression. ESCALATE in opposite direction may produce symmetric bleed collapse at extreme weights. Treat as boundary probe, not primary win candidate.

---

## Branch B — H144 TIED / Marginal MISS

H144 posts val PASS (< 6.1358%) but test_WSS 6.73–6.80% (marginal miss of ≤6.727% gate). Need additional −0.05 to −0.10pp from orthogonal mechanism.

---

### B1 — H156: Curvature-Stratified Loss Weighting

**Mechanism class**: Geometry-aware gradient weighting (untested)

**Rationale**: H133 tested curvature-stratified surface *sampling* (WIP). Curvature-stratified *loss weighting* is distinct — instead of resampling, assign higher per-point loss weight to high-curvature surface regions where WSS prediction is hardest (sharp leading edges, A-pillars, underbody transitions). This is zero-param, zero-architecture change; it operates on the existing loss reduction. High-curvature points are exactly where the heavy-tail WSS_z residuals live — the SOFTEN class failed because it *softened* these residuals, but weighting them *harder* (like ESCALATE does at the channel level) should align with the ESCALATE mechanism. Curvature is already computed for geometry features (#605 merged surface curvature features).

**Implementation summary**:
- Code change in `trainer_runtime.py` loss reduction: multiply per-point MSE by `1 + alpha * normalized_curvature` before mean
- `surface_x[...,5]` or equivalent curvature channel (verify index from #605 merged code)
- `alpha = 1.5` start; hyper-param controllable via new `--curvature-loss-weight-alpha FLOAT` flag
- ~10 lines of code change; no new parameters

**Expected impact**: −0.05 to −0.12pp test_WSS_z; smaller impact on tau_y/tau_x

**Risk**: If curvature correlates with model confidence (i.e., the model already pays more attention to high-curvature regions), this adds noise rather than signal. Curvature distribution skew may also amplify numerical instability at sharp corners.

---

### B2 — H157: Cosine LR Warm Restarts (SGDR-style)

**Mechanism class**: Optimization trajectory (untested in this programme)

**Rationale**: Current recipe uses a single cosine cycle (T_max=13 epochs). Cosine warm restarts (Loshchilov & Hutter, 2017) with T_max=4–5 epochs would produce 2–3 restarts over 13 epochs. Each restart forces the optimizer to escape flat regions that accumulate over long cosine tails — the slope-flattening pathology observed in the capacity-axis cohort may partly reflect late-train convergence to flat basins that restarts would escape. This is a pure optimization change: zero params, zero architecture, same total compute.

**Implementation summary**:
- `--lr-cosine-t-max 4` (3-epoch restart cycles: EP1-4, EP4-8, EP8-12, EP12-13 partial)
- Or `--lr-cosine-t-max 5` for 2 full cycles (EP1-5, EP5-10, EP10-13 partial)
- No code changes needed; flag already exists in `trainer_runtime.py`
- Try T_max=4 first; T_max=5 as follow-up if 4 shows mid-run spike

**Expected impact**: −0.04 to −0.10pp test_WSS; primarily through better late-train convergence

**Risk**: Restarts at EP4 and EP8 may destabilize EMA-averaged checkpoint selection — the early restart LR spike could hurt the best-checkpoint val metric used for test evaluation. Monitor val_abupt at step 32592 (EP3 gate) to confirm trajectory hasn't diverged.

---

### B3 — H158: AdamW + ESCALATE Compound (contingent on H149 alive)

**Mechanism class**: Optimizer swap × ESCALATE (orthogonal mechanism cross)

**Rationale**: H149 tests whether slope-flattening is Lion-specific. If AdamW produces a better test/val slope (meaning Lion sign-accumulation IS partially responsible for the pathology), then combining AdamW with the proven ESCALATE weight is the natural next step. Cross-channel bleed under Lion is a 3-class confirmed finding — AdamW's per-parameter second-moment scaling should reduce or eliminate the symmetric bleed signature, allowing higher tau_z weights without tau_y/tau_x regression.

**Implementation summary**:
- Contingent on H149 (AdamW standalone) showing alive signal: val_abupt < 6.20% or improved test_WSS slope vs H112
- `--optimizer adamw --lr 3e-4 --tau-z-loss-weight 6.0`
- No code changes; existing AdamW default in train.py (`optimizer: str = "adamw"`, `lr: float = 3e-4`)

**Expected impact**: −0.08 to −0.15pp test_WSS if AdamW eliminates cross-channel bleed; enables higher ESCALATE weights without bleed collapse

**Risk**: AdamW may underperform Lion on the primary WSS metric even with better slope characteristics — the H112 Lion recipe is tuned, and AdamW at 3e-4 may undershoot. If H149 shows AdamW is strictly worse than H112, skip this compound.

---

### B4 — H159: LR Warmup + Compressed Cosine

**Mechanism class**: Schedule refinement (untested combination)

**Rationale**: The current recipe has zero warmup (`lr_warmup_epochs=0`). Adding 1-epoch linear warmup while compressing the cosine to T_max=12 (avoiding the warmup overlap) is a small schedule refinement. Early training instability under the full Lion LR (`9e-5`) may be eroding the first-epoch gradient quality — warmup would reduce this. The combination is zero-param, zero-architecture, and the flags already exist. H144's projected win at EP9 suggests it has a slightly faster convergence trajectory; warmup might sharpen that further.

**Implementation summary**:
- `--lr-warmup-epochs 1 --lr-cosine-t-max 12`
- No code changes; both flags exist in `trainer_runtime.py`
- Note: kill threshold at step 10864 needs adjustment — warmup=1 recipe lands val_abupt ~33% at EP1 (per MEMORY.md), so the EP1 kill gate must be relaxed to step 21728 equivalent

**Expected impact**: −0.02 to −0.06pp test_WSS; primarily through early convergence stability

**Risk**: Small expected impact relative to run cost. Low priority vs. B1/B2/B3 which have clearer mechanisms. The warmup note in MEMORY.md confirms this changes the EP1 gate interpretation, which adds operational complexity.

---

### B5 — H160: Geometry-Family Curriculum Ordering

**Mechanism class**: Training data ordering / curriculum (untested)

**Rationale**: DrivAerML contains multiple aerodynamic geometry families (fastback, notchback, estate variants). Hard-to-predict geometries (complex underbody, aggressive fascia transitions) likely cluster by family. Training samples ordered from easy-to-hard families may produce better gradient quality in the critical early epochs where LR is highest — the model builds a stronger feature basis on clean geometries before encountering pathological WSS_z distributions. This is zero-param, zero-architecture, and pure data-loader ordering.

**Implementation summary**:
- Requires geometry family labels from DrivAerML dataset metadata
- Order training epoch samples: first 30% easy geometries (low WSS_z variance), then 70% hard
- Code change in `data/loader.py` sampler: add curriculum ordering flag `--curriculum-ordering easy-hard`
- Validate geometry family labels available in dataset metadata before implementing

**Expected impact**: −0.03 to −0.08pp test_WSS; hard to predict — no prior art in this codebase

**Risk**: If geometry families are not clearly separable by WSS difficulty, curriculum ordering becomes random shuffling with overhead. Requires metadata audit before implementing. Lower priority than B1-B3.

---

## Branch C — H144 Slope Catastrophe

H144 regresses on test_WSS (>6.752% with val PASS, or val FAIL entirely). ESCALATE class exhausted at tau_z=6.0. Pivot to genuinely new mechanism families.

---

### C1 — H161: AdamW Standalone Pivot (if H149 dead: AdamW+reduced LR)

**Mechanism class**: Optimizer root cause analysis

**Rationale**: If H144 produces slope catastrophe (val PASS but test regression), then the cross-channel bleed under Lion is the dominant mechanism blocking progress — 3-class confirmed, ESCALATE class exhausted. H149 (AdamW `--optimizer adamw --lr 3e-4`) is the direct falsification test. If H149 is simultaneously running, wait for its result before assigning. If H149 is also failing, run AdamW at reduced LR (`--lr 1e-4`) — the standard 3e-4 may be too high for this task given the H112 recipe was tuned at Lion `9e-5` (effective scale different due to sign-only vs. moment-based updates).

**Implementation summary**:
- Primary: `--optimizer adamw --lr 3e-4` (H149 recipe — run H149 first if not already done)
- Follow-up if H149 dead: `--optimizer adamw --lr 1e-4` (half-LR variant to match effective scale)
- No code changes

**Expected impact**: If alive: opens all closed mechanism classes for re-test without Lion bleed; test_WSS improvement of −0.05 to −0.20pp depending on bleed contribution magnitude

**Risk**: AdamW at 3e-4 may converge slower than Lion in early epochs — the 13-epoch budget may be insufficient. At step 32592 (EP3 gate), AdamW run should show val_abupt trending below 6.40%; if still above 6.50%, the LR may need adjustment.

---

### C2 — H162: Cosine Warm Restarts as Flat-Basin Escape

**Mechanism class**: Optimization trajectory (cold start, no ESCALATE dependency)

**Rationale**: If both ESCALATE and optimizer swap fail, the pathology may be a fundamental flat-basin problem in the loss landscape. Warm restarts (SGDR, `--lr-cosine-t-max 4`) force the optimizer out of basins that single-cosine tails converge to. The slope-flattening observed across the capacity-axis cohort (H120/H121/H125/H132) produced systematically shallower val-to-test slopes — this is characteristic of sharp minima that generalize poorly. Restarts at EP4 and EP8 may find flatter but deeper basins. Zero params, zero architecture.

**Implementation summary**:
- `--lr-cosine-t-max 4`
- One flag, no code changes
- Try with H112 base config (no ESCALATE) to isolate optimizer mechanism from loss-weight interaction

**Expected impact**: −0.04 to −0.10pp test_WSS; hard to predict without prior restart results in this codebase

**Risk**: Restarts produce LR spikes that may hurt EMA checkpoint quality. Run without ESCALATE first to cleanly attribute any gain to the restart mechanism.

---

### C3 — H163: Per-Point WSS Magnitude Loss Weighting (Hard-Example Mining)

**Mechanism class**: Loss formulation — adaptive hard-example focus (untested in this form)

**Rationale**: SOFTEN was closed because softening heavy-tail residuals removes the WSS_z signal. The inverse — *harder* weighting on large residuals — is the correct direction and is not the same as ESCALATE (which weights at the channel level, not the per-point level). Per-point weighting proportional to `|residual|^alpha` (focal-style, alpha=0.5) concentrates gradient on the hardest surface points without changing the channel-level loss balance. This is algebraically distinct from SOFTEN (which applies a concave function to reduce large residuals) and from ESCALATE (which scales the entire channel MSE). Never tested in this codebase.

**Implementation summary**:
- Code change in `trainer_runtime.py` loss computation: replace `F.mse_loss(pred, target)` with `((pred - target)**2 * (pred - target).abs().detach()**alpha).mean()` for WSS channels
- `alpha = 0.5` start (focal-style); add `--wss-focal-alpha FLOAT` flag
- Scope: apply only to WSS channels (tau_x, tau_y, tau_z), not VP or SP
- ~15 lines of code change; no new parameters

**Expected impact**: −0.05 to −0.15pp test_WSS_z (hard surface points are precisely where WSS_z deficits live)

**Risk**: Per-point magnitude weighting may destabilize training if extreme residuals (outliers) dominate early gradients. Add gradient clipping or start alpha=0.25. Distinct from Huber loss (#353, not merged) — Huber *softens* large residuals; this *hardens* them.

---

### C4 — H164: Spectral Loss on WSS Channels (Frequency-Domain Alignment)

**Mechanism class**: Loss formulation — spatial frequency alignment (untested)

**Rationale**: WSS distributions on automotive surfaces have characteristic spatial frequency content — smooth gradients over flat surfaces, sharp peaks near edges. Standard MSE operates pointwise and is blind to spatial correlation structure. A spectral auxiliary loss (e.g., FFT of surface patches) that penalizes frequency-domain mismatch could improve recovery of the fine-scale WSS_z patterns that drive the test deficit. Applied to surface mesh representations, this requires approximating spectral content via surface patch FFTs or graph Laplacian eigendecomposition. Zero additional parameters; moderate code complexity.

**Implementation summary**:
- Code change in `trainer_runtime.py`: compute auxiliary spectral loss on WSS surface predictions
- Surface patches of N×N points, 2D FFT, MSE in frequency domain with high-frequency emphasis
- `spectral_loss_weight = 0.1` (additive to main loss); `--spectral-loss-weight FLOAT` flag
- ~30 lines of code; no new parameters
- Note: requires surface point neighborhood indexing — verify mesh connectivity available in batch

**Expected impact**: Uncertain — no prior art in this codebase; could produce −0.05 to −0.15pp if spatial frequency information is a bottleneck

**Risk**: Complex implementation with multiple failure modes (FFT approximation on irregular meshes, patch boundary effects). High implementation risk — assign to student with clean code history. Lower priority than C1/C2/C3 until those are resolved.

---

### C5 — H165: EMA Decay Sweep to 0.9999 (Standalone)

**Mechanism class**: Late-train averaging quality

**Rationale**: H150 (nezuko) tests `--ema-decay 0.9999` standalone. EMA 0.9999 weights recent checkpoints much less than 0.999 — at 70,664 steps, effective window grows from ~1000 to ~10,000 steps. This may smooth out test-val slope differences if the slope catastrophe is partly checkpoint selection noise. Independent of ESCALATE and optimizer changes; pure late-averaging quality change. If H150 is already running, wait for its result; this entry exists to capture the follow-up (0.9999 × H144 compound if Branch A wins; or 0.9999 standalone if everything else fails).

**Implementation summary**:
- `--ema-decay 0.9999`
- One flag, no code changes
- EP1 kill gate drops entirely for ema=0.9999 (per MEMORY.md: EMA-aware kill threshold; recompute δ before quoting EP gates)

**Expected impact**: −0.02 to −0.06pp test_WSS; primarily through smoothing late-train variance

**Risk**: Very slow EMA convergence may hurt best-checkpoint identification in validation — if the EMA model lags the raw model by >5000 steps, the val_abupt plateau window may miss the true best. Low expected impact; treat as compound adjunct, not primary hypothesis.

---

## Priority Ranking Summary

### Branch A (H144 A WIN)
1. **A3 — H153 tau_z=8.0**: Single flag, directly extends proven response curve; highest info/cost ratio
2. **A1 — H151 joint escalation**: Compounds H144+H145 if both alive; highest win potential
3. **A2 — H152 H144×GradNorm**: Addresses cross-channel bleed risk of joint escalation; orthogonal safety net

### Branch B (H144 TIED)
1. **B1 — H156 curvature-stratified weighting**: Zero params, geometry-aware, directly addresses WSS_z hard-point deficit; never tried
2. **B2 — H157 warm restarts**: Pure optimization, single flag, may unlock flat basins without capacity addition
3. **B3 — H158 AdamW+ESCALATE**: Contingent on H149 alive; highest ceiling if Lion bleed is the blocking mechanism

### Branch C (H144 catastrophe)
1. **C1 — H161 AdamW pivot**: Root-cause test for Lion-specific bleed pathology; gates all subsequent decisions
2. **C3 — H163 per-point focal weighting**: Never tried, algebraically distinct from SOFTEN and ESCALATE; targets hard surface points precisely
3. **C2 — H162 warm restarts**: Pure optimization escape mechanism; applicable regardless of optimizer

---

## Decision Tree

```
H144 terminal (~09:20Z)
├── A WIN (test_WSS ≤ 6.727%)
│   ├── Run A3 (tau_z=8.0) immediately — single-flag response curve probe
│   ├── Run A1 (joint escalation) if H145 alive
│   ├── Run A2 (GradNorm compound) in parallel
│   └── A3 wins → A5 (tau_z=10.0 boundary probe)
│       A3 saturates → pivot to B1/B2 style refinements on H144 base
│
├── TIED / marginal MISS (val PASS, test 6.73–6.80%)
│   ├── Run B1 (curvature weighting) — untested, zero params
│   ├── Run B2 (warm restarts) — pure optimization, one flag
│   └── H149 (AdamW) alive → run B3 (AdamW+ESCALATE compound)
│       H149 dead → run B4 (LR warmup+compressed cosine)
│
└── SLOPE CATASTROPHE (test regression)
    ├── Run C1 (AdamW pivot) — confirm/deny Lion bleed as primary cause
    ├── Run C3 (per-point focal) — loss formulation escape
    └── C1 alive → C1×ESCALATE (reopen ESCALATE class under AdamW)
        C1 dead → C2 (warm restarts) + C4 (spectral loss, higher risk)
```

---

*Generated 2026-05-26 09:00Z for Wave 40 planning. H144 terminal result gates branch selection.*
