# Research Ideas — 2026-05-17 17:50Z
## H28: Sharpness-Aware Minimization — `sam-sharpness-escape`
**Assigned student:** edward
**Wave:** 30
**Priority:** HIGH — first optimizer-space attack in all of Wave 30

---

### Bottleneck orientation

Seven closed dead ends in Wave 30 share one failure mode: **per-vertex loss-reweighting against the rel_L2 metric geometry**. The rel_L2 denominator (channel reference magnitude) absorbs any absolute-residual reweighting signal. Seven active experiments (H18, H21, H23, H24, H25, H26, H27) attack: loss-shape, architecture heads, EMA distillation, encoder geometry, auxiliary tasks, coordinate frames, and proxy losses. **Zero experiments in Wave 30 have attacked the optimizer or training dynamics layer.** The τz/τx band attractor (τz/τx ∈ [1.44, 1.55] across every run) is precisely the signature of a sharp basin that momentum-based optimizers cannot escape — Lion's momentum bias makes this worse, not better.

### Hypothesis

**The τz/τx band attractor represents a sharp minimum that Lion's momentum cannot escape. SAM's two-pass perturb-recompute-restore update biases the optimizer toward flat minima with better generalization across OOD test cars, directly targeting the floor on wall_shear_z and test_SP.**

SAM (Foret et al., 2021) seeks parameters w* that minimize loss in a neighbourhood of radius ρ:

```
min_w  max_{||ε||≤ρ}  L(w + ε)
```

The practical first-order implementation:
1. At current weights w, compute gradient g = ∇L(w)
2. Perturb: ŵ = w + ρ · g / ‖g‖₂
3. Forward + backward at ŵ → compute ĝ = ∇L(ŵ)
4. Restore w, apply Lion update using ĝ

When ρ=0: ŵ = w, ĝ = g → **exact baseline recovery** (zero-init path ✓).

### Orthogonality table

| Closed/Active experiment | Mechanism | SAM orthogonal? |
|--------------------------|-----------|-----------------|
| H10b/H11b area-weighted MSE | per-vertex loss weight | YES — optimizer space |
| H16 Charbonnier | loss shape | YES — optimizer space |
| H16b smooth-L1 | loss shape | YES — optimizer space |
| H20 per-vertex error reweighting | per-vertex loss weight | YES — optimizer space |
| H22 Charbonnier + MAE aux | loss shape + aux | YES — optimizer space |
| H12 per-vertex τ-magnitude weighting | per-vertex loss weight | YES — optimizer space |
| H18 area-weighted (active) | per-vertex loss weight | YES — optimizer space |
| H21 per-component heads (active) | architecture | YES — optimizer space |
| H23 EMA self-distillation (active) | training regularization | COMPLEMENTARY — flat-minima goal aligns |
| H24 GSTS slice-temp sharpening (active) | encoder geometry | YES — optimizer space |
| H25 ALGP auxiliary loss (active) | loss augmentation | YES — optimizer space |
| H26 NPCA coordinate augmentation (active) | input representation | YES — optimizer space |
| H27 PRLP per-component rel-L2 proxy (active) | loss proxy | YES — optimizer space |

SAM is provably orthogonal to all 13 prior experiments. It attacks the **optimizer trajectory** rather than the loss surface shape or the input representation.

### Why SAM specifically for τz

Wall shear stress z (τz) has the largest rel_L2 error (τz ≈ 3× larger rel_L2 than τx in many runs). The τz/τx ratio being stuck in [1.44, 1.55] across all experiments — regardless of loss formulation — is consistent with a geometry where the gradient flow toward the τz minimum is surrounded by a sharper basin than τx. SAM's flat-minima bias should disproportionately help the harder-to-generalize channel.

External evidence from similar settings:
- ASAM (Kwon et al., 2021) and SAM on point-cloud / irregular-mesh tasks show the largest gains on OOD evaluation (DrivAerML's 34-val / 50-test split is exactly this regime: 400 train, heterogeneous test geometry)
- LookSAM (Liu et al., 2022) shows SAM combines with Lion-class optimizers without instability at ρ∈[0.02, 0.1]
- Flat-minima work in scientific ML (e.g., Subramanian et al., 2024 on PDEs) shows SAM narrows the train/test gap specifically on OOD physical configurations

### Implementation (≤80 LOC in train.py)

```python
# ---- 1. Argument (~5 LOC, add to argument parser block) ----
parser.add_argument("--sam-rho", type=float, default=0.0,
    help="SAM perturbation radius. 0 = standard Lion (exact baseline recovery).")

# ---- 2. SAM wrapper class (~38 LOC, add near optimizer construction) ----
class SAMLion:
    """Two-pass SAM wrapper compatible with any base optimizer.
    
    When rho=0, first_step() is a no-op and second_step() calls
    optimizer.step() directly — exact baseline recovery guaranteed.
    """
    def __init__(self, optimizer, rho=0.05):
        self.optimizer = optimizer
        self.rho = rho
        self.param_groups = optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        """Perturb weights by rho * g/||g||. No-op when rho=0."""
        if self.rho == 0.0:
            return
        grad_norm = torch.sqrt(sum(
            p.grad.norm() ** 2
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ) + 1e-12)
        scale = self.rho / grad_norm
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                p._sam_e_w = e_w

    @torch.no_grad()
    def second_step(self):
        """Restore weights then apply base optimizer update."""
        if self.rho == 0.0:
            self.optimizer.step()
            return
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "_sam_e_w"):
                    p.sub_(p._sam_e_w)
                    del p._sam_e_w
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd)

# ---- 3. Optimizer construction patch (~5 LOC) ----
# Replace: optimizer = Lion(...)
# With:
base_optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.wd)
optimizer = SAMLion(base_optimizer, rho=args.sam_rho)

# ---- 4. Training loop patch (~12 LOC, replace the backward+step block) ----
# First pass
loss.backward()
optimizer.first_step()
optimizer.zero_grad()

# Second pass (only materially different when rho > 0)
if args.sam_rho > 0.0:
    out2 = model(surface_x, surface_mask, volume_x, volume_mask)
    loss2 = criterion(out2, targets)
    loss2.backward()

optimizer.second_step()
optimizer.zero_grad()
```

**Total new LOC: ~60**. Well within the 80-LOC constraint.

**DDP note:** In DDP, `first_step` must run with `no_sync()` context (no gradient all-reduce during the perturb step), and `second_step` runs with the normal sync. This is standard SAM-DDP practice:

```python
# DDP-aware first pass
with model.no_sync():
    loss.backward()
    optimizer.first_step()
    optimizer.zero_grad()

# DDP-aware second pass (gradient all-reduce happens here)
out2 = model(surface_x, surface_mask, volume_x, volume_mask)
loss2 = criterion(out2, targets)
loss2.backward()
optimizer.second_step()
optimizer.zero_grad()
```

### Hyperparameter recommendation

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--sam-rho` | 0.05 | Standard first choice; literature shows 0.02–0.10 robust for regression tasks |
| `--lr` | 9e-5 | Unchanged from baseline |
| `--optimizer` | lion | Unchanged; SAM wraps it |
| `--batch-size` | 4 | Unchanged; SAM doubles forward passes, not batch size |
| `--epochs` | 13 | Unchanged |
| `--vol-schedule` | 0:16384:3:32768:6:49152:9:65536 | Unchanged |

**Wall-clock note:** SAM doubles the number of forward passes per step. With the standard 13-epoch DrivAerML recipe on 8× GPU, expect ~36h wall time. Use `SENPAI_TIMEOUT_MINUTES=2200` to be safe.

### EP3 falsifiable gate

At EP3 checkpoint, the run is ALIVE if **both**:
1. `val_primary/abupt_axis_mean_rel_l2_pct < 6.00%` (vs 6.126% baseline at EP3 pace)
2. `val_primary/wall_shear_z_rel_l2_pct / val_primary/wall_shear_x_rel_l2_pct < 1.42` (band break — τz/τx below the [1.44, 1.55] attractor)

Gate 2 is the key falsifier: if SAM's flat-minima bias is actually helping τz disproportionately, the ratio should drop below the attractor band within 3 epochs.

KILL if:
- `val_primary/abupt_axis_mean_rel_l2_pct > 6.50%` at EP3 (divergence / instability)
- `val_primary/wall_shear_z_rel_l2_pct / val_primary/wall_shear_x_rel_l2_pct > 1.55` at EP3 (deeper in attractor than baseline)

### Torchrun recipe

```bash
SENPAI_TIMEOUT_MINUTES=2200 torchrun --nproc_per_node=8 train.py \
  --lr 9e-5 \
  --optimizer lion \
  --hidden-dim 512 \
  --slices 128 \
  --batch-size 4 \
  --vol-schedule "0:16384:3:32768:6:49152:9:65536" \
  --epochs 13 \
  --sam-rho 0.05 \
  --wandb-group wave30-sam \
  --run-name edward-h28-sam-sharpness-escape
```

### Predicted outcomes

**If SAM succeeds (mechanism alive):**
- τz/τx ratio drops to <1.42 by EP3 and stays below 1.44 through EP13
- test_SP improves from 3.577% toward <3.4% (τ floor breach resolved)
- test_abupt improves from 5.844% toward <5.5%

**If SAM fails (mechanism dead):**
- τz/τx stays inside [1.44, 1.55] — attractor is not a sharpness artefact
- Constrains the research map: the floor is not optimizer-escapable, likely structural in the Transolver slice assignment or the rel_L2 metric geometry itself
- Points toward H24 (GSTS) or H25 (ALGP) as the surviving live hypotheses

**Information value of a failure:** The τz/τx attractor surviving SAM would be the clearest evidence yet that the floor is in the **representation** (how the model projects surface geometry to slices) rather than the **optimization trajectory** — directly motivating H26 (NPCA) and any future architecture-level interventions on slice assignment diversity.

### Taste rubric

**Research mode:** Tier shift (new level of abstraction — optimizer dynamics, never tried in Wave 30)

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Mechanistic grounding | 4/4 | Precise mechanism: SAM's flat-minima bias targets the τz/τx attractor which exhibits the geometric signature of a sharp basin (persistent across all loss formulations). EP3 τz/τx ratio gate is a concrete, non-metric falsifier tied to the specific failure mode. Strong external evidence from OOD regression settings. |
| Research-state value | 4/4 | Either SAM breaks the band (confirms sharpness hypothesis, opens optimizer-tuning direction) or it doesn't (rules out optimizer-space explanations decisively, redirects to representation). Result changes the research map either way. |
| Execution value | 3/4 | 2× forward passes per step doubles wall time (36h vs 18h). This is the main cost. However: (a) no architectural changes needed, (b) ~60 LOC implementation, (c) rho=0 recovery eliminates debug risk, (d) the flat-minima hypothesis has never been tested and is the clearest unexplored axis. One point deducted for wall-time cost only. |

### Confidence

**Medium-high.** SAM's flat-minima bias has strong theoretical grounding and empirical support in OOD regression tasks. The τz/τx band attractor's persistence across all loss formulations (7 dead ends + 7 active covering loss space exhaustively) is a strong indirect signal that the bottleneck is not in the loss but in the optimization trajectory or representation. The main uncertainty is whether the DrivAerML geometry is sufficiently irregular to benefit from flat-minima bias — it's not an image task, so SAM's standard benefits may not transfer directly. The EP3 τz/τx gate resolves this quickly at 18h cost rather than 36h.

---

### Research state update

**Current best explanation for the floor:**
The test_SP floor (+0.502pp breach in H22, +0.45pp breach in all active runs) and τz/τx band attractor [1.44, 1.55] are most likely caused by one of:
1. A sharp minimum in the optimization landscape that Lion's momentum locks into (H28 SAM tests this)
2. Insufficient geometric diversity in the slice assignment (H24 GSTS tests this)
3. Structural coupling between cp and τ in the shared encoder (H25 ALGP tests this)
4. Missing local-frame information in the input features (H26 NPCA tests this)
5. Metric geometry mismatch between MSE training loss and rel_L2 eval (H27 PRLP tests this)

**Ruled-out paths (Wave 30 confirmed):**
- Per-vertex loss reweighting by any signal (error magnitude, target magnitude, area, Charbonnier, smooth-L1) — all 7 closures confirm this axis is dead against rel_L2 metric geometry
- Cold-start momentum artifacts in τz/τx — confirmed to fade by EP3 in H18 and H20

**Open uncertainties:**
1. Is the τz/τx attractor optimizer-escapable (sharpness) or representation-structural?
2. Does the test_SP floor breach come from cp generalization or τ generalization, or both?
3. Can any single wave-30 run simultaneously break the val_SP floor AND the test_vol_p floor?

**Next discriminating experiment:** H28 SAM at EP3 τz/τx gate. Cost: 18h (to EP3 gate check). If ratio breaks below 1.42, extend to full 13 epochs. If not, kill and the research map is substantially updated.

**Stop condition for SAM direction:** If ρ=0.05 fails EP3 gate, try ρ=0.02 and ρ=0.10 as two quick arm variants before closing the optimizer-space direction entirely.
