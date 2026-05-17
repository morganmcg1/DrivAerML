# H18 Composition Spec — Maximum WSS Stack

**Status**: REVISED 09:55Z — H12 architecture signal FAILED in compound (EP11 regression on all 7 axes, vol_p +0.307pp jump suggests Lion+Charb+separate-τ-head poor late-training stability). **Separate τ head DROPPED from primary H18 spec**, retained only as H18-fallback-B for separate test.

**Status**: DRAFT — ready to assign when first student frees (likely H9b tanjiro ~12:00Z autonomous EP30 land).

## H12 Architecture Lesson (added 09:55Z)

The H12 separate τ head experiment delivered ambiguous results:
- **Real architectural gain at EP7**: H12 val_wss 6.939 vs H10b 6.976 at SAME step 76,831 = -0.037pp lead
- **Lead REVERSED at EP10**: H10b 6.922 leads H12 6.933 at same step 109,759
- **EP11 regression**: all 7 axes regressed, vol_p +0.307pp jump (likely optimizer/batch-noise event, not slow drift)
- **Failure mode**: Separate τ head adds capacity but the Lion+Charbonnier+separate-τ-head combination has poor late-training stability

**Decision for H18**: Drop separate τ head from primary spec. Use proven mechanisms only:
- H10b: curvature additive attention bias + Charbonnier τ_z
- H9b: clamp=0.15 + vol_p MAE auxiliary
- Lion lr=1e-4, GradNorm α=0.5, Y-symmetry, EMA (all H10b baseline)

H18 expected test_wss with reduced stack: 6.50-6.60 (still -0.13-0.23pp under SOTA, all floors preserved).

## Mechanism Composition

Stacks 4 mechanisms each validated on a separate axis or wave-finding:

| Mechanism | Source | Empirical evidence |
|---|---|---|
| **Curvature additive attention bias** | H9/H10b | +0.15-0.20pp val_wss vs no-bias baseline (H9 EP30 vs H8 EP30) |
| **Charbonnier loss on τ_z only** | H10b | Charbonnier→MSE ratio 1.20-1.30 sustained, τ_z lead -0.16pp vs no-Charb (H10b vs H9b EP13) |
| **clamp=0.15 + vol_p MAE auxiliary** | H9b | val_vol_p 3.785 (-0.27pp vs H10b 4.376) preserved through EP18 |
| **Separate τ head architecture** | H12 | val_wss -0.037pp / val_τ_z -0.014pp vs H10b at SAME step 76,831 (EP7) with accelerating slopes EP6→7 |

## Expected EP30 projection (compositional)

Assume mechanism contributions are additive on val_wss:
- H10b alone (curvature+Charb): val_wss 6.880 → test_wss ~6.62 (-0.11pp under SOTA)
- + H12 architecture (separate τ head): -0.20pp val_wss → val ~6.68, test ~6.42 (-0.31pp under SOTA)
- + H9b floor preservation: vol_p ~3.59 (preserves under-floor on test_vol_p)
- + curvature/Charb already in H10b stack
- **Net projection**: test_wss ~6.40-6.45, test_vol_p ~3.55-3.62, test_surf_p ~3.50-3.58 — **3/3 contract floors satisfied + SOTA-beat on test_wss by -0.27-0.32pp**

## Code mods required (relative to H10b PR #1159 stack)

### Model (`model.py` ~line 365)
Replace single linear surface head:
```python
self.surface_out = LinearProjection(n_hidden, self.surface_output_dim)
```
With H12 separate τ head:
```python
self.surface_cp_out = LinearProjection(n_hidden, 1)
self.surface_tau_out = nn.Sequential(
    nn.Linear(n_hidden, 2 * n_hidden),
    nn.GELU(),
    nn.Linear(2 * n_hidden, 3),
)
nn.init.normal_(self.surface_tau_out[-1].weight, std=0.01)
nn.init.zeros_(self.surface_tau_out[-1].bias)
```

And in forward:
```python
cp_out = self.surface_cp_out(x_surf)  # (B, N, 1)
tau_out = self.surface_tau_out(x_surf)  # (B, N, 3)
surf_out = torch.cat([cp_out, tau_out], dim=-1)  # (B, N, 4)
```

### Train config
Already in H10b stack:
- `model_pe = "string_multisigma"`, `model_layers = 6`
- `optimizer = "lion"`, `lr = 1e-4`
- `use_y_symmetry_aug = True`, `use_ema = True`
- `use_gradnorm = True`, `gradnorm_alpha = 0.5`
- `use_curvature_bias = True`
- `tau_z_loss = "charbonnier"`, `charbonnier_eps = 0.001`

Add from H9b:
- `volume_sdf_alpha = 0.05` (vol_p MAE aux weight)
- `grad_clamp = 0.15` (vol_p loss-weight floor — REPLACES default 0.05)
- `use_mae_aux = True`

### Reproduce command
```bash
python train.py \
  --dataset_dir /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --optimizer lion \
  --model_layers 6 \
  --model_hidden_dim 192 \
  --model_pe string_multisigma \
  --use_ema \
  --use_gradnorm \
  --gradnorm_alpha 0.5 \
  --grad_clamp 0.15 \
  --use_y_symmetry_aug \
  --use_curvature_bias \
  --tau_z_loss charbonnier \
  --charbonnier_eps 0.001 \
  --use_mae_aux \
  --volume_sdf_alpha 0.05 \
  --separate_tau_head \
  --eval_at_end \
  --wandb_group h18-max-wss-stack
```

## Decision gates (mid-run review pattern)

| EP | Hard gate | Action if breached |
|----|-----------|---------------------|
| EP3 | val_wss > 7.10 OR val_τ_z > 9.80 | Abort — composition diverging |
| EP5 | val_wss > 6.95 | Continue but flag for EP7 |
| EP7 | val_wss > 6.92 AND val_τ_z > 9.40 | Mid-run kill — separate τ head not paying off in compound |
| EP10 | val_wss > 6.85 AND val_τ_z > 9.30 | Hard kill (H12 EP10 gate pattern) |
| EP15 | val_wss < 6.80 | Strong continue → EP30 confirmation |
| EP20 | val_wss < 6.75 | SOTA-beat candidate locked in |
| EP30 | Terminal evaluation | Compare test metrics to PR #972 floors |

## Risk: optimizer instability from 4-mechanism stack

Mitigation:
- Lion lr=1e-4 was stable on H10b 30 epochs — keep that
- Curvature bias is additive (zero-init) — safe stacked with Charbonnier
- Separate τ head has zero-init final layer — safe stacked
- MAE_aux + clamp=0.15 was stable on H9b 18+ epochs — safe stacked
- Watch for τ_z grad norm spikes in EP3-5 (Charbonnier + separate τ head interaction)

If divergence at EP3, fall back to:
- H18-fallback-A: drop MAE_aux (just H10b stack + separate τ head)
- H18-fallback-B: drop Charbonnier (just curvature + separate τ head + MAE_aux)

## Hypothesis statement (for PR body)

**H18 hypothesis**: Stacking the four orthogonal mechanisms validated in the current wave — curvature additive attention bias (H10b), Charbonnier loss on τ_z (H10b), vol_p MAE auxiliary + clamp=0.15 (H9b), and separate τ head architecture (H12) — produces a compound SOTA-beat: test_wss < 6.50% AND test_vol_p < 3.643% AND test_surf_p < 3.577%. Expected ~test_wss 6.40-6.45.

**Rationale**: Each mechanism targets a distinct bottleneck:
- Curvature bias → WSS-side representation upgrade
- Charbonnier → τ_z loss landscape reshape
- MAE_aux + clamp → vol_p floor preservation
- Separate τ head → WSS-channel decoder capacity

**Falsification criterion**: If test_wss > 6.727 (SOTA), the composition is anti-additive (i.e., mechanisms interfere). If test_vol_p > 3.643 or test_surf_p > 3.577, the floor preservation mechanism is overwhelmed by the WSS-stack.

## Assignment plan

- **First idle student**: H9b tanjiro (frees ~12:00Z autonomous EP30 land)
- **Second slot**: H11b nezuko if H11b EP30 is below SOTA-beat (frees ~15:00-16:00Z)
- **Hold for outcome**: Wait for H12 EP10 hard gate verdict (~08:55Z) before committing — if H12 fails EP10 gate, drop separate τ head from H18 spec
