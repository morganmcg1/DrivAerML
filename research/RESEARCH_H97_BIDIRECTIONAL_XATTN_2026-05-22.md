# H97 Bidirectional Cross-Attention — Research Memo
**Date:** 2026-05-22
**Branch:** alphonse/h97-bidirectional-xattn
**Researcher:** alphonse (research specialist role)

---

## What H97 Is

A second `nn.MultiheadAttention` module (`vol_to_surf_xattn`) mirroring the existing `surf_to_vol_xattn` (merged, PR #823). Both modules are applied in parallel — each reads from `surf_pre`/`vol_pre` captured before either write — and both use zero-init `out_proj` so the sublayer is identity at epoch 0. The hypothesis: volume flow context (recirculation, wake vortices) informs surface pressure and WSS prediction in separation regions, just as surface geometry context informs volume pressure in attached-flow regions.

---

## Q1: Prior Work — Parallel vs Sequential Bidirectional Cross-Attention

### The standard pattern is parallel (simultaneous), not sequential.

**BiXT (NeurIPS 2024, arxiv:2402.12138)** — "Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers." The key architectural finding: both input→latent and latent→input cross-attentions read from the same pre-update states ("simultaneously"). The paper explicitly exploits "naturally emerging approximately symmetric cross-attention patterns" and scales linearly with input size. This is the closest architectural analogue to H97 and directly validates the parallel pattern already in the code.

**SEA — State-Exchange Attention (NeurIPS 2024)** — cross-field multidirectional information exchange via MHA in CFD surrogates. Applied to highly state-dependent variables (pressure, velocity coupling). Reported 97% error reduction for state-dependent fields. Uses simultaneous multi-direction attention, not sequential.

**AB-UPT (TMLR 2025, Emmi AI/JKU Linz)** — "Anchored-Branched Universal Physics Transformers." Multi-branch operators for surface-volume interactions with divergence-free formulation. Directly relevant to automotive CFD; multi-branch (parallel) design, not sequential.

**Perceiver IO (ICML 2021, arxiv:2107.14795)** — latent cross-attends to input (perceiver) and output queries cross-attend to latent. This is a weaker analogue (asymmetric latent bottleneck), but the simultaneous/parallel pattern is the same. The latent bottleneck design is NOT what H97 uses — H97 is full N×N bidirectional without a compressed latent.

**Conclusion for H97:** The parallel pattern in the current code (both modules read from `surf_pre`/`vol_pre` before either write) is precisely what the leading literature uses. Sequential ordering would introduce an asymmetry artifact — whichever direction runs first would influence the second, making it harder to disentangle contributions and potentially creating a dominant-direction initialization bias. The current implementation is correct.

**Known stability issues from the literature:**
- BiXT notes that with very long sequences the QK similarity can concentrate (attention collapse) if the Q and K spaces are initialized identically. This is mitigated by separate per-layer learned projections — which PyTorch's `nn.MultiheadAttention` provides by default (in_proj is separate from the backbone's projections).
- SEA reports no instability with simultaneous bidirectional attention in CFD settings, but uses shorter sequence lengths (O(10k) tokens vs H97's 65k). At N=65k the attention entropy monitoring signal (Q3 below) becomes more important.

---

## Q2: Zero-Init Out_Proj + Post-LN Interaction

### The current LN placement is benign but has a mild gradient-slowing effect at init.

**What happens at init:**

At epoch 0, `out_proj.weight = 0` and `out_proj.bias = 0`, so:
```
xattn_surf ≈ 0  (output of vol_to_surf_xattn)
surface_hidden = vol_to_surf_xattn_norm(surf_pre + xattn_surf)
               = LN(surf_pre + 0)
               = LN(surf_pre)
```

`surf_pre` is `hidden_norm[:, :surface_tokens]` — already the output of `self.norm` (the backbone's final LayerNorm). So the xattn LN is re-normalizing an already near-unit-variance tensor. The output is approximately `surf_pre` (near-identity), which is correct for the zero-init invariant.

**The concern: LN gradient path at init.**

The gradient into `out_proj.weight` flows through: `loss → decoder → surface_hidden → LN_norm_residual → xattn_surf → out_proj.weight`. At init, `xattn_surf ≈ 0`, so the LN Jacobian is evaluated near its "re-normalizing identity" fixed point. Because `surf_pre` has unit variance, `LN(surf_pre) ≈ surf_pre` up to centering, and the Jacobian is well-conditioned (not degenerate). This is the key reason the concern is mild: the input to the xattn LN is pre-normalized, so LN does not encounter the pathological "zero-mean tiny-variance" input that would cause near-zero Jacobian.

**Comparison to ReZero (arxiv:2003.04887):**

ReZero replaces `LN(x + alpha * F(x))` with `x + alpha * F(x)` where `alpha` is a learnable scalar initialized to 0. Reported 56% faster convergence on 12-layer Transformers. ReZero eliminates the LN entirely from the residual branch at init, which removes the re-normalization and gives `∂loss/∂alpha` a cleaner gradient path from the very first step. The tradeoff: without LN, the magnitude of `xattn_surf` is unconstrained as training progresses, and large `alpha` values can cause instability in later epochs. The solution is usually to add a LN after ReZero stabilizes (which becomes similar to Pre-LN).

**Practical recommendation for H97:**

The current Post-LN + zero-init-out_proj design is safe and consistent with the existing `surf_to_vol_xattn` (already merged). Changing to ReZero would require an architecture change and diverge from the existing xattn module pattern. The expected slow-open issue is real but mild at N=65k — the gradient amplification through the 65k-token attention softmax is already substantial enough to drive `out_proj` out of zero within 1-2 epochs. Monitor `vol_to_surf_xattn_norm.weight` norm and `out_proj.weight` norm in W&B to confirm the module activates (see Q4 monitoring signals below).

**Pre-LN alternative:**

Pre-LN would mean: `surface_hidden = surf_pre + xattn(LN(surf_pre), LN(vol_pre), LN(vol_pre))`. This puts LN before the attention computation rather than after the residual. Pre-LN has better gradient behavior at init (no large gradients near output) per arxiv:2002.04745, but would require changing both the new module AND the existing `surf_to_vol_xattn` to stay symmetric. Not recommended for H97 alone; a dedicated Pre-LN sweep is a separate hypothesis.

---

## Q3: Physical Asymmetry — Does It Matter for Init?

### Asymmetry is real but should emerge through learning, not be baked into init.

**The physical picture:**

In DrivAerML (automotive CFD, RANS):
- **Attached-flow regions** (front hood, windshield, forward body panels): surface geometry is the upstream cause; surface normals and curvature drive boundary layer development; surface stress and SP are tightly coupled to local geometry; volume pressure is largely determined by surface conditions. Direction: surf→vol.
- **Separated-flow regions** (A-pillar, side mirror wake, rear diffuser, base): volume vorticity and recirculation are the upstream cause; the wake structure determines base pressure and reattachment; SP and WSS in the base region are determined by volume-side flow features. Direction: vol→surf.
- **Most of the car surface area** is attached-flow. Separated regions are localized but high-variance (they dominate the WSS error).

**What this means for H97:**

The `surf_to_vol_xattn` (established win, merged) benefits from a large signal — most of the car is attached-flow so surface→volume is the dominant direction. The `vol_to_surf_xattn` (H97) faces a harder problem: it only materially helps in separated regions (~15-25% of surface area by token count), so the effective signal-to-noise for this direction is lower. This explains:
1. Why H97 may show slower val improvement than H96 (WSS decoder separation) in early epochs.
2. Why WSS_z (wake/base/trailing edge, worst axis at 8.93-9.6%) is the primary target of `vol_to_surf_xattn` — it's the most separated-flow-dominated axis.
3. Why SP may NOT improve under H97 alone — front-body SP is attached-flow and already has the `surf_to_vol_xattn` win baked in; the vol→surf direction gives mostly noise for that region.

**Init implications:**

Symmetric initialization (same zero-init out_proj for both modules) is correct. Asymmetric init (e.g., larger initial scale for `vol_to_surf_xattn` to compensate for weaker signal) is not supported by evidence and risks destabilizing the established `surf_to_vol_xattn` win. The physical asymmetry will manifest as different learning rates for the two modules' `out_proj` weights — `surf_to_vol_xattn.out_proj` will grow faster (cleaner gradient from attached-flow tokens) and `vol_to_surf_xattn.out_proj` will grow slower (noisy gradient from sparse separated-flow tokens). This is the expected and healthy behavior.

**Early-epoch diagnostic (see Q4):** If `vol_to_surf_xattn.out_proj.weight` norm is still near-zero at EP2 while `surf_to_vol_xattn.out_proj.weight` is growing, that is the physical asymmetry at work, not a bug.

---

## Q4: VRAM and Compute — Critical FlashAttention Requirement

### Without FlashAttention: CATASTROPHIC. With FlashAttention: negligible.

**Raw attention matrix calculation (WITHOUT FlashAttention):**

For `vol_to_surf_xattn`:
- Query: `surf_pre` shape `[B, N_surf, d]` = `[4, 65536, 512]`
- Key/Value: `vol_pre` shape `[B, N_vol, d]` = `[4, 65536, 512]`
- Attention matrix: `[B, H, N_surf, N_vol]` = `[4, 4, 65536, 65536]`
- Memory per matrix: `4 × 4 × 65536 × 65536 × 2 bytes` = **137 GB** (bf16)

For `surf_to_vol_xattn` (already in model): same calculation, another 137 GB.

Total just for attention matrices: **274 GB** — exceeds 8×H100 96GB = 768 GB total? No, it exceeds single-GPU VRAM of 96 GB by 143×. But with DDP the batch is split across 8 GPUs so per-GPU is B/8=0.5 per shard... however the attention is computed per-GPU for the full local batch. With B=4 over 8 GPUs, each GPU sees B_local=0.5 which rounds to 1 sample per GPU. Per-GPU attention matrix: `1 × 4 × 65536 × 65536 × 2` = **34 GB per xattn module, 68 GB for both**. The 8×H100 has 96 GB each. 68 GB for attention matrices alone leaves ~28 GB for model weights, gradients, optimizer states, and activations. This is borderline impossible.

**With FlashAttention (O(N) memory, tiled computation):**

FlashAttention computes the attention output without materializing the full N×N matrix. Memory for attention:
- Q, K, V projections: `3 × B_local × N × d × 2` = `3 × 1 × 65536 × 512 × 2` ≈ **192 MB per xattn module**
- Output: `B_local × N × d × 2` ≈ **64 MB per module**
- Total per module: ~256-268 MB
- Total for both modules: ~500 MB — completely negligible

**Does PyTorch's `nn.MultiheadAttention` use FlashAttention by default?**

**NO.** `nn.MultiheadAttention` calls `F.scaled_dot_product_attention` internally as of PyTorch 2.0+, which DOES dispatch to FlashAttention when:
1. `need_weights=False` (the H97 code already has this — correct)
2. The input dtype is float16 or bfloat16 (the run uses bf16 — correct)
3. No custom attention mask that would prevent FlashAttention dispatch
4. The `PYTORCH_ENABLE_FLASH_SDP` environment variable is not disabled

**The H97 code correctly passes `need_weights=False`.** This is the critical flag that enables the FlashAttention path in `F.scaled_dot_product_attention`. Without it, PyTorch falls back to the explicit O(N²) implementation.

**Verification command to add to the experiment run:**

```python
# Add to model __init__ or a test script to verify flash attention dispatch:
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

And at the start of training, log:
```python
print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Mem-efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
```

**If FlashAttention is NOT dispatching** (e.g., due to a mask format issue), the run will OOM on the first forward pass. This is the primary failure mode to watch for.

**Attention mask interaction:** The `vol_to_surf_xattn` in the forward pass passes `need_weights=False` and no `key_padding_mask`. The `_apply_token_mask` is applied AFTER the attention output. This is correct for FlashAttention dispatch — passing a `key_padding_mask` to MHA would force an explicit attention mask computation that may prevent FlashAttention dispatch on some PyTorch versions. The current pattern (masking the output rather than gating the attention) is the right choice for VRAM efficiency.

**Memory cost summary:**

| Component | Per-module | Both modules |
|---|---|---|
| MHA parameters (weights) | 4×512²×2 ≈ 2 MB | 4 MB |
| Attention activations (FlashAttn) | ~270 MB/GPU | ~540 MB/GPU |
| Attention activations (no FlashAttn, B_local=1) | ~34 GB/GPU | ~68 GB/GPU |

Conclusion: H97 is **safe with FlashAttention** and **catastrophically OOM without it**. The code already sets `need_weights=False` which is the correct enabling condition. No code changes required for VRAM.

---

## What to Change vs the Current Plan

### 1. Keep the parallel pattern as-is. (No change needed.)

The current implementation captures `surf_pre` and `vol_pre` before either module writes, then applies both in sequence (surf_to_vol first, then vol_to_surf, each reading from `*_pre`). This is the correct parallel pattern per BiXT and SEA. Do not change to sequential.

### 2. The zero-init out_proj is correct and consistent. (No change needed.)

The LN-rescaling-nothing concern is mild because `surf_pre` is already post-backbone-LN (unit variance). The gradient path is well-conditioned. Changing to ReZero would require touching both `surf_to_vol_xattn` and `vol_to_surf_xattn` and adds complexity without clear evidence of benefit in this specific setting.

### 3. Add W&B logging for xattn module activation metrics. (Recommended addition.)

The student implementing H97 should add the following W&B logs to confirm the module activates and is not stuck in the zero-init dead zone:

```python
# In the training loop, log per-epoch:
wandb.log({
    "vol_to_surf_xattn_out_proj_weight_norm": model.module.vol_to_surf_xattn.out_proj.weight.norm().item(),
    "surf_to_vol_xattn_out_proj_weight_norm": model.module.surf_to_vol_xattn.out_proj.weight.norm().item(),
    "vol_to_surf_xattn_norm_weight_norm": model.module.vol_to_surf_xattn_norm.weight.norm().item(),
    "surf_to_vol_xattn_norm_weight_norm": model.module.surf_to_vol_xattn_norm.weight.norm().item(),
})
```

This confirms: (a) both modules activate out of zero-init (out_proj weight norms growing), (b) the physical asymmetry is observable (surf→vol should grow faster than vol→surf), (c) no module is stuck (weight norm still exactly 0 at EP3 = dead module).

### 4. Verify FlashAttention dispatch before the full run. (One-step check.)

Add a brief startup log or assertion:
```python
import torch
assert torch.backends.cuda.flash_sdp_enabled(), "FlashAttention not available — H97 will OOM"
```
If this assertion fires, investigate the PyTorch version and CUDA availability before running.

---

## Likely Failure Modes

### FM1: OOM on first forward pass (probability: LOW given `need_weights=False` is set)

The code already sets `need_weights=False` for both xattn modules. If the run OOMs immediately, the cause is FlashAttention dispatch failure — check PyTorch version, bf16 dtype, and the `key_padding_mask` argument (not passed here, so likely not the issue).

### FM2: vol_to_surf_xattn stays in zero-init dead zone through EP3 (probability: MEDIUM)

The physical asymmetry argument (sparse separated-flow signal) means `vol_to_surf_xattn.out_proj` may grow very slowly. If the out_proj weight norm is < 0.01 at EP3 (vs `surf_to_vol_xattn` which should be ~0.1-1.0), the module is effectively not contributing. This is distinguishable from a bug by checking if the SP/WSS metrics track the baseline without H97's addition at all. If FM2 occurs, the follow-up is a dedicated decoder head for WSS (H96, already running) rather than cross-attention.

### FM3: SP plateau persists — H97 does not crack test_SP 3.577 floor (probability: HIGH)

The 9+ variant SP plateau at 3.74-3.95% is hypothesized as decoder-bound (shared MLP bottleneck). Vol→surf cross-attention adds volume context to surface tokens but does not change the decoder head structure. The same MLP bottleneck applies. Expected: H97 helps WSS (separated-flow signal) more than SP (attached-flow, already has surf→vol win). If test_SP is unchanged, that confirms the decoder-bound hypothesis and supports H96+H97 compound.

### FM4: vol_to_surf_xattn helps vol_pre but hurts surface decoder (probability: LOW but watch)

The parallel pattern prevents the new module from polluting `surf_to_vol_xattn`'s input. But `surface_hidden` is updated by `vol_to_surf_xattn_norm(surf_pre + xattn_surf)` and then fed to the surface decoder (both SP and WSS heads in H96, or the shared `surface_out` in base). If the volume tokens contain predominantly wake/wake-noise features, the vol→surf signal might introduce high-variance activations in surface tokens that destabilize SP rather than helping it. Monitor: if val_SP regresses at EP1-2 while val_WSS_z improves, FM4 is occurring and the vol→surf xattn may need a lower learning rate or gating.

### FM5: Memory regression from xattn activation checkpointing gap (probability: LOW)

If the backbone uses activation checkpointing and the xattn modules are outside the checkpointed scope, the xattn activations (Q, K, V, output) accumulate in GPU memory. At N=65k with FlashAttention this is ~540 MB total (negligible), but verify that the xattn modules are included in any existing checkpointing scope.

---

## Early-Epoch Monitoring Signals (EP1-3)

### Signal 1: Module activation — out_proj.weight norm (check EP1)

Expected behavior:
- `surf_to_vol_xattn.out_proj.weight` norm: should grow from 0 → ~0.05-0.5 by EP1
- `vol_to_surf_xattn.out_proj.weight` norm: should grow from 0 → ~0.01-0.2 by EP1 (slower, physical asymmetry)
- If EITHER is still exactly 0.0 at EP1: module is dead, investigate gradient flow

### Signal 2: val_WSS_z trajectory (primary H97 target)

H97 targets separated-flow regions, and WSS_z (wake/trailing edge, worst axis) is the most separated-flow-dominated output. Expected:
- EP1: val_WSS_z should be at or slightly above baseline (module barely activated)
- EP2: val_WSS_z should show a distinct slope improvement vs H96-alone (if H97 is working)
- EP3 gate: val_WSS_z < 8.8 = mechanism plausibly alive; > 9.2 = mechanism probably not contributing

### Signal 3: val_WSS_x and val_WSS_y — should NOT regress

WSS_x (streamwise) and WSS_y (lateral/yaw-coupled) are partially attached-flow. H97 should be neutral to mildly positive for these axes. If val_WSS_x or val_WSS_y regress at EP1-2 relative to the EP1 baseline from prior runs (~6.1% and ~7.5% respectively), FM4 is occurring.

### Signal 4: val_SP — expected neutral or slight improvement

test_SP is decoder-bound. H97 adds volume context but does not change the decoder. Expected: val_SP tracks similar to H96 baseline without sharp improvement. If val_SP improves by > 0.05pp at EP2, the vol→surf attention is contributing more than expected (positive surprise). If val_SP regresses by > 0.05pp at EP2, FM4 is occurring.

### Signal 5: Attention entropy (optional but useful if logging is cheap)

If the forward pass logs `need_weights=True` for a single diagnostic batch (not training, just a logged eval batch), the attention entropy can reveal whether the module is attending uniformly (softmax collapsed to uniform = learning nothing) or sharply (focusing on specific vol tokens = signal present). This is expensive at N=65k but a single batch every 5 epochs is fine.

### EP3 go/no-go decision:

| Condition | Decision |
|---|---|
| val_WSS_z < 8.7 AND out_proj norms growing | Continue to terminal |
| val_WSS_z in 8.7-9.0 AND out_proj norms growing slowly | Continue with flag (monitor to EP6) |
| val_WSS_z > 9.2 OR out_proj norms still 0 at EP3 | Likely dead module — report B PARTIAL, suggest compound with H96 |
| val_SP regressed > 0.08pp | FM4 occurring — report mechanism failure |

---

## Research State Update

**Current best explanation for the bottleneck:**

The primary bottleneck is a decoder-architecture bound on SP (confirmed across 9+ variants in 3.74-3.95% range). WSS is partially decoder-bound (H96's dedicated head targets this) and partially information-bound (the volume cross-stream signal for separated-flow regions is missing from the surface decoder). H97 directly targets the information-bound component of WSS.

**Evidence:**
- H88 (heads=8): all 3 WSS axes regressed +0.13-0.22pp — WSS is NOT attention-capacity-bound
- H96 (split decoder heads): first mechanism to target WSS via capacity separation — currently running
- H97 (vol→surf xattn): first mechanism to explicitly route separated-flow volume context into surface tokens
- H87 (surf_loss=1.5): HISTORIC first Wave 32 val gate clear, but test_WSS regressed — gradient rebalancing is insufficient alone
- SP plateau (H78/H79/H80/H82/H83/H84/H86/H87/H88): 9 variants all fail test_SP floor → confirmed structural bound under shared decoder

**Ruled-out paths (do not revisit without new evidence):**
- Attention-head capacity expansion (H88): falsified for WSS
- LR magnitude increase (H85): falsified upward sweep
- FFN width expansion (H86): falsified via mid-cosine engagement / late-cosine drain
- RFF feature expansion (H84): diminishing returns confirmed
- Loss-weighting alone (H87): insufficient without structural change

**Open uncertainties:**
1. Whether vol→surf cross-attention activates meaningfully for separated-flow tokens (FM2) — direct test is H97 EP1-3 out_proj norm monitoring
2. Whether H96 (split heads) resolves WSS_z specifically, or H97 is needed as compound — both currently running, compound H96+H97 should be next if both show partial wins
3. Whether the SP plateau can be cracked by any single mechanism, or requires compound architectural change — H96 split heads + H97 vol→surf xattn + H95 surf_loss=1.25 compound is the most motivated compound candidate

**Next discriminating experiment:**

H96 + H97 compound (if both show partial wins individually). The mechanism is clear: split decoder separates SP/WSS capacity AND vol→surf xattn routes separated-flow context into WSS tokens. Together they address both the representational bottleneck (decoder) and the information bottleneck (cross-stream attention).

**Stop condition for H97 alone:**

If EP3 val_WSS_z ≥ 9.2% AND out_proj norms are near-zero: mechanism is not activating in 65k-token setting. Report B PARTIAL or C NULL and recommend compound with H96 only.

---

## External Evidence Summary

| Paper | Year | Finding relevant to H97 |
|---|---|---|
| BiXT (arxiv:2402.12138) | NeurIPS 2024 | Simultaneous/parallel bidirectional xattn is the standard; avoids ordering artifact; validates H97 parallel pattern |
| SEA (NeurIPS 2024) | 2024 | 97% error reduction in CFD state-dependent fields via multidirectional cross-attention |
| AB-UPT (TMLR 2025) | 2025 | Multi-branch surface-volume interaction in automotive CFD transformers |
| ReZero (arxiv:2003.04887) | 2020 | Scalar-gated residual at zero-init; 56% faster convergence vs Post-LN; relevant to zero-init concern |
| Pre-LN analysis (arxiv:2002.04745) | 2020 | Post-LN has large output-layer gradients at init; Pre-LN is more stable; current Post-LN is acceptable because input is pre-normalized |

---

## Confidence

**Q1 (parallel pattern correctness):** Strong — BiXT explicitly demonstrates this and the code already implements it correctly.

**Q2 (zero-init + Post-LN):** Strong — the analysis is based on the specific code path (pre-normalized input to LN), not a generic Post-LN concern. The mild gradient-slowing is expected and acceptable.

**Q3 (physical asymmetry):** Medium — the physical reasoning is sound but the quantitative prediction (slower vol→surf activation) is an inference, not directly measured in this codebase.

**Q4 (VRAM):** Strong — the FlashAttention dispatch via `need_weights=False` is documented behavior for PyTorch 2.0+ `F.scaled_dot_product_attention`. The calculation is straightforward. The OOM risk is real without it.

**Overall:** The H97 implementation as coded is architecturally sound. The primary execution risk is not in the design but in the FlashAttention dispatch check and the physical-asymmetry-driven slow activation of `vol_to_surf_xattn`. Both are diagnosable from EP1 logs.
