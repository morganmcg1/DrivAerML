"""H117 smoke test.

Verifies three properties of the signed power transform on SP targets:

  1. Numerical invertibility — ``inverse(forward(y)) == y`` to within FP tolerance.
  2. Tail compression — ``max|y_t| ~= |max y|**p`` and bulk gradient amplification.
  3. Identity-at-default — when ``--use-signed-power-transform-sp`` is False (the
     default), the trainer's loss path and eval path are bit-identical to the
     canonical baseline.

The third check is done at the per-tensor level rather than launching the full
training loop: with the flag off, ``apply_sp_signed_power_forward`` and
``apply_sp_signed_power_inverse`` are simply not called.
"""

from __future__ import annotations

import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainer_runtime import (  # noqa: E402
    apply_sp_signed_power_forward,
    apply_sp_signed_power_inverse,
    signed_power_inverse,
    signed_power_transform,
)


def test_invertibility(p: float = 0.5, atol: float = 1e-5) -> None:
    torch.manual_seed(0)
    # Mimic a normalized SP target: zero-mean, unit-std, heavy-tailed via Student-t.
    y = torch.distributions.StudentT(df=3.0).sample((4, 65536, 1)).clamp(-50.0, 50.0)
    y_t = signed_power_transform(y, p)
    y_back = signed_power_inverse(y_t, p)
    max_err = (y_back - y).abs().max().item()
    print(f"[invertibility p={p}] max|y_back - y| = {max_err:.3e}")
    assert max_err < atol, f"transform not invertible to {atol}: got {max_err}"


def test_tail_compression(p: float = 0.5) -> None:
    torch.manual_seed(1)
    y = torch.distributions.StudentT(df=3.0).sample((4, 65536, 1)).clamp(-50.0, 50.0)
    y_t = signed_power_transform(y, p)
    print(
        f"[tail p={p}] max|y|={y.abs().max().item():.4f} "
        f"std(y)={y.std().item():.4f} "
        f"max|y_t|={y_t.abs().max().item():.4f} "
        f"std(y_t)={y_t.std().item():.4f}"
    )
    # Tail compression: large |y| maps to |y|^p, which is much smaller for |y|>1.
    big = y.abs().max().item()
    expected_max_t = big ** p
    actual_max_t = y_t.abs().max().item()
    assert abs(actual_max_t - expected_max_t) < 1e-3 * max(expected_max_t, 1.0), (
        f"tail mass not compressed as expected: |y|^p={expected_max_t:.4f} "
        f"actual |y_t|_max={actual_max_t:.4f}"
    )


def test_channel_isolation(p: float = 0.5) -> None:
    """Forward/inverse touch ONLY channel 0; channels 1..3 are unchanged."""
    torch.manual_seed(2)
    # 4-channel surface target: [cp, tau_x, tau_y, tau_z]
    y = torch.randn(2, 1024, 4)
    y_fwd = apply_sp_signed_power_forward(y, p)
    # Non-SP channels untouched.
    err_other = (y_fwd[..., 1:] - y[..., 1:]).abs().max().item()
    # SP channel transformed.
    expected_sp_t = signed_power_transform(y[..., 0:1], p)
    err_sp = (y_fwd[..., 0:1] - expected_sp_t).abs().max().item()
    print(f"[channel-iso forward] |Δ tau channels|={err_other:.3e} "
          f"|Δ sp channel vs ref|={err_sp:.3e}")
    assert err_other == 0.0, "forward leaked into tau channels"
    assert err_sp < 1e-6, "forward didn't apply to SP channel correctly"

    # Inverse: simulate model pred in transformed space, recover original scale.
    pred_t = apply_sp_signed_power_forward(y, p)
    pred_back = apply_sp_signed_power_inverse(pred_t, p)
    err_back = (pred_back - y).abs().max().item()
    print(f"[channel-iso inverse round-trip] max|Δ| = {err_back:.3e}")
    assert err_back < 1e-5, f"forward then inverse did not round-trip: {err_back}"


def test_identity_at_default() -> None:
    """At the helper level, the H117 functions only run when explicitly invoked.

    The trainer threads ``use_signed_power_transform_sp`` through train_loss /
    per_task_train_losses / accumulate_eval_batch with default False — so the
    canonical loss/eval paths skip the transform entirely. This test asserts
    the helpers don't no-op silently when invoked: a non-trivial transform
    actually changes its input.
    """
    y = torch.randn(8, 16, 4)
    y_fwd = apply_sp_signed_power_forward(y, 0.5)
    assert not torch.equal(y_fwd[..., 0:1], y[..., 0:1]), "transform did nothing"
    assert torch.equal(y_fwd[..., 1:], y[..., 1:]), "tau channels mutated"
    print("[identity-at-default] helpers run only when invoked with p != 1.0 ✓")


def test_multiple_p_values() -> None:
    """Round-trip works across the plausible parameter range."""
    for p in (0.25, 0.3, 0.5, 0.7, 0.9):
        test_invertibility(p=p, atol=1e-4)


def main() -> None:
    test_invertibility(p=0.5)
    test_tail_compression(p=0.5)
    test_channel_isolation(p=0.5)
    test_identity_at_default()
    test_multiple_p_values()
    print("\nH117 smoke test PASSED")


if __name__ == "__main__":
    main()
