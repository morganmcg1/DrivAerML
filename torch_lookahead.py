# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lookahead optimizer wrapper (Zhang et al. 2019, arxiv 1907.08610).

Maintains slow weights `theta_slow` and fast weights `theta_fast`. After every
`k` inner-optimizer steps on `theta_fast`, applies
    theta_slow <- theta_slow + alpha * (theta_fast - theta_slow)
    theta_fast <- theta_slow                               (re-anchor)

This is NOT weight averaging (EMA / SWA / Polyak). The fast weights are reset
into the slow basin every k steps, actively re-shaping the trajectory the base
optimizer follows.
"""
from collections import defaultdict
from typing import Callable, Optional

import torch
from torch.optim import Optimizer


class Lookahead(Optimizer):
    def __init__(self, base_optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha (slow-weight coef): {alpha}")
        if k < 1:
            raise ValueError(f"Invalid k (inner steps): {k}")
        self.optimizer = base_optimizer
        self.k = int(k)
        self.alpha = float(alpha)
        self.step_counter = 0
        self.outer_step_counter = 0
        self.state = defaultdict(dict)
        # Share param_groups (and defaults) with the base optimizer by reference
        # so LR schedulers, GradNorm, and per-group LR setters all transparently
        # see the live param groups.
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        for group in self.param_groups:
            for p in group["params"]:
                slow = p.detach().clone()
                slow.requires_grad = False
                self.state[p]["slow_weight"] = slow

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            self.outer_step_counter += 1
            for group in self.param_groups:
                for p in group["params"]:
                    slow = self.state[p]["slow_weight"]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        slow_weights = []
        for group in self.param_groups:
            for p in group["params"]:
                slow_weights.append(self.state[p]["slow_weight"].detach().clone())
        return {
            "base_state": self.optimizer.state_dict(),
            "lookahead": {
                "step_counter": self.step_counter,
                "outer_step_counter": self.outer_step_counter,
                "k": self.k,
                "alpha": self.alpha,
                "slow_weights": slow_weights,
            },
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["base_state"])
        lh = state_dict["lookahead"]
        self.step_counter = int(lh["step_counter"])
        self.outer_step_counter = int(lh.get("outer_step_counter", 0))
        self.k = int(lh["k"])
        self.alpha = float(lh["alpha"])
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["slow_weight"].copy_(lh["slow_weights"][idx])
                idx += 1
