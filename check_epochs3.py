"""
Estimate epoch from step count.
Baseline best run (vu4jsiic) reached ep~45.3 at step 807,025.
Steps per epoch ~= 807025 / 45.3 = ~17,817.
"""
import os

# Steps per epoch baseline estimate
# From best run: ep45.3 @ step 807025 → ~17,817 steps/epoch
# But different runs may have different steps/epoch based on batch size / dataset split

STEPS_PER_EPOCH = 17817  # baseline estimate

RUNS = {
    "hph6eaky": ("fern / PR#409 coord-norm fix", 463709),
    "5ifnf1wc": ("thorfinn / PR#382 6L/512d/8H", 227104),
    "4632xosf": ("kohaku / PR#417 EMA", 440784),
    "0xi2n4oo": ("alphonse / PR#437 6L/256d", 338215),
    "jj9r7x0o": ("senku / PR#442 OHEM", 374155),
    "vyhpqruv": ("tanjiro / PR#443 mirror+SW=2.0", 379800),
}

print(f"{'run_id':<12} {'label':<35} {'_step':>10} {'est_epoch':>10}")
print("-" * 75)
for run_id, (label, step) in RUNS.items():
    est_epoch = step / STEPS_PER_EPOCH
    print(f"{run_id:<12} {label:<35} {step:>10} {est_epoch:>10.1f}")
