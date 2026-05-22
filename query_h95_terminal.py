"""Terminal H95 results collection — comprehensive comparison vs H87 and baseline.

H95: surface_loss_weight=1.25 (ze0bohdu)
H87: surface_loss_weight=1.5 (jpspxktf) — previous best
#972 baseline: canonical surface_loss_weight=2.0 (56bcqp3m)
"""
import wandb
import numpy as np

api = wandb.Api()
ENTITY = "wandb-applied-ai-team"
PROJECT = "senpai-v1-drivaerml-ddp8"

H95_ID = "ze0bohdu"
H87_ID = "jpspxktf"
BASELINE_ID = "56bcqp3m"


def fetch(run_id):
    return api.run(f"{ENTITY}/{PROJECT}/{run_id}")


def get_summary(run, prefix=""):
    s = run.summary_metrics
    return {k: v for k, v in s.items() if (prefix == "" or k.startswith(prefix))}


def print_section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


run_h95 = fetch(H95_ID)
print_section(f"H95 RUN STATE — {H95_ID}")
print(f"State: {run_h95.state}")
s95 = run_h95.summary_metrics
print(f"Step: {s95.get('_step')} / 70664")
rt = s95.get('_runtime', 0)
if rt:
    print(f"Runtime: {rt / 3600:.2f}h")

# Terminal val/val_surface metrics (primary)
print_section("H95 — LATEST val/val_surface metrics")
val_keys = [
    'abupt_axis_mean_rel_l2_pct',
    'surface_pressure_rel_l2_pct',
    'volume_pressure_rel_l2_pct',
    'wall_shear_rel_l2_pct',
    'wall_shear_x_rel_l2_pct',
    'wall_shear_y_rel_l2_pct',
    'wall_shear_z_rel_l2_pct',
]
for k in val_keys:
    full = f'val/val_surface/{k}'
    v = s95.get(full)
    if v is not None:
        print(f"  val/val_surface/{k}: {v:.4f}")

print_section("H95 — TEST metrics (terminal only)")
test_keys_found = [k for k in s95.keys() if k.startswith('test')]
if test_keys_found:
    for k in sorted(test_keys_found):
        v = s95.get(k)
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
else:
    print("  (no test metrics yet — only present at terminal)")

print_section("H95 — Loss bucket effective magnitudes (mechanism check)")
loss_keys = ['loss/surface_loss', 'loss/volume_loss', 'loss/total_loss',
             'loss/surface_loss_weight', 'loss/volume_loss_weight',
             'loss/tau_y_loss_weight', 'loss/tau_z_loss_weight']
for k in loss_keys:
    v = s95.get(k)
    if v is not None:
        print(f"  {k}: {v}")
sl = s95.get('loss/surface_loss')
vl = s95.get('loss/volume_loss')
if sl and vl:
    print(f"  → surface_loss / volume_loss = {sl / vl:.3f}x (expected ~1.25x for H95)")

print_section("H95 — NaN counts / stability")
nan_keys = [k for k in s95.keys() if 'nan' in k.lower() and isinstance(s95.get(k), (int, float)) and s95.get(k)]
if nan_keys:
    for k in nan_keys:
        print(f"  {k}: {s95.get(k)}")
else:
    print("  No NaN events recorded ✓")

print_section("H95 — final grad norms")
gn_keys = ['train/grad/global_norm', 'train/grad/global_norm_pre_clip']
for k in gn_keys:
    v = s95.get(k)
    if v is not None:
        print(f"  {k}: {v:.4f}")

# ===== H87 baseline comparison =====
print_section(f"H87 SIBLING ({H87_ID}) — surface_loss_weight=1.5 — for direct comparison")
try:
    run_h87 = fetch(H87_ID)
    s87 = run_h87.summary_metrics
    print(f"State: {run_h87.state}")
    print("Val/val_surface terminal:")
    for k in val_keys:
        full = f'val/val_surface/{k}'
        v = s87.get(full)
        if v is not None:
            print(f"  val/val_surface/{k}: {v:.4f}")
    print("Test:")
    test_h87 = [k for k in s87.keys() if k.startswith('test')]
    for k in sorted(test_h87):
        v = s87.get(k)
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
    print("Loss buckets:")
    for k in ['loss/surface_loss', 'loss/volume_loss']:
        v = s87.get(k)
        if v is not None:
            print(f"  {k}: {v}")
    sl87 = s87.get('loss/surface_loss')
    vl87 = s87.get('loss/volume_loss')
    if sl87 and vl87:
        print(f"  → H87 ratio surface/volume = {sl87 / vl87:.3f}x")
except Exception as e:
    print(f"  Failed to load H87: {e}")

# ===== #972 baseline comparison =====
print_section(f"#972 baseline ({BASELINE_ID}) — surface_loss_weight=2.0 — canonical")
try:
    run_base = fetch(BASELINE_ID)
    sb = run_base.summary_metrics
    print(f"State: {run_base.state}")
    print("Val/val_surface terminal:")
    for k in val_keys:
        full = f'val/val_surface/{k}'
        v = sb.get(full)
        if v is not None:
            print(f"  val/val_surface/{k}: {v:.4f}")
    print("Test:")
    test_base = [k for k in sb.keys() if k.startswith('test')]
    for k in sorted(test_base):
        v = sb.get(k)
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
except Exception as e:
    print(f"  Failed to load baseline: {e}")

# ===== Slope analysis from history =====
print_section("H95 — val_abupt history (last 12 val reads)")
try:
    hist = run_h95.history(
        keys=['_step', 'val/val_surface/abupt_axis_mean_rel_l2_pct',
              'val/val_surface/surface_pressure_rel_l2_pct',
              'val/val_surface/volume_pressure_rel_l2_pct',
              'val/val_surface/wall_shear_rel_l2_pct'],
        pandas=False,
    )
    seen = [h for h in hist if h.get('val/val_surface/abupt_axis_mean_rel_l2_pct') is not None]
    for h in seen[-12:]:
        st = h.get('_step')
        ab = h.get('val/val_surface/abupt_axis_mean_rel_l2_pct')
        sp = h.get('val/val_surface/surface_pressure_rel_l2_pct')
        vp = h.get('val/val_surface/volume_pressure_rel_l2_pct')
        ws = h.get('val/val_surface/wall_shear_rel_l2_pct')
        print(f"  step={st}  abupt={ab:.3f}  SP={sp:.3f}  VP={vp:.3f}  WSS={ws:.3f}")
except Exception as e:
    print(f"  Failed: {e}")
