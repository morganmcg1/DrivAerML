"""Focused H95 W&B query for status checks."""
import wandb

api = wandb.Api()
run = api.run("wandb-applied-ai-team/senpai-v1-drivaerml-ddp8/ze0bohdu")
s = run.summary_metrics

print(f"State: {run.state}")
print(f"Step: {s.get('_step')} / 70664")
rt = s.get('_runtime', 0)
print(f"Runtime: {rt/3600:.2f}h" if rt else "Runtime: unknown")

print("\n=== Latest val/val_surface (primary) ===")
for k in ['abupt_axis_mean_rel_l2_pct', 'surface_pressure_rel_l2_pct',
          'volume_pressure_rel_l2_pct', 'wall_shear_rel_l2_pct',
          'wall_shear_x_rel_l2_pct', 'wall_shear_y_rel_l2_pct',
          'wall_shear_z_rel_l2_pct']:
    full = f'val/val_surface/{k}'
    v = s.get(full)
    if v is not None:
        print(f"  {k}: {v:.4f}")

print("\n=== Best checkpoint snapshot ===")
for k in sorted(s.keys()):
    if 'best_checkpoint' in k and isinstance(s.get(k), (int, float)):
        print(f"  {k}: {s.get(k)}")

print("\n=== test/* metrics (if any) ===")
test_keys = [k for k in s.keys() if k.startswith('test')]
for k in sorted(test_keys):
    v = s.get(k)
    if isinstance(v, (int, float)):
        print(f"  {k}: {v}")
if not test_keys:
    print("  (none yet — test eval only runs at end)")

print("\n=== loss/* snapshot ===")
for k in sorted(s.keys()):
    if k.startswith('loss/') and isinstance(s.get(k), (int, float)):
        print(f"  {k}: {s.get(k):.6f}")

print("\n=== NaN counts ===")
for k in sorted(s.keys()):
    if 'nan' in k.lower() and isinstance(s.get(k), (int, float)) and s.get(k):
        print(f"  {k}: {s.get(k)}")

print("\n=== Last grad/global_norm ===")
for k in sorted(s.keys()):
    if 'global_norm' in k and k.startswith('train/grad'):
        print(f"  {k}: {s.get(k)}")
