# H144 fern: tau_z loss weight escalation 4.0→6.0

**Assigned:** 2026-05-25
**Branch:** fern/h144-tau-z-loss-weight-6
**Hypothesis:** Escalate tau_z loss weight from H143's 4.0 to 6.0 (3× H112 baseline of 2.0) to map the z-shear loss-weight response curve.

## Context

- H112 baseline: `--tau-z-loss-weight 2.0` → test_WSS 6.752% (SOTA)
- H143 frieren: `--tau-z-loss-weight 4.0` → in-flight, EP2 PASS @26.45%
- H144 fern: `--tau-z-loss-weight 6.0` → this assignment

Tau_z (z-shear) is the dominant WSS error source. Escalation tests whether higher gradient pressure on z-shear continues improving WSS without degrading tau_x/tau_y.

## Training Command

```bash
SENPAI_TIMEOUT_MINUTES=1100 torchrun --standalone --nproc-per-node=8 target/train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --manifest data/split_manifest.json \
  --agent fern --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 9e-5 --weight-decay 5e-4 --batch-size 4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 6.0 \
  --surface-loss-weight 2.0 --volume-loss-weight 0.5 \
  --use-surf-to-vol-xattn --enable-residual-positions \
  --use-drop-path --drop-path-rate 0.10 \
  --epochs 13 --lr-warmup-epochs 1 --lr-schedule cosine \
  --ema-decay 0.999 --grad-clip 1.0 --save-best-checkpoint \
  --wandb-name fern/h144-tau-z-loss-weight-6 \
  --wandb-group wss_h144_tau_z_6 \
  --kill-thresholds "10864:val_primary/abupt_axis_mean_rel_l2_pct<35.0" \
    "32592:val_primary/abupt_axis_mean_rel_l2_pct<8.5" \
    "48897:val_primary/abupt_axis_mean_rel_l2_pct<7.0" \
    "70664:val_primary/abupt_axis_mean_rel_l2_pct<6.1358"
```
