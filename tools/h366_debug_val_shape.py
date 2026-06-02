"""H366 debug: reproduce the EP14 val shape mismatch.

Build a real val loader, iterate one batch through the model with kNN bias
enabled, and print the shapes of surface_x, volume_x, knn_idx, hidden, and
slice_logits inside the first attention layer.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from model import SurfaceTransolver
from data.loader import load_data, pad_collate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from data.loader import DEFAULT_MANIFEST
    _, val_splits, _, _ = load_data(
        manifest_path=str(DEFAULT_MANIFEST),
        root=None,
        train_surface_points=40_000,
        eval_surface_points=40_000,
        train_volume_points=40_000,
        eval_volume_points=40_000,
        debug=False,
    )
    val_ds = val_splits["val_surface"]
    print(f"val_ds len = {len(val_ds)}")

    import sys as _sys
    bs = int(_sys.argv[1]) if len(_sys.argv) > 1 else 2
    print(f"batch_size = {bs}")
    loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=pad_collate, num_workers=0)

    model = SurfaceTransolver(
        n_layers=5,
        n_hidden=512,
        dropout=0.0,
        n_head=4,
        mlp_ratio=4,
        slice_num=128,
        rff_num_features=16,
        rff_sigma=1.0,
        rff_init_sigmas=[0.25, 0.5, 1.0, 2.0, 4.0],
        pos_encoding_mode="string_separable",
        use_qk_norm=True,
        use_surf_to_vol_xattn=True,
        drop_path_max=0.10,
        use_knn_attention_bias=True,
        knn_attention_k=32,
    ).to(device)
    model.eval()


    n_done = 0
    seen_shapes = {}
    fail_examples = []
    fail_count = 0
    for batch in loader:
        # Probe each unique shape pattern through the model once
        sx2 = tuple(batch.surface_x.shape)
        vx2 = tuple(batch.volume_x.shape)
        sm2 = tuple(batch.surface_mask.sum(dim=1).tolist())
        key2 = (sx2, vx2, sm2)
        if key2 not in fail_examples and len(fail_examples) < 200:
            fail_examples.append(key2)
            batch_d = batch.to(device)
            try:
                with torch.no_grad():
                    model(
                        surface_x=batch_d.surface_x,
                        surface_mask=batch_d.surface_mask,
                        volume_x=batch_d.volume_x,
                        volume_mask=batch_d.volume_mask,
                    )
            except Exception as e:
                fail_count += 1
                if fail_count <= 8:
                    print(f"FAIL on {sx2} surface_mask={sm2}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
        sx = tuple(batch.surface_x.shape)
        vx = tuple(batch.volume_x.shape)
        sm = tuple(batch.surface_mask.sum(dim=1).tolist())
        vm = tuple(batch.volume_mask.sum(dim=1).tolist())
        key = (sx, vx, sm, vm)
        n_done += 1
        if key in seen_shapes:
            seen_shapes[key] += 1
            continue
        seen_shapes[key] = 1
        if n_done % 500 == 1:
            print(f"  ...{n_done} batches scanned, {len(seen_shapes)} unique shape patterns")
    print(f"\nDone scanning. {n_done} batches, {len(seen_shapes)} unique shape patterns.")
    print("\n--- Unique shape patterns ---")
    for key, count in sorted(seen_shapes.items(), key=lambda x: -x[1]):
        sx, vx, sm, vm = key
        print(f"  count={count}: surface_x={sx} (mask sums={sm}), volume_x={vx} (mask sums={vm})")


if __name__ == "__main__":
    main()
