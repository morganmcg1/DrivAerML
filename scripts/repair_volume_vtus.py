# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Audit and repair DrivAerML volume VTU files split across Hugging Face parts."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VolumeParts:
    run_id: int
    run_dir: Path
    merged: Path
    parts: list[Path]

    @property
    def merged_size(self) -> int | None:
        return self.merged.stat().st_size if self.merged.exists() else None

    @property
    def parts_size(self) -> int:
        return sum(path.stat().st_size for path in self.parts)

    @property
    def needs_repair(self) -> bool:
        return bool(self.parts) and self.merged_size != self.parts_size


def find_volume_parts(raw_root: Path) -> list[VolumeParts]:
    volumes: list[VolumeParts] = []
    for run_dir in sorted(raw_root.glob("run_*"), key=lambda path: int(path.name.split("_")[1])):
        run_id = int(run_dir.name.split("_")[1])
        parts = sorted(run_dir.glob(f"volume_{run_id}.vtu.*.part"))
        if not parts:
            continue
        volumes.append(
            VolumeParts(
                run_id=run_id,
                run_dir=run_dir,
                merged=run_dir / f"volume_{run_id}.vtu",
                parts=parts,
            )
        )
    return volumes


def hardlink_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        dst_root = dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)
        for directory in dirs:
            (dst_root / directory).mkdir(exist_ok=True)
        for filename in files:
            src_path = root_path / filename
            dst_path = dst_root / filename
            if src_path.is_symlink():
                os.symlink(os.readlink(src_path), dst_path)
            else:
                os.link(src_path, dst_path)


def concat_parts(volume: VolumeParts) -> None:
    if not volume.parts:
        raise FileNotFoundError(f"No parts found for {volume.merged}")
    tmp = volume.merged.with_suffix(volume.merged.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    if volume.merged.exists():
        volume.merged.unlink()
    with tmp.open("wb") as out:
        for part in volume.parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, out, length=64 * 1024 * 1024)
    tmp.replace(volume.merged)
    if volume.merged.stat().st_size != volume.parts_size:
        raise IOError(
            f"Repaired size mismatch for {volume.merged}: "
            f"{volume.merged.stat().st_size} vs {volume.parts_size}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="Existing raw DrivAerML root.")
    parser.add_argument("--fixed-root", default="", help="Optional new root to hardlink-copy before repair.")
    parser.add_argument("--repair", action="store_true", help="Concatenate mismatched volume_i.vtu files.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    if args.fixed_root:
        fixed_root = Path(args.fixed_root)
        if not fixed_root.exists():
            hardlink_tree(raw_root, fixed_root)
        root = fixed_root
    else:
        root = raw_root

    volumes = find_volume_parts(root)
    mismatched = [volume for volume in volumes if volume.needs_repair]
    for volume in volumes:
        status = "BAD" if volume.needs_repair else "ok"
        print(
            f"{status} run_{volume.run_id}: "
            f"merged={volume.merged_size} parts={volume.parts_size} n_parts={len(volume.parts)}"
        )

    if args.repair:
        for volume in mismatched:
            concat_parts(volume)
            print(f"repaired run_{volume.run_id}")
        remaining = [volume for volume in find_volume_parts(root) if volume.needs_repair]
        if remaining:
            raise RuntimeError(f"Remaining mismatched VTUs: {[v.run_id for v in remaining]}")


if __name__ == "__main__":
    main()
