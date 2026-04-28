# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Validate packaged DrivAerML arrays and write a point-count cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.loader import DEFAULT_MANIFEST, DrivAerMLCaseStore
from data.split_utils import write_json

DEFAULT_OUTPUT = Path("/mnt/new-pvc/datasets/drivaerml/point_counts.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--root", default="")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    store = DrivAerMLCaseStore(
        manifest_path=args.manifest,
        root=args.root or None,
    )
    payload: dict[str, dict[str, int]] = {}
    all_ids = []
    for split in ("train", "val", "test"):
        all_ids.extend(store.case_ids(split))
    if args.limit > 0:
        all_ids = all_ids[: args.limit]

    for i, case_id in enumerate(all_ids, start=1):
        payload[case_id] = store.case_point_counts(case_id)
        if i % 25 == 0 or i == len(all_ids):
            print(f"Checked {i}/{len(all_ids)} cases")

    write_json(args.out, {"root": str(store.root), "counts": payload})
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
