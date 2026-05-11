# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Build `data/split_manifest.json` from the processed DrivAerML PVC manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from split_utils import expand_pvc_candidates, ensure_disjoint, rewrite_under_pvc_mount, write_json
else:
    from data.split_utils import expand_pvc_candidates, ensure_disjoint, rewrite_under_pvc_mount, write_json

DEFAULT_CASE_MANIFEST = "/mnt/pvc/Processed/drivaerml_processed_fixed_20260511/manifest.csv"
DEFAULT_CASE_MANIFEST_FULL = "/mnt/pvc/Processed/drivaerml_processed_fixed_20260511/manifest_full_failed10_included.csv"
DEFAULT_CASE_ROOT = "/mnt/pvc/Processed/drivaerml_processed_fixed_20260511"
DEFAULT_CASE_ROOT_CANDIDATES = [
    "/mnt/pvc/Processed/drivaerml_processed_fixed_20260511",
    "/mnt/new-pvc/Processed/drivaerml_processed_fixed_20260511",
]
DEFAULT_OUTPUT = Path(__file__).with_name("split_manifest.json")


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_manifest(
    case_manifest_path: str,
    case_full_manifest_path: str,
    case_root: str,
    case_root_candidates: list[str],
    *,
    case_manifest_read_path: str | None = None,
    case_full_manifest_read_path: str | None = None,
) -> dict:
    case_rows = _read_csv_rows(case_manifest_read_path or case_manifest_path)
    case_full_rows = _read_csv_rows(case_full_manifest_read_path or case_full_manifest_path)

    case_splits = {
        split: [row["case_id"] for row in case_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    current_ids = {row["case_id"] for row in case_rows}
    full_ids = {row["case_id"] for row in case_full_rows}
    excluded_case_ids = sorted(full_ids - current_ids)

    return {
        "dataset": "DrivAerML",
        "manifest_version": 2,
        "source_case_manifest": case_manifest_path,
        "source_case_manifest_full": case_full_manifest_path,
        "case_root": case_root,
        "case_root_candidates": case_root_candidates,
        "case_splits": case_splits,
        "case_split_counts": {k: len(v) for k, v in case_splits.items()},
        "excluded_case_ids": excluded_case_ids,
        "excluded_case_count": len(excluded_case_ids),
        "notes": [
            "Case splits come directly from the packaged processed manifest.csv.",
            "The same train/val/test case IDs are used for both surface and volume fields.",
            "manifest_full_failed10_included.csv is used to verify repaired public cases are present.",
        ],
    }


def verify_manifest(manifest: dict) -> None:
    case_counts = manifest["case_split_counts"]
    if case_counts != {"train": 400, "val": 34, "test": 50}:
        raise ValueError(f"Unexpected DrivAerML case split counts: {case_counts}")
    ensure_disjoint(manifest["case_splits"])
    if manifest["excluded_case_count"] != 0:
        raise ValueError(f"Expected 0 excluded DrivAerML cases, got {manifest['excluded_case_count']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", default=DEFAULT_CASE_MANIFEST)
    parser.add_argument("--case-manifest-full", default=DEFAULT_CASE_MANIFEST_FULL)
    parser.add_argument("--case-root", default=DEFAULT_CASE_ROOT)
    parser.add_argument("--case-root-candidate", action="append", default=[])
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    case_root_candidates = args.case_root_candidate or DEFAULT_CASE_ROOT_CANDIDATES
    manifest = build_manifest(
        case_manifest_path=str(args.case_manifest),
        case_full_manifest_path=str(args.case_manifest_full),
        case_root=str(args.case_root),
        case_root_candidates=expand_pvc_candidates(case_root_candidates),
        case_manifest_read_path=str(rewrite_under_pvc_mount(args.case_manifest)),
        case_full_manifest_read_path=str(rewrite_under_pvc_mount(args.case_manifest_full)),
    )
    verify_manifest(manifest)
    write_json(args.out, manifest)
    print(f"Wrote {args.out}")
    print("Case splits:", manifest["case_split_counts"])
    print("Excluded cases:", manifest["excluded_case_count"], manifest["excluded_case_ids"])


if __name__ == "__main__":
    main()
