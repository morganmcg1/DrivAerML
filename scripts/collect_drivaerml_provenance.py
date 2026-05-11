# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Collect per-case DrivAerML preprocessing sidecars into root manifests."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", default="")
    parser.add_argument("--raw-root", default="")
    parser.add_argument("--out", default="volume_manifest.csv")
    args = parser.parse_args()

    case_rows = read_csv_rows(args.root / "manifest.csv")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for case in case_rows:
        case_id = case["case_id"]
        sidecar = args.root / case_id / "volume_provenance.json"
        if not sidecar.exists():
            missing.append(case_id)
            continue
        rows.append(json.loads(sidecar.read_text()))
    if missing:
        raise FileNotFoundError(f"Missing volume_provenance.json for {len(missing)} cases: {missing}")

    write_csv_rows(args.root / args.out, rows)
    payload = {
        "dataset": "DrivAerML",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_processed_root": args.source_root,
        "raw_root": args.raw_root,
        "output_root": str(args.root),
        "case_count": len(case_rows),
        "processed_case_count": len(rows),
        "processor_versions": sorted({row["processor_version"] for row in rows}),
        "notes": [
            "Surface arrays are resolved from the source processed root and hardlinked or copied.",
            "Volume arrays are regenerated for every case from complete raw VTU files.",
            "No synthetic inside-body samples are added.",
        ],
    }
    (args.root / "preprocessing_provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.root / args.out}")
    print(f"Wrote {args.root / 'preprocessing_provenance.json'}")


if __name__ == "__main__":
    main()
