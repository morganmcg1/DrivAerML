# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Audit DrivAerML per-case SDF distributions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from data.loader import DEFAULT_MANIFEST, DrivAerMLCaseStore, REQUIRED_RESTORED_CASE_IDS, _resolve_artifact_path

NEAR_SURFACE_ABS = 0.1
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_sdf_stats(store: DrivAerMLCaseStore, split: str, case_id: str) -> dict[str, Any]:
    path = _resolve_artifact_path(store.root / case_id / "volume_sdf.npy")
    sdf = np.load(path, mmap_mode="r")
    values = np.asarray(sdf).reshape(-1)
    count = int(values.size)
    negative_count = int(np.count_nonzero(values < 0.0))
    near_count = int(np.count_nonzero(np.abs(values) < NEAR_SURFACE_ABS))
    quantiles = np.quantile(values, QUANTILES)

    row: dict[str, Any] = {
        "case_id": case_id,
        "split": split,
        "restored_case": case_id in REQUIRED_RESTORED_CASE_IDS,
        "rows": count,
        "sdf_min": float(np.min(values)),
        "sdf_max": float(np.max(values)),
        "sdf_mean": float(np.mean(values, dtype=np.float64)),
        "sdf_std": float(np.std(values, dtype=np.float64)),
        "sdf_negative_count": negative_count,
        "sdf_negative_frac": float(negative_count / count),
        "sdf_near_surface_abs_0p1_count": near_count,
        "sdf_near_surface_abs_0p1_frac": float(near_count / count),
        "sdf_path": str(path),
    }
    for quantile, value in zip(QUANTILES, quantiles, strict=True):
        row[f"sdf_q{int(quantile * 100):02d}"] = float(value)
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    neg = np.asarray([row["sdf_negative_frac"] for row in rows], dtype=np.float64)
    mins = np.asarray([row["sdf_min"] for row in rows], dtype=np.float64)
    restored = [row for row in rows if row["restored_case"]]
    normal = [row for row in rows if not row["restored_case"]]
    summary = {
        "case_count": len(rows),
        "negative_frac_min": float(np.min(neg)),
        "negative_frac_median": float(np.median(neg)),
        "negative_frac_max": float(np.max(neg)),
        "sdf_min_min": float(np.min(mins)),
        "sdf_min_median": float(np.median(mins)),
        "sdf_min_max": float(np.max(mins)),
        "restored_case_count": len(restored),
    }
    if restored:
        restored_neg = np.asarray([row["sdf_negative_frac"] for row in restored], dtype=np.float64)
        restored_min = np.asarray([row["sdf_min"] for row in restored], dtype=np.float64)
        summary.update(
            {
                "restored_negative_frac_min": float(np.min(restored_neg)),
                "restored_negative_frac_median": float(np.median(restored_neg)),
                "restored_negative_frac_max": float(np.max(restored_neg)),
                "restored_sdf_min_median": float(np.median(restored_min)),
            }
        )
    if normal:
        normal_neg = np.asarray([row["sdf_negative_frac"] for row in normal], dtype=np.float64)
        normal_min = np.asarray([row["sdf_min"] for row in normal], dtype=np.float64)
        summary.update(
            {
                "non_restored_negative_frac_min": float(np.min(normal_neg)),
                "non_restored_negative_frac_median": float(np.median(normal_neg)),
                "non_restored_negative_frac_max": float(np.max(normal_neg)),
                "non_restored_sdf_min_median": float(np.median(normal_min)),
            }
        )
    return summary


def render_markdown(root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lowest = sorted(rows, key=lambda row: row["sdf_negative_frac"])[:15]
    restored = [row for row in rows if row["restored_case"]]
    lines = [
        "# DrivAerML SDF QA Report",
        "",
        f"- Root: `{root}`",
        f"- Cases audited: {summary['case_count']}",
        f"- Negative SDF fraction median: `{summary['negative_frac_median']:.6g}`",
        f"- Negative SDF fraction range: `{summary['negative_frac_min']:.6g}` to `{summary['negative_frac_max']:.6g}`",
        f"- SDF minimum median: `{summary['sdf_min_median']:.6g}`",
        "",
        "## Restored Cases",
        "",
    ]
    if restored:
        lines.extend(
            [
                "| case | split | rows | neg frac | sdf min | near |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(restored, key=lambda item: item["case_id"]):
            lines.append(
                "| {case_id} | {split} | {rows} | {sdf_negative_frac:.6g} | "
                "{sdf_min:.6g} | {sdf_near_surface_abs_0p1_frac:.6g} |".format(**row)
            )
    else:
        lines.append("No restored cases were present in this root.")

    lines.extend(
        [
            "",
            "## Lowest Negative-SDF Fractions",
            "",
            "| case | split | restored | rows | neg frac | sdf min |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in lowest:
        lines.append(
            "| {case_id} | {split} | {restored_case} | {rows} | "
            "{sdf_negative_frac:.6g} | {sdf_min:.6g} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report is diagnostic only. A low negative-SDF fraction is not automatically a bad SDF value; "
            "it usually means the sampled volume points contain few or no inside-body points. The canonical "
            "raw-only dataset intentionally does not synthesize inside-body rows.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--root", default="")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--md-out", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    store = DrivAerMLCaseStore(manifest_path=args.manifest, root=args.root or None)
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for case_id in store.case_ids(split):
            rows.append(case_sdf_stats(store, split, case_id))
            if len(rows) % 25 == 0:
                print(f"Audited {len(rows)} cases")

    summary = summarize_rows(rows)
    csv_out = Path(args.csv_out) if args.csv_out else store.root / "sdf_qa.csv"
    md_out = Path(args.md_out) if args.md_out else store.root / "SDF_QA_REPORT.md"
    json_out = Path(args.json_out) if args.json_out else store.root / "sdf_qa_summary.json"

    write_csv_rows(csv_out, rows)
    md_out.write_text(render_markdown(store.root, rows, summary) + "\n")
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {csv_out}")
    print(f"Wrote {md_out}")
    print(f"Wrote {json_out}")


if __name__ == "__main__":
    main()
