#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Re-evaluate closed DrivAerML experiment artifacts on the rawcanon dataset.

This script is intentionally self-contained so it can be mounted into a short
Kubernetes job. For each candidate it checks out the PR head that produced the
model, overlays the fixed rawcanon data-loading package, downloads the W&B model
artifact, and logs fresh validation/test metrics.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml


DEFAULT_REPO_URL = "https://github.com/morganmcg1/DrivAerML.git"
DEFAULT_DATA_FIX_REF = "codex/drivaerml-case-splits-data-fix"
DEFAULT_RAWCANON_ROOT = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
DEFAULT_PROJECT = "senpai-v1-drivaerml-ddp8"
DEFAULT_ENTITY = "wandb-applied-ai-team"


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    display = " ".join(cmd)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        display = display.replace(token, "***")
    print("+", display, flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_candidate(value: str, *, index: int | None = None, run_id: str = "") -> dict[str, Any]:
    path = Path(value)
    if path.exists():
        loaded = json.loads(path.read_text())
    else:
        loaded = json.loads(value)
    if isinstance(loaded, list):
        if run_id:
            for candidate in loaded:
                if candidate.get("run_id") == run_id:
                    return candidate
            raise KeyError(f"No candidate with run_id={run_id!r}")
        if index is None:
            raise ValueError("--candidate-index or --candidate-run-id is required when candidate-json is a list")
        return loaded[index]
    return loaded


def authenticated_repo_url(repo_url: str) -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token or not repo_url.startswith("https://github.com/"):
        return repo_url
    return repo_url.replace("https://github.com/", f"https://{token}@github.com/", 1)


def checkout_candidate_repo(
    *,
    candidate: dict[str, Any],
    work_root: Path,
    repo_url: str,
    data_fix_ref: str,
) -> Path:
    pr_number = int(candidate["pr"])
    repo_dir = work_root / f"repo-pr{pr_number}-{candidate['run_id']}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    run(["git", "clone", "--no-tags", "--depth", "1", authenticated_repo_url(repo_url), str(repo_dir)])
    run(["git", "remote", "set-url", "origin", repo_url], cwd=repo_dir)
    run(["git", "fetch", "--depth", "1", "origin", f"pull/{pr_number}/head"], cwd=repo_dir)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo_dir)
    candidate["pr_head_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
    ).strip()
    run(["git", "fetch", "--depth", "1", "origin", data_fix_ref], cwd=repo_dir)
    run(["git", "checkout", "FETCH_HEAD", "--", "data"], cwd=repo_dir)
    candidate["data_fix_ref"] = data_fix_ref
    return repo_dir


def clear_repo_modules() -> None:
    for name in list(sys.modules):
        if name == "train" or name == "model" or name == "trainer_runtime" or name == "data" or name.startswith("data."):
            sys.modules.pop(name, None)


def import_repo(repo_dir: Path):
    clear_repo_modules()
    sys.path.insert(0, str(repo_dir))
    train = importlib.import_module("train")
    runtime = importlib.import_module("trainer_runtime")
    return train, runtime


def make_config(train_module, raw_config: dict[str, Any], args: argparse.Namespace):
    config_class = getattr(train_module, "Config")
    config = config_class()
    if is_dataclass(config):
        allowed = {field.name for field in fields(config)}
    else:
        allowed = set(vars(config))
    for key, value in raw_config.items():
        if key in allowed:
            setattr(config, key, value)
    config.manifest = "data/split_manifest.json"
    config.data_root = args.rawcanon_root
    config.output_dir = str(args.output_dir)
    config.batch_size = args.batch_size
    if hasattr(config, "num_workers"):
        config.num_workers = args.num_workers
    if hasattr(config, "pin_memory"):
        config.pin_memory = True
    if hasattr(config, "persistent_workers"):
        config.persistent_workers = args.num_workers > 0
    if hasattr(config, "prefetch_factor"):
        config.prefetch_factor = 2
    if hasattr(config, "compile_model"):
        config.compile_model = False
    if hasattr(config, "debug"):
        config.debug = args.debug
    if hasattr(config, "wandb_group"):
        config.wandb_group = args.wandb_group
    if hasattr(config, "wandb_name"):
        config.wandb_name = f"rawcanon-retest/pr{args.current_pr}-{args.current_run_id}"
    if hasattr(config, "agent"):
        config.agent = "rawcanon-retest"
    if hasattr(config, "amp_mode") and args.amp_mode:
        config.amp_mode = args.amp_mode
    return config


def download_artifact(candidate: dict[str, Any], *, artifact_root: Path, entity: str, project: str) -> Path:
    api = wandb.Api(timeout=60)
    artifact_name = candidate["artifact"]
    if "/" not in artifact_name:
        artifact_name = f"{entity}/{project}/{artifact_name}"
    artifact = api.artifact(artifact_name, type="model")
    artifact_dir = Path(artifact.download(root=str(artifact_root / candidate["run_id"])))
    checkpoint = artifact_dir / "checkpoint.pt"
    config_yaml = artifact_dir / "config.yaml"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Artifact {artifact_name} is missing checkpoint.pt")
    if not config_yaml.exists():
        raise FileNotFoundError(f"Artifact {artifact_name} is missing config.yaml")
    candidate["artifact_download_dir"] = str(artifact_dir)
    candidate["artifact_qualified_name"] = artifact_name
    return artifact_dir


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        stripped = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "_orig_mod."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            stripped[key] = value
        model.load_state_dict(stripped, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {"model": state_dict}


def flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def evaluate_candidate(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    torch.set_num_threads(max(1, min(args.torch_threads, os.cpu_count() or args.torch_threads)))
    args.current_pr = int(candidate["pr"])
    args.current_run_id = candidate["run_id"]
    work_root = args.work_root / f"pr{candidate['pr']}-{candidate['run_id']}"
    work_root.mkdir(parents=True, exist_ok=True)
    print(f"Preparing PR #{candidate['pr']} run {candidate['run_id']}", flush=True)
    repo_dir = checkout_candidate_repo(
        candidate=candidate,
        work_root=work_root,
        repo_url=args.repo_url,
        data_fix_ref=args.data_fix_ref,
    )
    print("Downloading model artifact", flush=True)
    artifact_dir = download_artifact(
        candidate,
        artifact_root=args.artifact_root,
        entity=args.entity,
        project=args.project,
    )
    raw_config = yaml.safe_load((artifact_dir / "config.yaml").read_text()) or {}

    os.chdir(repo_dir)
    print("Importing PR code", flush=True)
    train_module, runtime = import_repo(repo_dir)
    print("Building model", flush=True)
    config = make_config(train_module, raw_config, args)
    model = train_module.build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loading checkpoint on {device}", flush=True)
    checkpoint = load_checkpoint(model, artifact_dir / "checkpoint.pt", device)

    print("Building rawcanon loaders", flush=True)
    _, val_loaders, test_loaders, stats = runtime.make_loaders(config, distributed_state=None)
    transform = runtime.TargetTransform(**stats)
    amp_mode = getattr(config, "amp_mode", "none")

    print("Evaluating validation split", flush=True)
    val_metrics = {
        split: runtime.evaluate_split(model, loader, transform, device, amp_mode=amp_mode)
        for split, loader in val_loaders.items()
    }
    print("Evaluating test split", flush=True)
    test_metrics = {
        split: runtime.evaluate_split(model, loader, transform, device, amp_mode=amp_mode)
        for split, loader in test_loaders.items()
    }

    primary_val = val_metrics["val_surface"]
    primary_test = test_metrics["test_surface"]
    result = {
        "status": "ok",
        "pr": int(candidate["pr"]),
        "title": candidate.get("title", ""),
        "run_id": candidate["run_id"],
        "run_url": candidate.get("run_url", ""),
        "artifact": candidate["artifact"],
        "artifact_qualified_name": candidate.get("artifact_qualified_name", ""),
        "artifact_aliases": candidate.get("artifact_aliases", []),
        "pr_head_sha": candidate.get("pr_head_sha"),
        "data_fix_ref": args.data_fix_ref,
        "rawcanon_root": args.rawcanon_root,
        "eval_surface_points": getattr(config, "eval_surface_points", None),
        "eval_volume_points": getattr(config, "eval_volume_points", None),
        "batch_size": getattr(config, "batch_size", None),
        "old_val_abupt": candidate.get("old_val_abupt"),
        "old_val_volume": candidate.get("old_val_volume"),
        "old_test_abupt": candidate.get("old_test_abupt"),
        "old_test_volume": candidate.get("old_test_volume"),
        "rawcanon_val_abupt": primary_val["abupt_axis_mean_rel_l2_pct"],
        "rawcanon_val_volume": primary_val["volume_pressure_rel_l2_pct"],
        "rawcanon_test_abupt": primary_test["abupt_axis_mean_rel_l2_pct"],
        "rawcanon_test_volume": primary_test["volume_pressure_rel_l2_pct"],
        "rawcanon_val_metrics": primary_val,
        "rawcanon_test_metrics": primary_test,
        "checkpoint_keys": sorted(k for k in checkpoint.keys() if isinstance(k, str)),
        "elapsed_seconds": time.time() - start,
    }

    if not args.no_wandb:
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.wandb_group,
            job_type="rawcanon-retest",
            name=f"rawcanon-retest-pr{candidate['pr']}-{candidate['run_id']}",
            tags=["rawcanon-retest", f"pr:{candidate['pr']}", f"source:{candidate['run_id']}"],
            config={
                "candidate": candidate,
                "raw_config": raw_config,
                "rawcanon_root": args.rawcanon_root,
                "data_fix_ref": args.data_fix_ref,
                "eval_harness": "scripts/retest_rawcanon_wandb_artifacts.py",
            },
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        log = {
            **flatten_metrics("rawcanon_val_primary", primary_val),
            **flatten_metrics("rawcanon_test_primary", primary_test),
            "elapsed_seconds": result["elapsed_seconds"],
        }
        for split, metrics in val_metrics.items():
            log.update(flatten_metrics(f"rawcanon_val/{split}", metrics))
        for split, metrics in test_metrics.items():
            log.update(flatten_metrics(f"rawcanon_test/{split}", metrics))
        wandb.log(log)
        wandb.summary.update(
            {
                **log,
                "source_pr": int(candidate["pr"]),
                "source_run_id": candidate["run_id"],
                "source_artifact": candidate["artifact"],
                "old_test_primary/abupt_axis_mean_rel_l2_pct": candidate.get("old_test_abupt"),
                "old_test_primary/volume_pressure_rel_l2_pct": candidate.get("old_test_volume"),
                "rawcanon_root": args.rawcanon_root,
                "pr_head_sha": candidate.get("pr_head_sha"),
            }
        )
        result["wandb_eval_run_id"] = run.id
        result["wandb_eval_run_url"] = run.url
        wandb.finish()
    return result


def append_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "rawcanon_retest_results.jsonl"
    with jsonl_path.open("a") as f:
        f.write(json_dumps(result) + "\n")

    csv_path = output_dir / "rawcanon_retest_results.csv"
    row = {
        key: result.get(key)
        for key in (
            "status",
            "pr",
            "title",
            "run_id",
            "artifact",
            "old_val_abupt",
            "old_val_volume",
            "old_test_abupt",
            "old_test_volume",
            "rawcanon_val_abupt",
            "rawcanon_val_volume",
            "rawcanon_test_abupt",
            "rawcanon_test_volume",
            "wandb_eval_run_id",
            "wandb_eval_run_url",
            "elapsed_seconds",
        )
    }
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--candidate-run-id", default="")
    parser.add_argument("--repo-url", default=os.environ.get("REPO_URL", DEFAULT_REPO_URL))
    parser.add_argument("--data-fix-ref", default=os.environ.get("DATA_FIX_REF", DEFAULT_DATA_FIX_REF))
    parser.add_argument("--rawcanon-root", default=os.environ.get("RAWCANON_ROOT", DEFAULT_RAWCANON_ROOT))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", "rawcanon-retest-20260511"))
    parser.add_argument("--work-root", type=Path, default=Path("/workspace/rawcanon-retest"))
    parser.add_argument("--artifact-root", type=Path, default=Path("/workspace/rawcanon-retest/artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/new-pvc/Reports/drivaerml_rawcanon_retest_20260511"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--amp-mode", default="bf16")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    candidate = load_candidate(
        args.candidate_json,
        index=args.candidate_index,
        run_id=args.candidate_run_id,
    )
    result: dict[str, Any]
    try:
        result = evaluate_candidate(candidate, args)
    except Exception as exc:
        result = {
            "status": "error",
            "pr": int(candidate.get("pr", -1)),
            "title": candidate.get("title", ""),
            "run_id": candidate.get("run_id", ""),
            "artifact": candidate.get("artifact", ""),
            "error": repr(exc),
        }
        print(json_dumps(result), flush=True)
        append_outputs(result, args.output_dir)
        raise
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    append_outputs(result, args.output_dir)


if __name__ == "__main__":
    main()
