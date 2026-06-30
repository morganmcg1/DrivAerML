#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Evaluate W&B model artifacts on a full-resolution DrivAerML test root.

This is a test-only harness for the reporting pass where volume arrays have
been regenerated with ``--sample-ratio 1.0``. Evaluation uses the repo's
deterministic ``eval_chunk`` point views, so every loaded surface and volume
point is evaluated exactly once with no duplicate prediction averaging.

For full DrivAerML cases, the historical dataset's row-indexed ``np.load``
path is prohibitively slow because each chunk performs scattered mmap reads
from PVC. The default path below loads one full case at a time in each rank and
then serves the same deterministic chunk indices from memory.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def drop_local_wandb_shadow() -> None:
    """Keep repo-local ``wandb/`` run directories from shadowing the SDK."""

    cwd = Path.cwd().resolve()
    for entry in list(sys.path):
        base = Path(entry or cwd).resolve()
        if (base / "wandb").is_dir() and base in {cwd, REPO_ROOT}:
            sys.path.remove(entry)


drop_local_wandb_shadow()
try:
    import wandb
except ImportError as exc:  # pragma: no cover - exercised in runtime images
    raise SystemExit("The W&B SDK is required: install the project dependencies first.") from exc


DEFAULT_REPO_URL = "https://github.com/morganmcg1/DrivAerML.git"
DEFAULT_DATA_FIX_REF = "codex/drivaerml-case-splits-data-fix"
DEFAULT_ENTITY = "wandb-applied-ai-team"
DEFAULT_PROJECT = "senpai-v1-drivaerml-ddp8"
DEFAULT_SELECTION_METRIC = "val_primary/surface_pressure_rel_l2_pct"
RUN_URL_RE = re.compile(r"/runs/([^/?#]+)")


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    display = " ".join(cmd)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        display = display.replace(token, "***")
    print("+", display, flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def authenticated_repo_url(repo_url: str) -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token or not repo_url.startswith("https://github.com/"):
        return repo_url
    return repo_url.replace("https://github.com/", f"https://{token}@github.com/", 1)


def run_id_from_value(value: str) -> str:
    match = RUN_URL_RE.search(value)
    if match:
        return match.group(1)
    return value.rstrip("/").split("/")[-1]


def read_candidate_json(value: str) -> list[dict[str, Any]]:
    path = Path(value)
    loaded = json.loads(path.read_text() if path.exists() else value)
    if isinstance(loaded, list):
        return [dict(item) for item in loaded]
    return [dict(loaded)]


def load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for value in args.candidate_json:
        candidates.extend(read_candidate_json(value))
    for value in args.run:
        candidates.append({"run_id": run_id_from_value(value), "run_url": value})
    if not candidates:
        raise ValueError("Provide at least one --run or --candidate-json.")
    for candidate in candidates:
        if "run_id" not in candidate:
            raise ValueError(f"Candidate is missing run_id: {candidate}")
    return candidates


def clear_repo_modules() -> None:
    for name in list(sys.modules):
        if name in {"train", "model", "trainer_runtime", "data"} or name.startswith("data."):
            sys.modules.pop(name, None)


def checkout_candidate_repo(
    *,
    candidate: dict[str, Any],
    args: argparse.Namespace,
    rank: int,
) -> Path:
    if args.use_current_code:
        return REPO_ROOT

    code_ref = candidate.get("code_ref") or args.code_ref
    if code_ref:
        safe_ref = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(code_ref)).strip("-")
        repo_dir = args.work_root / f"rank{rank}" / f"repo-{safe_ref}-{candidate['run_id']}"
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "clone", "--no-tags", "--depth", "1", authenticated_repo_url(args.repo_url), str(repo_dir)])
        run_cmd(["git", "remote", "set-url", "origin", args.repo_url], cwd=repo_dir)
        run_cmd(["git", "fetch", "--depth", "1", "origin", str(code_ref)], cwd=repo_dir)
        run_cmd(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo_dir)
        candidate["code_ref"] = str(code_ref)
        candidate["code_ref_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
        return repo_dir

    pr_number = candidate.get("pr")
    if pr_number is None:
        raise ValueError(
            f"Run {candidate['run_id']} needs a PR number in candidate JSON, "
            "a --code-ref, or --use-current-code if the current checkout matches the checkpoint."
        )

    repo_dir = args.work_root / f"rank{rank}" / f"repo-pr{int(pr_number)}-{candidate['run_id']}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["git", "clone", "--no-tags", "--depth", "1", authenticated_repo_url(args.repo_url), str(repo_dir)])
    run_cmd(["git", "remote", "set-url", "origin", args.repo_url], cwd=repo_dir)
    run_cmd(["git", "fetch", "--depth", "1", "origin", f"pull/{int(pr_number)}/head"], cwd=repo_dir)
    run_cmd(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo_dir)
    candidate["pr_head_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
    ).strip()
    run_cmd(["git", "fetch", "--depth", "1", "origin", args.data_fix_ref], cwd=repo_dir)
    run_cmd(["git", "checkout", "FETCH_HEAD", "--", "data"], cwd=repo_dir)
    candidate["data_fix_ref"] = args.data_fix_ref
    return repo_dir


def import_repo(repo_dir: Path):
    clear_repo_modules()
    sys.path.insert(0, str(repo_dir))
    train = importlib.import_module("train")
    runtime = importlib.import_module("trainer_runtime")
    data_loader = importlib.import_module("data.loader")
    return train, runtime, data_loader


def infer_pr_from_run(run, candidate: dict[str, Any]) -> None:
    if candidate.get("pr") is not None:
        return
    for key in ("pr", "github_pr", "pull_request", "source_pr"):
        value = run.config.get(key) if hasattr(run, "config") else None
        if value is not None:
            candidate["pr"] = int(str(value).removeprefix("#"))
            return
    for tag in getattr(run, "tags", []) or []:
        if isinstance(tag, str) and tag.startswith("pr:"):
            candidate["pr"] = int(tag.split(":", 1)[1])
            return


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def best_epoch_for_metric(run, metric: str) -> tuple[int | None, float | None]:
    best_epoch: int | None = None
    best_value: float | None = None
    keys = [metric, "epoch", "_step", "global_step"]
    for row in run.scan_history(keys=keys, page_size=1000):
        value = finite_float(row.get(metric))
        if value is None:
            continue
        epoch_value = finite_float(row.get("epoch"))
        if epoch_value is None:
            continue
        if best_value is None or value < best_value:
            best_value = value
            best_epoch = int(round(epoch_value))
    return best_epoch, best_value


def qualified_artifact_name(name: str, *, entity: str, project: str) -> str:
    if "/" in name:
        return name
    return f"{entity}/{project}/{name}"


def artifact_from_candidate(api, candidate: dict[str, Any], args: argparse.Namespace):
    artifact_name = candidate.get("artifact")
    if artifact_name:
        return api.artifact(
            qualified_artifact_name(str(artifact_name), entity=args.entity, project=args.project),
            type="model",
        )
    return None


def select_artifact(api, run, candidate: dict[str, Any], args: argparse.Namespace):
    direct = artifact_from_candidate(api, candidate, args)
    if direct is not None:
        candidate["artifact_selection"] = "candidate-json"
        return direct

    artifacts = [artifact for artifact in run.logged_artifacts() if artifact.type == "model"]
    if not artifacts:
        raise ValueError(f"Run {candidate['run_id']} has no logged model artifacts.")

    best_epoch, best_value = best_epoch_for_metric(run, args.selection_metric)
    candidate["selection_metric"] = args.selection_metric
    candidate["selection_metric_best_epoch"] = best_epoch
    candidate["selection_metric_best_value"] = best_value

    if best_epoch is not None:
        wanted_alias = f"epoch-{best_epoch}"
        for artifact in artifacts:
            if wanted_alias in artifact.aliases:
                candidate["artifact_selection"] = wanted_alias
                return artifact

    for artifact in artifacts:
        if args.artifact_alias in artifact.aliases:
            candidate["artifact_selection"] = args.artifact_alias
            if best_epoch is not None and f"epoch-{best_epoch}" not in artifact.aliases:
                message = (
                    f"Run {candidate['run_id']} best {args.selection_metric} epoch is {best_epoch}, "
                    f"but no matching artifact alias was found; {args.artifact_alias!r} artifact "
                    f"has aliases {artifact.aliases}."
                )
                if not args.allow_alias_fallback:
                    raise ValueError(message)
                candidate["artifact_selection_warning"] = message
            return artifact

    if len(artifacts) == 1 and args.allow_alias_fallback:
        candidate["artifact_selection"] = "single-model-artifact"
        candidate["artifact_selection_warning"] = (
            f"Using sole model artifact {artifacts[0].name}; no {args.artifact_alias!r} alias found."
        )
        return artifacts[0]

    raise ValueError(
        f"Could not select an artifact for run {candidate['run_id']}. "
        f"Available: {[(artifact.name, artifact.aliases) for artifact in artifacts]}"
    )


def download_artifact(artifact, *, artifact_root: Path, run_id: str, rank: int) -> Path:
    artifact_dir = Path(artifact.download(root=str(artifact_root / f"rank{rank}" / run_id)))
    checkpoint = artifact_dir / "checkpoint.pt"
    config_yaml = artifact_dir / "config.yaml"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Artifact {artifact.name} is missing checkpoint.pt")
    if not config_yaml.exists():
        raise FileNotFoundError(f"Artifact {artifact.name} is missing config.yaml")
    return artifact_dir


def resolve_manifest_path(manifest: str, repo_dir: Path) -> str:
    path = Path(manifest)
    if path.is_absolute():
        return str(path)
    repo_path = repo_dir / path
    return str(repo_path if repo_path.exists() else path)


def make_config(train_module, raw_config: dict[str, Any], args: argparse.Namespace, repo_dir: Path):
    config_class = getattr(train_module, "Config")
    config = config_class()
    allowed = {field.name for field in fields(config)} if is_dataclass(config) else set(vars(config))
    for key, value in raw_config.items():
        if key in allowed:
            setattr(config, key, value)
    config.manifest = resolve_manifest_path(args.manifest, repo_dir)
    config.data_root = args.full_test_root
    config.output_dir = str(args.output_dir)
    config.batch_size = args.batch_size
    if hasattr(config, "eval_surface_points"):
        config.eval_surface_points = args.eval_surface_points
    if hasattr(config, "eval_volume_points"):
        config.eval_volume_points = args.eval_volume_points
    if hasattr(config, "num_workers"):
        config.num_workers = args.num_workers
    if hasattr(config, "pin_memory"):
        config.pin_memory = True
    if hasattr(config, "persistent_workers"):
        config.persistent_workers = args.num_workers > 0
    if hasattr(config, "prefetch_factor"):
        config.prefetch_factor = args.prefetch_factor
    if hasattr(config, "compile_model"):
        config.compile_model = False
    if hasattr(config, "debug"):
        config.debug = False
    if hasattr(config, "amp_mode") and args.amp_mode:
        config.amp_mode = args.amp_mode
    return config


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        stripped = {}
        for key, value in state_dict.items():
            cleaned = key
            for prefix in ("module.", "_orig_mod."):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :]
            stripped[cleaned] = value
        model.load_state_dict(stripped, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {"model": state_dict}


class CachedEvalChunkDataset(torch.utils.data.Dataset):
    """Serve deterministic eval chunks from a one-case in-process cache."""

    def __init__(
        self,
        case_ids: list[str],
        *,
        store,
        data_loader,
        max_surface_points: int,
        max_volume_points: int,
    ):
        self.case_ids = list(case_ids)
        self.store = store
        self.data_loader = data_loader
        self.max_surface_points = max_surface_points
        self.max_volume_points = max_volume_points
        self.views = self._build_views()
        self._cached_case_id: str | None = None
        self._cached_case = None

    def __len__(self) -> int:
        return len(self.views)

    @staticmethod
    def _view_count(total: int, points_per_view: int) -> int:
        if points_per_view <= 0 or total <= points_per_view:
            return 1
        return max(1, math.ceil(total / points_per_view))

    def _build_views(self) -> list[dict[str, int | str]]:
        views: list[dict[str, int | str]] = []
        for case_id in self.case_ids:
            counts = self.store.case_point_counts(case_id)
            surface_views = self._view_count(counts["n_surface"], self.max_surface_points)
            volume_views = self._view_count(counts["n_volume"], self.max_volume_points)
            view_count = max(surface_views, volume_views)
            for view_index in range(view_count):
                views.append(
                    {
                        "case_id": case_id,
                        "view_index": view_index,
                        "view_count": view_count,
                        "surface_view_count": surface_views,
                        "volume_view_count": volume_views,
                        "n_surface": int(counts["n_surface"]),
                        "n_volume": int(counts["n_volume"]),
                    }
                )
        return views

    @staticmethod
    def _indices(
        *,
        total: int,
        count: int,
        view_index: int,
        group_view_count: int,
    ) -> torch.Tensor | None:
        if view_index >= group_view_count:
            return torch.empty(0, dtype=torch.long)
        if count <= 0 or total <= count:
            return None if view_index == 0 else torch.empty(0, dtype=torch.long)
        return torch.arange(view_index, total, group_view_count, dtype=torch.long)

    def _load_full_case(self, case_id: str):
        if self._cached_case_id != case_id:
            rank = int(os.environ.get("RANK", "0"))
            print(f"[rank {rank}] caching full case {case_id}", flush=True)
            self._cached_case = self.store.load_case(case_id)
            self._cached_case_id = case_id
        return self._cached_case

    @staticmethod
    def _take_rows(tensor: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
        if rows is None:
            return tensor
        return tensor.index_select(0, rows)

    def __getitem__(self, idx: int):
        view = self.views[idx]
        case_id = str(view["case_id"])
        view_index = int(view["view_index"])
        case = self._load_full_case(case_id)
        surface_idx = self._indices(
            total=int(view["n_surface"]),
            count=self.max_surface_points,
            view_index=view_index,
            group_view_count=int(view["surface_view_count"]),
        )
        volume_idx = self._indices(
            total=int(view["n_volume"]),
            count=self.max_volume_points,
            view_index=view_index,
            group_view_count=int(view["volume_view_count"]),
        )

        metadata = dict(case.metadata)
        if "surface_curvature" in metadata:
            metadata["surface_curvature"] = self._take_rows(metadata["surface_curvature"], surface_idx)
        metadata["n_surface_full"] = int(view["n_surface"])
        metadata["n_surface_loaded"] = int(
            case.surface_x.shape[0] if surface_idx is None else surface_idx.numel()
        )
        metadata["surface_view_index"] = view_index
        metadata["surface_view_count"] = int(view["surface_view_count"])
        metadata["surface_sampling_mode"] = "eval_chunk"
        metadata["n_volume_full"] = int(view["n_volume"])
        metadata["n_volume_loaded"] = int(
            case.volume_x.shape[0] if volume_idx is None else volume_idx.numel()
        )
        metadata["volume_view_index"] = view_index
        metadata["volume_view_count"] = int(view["volume_view_count"])
        metadata["volume_sampling_mode"] = "eval_chunk"
        metadata["joint_view_count"] = int(view["view_count"])

        return self.data_loader.DrivAerMLCase(
            case_id=case.case_id,
            surface_x=self._take_rows(case.surface_x, surface_idx),
            surface_y=self._take_rows(case.surface_y, surface_idx),
            volume_x=self._take_rows(case.volume_x, volume_idx),
            volume_y=self._take_rows(case.volume_y, volume_idx),
            metadata=metadata,
        )


def load_curvature_stats_for_eval(runtime, args: argparse.Namespace) -> dict[str, Any]:
    candidate_paths = []
    if args.curvature_stats:
        candidate_paths.append(Path(args.curvature_stats))
    candidate_paths.append(REPO_ROOT / "curvature_proxy_stats_k16_v1.json")
    for path in candidate_paths:
        if path.exists():
            return json.loads(path.read_text())
    return runtime.load_curvature_stats()


def make_case_store(data_loader, runtime, config, args: argparse.Namespace):
    if getattr(config, "use_curvature_attention_bias", False):
        stats = load_curvature_stats_for_eval(runtime, args)
        return runtime.CurvatureAugmentedCaseStore(
            manifest_path=config.manifest,
            root=config.data_root,
            stats=stats,
        )
    return data_loader.DrivAerMLCaseStore(manifest_path=config.manifest, root=config.data_root)


def build_loader_from_dataset(runtime, dataset, config, *, distributed_state, use_runtime_sampler: bool):
    if use_runtime_sampler:
        return runtime.eval_loader_for_dataset(dataset, config, distributed_state=distributed_state)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **runtime.loader_kwargs(config),
    )


def build_test_loader(data_loader, runtime, config, args: argparse.Namespace, distributed_state):
    store = make_case_store(data_loader, runtime, config, args)
    case_ids = store.case_ids(args.split)
    if args.case_id:
        wanted = set(args.case_id)
        missing = sorted(wanted - set(case_ids))
        if missing:
            raise ValueError(f"Requested case IDs are not in split {args.split}: {missing}")
        case_ids = [case_id for case_id in case_ids if case_id in wanted]
    if args.limit_cases > 0:
        case_ids = case_ids[: args.limit_cases]
    rank_case_ids = case_ids
    use_runtime_sampler = True
    if args.cache_full_cases and args.case_shard_cache and distributed_state.enabled:
        rank_case_ids = case_ids[distributed_state.rank :: distributed_state.world_size]
        use_runtime_sampler = False
    if args.cache_full_cases:
        dataset = CachedEvalChunkDataset(
            rank_case_ids,
            store=store,
            data_loader=data_loader,
            max_surface_points=config.eval_surface_points,
            max_volume_points=config.eval_volume_points,
        )
    else:
        dataset = data_loader.DrivAerMLSurfaceDataset(
            rank_case_ids,
            store=store,
            max_surface_points=config.eval_surface_points,
            max_volume_points=config.eval_volume_points,
            sampling_mode="eval_chunk",
        )
    loader = build_loader_from_dataset(
        runtime,
        dataset,
        config,
        distributed_state=distributed_state,
        use_runtime_sampler=use_runtime_sampler,
    )
    stats = data_loader.target_stats_from_normalizers(store)
    return loader, stats, case_ids


def flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json_dumps(payload) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_average(results: Iterable[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        metrics = result.get("full_test_metrics", {})
        for key, value in metrics.items():
            number = finite_float(value)
            if number is None:
                continue
            sums[key] = sums.get(key, 0.0) + number
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums) if counts[key] > 0}


def evaluate_candidate(candidate: dict[str, Any], args: argparse.Namespace, api, state) -> dict[str, Any]:
    start = time.time()
    run = api.run(f"{args.entity}/{args.project}/{candidate['run_id']}")
    candidate.setdefault("run_url", run.url)
    infer_pr_from_run(run, candidate)
    artifact = select_artifact(api, run, candidate, args)
    candidate["artifact"] = artifact.name
    candidate["artifact_aliases"] = list(artifact.aliases)

    repo_dir = checkout_candidate_repo(candidate=candidate, args=args, rank=state.rank)
    artifact_dir = download_artifact(
        artifact,
        artifact_root=args.artifact_root,
        run_id=candidate["run_id"],
        rank=state.rank,
    )
    raw_config = yaml.safe_load((artifact_dir / "config.yaml").read_text()) or {}

    train_module, runtime, data_loader = import_repo(repo_dir)
    config = make_config(train_module, raw_config, args, repo_dir)
    model = train_module.build_model(config).to(state.device)
    checkpoint = load_checkpoint(model, artifact_dir / "checkpoint.pt", state.device)

    loader, stats, case_ids = build_test_loader(data_loader, runtime, config, args, state)
    transform = runtime.TargetTransform(**stats)
    metrics = runtime.evaluate_split(
        model,
        loader,
        transform,
        state.device,
        amp_mode=getattr(config, "amp_mode", "none"),
        distributed_state=state,
    )

    if not state.is_main:
        return {"status": "rank_complete", "run_id": candidate["run_id"]}

    return {
        "status": "ok",
        "run_id": candidate["run_id"],
        "run_url": candidate.get("run_url", ""),
        "pr": candidate.get("pr"),
        "pr_head_sha": candidate.get("pr_head_sha"),
        "code_ref": candidate.get("code_ref"),
        "code_ref_sha": candidate.get("code_ref_sha"),
        "artifact": artifact.name,
        "artifact_aliases": list(artifact.aliases),
        "artifact_selection": candidate.get("artifact_selection"),
        "artifact_selection_warning": candidate.get("artifact_selection_warning", ""),
        "selection_metric": candidate.get("selection_metric", args.selection_metric),
        "selection_metric_best_epoch": candidate.get("selection_metric_best_epoch"),
        "selection_metric_best_value": candidate.get("selection_metric_best_value"),
        "full_test_root": args.full_test_root,
        "manifest": args.manifest,
        "split": args.split,
        "case_count": len(case_ids),
        "eval_surface_points": getattr(config, "eval_surface_points", None),
        "eval_volume_points": getattr(config, "eval_volume_points", None),
        "batch_size": getattr(config, "batch_size", None),
        "world_size": state.world_size,
        "cache_full_cases": args.cache_full_cases,
        "case_shard_cache": args.case_shard_cache,
        "use_curvature_attention_bias": bool(getattr(config, "use_curvature_attention_bias", False)),
        "curvature_stats": args.curvature_stats or str(REPO_ROOT / "curvature_proxy_stats_k16_v1.json"),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_selection_metric": checkpoint.get("selection_metric") if isinstance(checkpoint, dict) else None,
        "checkpoint_keys": sorted(k for k in checkpoint if isinstance(k, str)) if isinstance(checkpoint, dict) else [],
        "full_test_metrics": metrics,
        "elapsed_seconds": time.time() - start,
    }


def init_state_from_runtime() -> Any:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import trainer_runtime

    return trainer_runtime.init_distributed()


def cleanup_state(state: Any) -> None:
    import trainer_runtime

    trainer_runtime.cleanup_distributed(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="W&B run ID or run URL. Repeatable.")
    parser.add_argument("--candidate-json", action="append", default=[], help="Candidate JSON object/list or path.")
    parser.add_argument("--full-test-root", required=True, help="Processed root with full-resolution test case arrays.")
    parser.add_argument("--manifest", default="data/split_manifest.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--selection-metric", default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--artifact-alias", default="best")
    parser.add_argument("--allow-alias-fallback", action="store_true")
    parser.add_argument("--repo-url", default=os.environ.get("REPO_URL", DEFAULT_REPO_URL))
    parser.add_argument("--data-fix-ref", default=os.environ.get("DATA_FIX_REF", DEFAULT_DATA_FIX_REF))
    parser.add_argument("--use-current-code", action="store_true")
    parser.add_argument("--code-ref", default=os.environ.get("CODE_REF", ""))
    parser.add_argument("--work-root", type=Path, default=Path("/workspace/full-test-eval"))
    parser.add_argument("--artifact-root", type=Path, default=Path("/workspace/full-test-eval/artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/new-pvc/Reports/drivaerml_full_test_eval"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-surface-points", type=int, default=40_000)
    parser.add_argument("--eval-volume-points", type=int, default=40_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp-mode", default="bf16")
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", "full-test-eval"))
    parser.set_defaults(cache_full_cases=True, case_shard_cache=True)
    parser.add_argument("--cache-full-cases", dest="cache_full_cases", action="store_true")
    parser.add_argument("--no-cache-full-cases", dest="cache_full_cases", action="store_false")
    parser.add_argument("--case-shard-cache", dest="case_shard_cache", action="store_true")
    parser.add_argument("--no-case-shard-cache", dest="case_shard_cache", action="store_false")
    parser.add_argument("--curvature-stats", default=os.environ.get("CURVATURE_STATS", ""))
    parser.add_argument("--no-wandb-log", action="store_true")
    args = parser.parse_args()

    candidates = load_candidates(args)
    state = init_state_from_runtime()
    api = wandb.Api(timeout=60)
    results: list[dict[str, Any]] = []
    try:
        for candidate in candidates:
            result = evaluate_candidate(candidate, args, api, state)
            if state.is_main:
                results.append(result)
                append_jsonl(args.output_dir / "full_test_eval_results.jsonl", result)
                print(json.dumps(result, indent=2, sort_keys=True), flush=True)

        if state.is_main:
            average = {
                "status": "ok",
                "run_count": len(results),
                "run_ids": [result["run_id"] for result in results],
                "full_test_metric_average": metric_average(results),
            }
            (args.output_dir / "full_test_eval_average.json").write_text(
                json.dumps(average, indent=2, sort_keys=True) + "\n"
            )
            write_csv(
                args.output_dir / "full_test_eval_results.csv",
                [
                    {
                        "run_id": result["run_id"],
                        "pr": result.get("pr"),
                        "artifact": result["artifact"],
                        "artifact_selection": result["artifact_selection"],
                        "case_count": result["case_count"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        **flatten_metrics("full_test", result["full_test_metrics"]),
                    }
                    for result in results
                ],
            )
            print(json.dumps(average, indent=2, sort_keys=True), flush=True)

            if not args.no_wandb_log:
                run = wandb.init(
                    entity=args.entity,
                    project=args.project,
                    group=args.wandb_group,
                    job_type="full-test-eval",
                    name=f"full-test-eval-{int(time.time())}",
                    config={
                        "candidates": candidates,
                        "full_test_root": args.full_test_root,
                        "selection_metric": args.selection_metric,
                        "eval_surface_points": args.eval_surface_points,
                        "eval_volume_points": args.eval_volume_points,
                        "world_size": state.world_size,
                        "cache_full_cases": args.cache_full_cases,
                        "case_shard_cache": args.case_shard_cache,
                    },
                    mode=os.environ.get("WANDB_MODE", "online"),
                )
                log = flatten_metrics("full_test_average", average["full_test_metric_average"])
                wandb.log(log)
                wandb.summary.update(log)
                wandb.summary.update({"source_run_ids": average["run_ids"], "full_test_root": args.full_test_root})
                run.finish()
    finally:
        cleanup_state(state)


if __name__ == "__main__":
    main()
