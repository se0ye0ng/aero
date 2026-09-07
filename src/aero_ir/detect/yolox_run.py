"""Content-addressed lifecycle for FLIR YOLOX timing and reportable training runs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from aero_ir.data.flir import verify_detector_manifest
from aero_ir.data.preprocess import verify_preprocess_spec
from aero_ir.utils.manifest import RunManifest, canonical_hash, file_sha256

RUN_SPEC_SCHEMA = 1
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
DEFAULT_CODE_PATHS = (
    "pyproject.toml",
    "requirements/yolox.txt",
    "requirements/yolox-runtime.txt",
    "src/aero_ir/data/flir.py",
    "src/aero_ir/data/preprocess.py",
    "src/aero_ir/detect/evaluate.py",
    "src/aero_ir/detect/flir_yolox.py",
    "src/aero_ir/detect/yolox_evaluator.py",
    "src/aero_ir/detect/yolox_flir_exp.py",
    "src/aero_ir/detect/yolox_run.py",
    "src/aero_ir/detect/yolox_trainer.py",
    "src/aero_ir/utils/manifest.py",
    "scripts/run_flir_yolox.py",
)


def _artifact(path: str | Path) -> dict[str, str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _verify_artifact(name: str, artifact: dict) -> None:
    if set(artifact) != {"path", "sha256"}:
        raise ValueError(f"run-spec artifact {name!r} must contain path and sha256")
    path = Path(artifact["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != artifact["sha256"]:
        raise ValueError(f"run-spec artifact {name!r} hash mismatch")


def _git_state(repository_root: Path) -> dict[str, str | bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "sha": sha.stdout.strip() if sha.returncode == 0 else "unknown",
        "dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def prepare_yolox_run_spec(
    *,
    mode: str,
    run_id: str,
    repository_root: str | Path,
    dataset_root: str | Path,
    dataset_manifest: str | Path,
    preprocess: str | Path,
    prepared_root: str | Path,
    output_root: str | Path,
    seed: int = 0,
    epochs: int = 300,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 8,
    effective_batch_size: int = 64,
    workers: int = 2,
    max_train_iters: int = 0,
    timing_warmup_iters: int = 0,
    print_interval: int = 10,
    eval_interval: int = 10,
    fp16: bool = True,
    code_paths: tuple[str, ...] = DEFAULT_CODE_PATHS,
) -> dict:
    """Freeze and validate every input needed before a GPU process may start."""
    if mode not in {"timing", "train"}:
        raise ValueError("mode must be timing or train")
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and dash")
    if min(epochs, batch_size, gradient_accumulation_steps, effective_batch_size, workers) <= 0:
        raise ValueError("epochs, batch sizes, accumulation, and workers must be positive")
    if batch_size * gradient_accumulation_steps != effective_batch_size:
        raise ValueError("batch_size x gradient_accumulation_steps must equal effective_batch_size")
    if mode == "timing":
        if epochs <= 15:
            raise ValueError("timing mode requires epochs > 15 so normal augmentation stays active")
        if max_train_iters <= 0 or max_train_iters % gradient_accumulation_steps:
            raise ValueError(
                "timing max_train_iters must be positive and divisible by accumulation"
            )
        if not 0 <= timing_warmup_iters < max_train_iters:
            raise ValueError("timing warm-up must leave measured iterations")
    elif max_train_iters or timing_warmup_iters:
        raise ValueError("bounded iterations and timing warm-up are timing-only settings")

    repository_root = Path(repository_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    dataset_manifest = Path(dataset_manifest).resolve()
    preprocess = Path(preprocess).resolve()
    prepared_root = Path(prepared_root).resolve()
    output_root = Path(output_root).resolve()
    if "\n" in str(dataset_root) or "\r" in str(dataset_root):
        raise ValueError("dataset_root contains a newline")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    manifest_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    verify_detector_manifest(manifest_payload)
    preprocess_payload = json.loads(preprocess.read_text(encoding="utf-8"))
    verify_preprocess_spec(
        preprocess_payload,
        expected_manifest_sha256=manifest_payload["manifest_sha256"],
    )
    preflight_path = prepared_root / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("gate") != "pass":
        raise ValueError("prepared FLIR YOLOX preflight gate did not pass")

    annotations: dict[str, dict[str, str]] = {}
    for split in ("train", "val"):
        path = prepared_root / "annotations" / f"{split}.json"
        artifact = _artifact(path)
        expected = preflight.get("annotations", {}).get(split, {}).get("sha256")
        if artifact["sha256"] != expected:
            raise ValueError(f"full {split} annotation hash differs from preflight")
        annotation = json.loads(path.read_text(encoding="utf-8"))
        info = annotation.get("info", {})
        if info.get("engineering_smoke_only"):
            raise ValueError(f"{split} annotation is an engineering subset")
        if info.get("aero_manifest_sha256") != manifest_payload["manifest_sha256"]:
            raise ValueError(f"{split} annotation refers to a different detector manifest")
        annotations[split] = artifact

    code = {path.replace("/", "__"): _artifact(repository_root / path) for path in code_paths}
    resolved_config = {
        "mode": mode,
        "run_id": run_id,
        "dataset": "flir_urban",
        "representation": "analytics16_fixed_train_window",
        "train_annotation": "train.json",
        "val_annotation": "val.json",
        "detector": "yolox-s",
        "yolox_version": "0.3.0",
        "initialization": "scratch",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "workers": workers,
        "max_train_iters": max_train_iters,
        "timing_warmup_iters": timing_warmup_iters,
        "print_interval": print_interval,
        "eval_interval": eval_interval,
        "fp16": fp16,
        "augmentation": "yolox_default_mosaic_mixup_multiscale",
        "determinism": {
            "worker_seed": "run_seed_plus_worker_id",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            "pythonhashseed": str(seed),
        },
        "save_history_checkpoints": False,
    }
    spec = {
        "schema_version": RUN_SPEC_SCHEMA,
        "kind": "flir_yolox_run_spec",
        "scientific_status": (
            "engineering_only_not_reportable" if mode == "timing" else "reportable_candidate"
        ),
        "run_id": run_id,
        "repository_root": str(repository_root),
        "dataset_root": str(dataset_root),
        "prepared_root": str(prepared_root),
        "output_root": str(output_root),
        "exp_file": str(repository_root / "src/aero_ir/detect/yolox_flir_exp.py"),
        "config_hash": canonical_hash(resolved_config),
        "resolved_config": resolved_config,
        "dataset_ids": {
            "manifest_sha256": manifest_payload["manifest_sha256"],
            "preprocess_sha256": preprocess_payload["preprocess_sha256"],
        },
        "inputs": {
            "dataset_manifest": _artifact(dataset_manifest),
            "preprocess": _artifact(preprocess),
            "prepared_preflight": _artifact(preflight_path),
            "train_annotations": annotations["train"],
            "val_annotations": annotations["val"],
        },
        "code": code,
        "git": _git_state(repository_root),
        "prepared_at": datetime.now(UTC).isoformat(),
    }
    spec["spec_sha256"] = canonical_hash(spec)
    return spec


def save_yolox_run_spec(spec: dict, path: str | Path) -> str:
    verify_yolox_run_spec(spec, verify_files=True)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite run spec: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(spec["spec_sha256"])


def load_yolox_run_spec(path: str | Path, *, verify_files: bool = True) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_yolox_run_spec(spec, verify_files=verify_files)
    return spec


def verify_yolox_run_spec(spec: dict, *, verify_files: bool = True) -> None:
    if spec.get("schema_version") != RUN_SPEC_SCHEMA:
        raise ValueError("unsupported YOLOX run-spec schema")
    recorded = spec.get("spec_sha256")
    unsigned = {key: value for key, value in spec.items() if key != "spec_sha256"}
    if not isinstance(recorded, str) or canonical_hash(unsigned) != recorded:
        raise ValueError("YOLOX run-spec hash mismatch")
    if canonical_hash(spec.get("resolved_config", {})) != spec.get("config_hash"):
        raise ValueError("YOLOX run-spec resolved config hash mismatch")
    if verify_files:
        for group in ("inputs", "code"):
            for name, artifact in spec.get(group, {}).items():
                _verify_artifact(name, artifact)


def _relative_artifact(path: Path, run_dir: Path) -> dict[str, str]:
    return {
        "path": os.path.relpath(path.resolve(), run_dir.resolve()),
        "sha256": file_sha256(path),
    }


def _verify_metrics_payload(payload: dict) -> None:
    recorded = payload.get("metrics_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "metrics_sha256"}
    if not isinstance(recorded, str) or canonical_hash(unsigned) != recorded:
        raise ValueError("YOLOX metrics payload hash mismatch")
    if payload.get("status") != "pass":
        raise ValueError("YOLOX metrics status did not pass")


def finalize_yolox_run(
    spec_path: str | Path,
    run_dir: str | Path,
    *,
    started_at: str,
) -> Path:
    """Bind completed GPU outputs and complete metrics into a replayable manifest."""
    spec_path = Path(spec_path).resolve()
    spec = load_yolox_run_spec(spec_path)
    run_dir = Path(run_dir).resolve()
    mode = spec["resolved_config"]["mode"]
    metrics_name = "timing_metrics.json" if mode == "timing" else "metrics.json"
    metrics_path = run_dir / metrics_name
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    _verify_metrics_payload(metrics_payload)
    runtime_path = run_dir / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_recorded = runtime.get("runtime_sha256")
    runtime_unsigned = {key: value for key, value in runtime.items() if key != "runtime_sha256"}
    if (
        not isinstance(runtime_recorded, str)
        or canonical_hash(runtime_unsigned) != runtime_recorded
    ):
        raise ValueError("YOLOX runtime payload hash mismatch")

    artifacts: dict[str, dict[str, str]] = {
        "run_spec": _relative_artifact(spec_path, run_dir),
        "metrics": _relative_artifact(metrics_path, run_dir),
        "train_log": _relative_artifact(run_dir / "train_log.txt", run_dir),
        "runtime": _relative_artifact(runtime_path, run_dir),
    }
    for name, artifact in {**spec["inputs"], **spec["code"]}.items():
        artifacts[f"input__{name}"] = _relative_artifact(Path(artifact["path"]), run_dir)
    event_files = sorted((run_dir / "tensorboard").glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise ValueError(f"expected one TensorBoard event file, found {len(event_files)}")
    artifacts["tensorboard"] = _relative_artifact(event_files[0], run_dir)

    if mode == "train":
        for name in ("predictions.json", "latest_ckpt.pth", "last_epoch_ckpt.pth"):
            artifacts[name] = _relative_artifact(run_dir / name, run_dir)
        best = run_dir / "best_ckpt.pth"
        if best.is_file():
            artifacts["best_ckpt.pth"] = _relative_artifact(best, run_dir)
        comparison_metrics = {
            key: metrics_payload[key]
            for key in (
                "status",
                "evaluator",
                "iou_type",
                "images",
                "detections",
                "map_50_95",
                "map_50",
                "mar_50_95",
                "mar_50",
                "per_class",
            )
        }
    else:
        comparison_metrics = {
            key: metrics_payload[key]
            for key in (
                "status",
                "kind",
                "full_dataset_images",
                "full_iterations_per_epoch",
                "executed_iterations",
                "executed_optimizer_steps",
                "configured_epochs",
                "executed_epochs",
                "microbatch_size",
                "gradient_accumulation_steps",
                "effective_batch_size",
                "normal_augmentation",
                "all_losses_finite",
            )
        }

    environment = {
        "python_executable": sys.executable,
        "python_version": runtime["python_version"],
        "torch_version": runtime["torch_version"],
        "torch_cuda_version": runtime["torch_cuda_version"],
        "yolox_version": "0.3.0",
        "determinism": runtime.get("determinism", {}),
    }
    hardware = {
        "host": runtime["host"],
        "cpu_count": os.cpu_count(),
        "cuda_available": True,
        "device_name": runtime["device_name"],
        "peak_cuda_memory_mib": runtime["peak_cuda_memory_mib"],
    }
    repository_root = Path(spec["repository_root"])
    command_spec = os.path.relpath(spec_path, repository_root)
    manifest = RunManifest(
        run_id=spec["run_id"],
        config_hash=spec["config_hash"],
        experiment=f"flir_yolox_{mode}",
        seed=int(spec["resolved_config"]["seed"]),
        dataset_checksums=spec["dataset_ids"],
        resolved_config=spec["resolved_config"],
        command=[
            sys.executable,
            "scripts/run_flir_yolox.py",
            "execute",
            "--spec",
            command_spec,
        ],
        command_environment={
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            "PYTHONHASHSEED": str(spec["resolved_config"]["seed"]),
        },
        unset_environment=["LD_LIBRARY_PATH"],
        working_directory=str(repository_root),
        metrics_path=metrics_name,
        replay_metrics_path=f"{spec['run_id']}/{metrics_name}",
        metrics=comparison_metrics,
        artifacts=artifacts,
        environment=environment,
        hardware=hardware,
        git_sha=str(spec["git"]["sha"]),
        git_dirty=bool(spec["git"]["dirty"]),
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest.save(manifest_path)
    return manifest_path


def execute_yolox_run(spec_path: str | Path) -> Path:
    """Execute a verified spec and finalize its outputs; this is the only GPU boundary."""
    spec_path = Path(spec_path).resolve()
    spec = load_yolox_run_spec(spec_path)
    config = spec["resolved_config"]
    output_root = Path(os.environ.get("AERO_REPLAY_OUTPUT_ROOT", spec["output_root"])).resolve()
    run_dir = output_root / spec["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {run_dir}")

    os.environ.pop("LD_LIBRARY_PATH", None)
    environment = dict(os.environ)
    for stale in ("AERO_YOLOX_METRICS_PATH", "AERO_YOLOX_PREDICTIONS_PATH"):
        environment.pop(stale, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            "PYTHONHASHSEED": str(config["seed"]),
            "PYTHONPATH": str(Path(spec["repository_root"]) / "src"),
            "AERO_FLIR_ROOT": spec["dataset_root"],
            "AERO_FLIR_YOLOX_ROOT": spec["prepared_root"],
            "AERO_FLIR_PREPROCESS": spec["inputs"]["preprocess"]["path"],
            "AERO_YOLOX_OUTPUT": str(output_root),
            "AERO_YOLOX_RUN_MODE": config["mode"],
            "AERO_YOLOX_MAX_EPOCHS": str(config["epochs"]),
            "AERO_YOLOX_EVAL_INTERVAL": str(config["eval_interval"]),
            "AERO_YOLOX_PRINT_INTERVAL": str(config["print_interval"]),
            "AERO_YOLOX_WORKERS": str(config["workers"]),
            "AERO_YOLOX_SEED": str(config["seed"]),
            "AERO_YOLOX_GRAD_ACCUM": str(config["gradient_accumulation_steps"]),
            "AERO_YOLOX_EFFECTIVE_BATCH": str(config["effective_batch_size"]),
            "AERO_YOLOX_MAX_TRAIN_ITERS": str(config["max_train_iters"]),
            "AERO_YOLOX_TIMING_WARMUP_ITERS": str(config["timing_warmup_iters"]),
            "AERO_YOLOX_TRAIN_ANN": config["train_annotation"],
            "AERO_YOLOX_VAL_ANN": config["val_annotation"],
            "AERO_YOLOX_SAVE_HISTORY": "1" if config["save_history_checkpoints"] else "0",
        }
    )
    if config["mode"] == "train":
        environment["AERO_YOLOX_METRICS_PATH"] = str(run_dir / "metrics.json")
        environment["AERO_YOLOX_PREDICTIONS_PATH"] = str(run_dir / "predictions.json")

    command = [
        sys.executable,
        "-m",
        "yolox.tools.train",
        "-f",
        spec["exp_file"],
        "-d",
        "1",
        "-b",
        str(config["batch_size"]),
    ]
    if config["fp16"]:
        command.append("--fp16")
    command.extend(["-expn", spec["run_id"], "-l", "tensorboard"])
    started_at = datetime.now(UTC).isoformat()
    subprocess.run(
        command,
        cwd=spec["repository_root"],
        env=environment,
        check=True,
    )
    return finalize_yolox_run(spec_path, run_dir, started_at=started_at)
