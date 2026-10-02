"""One matched optimizer update per arm; not a shortened research experiment."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from aero_ir.detect.flir_yolox import build_flir_yolox_dataset
from aero_ir.utils.manifest import file_sha256
from scripts.prepare_flir_mixture_smoke import PREPROCESS, ROOT

INPUT = Path("experiments/flir_mixture_loader_smoke_01/report.json")
INPUT_SHA = "ae2ff26d32ca4f8ed83212dbe8948a64a4b44c892064a0321454c05ca4bcfefd"
ARMS = ("real_only", "real_plus_procedural")


def batches(size=32, seed=0):
    if size != 32:
        raise ValueError("fixed 32-image engineering panel required")
    order = np.random.default_rng(seed).permutation(size).tolist()
    return [(order + order)[i : i + 8] for i in range(0, 64, 8)]


def digest_state(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def preflight():
    if file_sha256(INPUT) != INPUT_SHA:
        raise ValueError("loader input report changed")
    report = json.loads(INPUT.read_text())
    hashes = report["input_and_source_sha256"].copy()
    hashes[str(INPUT)] = INPUT_SHA
    for arm in ARMS:
        data = report["arms"][arm]
        if data["images"] != 32:
            raise ValueError("unmatched image budget")
        path = INPUT.parent / data["annotation_file"]
        if not path.resolve().is_relative_to(INPUT.parent.resolve()):
            raise ValueError("unsafe annotation path")
        hashes[str(path)] = data["sha256"]
    for path, sha in hashes.items():
        if file_sha256(path) != sha:
            raise ValueError(f"changed input: {path}")
    import yolox

    for path in Path(yolox.__file__).parent.rglob("*.py"):
        hashes[str(path)] = file_sha256(path)
    for path in (Path(__file__), Path("scripts/run_flir_mixture_gpu_smoke.sh")):
        hashes[str(path)] = file_sha256(path)
    return report, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    report, hashes = preflight()
    if args.action == "preflight":
        print(
            json.dumps(
                dict(
                    arms=ARMS,
                    microbatches=8,
                    effective_batch=64,
                    optimizer_updates_per_arm=1,
                    gpu_run=False,
                )
            )
        )
        return
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    from yolox.data import TrainTransform
    from yolox.exp import Exp

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    max_labels = max(x["labels"] for arm in ARMS for x in report["arms"][arm]["loaded"])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    initial, initial_sha = None, None
    results = {}
    for arm in ARMS:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        exp = Exp()
        exp.num_classes, exp.depth, exp.width, exp.warmup_epochs = 6, 0.33, 0.5, 0
        model = exp.get_model()
        if initial is None:
            initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            initial_sha = digest_state(initial)
        model.load_state_dict(initial, strict=True)
        if digest_state(model.state_dict()) != initial_sha:
            raise ValueError("initial weights differ between arms")
        model.cuda().train()
        optimizer = exp.get_optimizer(64)
        optimizer.zero_grad(set_to_none=True)
        parameter_before = digest_state(dict(model.named_parameters()))
        dataset = build_flir_yolox_dataset(
            image_root=ROOT,
            prepared_root=INPUT.parent,
            annotation_file=Path(report["arms"][arm]["annotation_file"]).name,
            preprocess_path=PREPROCESS,
            transform=TrainTransform(max_labels=max_labels, flip_prob=0, hsv_prob=0),
        )
        trace = []
        for batch_index, indices in enumerate(batches(len(dataset))):
            loaded = [dataset[index] for index in indices]
            images = torch.from_numpy(np.stack([x[0] for x in loaded])).cuda()
            targets = torch.from_numpy(np.stack([x[1] for x in loaded])).cuda()
            with torch.autocast("cuda", enabled=False):
                output = model(images, targets)
                loss = output["total_loss"]
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / 8).backward()
            trace.append(
                dict(
                    microbatch=batch_index,
                    indices=indices,
                    image_ids=[int(np.asarray(x[3]).item()) for x in loaded],
                    retained_targets=int(((targets[:, :, 3] > 0) & (targets[:, :, 4] > 0)).sum()),
                    losses={
                        k: float(v.detach()) if torch.is_tensor(v) else float(v)
                        for k, v in output.items()
                    },
                )
            )
            print(
                f"{arm} microbatch {batch_index + 1}/8 loss={float(loss.detach()):.5f}", flush=True
            )
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError("missing or nonfinite gradients")
        optimizer.step()
        parameter_after = digest_state(dict(model.named_parameters()))
        if parameter_after == parameter_before:
            raise ValueError("optimizer did not update parameters")
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError("nonfinite updated model")
        destination = args.out_dir / f"{arm}_engineering_only.pth"
        torch.save(
            dict(kind="engineering_smoke_not_research_checkpoint", model_state=state), destination
        )
        replay = torch.load(destination, map_location="cpu", weights_only=True)["model_state"]
        if digest_state(replay) != digest_state(state):
            raise ValueError("checkpoint serialization differs")
        results[arm] = dict(
            initial_state_sha256=initial_sha,
            trace=trace,
            optimizer_updates=1,
            gradient_is_finite=True,
            parameters_before_sha256=parameter_before,
            parameters_after_sha256=parameter_after,
            checkpoint=destination.name,
            checkpoint_sha256=file_sha256(destination),
        )
        del model, exp, optimizer, gradients, output, loss, images, targets
        torch.cuda.empty_cache()
    if preflight()[1] != hashes:
        raise ValueError("inputs changed during smoke")
    result = dict(
        arms=results,
        input_and_source_sha256=hashes,
        settings=dict(
            microbatch=8,
            accumulation=8,
            effective_batch=64,
            unique_images=32,
            epochs_equivalent=2,
            optimizer_updates=1,
            fp16=False,
            augmentations=False,
            max_labels=max_labels,
            learning_rate=0.01,
            ema=False,
            warmup=False,
        ),
        gpu=torch.cuda.get_device_name(0),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        registration_qualified=False,
        full_experiment_completed=False,
        limitations=[
            "One update per arm tests plumbing, not detection quality.",
            "No AP or validation/test inference and no full-training claim.",
            "No augmentation/EMA/warmup; not a shortened formal protocol.",
        ],
    )
    with (args.out_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"wrote {args.out_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
