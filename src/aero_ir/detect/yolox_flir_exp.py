"""Pinned YOLOX-s experiment definition for manifest-locked FLIR analytics16 inputs."""

from __future__ import annotations

import math
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from yolox.data import (
    DataLoader,
    InfiniteSampler,
    MosaicDetection,
    TrainTransform,
    ValTransform,
    YoloBatchSampler,
)
from yolox.exp import Exp as YOLOXExp
from yolox.utils import wait_for_the_master

from aero_ir.detect.flir_yolox import build_flir_yolox_dataset
from aero_ir.detect.yolox_evaluator import CompleteCOCOEvaluator
from aero_ir.detect.yolox_trainer import AccumulatingTrainer


def deterministic_worker_init(worker_id: int, *, base_seed: int) -> None:
    """Seed every augmentation RNG deterministically for one data-loader worker."""
    worker_seed = (int(base_seed) + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def enforce_deterministic_torch() -> None:
    """Undo YOLOX's benchmark override and reject nondeterministic Torch operations."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class Exp(YOLOXExp):
    """Official YOLOX-s recipe with only dataset and smoke-budget overrides."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = int(os.environ.get("AERO_YOLOX_NUM_CLASSES", "6"))
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.data_num_workers = int(os.environ.get("AERO_YOLOX_WORKERS", "4"))
        self.max_epoch = int(os.environ.get("AERO_YOLOX_MAX_EPOCHS", "300"))
        self.warmup_epochs = min(5, max(self.max_epoch - 1, 0))
        self.no_aug_epochs = min(15, self.max_epoch)
        self.eval_interval = int(os.environ.get("AERO_YOLOX_EVAL_INTERVAL", "1"))
        self.print_interval = int(os.environ.get("AERO_YOLOX_PRINT_INTERVAL", "10"))
        self.save_history_ckpt = os.environ.get("AERO_YOLOX_SAVE_HISTORY", "0") == "1"
        self.gradient_accumulation_steps = int(os.environ.get("AERO_YOLOX_GRAD_ACCUM", "8"))
        self.effective_batch_size = int(os.environ.get("AERO_YOLOX_EFFECTIVE_BATCH", "64"))
        if self.gradient_accumulation_steps <= 0 or self.effective_batch_size <= 0:
            raise ValueError("YOLOX accumulation and effective batch must be positive")
        self.run_mode = os.environ.get("AERO_YOLOX_RUN_MODE", "train")
        self.max_train_iters = int(os.environ.get("AERO_YOLOX_MAX_TRAIN_ITERS", "0"))
        self.timing_warmup_iters = int(os.environ.get("AERO_YOLOX_TIMING_WARMUP_ITERS", "0"))
        if self.run_mode not in {"train", "timing"}:
            raise ValueError("AERO_YOLOX_RUN_MODE must be train or timing")
        if self.run_mode == "timing" and self.max_train_iters <= 0:
            raise ValueError("timing mode requires positive AERO_YOLOX_MAX_TRAIN_ITERS")
        if self.run_mode == "train" and self.max_train_iters:
            raise ValueError("bounded iterations are allowed only in timing mode")
        self.seed = int(os.environ.get("AERO_YOLOX_SEED", "0"))
        self.output_dir = os.environ.get("AERO_YOLOX_OUTPUT", "experiments/yolox_runs")
        self.image_root = os.environ["AERO_FLIR_ROOT"]
        if "\n" in self.image_root or "\r" in self.image_root:
            raise ValueError(
                f"AERO_FLIR_ROOT contains a newline; set it on one shell line: {self.image_root!r}"
            )
        if not Path(self.image_root).is_dir():
            raise FileNotFoundError(f"AERO_FLIR_ROOT is not a directory: {self.image_root!r}")
        self.prepared_root = os.environ.get("AERO_FLIR_YOLOX_ROOT", "experiments/flir_yolox")
        self.preprocess_path = os.environ.get(
            "AERO_FLIR_PREPROCESS", "experiments/flir_preprocess.json"
        )
        self.train_ann = os.environ.get("AERO_YOLOX_TRAIN_ANN", "train.json")
        self.val_ann = os.environ.get("AERO_YOLOX_VAL_ANN", "val.json")
        self.exp_name = "aero_flir_yolox_s"

    def get_optimizer(self, batch_size):
        """Scale optimizer hyperparameters by the declared effective batch."""
        del batch_size
        return super().get_optimizer(self.effective_batch_size)

    def get_lr_scheduler(self, lr, iters_per_epoch):
        """Advance the upstream schedule once per optimizer step."""
        del lr
        optimizer_steps = math.ceil(iters_per_epoch / self.gradient_accumulation_steps)
        effective_lr = self.basic_lr_per_img * self.effective_batch_size
        return super().get_lr_scheduler(effective_lr, optimizer_steps)

    def get_trainer(self, args):
        enforce_deterministic_torch()
        return AccumulatingTrainer(self, args)

    def _dataset(self, annotation_file: str, transform):
        return build_flir_yolox_dataset(
            image_root=self.image_root,
            prepared_root=self.prepared_root,
            annotation_file=annotation_file,
            preprocess_path=self.preprocess_path,
            image_size=self.input_size,
            transform=transform,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        if cache_img:
            raise ValueError("image caching is disabled for manifest-locked analytics16 inputs")
        with wait_for_the_master():
            dataset = self._dataset(
                self.train_ann,
                TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
            )
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        self.dataset = dataset
        if is_distributed:
            batch_size //= dist.get_world_size()
        sampler = InfiniteSampler(len(dataset), seed=self.seed)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        return DataLoader(
            dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            worker_init_fn=partial(deterministic_worker_init, base_seed=self.seed),
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        if testdev:
            raise ValueError("test split is not available to the Phase 1 training adapter")
        dataset = self._dataset(self.val_ann, ValTransform(legacy=legacy))
        if is_distributed:
            batch_size //= dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=batch_size,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        metrics_path = os.environ.get("AERO_YOLOX_METRICS_PATH")
        predictions_path = os.environ.get("AERO_YOLOX_PREDICTIONS_PATH")
        if bool(metrics_path) != bool(predictions_path):
            raise ValueError("metrics and predictions paths must be configured together")
        if not metrics_path:
            return super().get_evaluator(batch_size, is_distributed, testdev, legacy)
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        return CompleteCOCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
        )
