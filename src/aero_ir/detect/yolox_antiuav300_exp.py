"""Pinned YOLOX-s engineering experiment for prepared Anti-UAV300 native IR."""

from __future__ import annotations

import math
import os
from functools import partial
from pathlib import Path

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

from aero_ir.detect.antiuav300_yolox import build_antiuav300_yolox_dataset
from aero_ir.detect.yolox_evaluator import CompleteCOCOEvaluator
from aero_ir.detect.yolox_flir_exp import deterministic_worker_init, enforce_deterministic_torch
from aero_ir.detect.yolox_trainer import AccumulatingTrainer


class Exp(YOLOXExp):
    """YOLOX-s with a one-class, native-IR, engineering-only data contract."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 1
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.data_num_workers = int(os.environ.get("AERO_YOLOX_WORKERS", "4"))
        self.max_epoch = int(os.environ.get("AERO_YOLOX_MAX_EPOCHS", "1"))
        self.warmup_epochs = 0
        self.no_aug_epochs = self.max_epoch
        self.eval_interval = int(os.environ.get("AERO_YOLOX_EVAL_INTERVAL", "1"))
        self.print_interval = int(os.environ.get("AERO_YOLOX_PRINT_INTERVAL", "10"))
        self.save_history_ckpt = False
        self.gradient_accumulation_steps = int(os.environ.get("AERO_YOLOX_GRAD_ACCUM", "2"))
        self.effective_batch_size = int(os.environ.get("AERO_YOLOX_EFFECTIVE_BATCH", "64"))
        if self.gradient_accumulation_steps <= 0 or self.effective_batch_size <= 0:
            raise ValueError("YOLOX accumulation and effective batch must be positive")
        self.run_mode = "train"
        self.max_train_iters = 0
        self.timing_warmup_iters = 0
        self.seed = int(os.environ.get("AERO_YOLOX_SEED", "0"))
        self.output_dir = os.environ.get("AERO_YOLOX_OUTPUT", "experiments/yolox_runs")
        self.prepared_root = os.environ.get(
            "AERO_ANTIUAV300_IR_PREPARED", "experiments/antiuav300_ir_yolox"
        )
        if "\n" in self.prepared_root or "\r" in self.prepared_root:
            raise ValueError("AERO_ANTIUAV300_IR_PREPARED contains a newline")
        if not Path(self.prepared_root).is_dir():
            raise FileNotFoundError(
                f"prepared Anti-UAV300 IR root is not a directory: {self.prepared_root!r}"
            )
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.exp_name = "aero_antiuav300_native_ir_yolox_s"

    def get_optimizer(self, batch_size):
        del batch_size
        return super().get_optimizer(self.effective_batch_size)

    def get_lr_scheduler(self, lr, iters_per_epoch):
        del lr
        optimizer_steps = math.ceil(iters_per_epoch / self.gradient_accumulation_steps)
        effective_lr = self.basic_lr_per_img * self.effective_batch_size
        return super().get_lr_scheduler(effective_lr, optimizer_steps)

    def get_trainer(self, args):
        enforce_deterministic_torch()
        return AccumulatingTrainer(self, args)

    def _dataset(self, annotation_file: str, transform):
        return build_antiuav300_yolox_dataset(
            prepared_root=Path(self.prepared_root),
            annotation_file=annotation_file,
            image_size=self.input_size,
            transform=transform,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        if cache_img:
            raise ValueError("image caching is disabled for manifest-locked inputs")
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
            raise ValueError("test split is prohibited for the native-IR engineering smoke")
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
        if not metrics_path or not predictions_path:
            raise ValueError("Anti-UAV300 smoke requires metrics and predictions paths")
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
            metrics_kind="antiuav300_native_ir_yolox_coco_metrics",
        )
