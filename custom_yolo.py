from collections import Counter
from copy import copy, deepcopy
import csv
from datetime import datetime
from pathlib import Path
import torch
from ultralytics import YOLO, __version__
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.cfg import TASK2DATA
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import (
    DEFAULT_CFG,
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)
from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16
from ultralytics.utils.metrics import DetMetrics
from custom_eval import eval_video, BINARY_KEYS, get_test_files, TEST_DIR
import json
import torch.nn as nn


def my_eval(model, epoch, batch, bias_path):
    test_files = get_test_files(TEST_DIR)
    model.model.eval()
    res = eval_video(model, test_files, 1e-4, 0.7, BINARY_KEYS)
    res["epoch"] = epoch
    res["batch"] = batch
    with open(str(bias_path).replace("csv", "jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    model.model.train()


def on_train_start_callback(self: DetectionTrainer):
    self.validator.metrics.epoch = self.start_epoch
    self.validator.metrics.batch = 0


def on_epoch_end_callback(self: DetectionTrainer):
    self.validator.metrics.epoch += 1
    self.validator.metrics.batch = 0


def on_train_batch_end_callback(self: DetectionTrainer):
    if (
        self.n_batches is not None
        and self.validator.metrics.batch % self.n_batches == 0
    ):
        self.save_model()
        # self.validate()
        # my_eval(
        #     self.parent,
        #     self.validator.metrics.epoch,
        #     self.validator.metrics.batch,
        #     self.validator.metrics.bias_path,
        # )
    self.validator.metrics.batch += 1


class MyDetectionTrainer(DetectionTrainer):

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks=None,
        n_batches=10,
    ):
        self.n_batches = n_batches
        super().__init__(cfg, overrides, _callbacks)

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return MyDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            bias_path=self.save_dir / "bias.csv",
        )

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        import pandas as pd  # scope for faster 'import ultralytics'

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        try:
            train_results = {
                k.strip(): v
                for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()
            }
        except Exception:
            train_results = None
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(
                    deepcopy(self.optimizer.state_dict())
                ),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": train_results,
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        mini_checkpoint = (
            self.wdir / f"epoch{self.epoch}-batch{self.validator.metrics.batch}.pt"
        )
        mini_checkpoint.write_bytes(
            serialized_ckpt
        )  # SAVE CHECKPOINT AT BATCH INTERVAL

        print(f"saved mini checkpoint: {mini_checkpoint}")

        # Save checkpoints
        if self.validator.metrics.batch == len(self.train_loader):
            self.last.write_bytes(serialized_ckpt)  # save last.pt
            if self.best_fitness == self.fitness:
                self.best.write_bytes(serialized_ckpt)  # save best.pt
            if (
                (self.save_period > 0)
                and (self.epoch > 0)
                and (self.epoch % self.save_period == 0)
            ):
                (self.wdir / f"epoch{self.epoch}.pt").write_bytes(
                    serialized_ckpt
                )  # save epoch, i.e. 'epoch3.pt'


class MyDetectionValidator(DetectionValidator):

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        pbar=None,
        args=None,
        _callbacks=None,
        bias_path=None,
    ):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = MyDetMetrics(
            save_dir=self.save_dir, on_plot=self.on_plot, bias_path=bias_path
        )  # HERE
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling


class MyDetMetrics(DetMetrics):
    def __init__(
        self, save_dir=Path("."), plot=False, on_plot=None, names=(), bias_path=None
    ) -> None:
        self.bias_path = bias_path
        with open(self.bias_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "batch",
                    "num_boxes",
                    "num_planes",
                    "num_birds",
                    "plane_conf",
                    "bird_conf",
                    "plane_guesses",
                    "bird_guesses",
                ]
            )
        super().__init__(save_dir, plot, on_plot, names)

    def process(self, tp, conf, pred_cls, target_cls):
        # DO SOMETHING TO WRITE THE RESULTS
        print(len(conf))
        print(len(pred_cls))
        print(len(target_cls))
        plane_conf_tot = [c for c, l in zip(conf, pred_cls) if l == 0]
        bird_conf_tot = [c for c, l in zip(conf, pred_cls) if l == 1]
        plane_conf = sum(plane_conf_tot) / len(plane_conf_tot)
        bird_conf = sum(bird_conf_tot) / len(bird_conf_tot)
        print(f"Avg. plane conf: {plane_conf}")
        print(f"Avg. bird conf: {bird_conf}")
        num_planes = len([l for l in target_cls if l == 0])
        num_plane_guesses = len([l for l in pred_cls if l == 0])
        num_birds = len([l for l in target_cls if l == 1])
        num_birds_guesses = len([l for l in pred_cls if l == 1])
        print(f"num_plane labels: {num_planes} ({100 * num_planes/len(target_cls)}%)")
        print(
            f"num_plane guesses: {num_plane_guesses} ({100 * num_plane_guesses / len(pred_cls)})"
        )

        print(f"num_bird labels: {num_birds} ({100 * num_birds/len(target_cls)}%)")
        print(
            f"num_bird guesses: {num_birds_guesses} ({100 * num_birds_guesses / len(pred_cls)})"
        )

        with open(self.bias_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.epoch,
                    self.batch,
                    len(pred_cls),
                    num_planes,
                    num_birds,
                    plane_conf,
                    bird_conf,
                    num_plane_guesses,
                    num_birds_guesses,
                ]
            )

        return super().process(tp, conf, pred_cls, target_cls)
