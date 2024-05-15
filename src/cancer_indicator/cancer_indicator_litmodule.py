import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAUROC, MulticlassF1Score
from torchmetrics.classification.accuracy import MulticlassAccuracy

from src import utils

log = utils.get_pylogger(__name__)


class CancerIndicatorLitModule(LightningModule):
    """A LightningModule organizes your PyTorch code into 6 sections:

        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        n_classes: int,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        use_weights: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # TODO derive this from the metrics that will be saved
        self.use_weights = use_weights

        # We need to define a "dummy" criterion here, because during evaluation/predict
        # lightning tries to restore the criterion from the ckpt and fails if it doesn't exist.
        # We need to overwrite with the actual criterion though in on_fit_start(), because
        # in the init the datamodule and therefore the label distribution is not yet available.
        # TODO: find a better way to do this (there seems to be no consensus on how to do this)
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.rand(n_classes) if self.use_weights else None
        )

        # TODO: create analogue to surival (val/acc, etc.)
        self.metrics = torch.nn.ModuleDict(
            {
                "val_acc_best": MaxMetric(),
                "val_auroc_best": MaxMetric(),
                "val_f1_best": MaxMetric(),
            }
        )
        for mode in ["train", "val", "test"]:
            self.metrics[f"{mode}_acc"] = MulticlassAccuracy(num_classes=n_classes)
            self.metrics[f"{mode}_auroc"] = MulticlassAUROC(num_classes=n_classes)
            self.metrics[f"{mode}_f1"] = MulticlassF1Score(num_classes=n_classes)
            self.metrics[f"{mode}_loss"] = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_fit_start(self):
        # https://discuss.pytorch.org/t/how-to-set-bceloss-with-weight-in-pytorch-lightning/130398
        if self.use_weights:
            self.label_dfs = self.trainer.datamodule.label_dfs.copy()
            sample_weights = torch.Tensor(get_class_weights(self.label_dfs["train"])).to(
                self.device
            )

        self.criterion = torch.nn.CrossEntropyLoss(
            weight=sample_weights if self.use_weights else None
        )

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        for metric in ["acc", "auroc", "f1"]:
            self.metrics[f"val_{metric}_best"].reset()

    def step(self, x: Any, y: Any):
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        return loss, probs, preds, logits, y

    def shared_step(self, batch: Any, mode: str):
        data, targets, *args = batch
        loss, probs, preds, logits, targets = self.step(data, targets)

        self.metrics[f"{mode}_loss"](loss)
        self.metrics[f"{mode}_acc"](preds, targets)
        self.metrics[f"{mode}_auroc"](probs, targets)
        self.metrics[f"{mode}_f1"](preds, targets)

        for metric in ["acc", "auroc", "f1", "loss"]:
            self.log(
                f"{mode}/{metric}",
                self.metrics[f"{mode}_{metric}"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return {
            "loss": loss,
            "probs": probs,
            "preds": preds,
            "logits": logits,
            "targets": targets,
        } | self._get_meta(args)

    def shared_on_epoch_end(self, outputs: List[Any], mode: str):
        pass

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs: List[Any]):
        self.shared_on_epoch_end(outputs, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        if self.trainer.sanity_checking and hasattr(
            self.trainer.datamodule, "patch_mask_preparer"
        ):
            random_idx = np.random.randint(len(self.trainer.val_dataloaders[0].dataset))

            self.trainer.datamodule.patch_mask_preparer._save_plot_overview(
                os.path.join(self.trainer._default_root_dir, "overview"),
                self.trainer.datamodule.coords_df,
                random_idx,
            )

        return self.shared_step(batch, "val")

    def validation_epoch_start(self, inputs: List[Any]):
        if self.trainer.sanity_checking:
            log.info("SANITY CHECKING")

    def validation_epoch_end(self, outputs: List[Any]):
        self.shared_on_epoch_end(outputs, "val")
        for metric in ["acc", "auroc", "f1"]:
            res = self.metrics[f"val_{metric}"].compute()
            self.metrics[f"val_{metric}_best"](res)
            self.log(
                f"val/{metric}_best", self.metrics[f"val_{metric}_best"].compute(), prog_bar=True
            )

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs: List[Any]):
        self.shared_on_epoch_end(outputs, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y, *args = batch
        x = self.forward(x)
        return {"x": x, "y": y} | self._get_meta(args)

    def predict_epoch_end(self, outputs: List[Any]):
        return utils.output_collate(outputs)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())

        out_dict = {"optimizer": optimizer}

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            out_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }

        return out_dict

    def on_train_end(self) -> None:
        """Called at the end of training."""
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                save_best_weights(callback)

    def _get_meta(self, data_args):
        """Called after the forward pass of the network."""
        meta = {}
        if len(data_args) > 0:
            data_meta = data_args[0]
            meta["data_meta"] = data_meta
        if hasattr(self.net, "get_meta"):
            model_meta = self.net.get_meta()
            meta["model_meta"] = model_meta
        return meta

    @staticmethod
    def process_logits(predictions):
        tmp = {}
        logits = torch.stack(predictions["x"])
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        tmp["pred_cls"] = [p.item() for p in preds]
        for i in range(probs.shape[1]):
            tmp[f"prob_cls_{i}"] = [p[i].item() for p in probs]
        predictions["x"] = tmp
        return predictions


def get_class_weights(label_df: pd.DataFrame) -> pd.Series:
    counts = 1 / label_df.value_counts(normalize=True).sort_index()
    return counts / counts.sum()


from copy import copy
def save_best_weights(checkpoint_callback: ModelCheckpoint):
    """Used to extract the model weights of the best lightning checkpoint into raw pytorch
    checkpoint, so that it can be loaded even when non-architecture code changes."""
    ckpt_path = copy(checkpoint_callback.best_model_path)
    ckpt = torch.load(ckpt_path)
    weights_path = ckpt_path.replace(".ckpt", ".pt")
    torch.save(ckpt["state_dict"], weights_path)
    log.info(f"Saved best model weights to {weights_path}")
