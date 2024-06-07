from typing import Any

import torch

from torch.optim import Adam

from pytorch_lightning import LightningModule

from src.architecture import AdversarialNet

from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAUROC


class LitModuleClasAdversarial(LightningModule):
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
        lr=1e-3,
        lambda_adv: float = 0.5,
    ):
        super().__init__()

        self.net = AdversarialNet()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

    def setup(self, stage: str) -> None:
        if stage != "predict":

            self.criterion_clas = torch.nn.CrossEntropyLoss()
            self.criterion_adv = torch.nn.CrossEntropyLoss()

            # build metric objects
            self.metrics = torch.nn.ModuleDict()
            for mode in ["train", "val", "test"]:
                self.metrics[f"{mode}_loss"] = MeanMetric()
                self.metrics[f"{mode}_loss_clas"] = MeanMetric()
                self.metrics[f"{mode}_loss_adv"] = MeanMetric()
                self.metrics[f"{mode}_auroc_clas"] = MulticlassAUROC(
                    num_classes=2, average="weighted"
                )
                self.metrics[f"{mode}_auroc_adv"] = MulticlassAUROC(
                    num_classes=3, average="weighted"
                )
                

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def _log_metrics(self, mode: str, head: str, loss, probs, targets):
        self.metrics[f"{mode}_loss_{head}"](loss)
        self.metrics[f"{mode}_auroc_{head}"](probs, targets)
        for metric in ["loss", "auroc",]:
            self.log(
                f"{mode}/{metric}_{head}",
                self.metrics[f"{mode}_{metric}_{head}"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def _step(self, x_hat, y, criterion):
        loss = criterion(x_hat, y)

        preds = torch.argmax(x_hat, dim=1)
        probs = torch.softmax(x_hat, dim=1)

        return loss, preds, probs

    def step(self, batch: Any, mode: str):
        x, y_clas, y_adv, data_meta = batch
        x_hat_clas, x_hat_adv, features = self.forward(x)

        # clas
        loss_clas, preds_clas, probs_clas = self._step(x_hat_clas, y_clas, self.criterion_clas)

        self._log_metrics(mode, "clas", loss_clas, probs_clas, y_clas)
        
        # adv
        loss_adv, preds_adv, probs_adv = self._step(x_hat_adv, y_adv, self.criterion_adv)

        self._log_metrics(mode, "adv", loss_adv, probs_adv, y_adv)

        # combined
        total_loss = loss_clas + self.hparams.lambda_adv * loss_adv

        self.metrics[f"{mode}_loss"](total_loss)
        self.log(
            f"{mode}/loss",
            self.metrics[f"{mode}_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        out_dict = {
            "loss": total_loss,
            "loss_clas": loss_clas,
            "logits_clas": x_hat_clas,
            "targets_clas": y_clas,
            "preds_clas": preds_clas,
            "probs_clas": probs_clas,
        }
        out_dict |= {
            "loss_adv": loss_adv,
            "logits_adv": x_hat_adv,
            "targets_adv": y_adv,
            "preds_adv": preds_adv,
            "probs_adv": probs_adv,
        }

        return out_dict | {"data_meta": data_meta} | {"model_meta": {"features": features.detach()}}

    def training_step(self, batch: Any, batch_idx: int):
        return self.step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.step(batch, "test")
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y_clas, y_adv, data_meta = batch
        x_hat_clas, x_hat_adv, features = self.forward(x)

        return {
            "x": {"clas": x_hat_clas, "adv": x_hat_adv},
            "y": {"clas": y_clas, "adv": y_adv},
        } | {"data_meta": data_meta} | {"model_meta": {"features": features.detach()}}

    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=self.hparams.lr)
        return {"optimizer": optimizer}

