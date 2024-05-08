from typing import Any, List

import torch

from torch.optim import Adam

from pytorch_lightning import LightningModule

from src.architecture import AdversarialNet


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


    def forward(self, x: torch.Tensor):
        return self.net(x)

    def _step(self, x_hat, y, criterion):
        loss = criterion(x_hat, y)

        preds = torch.argmax(x_hat, dim=1)
        probs = torch.softmax(x_hat, dim=1)

        return loss, preds, probs

    def step(self, batch: Any, mode: str):
        x, y_clas, y_adv, *args = batch

        x_hat_clas, x_hat_adv = self.forward(x)

        # clas
        loss_bin_clas, preds_bin_clas, probs_bin_clas = self._step(x_hat_clas, y_clas, self.criterion_clas)

        # adv
        loss_adv, preds_adv, probs_adv = self._step(x_hat_adv, y_adv, self.criterion_adv)

        total_loss = loss_bin_clas + self.hparams.lambda_adv * loss_adv

        out_dict = {
            "loss": total_loss,
            "loss_bin_clas": loss_bin_clas,
            "logits_bin_clas": x_hat_clas,
            "targets_bin_clas": y_clas,
            "preds_bin_clas": preds_bin_clas,
            "probs_bin_clas": probs_bin_clas,
        }
        out_dict |= {
            "loss_adv": loss_adv,
            "logits_adv": x_hat_adv,
            "targets_adv": y_adv,
            "preds_adv": preds_adv,
            "probs_adv": probs_adv,
        }
        return out_dict

    def training_step(self, batch: Any, batch_idx: int):
        return self.step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.step(batch, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y_clas, y_adv, *args = batch
        x_hat_clas, x_hat_adv = self.forward(x)

        return {
            "x": {"clas": x_hat_clas, "adv": x_hat_adv},
            "y": {"clas": y_clas, "adv": y_adv},
        }

    def predict_epoch_end(self, outputs: List[Any]):
        return utils.output_collate(outputs)

    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=1e-3)

        out_dict = {"optimizer": optimizer}

        return out_dict
