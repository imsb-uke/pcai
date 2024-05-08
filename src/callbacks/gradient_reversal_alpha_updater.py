import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.callback import Callback


class GradientReversalAlphaUpdater(Callback):
    def __init__(self, max_epochs: int, log: bool = True):
        self.max_epochs = max_epochs
        self.log = log

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        p = (trainer.current_epoch + 1) / self.max_epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        # adjust module tree accordingly
        pl_module.net.adv_part.gradient_reverse.alpha = alpha

        if self.log:
            trainer.logger.log_metrics({"train/adv_alpha": alpha}, step=trainer.global_step)
