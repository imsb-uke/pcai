import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

from src.architecture.components import AdversarialModule, BagAggregationModule, BagCnnEncoderModule, BagSelfAttentionModule, ClassificationTargetModule

class AdversarialNet(LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            BagCnnEncoderModule(),
            BagSelfAttentionModule(),
            BagAggregationModule(),
        )
        self.clas_head = ClassificationTargetModule()
        self.adv_head = nn.Sequential(AdversarialModule(), ClassificationTargetModule())

    def forward(self, x):
        shared = self.shared(x)
        return self.clas_head(shared), self.adv_head(shared)