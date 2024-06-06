import torch
import torch.nn as nn
from collections import OrderedDict

from pytorch_lightning import LightningModule

from src.architecture.components import AdversarialModule, BagAggregationModule, BagCnnEncoderModule, BagSelfAttentionModule, ClassificationTargetModule

class AdversarialNet(LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.shared_part = nn.Sequential(OrderedDict([("encoder", BagCnnEncoderModule()), ("self_attention", BagSelfAttentionModule()), ("aggregation", BagAggregationModule())]))

        self.clas_part = nn.Sequential(OrderedDict([("clas_head", ClassificationTargetModule(num_classes=2))]))
        self.adv_part = nn.Sequential(OrderedDict([("gradient_reverse", AdversarialModule()), ("adv_head", ClassificationTargetModule(num_classes=3))]))

    def forward(self, x):
        shared = self.shared_part(x)
        return self.clas_part(shared), self.adv_part(shared), shared
    