import torch
import torch.nn as nn

from .components import EfficientnetBackbone, ClassificationTargetModule

def get_cancer_indicator_net(effnet_kwargs, target_kwargs):
    return nn.Sequential(
        EfficientnetBackbone(**effnet_kwargs),
        ClassificationTargetModule(**target_kwargs)
    )