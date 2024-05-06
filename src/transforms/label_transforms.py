from typing import Dict, Optional

import torch
from torchvision.transforms import transforms


def to_long_tensor(x):
    return torch.tensor(x, dtype=torch.long)

class ToBinaryRelapseLabel(torch.nn.Module):
    def __init__(self, cutoff_months):
        super().__init__()
        self.cutoff_months=cutoff_months

    def forward(self, x):
        if x[0] < self.cutoff_months and x[1] == 1:
            return 1
        else:
            return 0
    
class ToDomainLabel(torch.nn.Module):
    def __init__(self, domain_mapping):
        super().__init__()
        self.domain_mapping=domain_mapping

    def forward(self, x):
        return self.domain_mapping[x]

class LabelTransformsClas(transforms.Compose):
    def __init__(self, cutoff_months: Optional[int] = None):

        cutoff_months = cutoff_months or 60

        transforms = [
            ToBinaryRelapseLabel(cutoff_months),
            to_long_tensor,
            torch.squeeze,
        ]
        super().__init__(transforms)

class LabelTransformsDomain(transforms.Compose):
    def __init__(self, domain_mapping: Optional[Dict] = None, dummy_domain_value: int = 0):

        domain_mapping = domain_mapping or {}

        domain_mapping = domain_mapping | {'dummy_domain': dummy_domain_value}

        transforms = [
            ToDomainLabel(domain_mapping),
            to_long_tensor,
            torch.squeeze,
        ]
        super().__init__(transforms)