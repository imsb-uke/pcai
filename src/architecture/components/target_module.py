import torch
import torch.nn as nn


class ClassificationTargetModule(nn.Module):
    """Classification target module.

    Expects latent input of shape BxL and outputs predictions shape Bxnum_classes.
    """

    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 100,
        drop_rate: float = 0.5,
        num_classes: int = 2,
    ):

        super().__init__()

        self.classification_fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(input_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(drop_rate),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor):
        # in: BxL

        x = self.classification_fc(inputs)

        # out: Bxnum_classes
        return x
