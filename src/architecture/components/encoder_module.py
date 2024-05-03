import torch
import torch.nn as nn

import torchvision.models as tvmodels


class BagCnnEncoderModule(nn.Module):
    """Bag CNN encoder module.

    Expects image-patch input of shape BxNxCxHxW and outputs latent encodings of shape BxNxL.
    """

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__()

        self.backbone = EfficientnetBackbone(**kwargs)

    def forward(self, inputs: torch.Tensor):
        """inputs shape: BxNxCxHxW this method only works for equal number of patches per batch."""
        # in: BxNxCxHxW
        input_shape = inputs.shape
        # merge batch and patch dimensions for encoder
        x = inputs.contiguous().view(-1, *input_shape[2:])  # B*NxCxHxW
        x = self.backbone(x)  # B*NxL
        # split batch and patch dimensions for MIL
        x = x.contiguous().view(-1, input_shape[1], *x.shape[1:])  # BxNxL
        # out: BxNxL
        return x



class EfficientnetBackbone(nn.Module):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        
        super().__init__()

        full_cnn = self._get_full_cnn(**kwargs)

        self.backbone_features = self._get_feature_extractor(full_cnn)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.kwargs = kwargs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone_features(inputs)
        x = self.global_pool(x)
        x = self.flatten(x)

        return x

    def _get_full_cnn(self, **kwargs):
        return tvmodels.efficientnet_b0(**kwargs)

    def _get_feature_extractor(self, model: nn.Module):
        return model.features