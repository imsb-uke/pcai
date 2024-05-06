from typing import Optional

import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import (
    AugMix,
    Resize,
    transforms,
)

from .normalize import Normalize


def numpy_to_tensor(x):
    return torch.from_numpy(x)


def div_by_255(x):
    return x / 255


class ImageTransformsAugMix(transforms.Compose):
    def __init__(
        self,
        normalize_imagenet: bool = True,
        augment: bool = False,
        resize_size: Optional[int] = None,
    ):

        transforms = [numpy_to_tensor]

        if resize_size is not None:
            transforms.extend(
                [
                    Resize(size=resize_size),
                ]
            )

        if augment:
            transforms.extend(
                [
                    AugMix(),
                ]
            )

        transforms.extend(
            [
                div_by_255,
            ]
        )

        if normalize_imagenet:
            transforms.extend(
                [
                    Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ]
            )

        super().__init__(transforms)
