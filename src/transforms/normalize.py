from typing import List

import torch
from torch import Tensor
from torchvision.utils import _log_api_usage_once


class Normalize(torch.nn.Module):
    """Normalize a tensor image bag with mean and standard deviation.

    Same as the default torchvision.transforms.Normalize, but for image bags with bagsize as first dim.

    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean: List[float], std: List[float], inplace: bool = False):
        super().__init__()
        _log_api_usage_once(self)
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image bag to be normalized. Expects shape (.., C, H, W).

        Returns:
            Tensor: Normalized Tensor image bag.
        """
        tensor = self.F_normalize(tensor, self.mean, self.std, self.inplace)
        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

    def F_normalize(
        self, tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False
    ) -> Tensor:
        """Normalize a float tensor image with mean and standard deviation.

        .. note::
            This transform acts out of place by default, i.e., it does not mutates the input tensor.

        See :class:`~torchvision.transforms.Normalize` for more details.

        Args:
            tensor (Tensor): Float tensor bag of size (.., C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation inplace.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(self.F_normalize)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"img should be Tensor Image. Got {type(tensor)}")

        return self.F_t_normalize(tensor, mean=mean, std=std, inplace=inplace)

    def F_t_normalize(
        self, tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False
    ) -> Tensor:
        self._assert_image_tensor(tensor)

        if not tensor.is_floating_point():
            raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(
                f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
            )

        view_shape = [-1, 1, 1]
        # add extra dimensions to the mean and std vectors to allow broadcasting
        # if the tensor has more than 3 dimensions [..., C, H, W]
        if len(tensor.shape) > 3:
            for _ in range(len(tensor.shape) - 3):
                view_shape.insert(0, 1)
        view_shape = tuple(view_shape)

        if mean.ndim == 1:
            mean = mean.view(view_shape)
        if std.ndim == 1:
            std = std.view(view_shape)
        tensor.sub_(mean).div_(std)
        return tensor

    def _is_tensor_a_torch_image(self, x: Tensor) -> bool:
        return x.ndim >= 2

    def _assert_image_tensor(self, img: Tensor) -> None:
        if not self._is_tensor_a_torch_image(img):
            raise TypeError("Tensor is not a torch image.")


class DeNormalize(Normalize):
    def __init__(self, mean, std):
        mean = [-m / s for m, s in zip(mean, std)]
        std = [1 / s for s in std]
        super().__init__(mean, std)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
