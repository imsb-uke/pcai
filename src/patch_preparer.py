from copy import deepcopy
from functools import reduce
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch


class PatchPreparer:
    def __init__(self, mask_path: str):
        self.masks = self._load_masks_npz(mask_path)

        self.mask_max_value = self.masks["meta"]["mask_max_value"]
        self.mask_dsf = self.masks["meta"]["mask_downsample_factor"]
        self.orig_slide_shape = self.masks["meta"]["original_slide_shape"]


    def _load_masks_npz(self, mask_path: str) -> Dict[str, np.ndarray]:
        masks = dict(np.load(mask_path, allow_pickle=True))
        for key, value in masks.items():
            if value.dtype == "object":
                masks[key] = value.item()
        return masks

    def _get_combined_filter_mask(
        self, filter_masks: Dict[str, int], patch_size: int, img_mag_dsf: int
    ):
        f_masks = []
        for name, thresh in deepcopy(filter_masks).items():
            mask = self.masks[name]

            mask = cv2.resize(
                mask,
                dsize=(
                    self.orig_slide_shape[1] // self.mask_dsf,
                    self.orig_slide_shape[0] // self.mask_dsf,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            self._verify_mask_dsf(mask, name)

            patch_mask = self._create_patch_mask(mask, patch_size, img_mag_dsf)
            patch_mask = self._threshold_mask(patch_mask, thresh)
            f_masks.append(patch_mask)
        return reduce(lambda x, y: x * y, f_masks)

    def _get_combined_value_mask(self, value_masks: List[str], patch_size: int, img_mag_dsf: int):
        v_masks = []
        for name in deepcopy(value_masks):
            mask = self.masks[name]
            self._verify_mask_dsf(mask, name)
            patch_mask = self._create_patch_mask(mask, patch_size, img_mag_dsf)
            v_masks.append(patch_mask)
        return reduce(lambda x, y: x * y, v_masks)

    def get_patch_info_df(
        self,
        img_mag_dsf: int,
        patch_size: int,
        filter_masks: Dict[str, int],
        value_masks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        combined_filter_mask = self._get_combined_filter_mask(
            filter_masks, patch_size, img_mag_dsf
        )

        patch_coords = combined_filter_mask.nonzero()

        patch_df = pd.DataFrame(
            {
                "row": patch_coords[:, 0],
                "col": patch_coords[:, 1],
            }
        ).astype({"row": int, "col": int})

        if value_masks is not None:
            combined_value_mask = self._get_combined_value_mask(
                value_masks, patch_size, img_mag_dsf
            )
            patch_df = patch_df.assign(
                value=combined_value_mask[combined_filter_mask > 0].flatten()
            ).astype(float)

        return patch_df

    def _create_patch_mask(
        self, mask: np.array, patch_size: int, img_mag_dsf: int
    ) -> torch.Tensor:
        kernel_size = int(patch_size // self.mask_dsf * img_mag_dsf)
        mask_tensor = torch.tensor(mask / self.mask_max_value, dtype=torch.float32)
        return torch.nn.AvgPool2d(kernel_size=kernel_size)(mask_tensor.unsqueeze(0)).squeeze(0)

    def _threshold_mask(self, patch_mask: torch.Tensor, thresh: float) -> torch.Tensor:
        self._verify_patch_selection_thresh(thresh)
        return (patch_mask >= thresh).float()

    # verification

    def _verify_mask_dsf(self, mask: np.array, name: str):
        mask_dsf_calc = self.orig_slide_shape[0] // mask.shape[0]
        calc_mask_size = tuple(np.array(self.orig_slide_shape) // self.mask_dsf)
        assert self.mask_dsf == mask_dsf_calc, (
            f"Mask '{name}': Meta info mask downsample factor is {self.mask_dsf} but "
            + f"calculated mask size {calc_mask_size} does not match actual size {mask.shape}!"
        )

    def _verify_patch_selection_thresh(self, thresh: Union[float, None]):
        assert (
            0 <= thresh <= 1 or thresh is None
        ), f"filter mask thresh value {thresh} must be float between 0 and 1 or None! "

 