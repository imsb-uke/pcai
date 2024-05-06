from typing import List, Optional

import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
import torch

from src.sampling import limit_sample_df, random_sample_df, top_sample_df
from src.patch_preparer import PatchPreparer
from src.patch_loader import OpenslidePatchLoader

class PatchDataset(Dataset):
    """Load multimasked images from disk."""

    def __init__(
        self,
        info_df: pd.DataFrame,
        image_base_dir: str,
        mask_base_dir: str,
        patch_size: int,
        label_cols_clas: List[str],
        label_col_domain: Optional[str] = None,
        img_mag_dsf: int = 1,
        mask_suffix: str = "_masks.npz",
        strategy: str = "all",
        filter_masks = None,
        value_masks = None,
        n_patches= None,
        image_transforms = None,
        label_transforms_clas = None,
        label_transforms_domain = None,
        allow_empty = False,
    ):
        self.info_df = info_df

        self.filter_masks = filter_masks or {"tissue_raw": 0.1}
        self.value_masks = value_masks

        self.image_base_dir = image_base_dir
        self.mask_base_dir = mask_base_dir
        self.mask_suffix = mask_suffix

        self.label_cols_clas = label_cols_clas
        self.label_col_domain = label_col_domain

        self.img_mag_dsf = img_mag_dsf
        self.patch_size = patch_size
        self.strategy = strategy
        self.n_patches = n_patches

        # Store sample ids individually
        self.sample_ids = self.info_df.sample_id.unique()

        if (not allow_empty) and (len(self.sample_ids) <= 0):
            raise ValueError("No sample ids found")

        if self.n_patches is None:
            print("Training with variable bag-size requires batch-size of 1!")

        self.image_transforms = image_transforms
        self.label_transforms_clas = label_transforms_clas
        self.label_transforms_domain = label_transforms_domain

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample_id = self.sample_ids[idx]

        patch_bag, label_clas, label_domain, meta_dict = self.get_sample(sample_id)

        if self.image_transforms is not None:
            patch_bag = self.image_transforms(patch_bag)

        if self.label_transforms_clas is not None:
            label_clas = self.label_transforms_clas(label_clas)

        if self.label_transforms_domain is not None:
            label_domain = self.label_transforms_domain(label_domain)

        return patch_bag, label_clas, label_domain, meta_dict

    def get_sample(self, sample_id) -> torch.Tensor:
        sample_info = self.info_df.loc[self.info_df.sample_id == sample_id]

        initial_dsf = int(sample_info.initial_dsf.values[0]) if "initial_dsf" in sample_info else 1
        rel_slide_path = sample_info.filepath.values[0]

        patch_info_df = self.select_patches(rel_slide_path, initial_dsf=initial_dsf)

        if len(patch_info_df) == 0:
            raise ValueError(f"Zero patches found for sample {sample_id}!")

        patch_bag, max_rows, max_cols = self.load_patches(
            rel_slide_path, patch_info_df, initial_dsf=initial_dsf
        )

        label_clas = sample_info[self.label_cols_clas].values[0]
        label_domain = (
            sample_info[self.label_col_domain].values[0].item() if self.label_col_domain else 'dummy_domain'
        )

        # build meta dict
        meta_dict = sample_info.to_dict("records")[0]
        meta_dict["patch_info"] = patch_info_df.values
        meta_dict["max_rows"] = max_rows
        meta_dict["max_cols"] = max_cols
        meta_dict["patch_size"] = self.patch_size
        meta_dict["img_mag_dsf"] = self.img_mag_dsf
        meta_dict["initial_dsf"] = initial_dsf

        return patch_bag, label_clas, label_domain, meta_dict

    def select_patches(self, rel_slide_path: str, initial_dsf: int = 1) -> pd.DataFrame:
        # get all valid patch coordinates based on masks and patch selection config

        patch_preparer = PatchPreparer(
            mask_path=os.path.join(self.mask_base_dir, rel_slide_path + self.mask_suffix)
        )

        patch_info_df = patch_preparer.get_patch_info_df(
            filter_masks=self.filter_masks,
            value_masks=self.value_masks,
            img_mag_dsf=initial_dsf * self.img_mag_dsf,
            patch_size=self.patch_size,
        )

        # over/undersample patch coordinates if n_patches is set
        if self.strategy == "random":
            patch_info_df = random_sample_df(patch_info_df, self.n_patches)
        elif self.strategy == "top":
            patch_info_df = top_sample_df(patch_info_df, self.n_patches)
        elif self.strategy == "limit":
            patch_info_df = limit_sample_df(patch_info_df, self.n_patches)
        elif self.strategy == "all":
            patch_info_df = patch_info_df
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

        return patch_info_df

    def load_patches(
        self, rel_slide_path: str, patch_info_df: pd.DataFrame, initial_dsf: int = 1
    ) -> np.ndarray:
        # load actual patches
        patch_loader = OpenslidePatchLoader(
            filepath=os.path.join(self.image_base_dir, rel_slide_path),
            patch_size=self.patch_size,
            downsample_rate=initial_dsf * self.img_mag_dsf,
        )

        patch_bag = patch_loader.get_patches(coords=patch_info_df[["row", "col"]].values)

        return patch_bag, patch_loader.max_rows, patch_loader.max_cols