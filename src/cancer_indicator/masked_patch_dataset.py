import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose

from src import utils
from src.patch_loader import OpenslidePatchLoader

log = utils.get_pylogger(__name__)


class MaskedPatchDataset(Dataset):
    """Load masked images and masked labels from disk."""

    def __init__(
        self,
        coords_df,
        image_base_dir,
        patch_size,
        undersample_majority_label=False,
        image_transforms: Optional[Compose] = None,
        label_transforms: Optional[Compose] = None,
        n_patches_per_label: Union[Dict, float, int, None] = None,
        seed=None,
    ):
        self.image_base_dir = image_base_dir
        self.patch_size = patch_size

        self.undersample_majority_label = undersample_majority_label
        self.seed = seed
        self.n_patches_per_label = n_patches_per_label

        self.image_transforms = image_transforms
        self.label_transforms = label_transforms

        self.coords_df = coords_df

        if self.n_patches_per_label is not None:
            self._sample_coords_df()

        if self.undersample_majority_label:
            self._undersample_majority_label()

        self.no_pos_labels = self.coords_df.loc[lambda df_: df_[self.label_col] == 1].shape[0]
        self.no_neg_labels = self.coords_df.loc[lambda df_: df_[self.label_col] == 0].shape[0]

    # attributes
    label_col = "label"

    # collection functions

    def __getitem__(self, idx: int):
        curr_patch_info = self.coords_df.iloc[idx]

        patch_loader = OpenslidePatchLoader(
            os.path.join(self.image_base_dir, curr_patch_info["filename"]),
            self.patch_size,
        )

        patch = patch_loader.get_patch(curr_patch_info["row"], curr_patch_info["col"])
        label = curr_patch_info["label"].astype("long")

        if self.image_transforms:
            patch = self.image_transforms(patch)

        if self.label_transforms:
            label = self.label_transforms(label)

        return patch, label, {"patch_info": curr_patch_info.to_dict()}

    def __len__(self):
        return len(self.coords_df)

    def __repr__(self):
        undersample_suffix = "(↓)" if self.undersample_majority_label else ""
        result = (
            f"{self.__class__.__name__}("
            f"{self.coords_df.filename.unique().shape[0]} image(s), "
            f"{len(self.coords_df)} patch(es) "
        )

        if not self.undersample_majority_label:
            result += f"[{self.no_pos_labels} pos, {self.no_neg_labels} neg]"
        else:
            result += (
                f"[{self.no_pos_labels}{undersample_suffix if self.majority_label==1 else ''} pos, "
                f"{self.no_neg_labels}{undersample_suffix if self.majority_label==0 else ''} neg]"
            )

        result += ")"
        return result

    # other

    def get_label_df(self):
        return self.coords_df[[self.label_col]]

    def _sample_coords_df(self):
        keep_patches_df = pd.DataFrame()

        if np.isscalar(self.n_patches_per_label):
            keep_patches_df = (
                self.coords_df.groupby("label")
                .apply(
                    lambda df_: df_.sample(int(self.n_patches_per_label), random_state=self.seed)
                )
                .droplevel(0)
            )
        else:
            for label, n_samples_of_label in self.n_patches_per_label.items():
                tmp_label_keep_patches_df = self.coords_df.loc[
                    lambda df_: df_["label"] == label
                ].sample(int(n_samples_of_label), random_state=self.seed)
                keep_patches_df = pd.concat(
                    [
                        keep_patches_df,
                        tmp_label_keep_patches_df,
                    ]
                )
        self._filter_coords_df(keep_patches_df)

    def _filter_coords_df(self, keep_coords_df):
        if not hasattr(self, "unused_patches_df"):
            self.unused_patches_df = pd.DataFrame()

        self.unused_patches_df = pd.concat(
            [
                self.unused_patches_df,
                self.coords_df[lambda df_: ~df_.index.isin(keep_coords_df.index)],
            ]
        )

        self.coords_df = keep_coords_df

    def _undersample_majority_label(self):
        self.n_minority_labels = self.coords_df["label"].value_counts().min()
        self.majority_label = self.coords_df["label"].value_counts().idxmax()
        patch_coords_df_undersampled = (
            self.coords_df.groupby("label", as_index=False)
            .apply(lambda df_: df_.sample(n=self.n_minority_labels, random_state=self.seed))
            .droplevel(0)  # remove groupby multiindex
            .sort_index()
        )

        # make sure to delete patches that were disselected in masks
        self._filter_patches(patch_coords_df_undersampled)
