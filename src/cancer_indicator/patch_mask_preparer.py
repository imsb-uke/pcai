import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.patch_loader import OpenslidePatchLoader
from src.utils import append_suffix_before_ext, get_printlogger, plot_mask_overlay

log = get_printlogger(__name__)

class PatchMaskPreparer:
    def __init__(
        self,
        image_base_dir,
        mask_base_dir,
        patch_size=256,
        fg_mask_channel=0,
        fg_mask_invert=False,
        fg_mask_threshold=0.5,
        mask_downsample_rate=16,
        label_mask_channel=1,
        label_mask_invert=False,
        label_mask_threshold=0.5,
        mask_filename_suffix="_mask",
        remove_mixed_label_patches=False,
    ):
        self.image_base_dir = str(image_base_dir)
        self.mask_base_dir = str(mask_base_dir)
        self.patch_size = patch_size
        self.mask_downsample_rate = mask_downsample_rate
        self.fg_mask_channel = fg_mask_channel
        self.fg_mask_invert = fg_mask_invert
        self.fg_mask_threshold = fg_mask_threshold
        self.label_mask_channel = label_mask_channel
        self.label_mask_invert = label_mask_invert
        self.label_mask_threshold = label_mask_threshold
        self.mask_filename_suffix = mask_filename_suffix
        self.remove_mixed_label_patches = remove_mixed_label_patches

        self.downsampled_patch_size = self.patch_size // self.mask_downsample_rate

    def get_filenames(self, filter_='.tiff'):
        # return os.listdir(self.image_base_dir)
        return self._validate_filenames(
            [fn for fn in os.listdir(self.image_base_dir) if fn.endswith(filter_)]
        )

    def _create_patch_mask(self, image_filename):

        # load mask from disk
        mask_filepath = os.path.join(
            self.mask_base_dir, append_suffix_before_ext(image_filename, self.mask_filename_suffix)
        )
        try:
            mask = np.array(Image.open(mask_filepath))
        except ValueError as exc:
            log.error(msg=f"Could not open mask file '{image_filename}'.")
            raise exc

        # invert masks if necessary
        if self.fg_mask_invert:
            fg_mask = mask[:, :, self.fg_mask_channel]
            mask[:, :, self.fg_mask_channel] = fg_mask.max() - fg_mask

        if self.label_mask_invert:
            label_mask = mask[:, :, self.label_mask_channel]
            mask[:, :, self.label_mask_channel] = label_mask.max() - label_mask

        # get patch averages
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        patch_mask_avg = torch.nn.AvgPool2d(kernel_size=self.downsampled_patch_size)(
            mask_tensor.swapaxes(-1, 0)
        ).swapaxes(0, -1)

        # extract fg and label mask:
        patch_fg_mask = patch_mask_avg[..., self.fg_mask_channel]
        patch_label_mask = patch_mask_avg[..., self.label_mask_channel]

        return patch_fg_mask, patch_label_mask

    def get_patch_coords(self, image_filenames, label_col="label"):

        valid_filenames = self._validate_filenames(image_filenames)

        # load
        cache_filepath = os.path.join(".cache", "/".join(self.image_base_dir.split("/")[-2:]))
        os.makedirs(cache_filepath, exist_ok=True)
        coords_filename = (
            f".{self.patch_size=}."
            f"{self.fg_mask_threshold=}.{self.label_mask_threshold=}"
            f".pkl".replace("self.", "")
        )

        if os.path.exists(os.path.join(cache_filepath, coords_filename)):
            log.info("Loading Patch Coords from cache...")
            coords_df = pd.read_pickle(os.path.join(cache_filepath, coords_filename))
        else:
            coords_df = pd.DataFrame(columns=["filename", "row", "col", "value", label_col])

        uncached_filenames = set(valid_filenames) - set(coords_df.filename.values)
        for _, image_filename in enumerate(tqdm(uncached_filenames, "Loading Uncached Patches")):
            # transform masks to coords by keeping only those in patch_fg_mask
            # with patch_label_mask as label
            try:
                patch_fg_mask, patch_label_mask = self._create_patch_mask(image_filename)

            except ValueError:
                log.error(msg=f"Could not create patch mask for '{image_filename}'.")
                continue

            # thresholding
            patch_fg_mask_thresh = (patch_fg_mask > self.fg_mask_threshold).float()

            patch_label_mask_norm = patch_label_mask / patch_fg_mask
            patch_label_mask_thresh = (patch_label_mask_norm > self.label_mask_threshold).float()

            if self.remove_mixed_label_patches:
                mixed_label_mask = (patch_label_mask_norm > 0) & (patch_label_mask_norm < 1)
                patch_fg_mask_thresh[mixed_label_mask] = 0
                patch_label_mask_thresh[mixed_label_mask] = np.nan

            # to coords based df
            patch_coords = patch_fg_mask_thresh.nonzero()

            tmp_patch_coords_df = pd.DataFrame(
                {
                    "row": patch_coords[:, 0],
                    "col": patch_coords[:, 1],
                    # continuous label value
                    "value": patch_label_mask_norm[patch_fg_mask_thresh == 1],
                    label_col: patch_label_mask_thresh[patch_fg_mask_thresh == 1],
                }
            ).assign(filename=image_filename)

            coords_df = pd.concat([coords_df, tmp_patch_coords_df])

        coords_df.reset_index(drop=True, inplace=True)

        # save
        if len(uncached_filenames) > 0:
            log.info(f"Saving {len(uncached_filenames)} updated Patch Coords to cache...")
            coords_df.to_pickle(
                os.path.join(cache_filepath, coords_filename),
            )

        return coords_df.loc[lambda df_: df_.filename.isin(valid_filenames)]

    def _validate_filenames(self, image_filenames):
        """make sure to only include files with mask."""
        image_filenames_cleaned = []
        for image_filename in image_filenames:
            if not os.path.exists(os.path.join(self.image_base_dir, image_filename)):
                log.debug(
                    f"Removing image {image_filename} since it does not exist"
                    f"in '{self.image_base_dir}'"
                )
                continue

            mask_filename = append_suffix_before_ext(image_filename, self.mask_filename_suffix)

            if not os.path.exists(os.path.join(self.mask_base_dir, mask_filename)):
                log.debug(
                    f"Removing image '{image_filename}' since mask '{mask_filename}' "
                    f"does not exist in '{self.mask_base_dir}'"
                )
                continue
            image_filenames_cleaned.append(image_filename)

        n_removed = len(image_filenames) - len(image_filenames_cleaned)
        log.info(f"Removed {n_removed} image/mask pairs. (not found)")

        return image_filenames_cleaned

    # plotting

    def _plot_patch_mask_overview(
        self,
        image_coords_df,
        tn_downsample_rate=8,
        label_col="label",
        cmap=None,
        vmin=None,
        vmax=None,
        alpha=0.5,
        ax=None,
        plot_grid=True,
        **plot_kwargs,
    ):
        ax = ax or plt.gca()

        filenames = image_coords_df.filename.unique()

        if len(filenames) > 1:
            raise ValueError("'image_coords_df' must only contain one filename.")
        
        filename = filenames[0]

        patch_loader = OpenslidePatchLoader(
            os.path.join(self.image_base_dir, filename),
            self.patch_size,
            channel_is_first_axis=False,
        )

        if plot_grid:
            default_params = dict(
                every_k_coordinates=5,
            )
            patch_loader.plot_patch_overview(
                tn_downsample_rate=tn_downsample_rate,
                ax=ax,
                **(default_params | plot_kwargs),
            )
        else:
            ax.imshow(patch_loader.get_image(tn_downsample_rate=tn_downsample_rate))

        tmp_image_coords = image_coords_df.loc[lambda df_: df_.filename == filename]

        combined_mask = -np.empty([patch_loader.max_rows, patch_loader.max_cols], dtype=np.float32)
        combined_mask[:] = np.nan

        for _, coord in tmp_image_coords.iterrows():
            combined_mask[coord.row, coord.col] = coord[label_col]

        new_size = (
            combined_mask.shape[1] * (self.patch_size // tn_downsample_rate),
            combined_mask.shape[0] * (self.patch_size // tn_downsample_rate),
        )

        combined_mask_resized = cv2.resize(
            np.array(combined_mask), dsize=new_size, interpolation=cv2.INTER_NEAREST
        )
        combined_mask_resized = np.pad(
            combined_mask_resized,
            ((1, 0), (1, 0)),
            mode="constant",
            constant_values=np.nan,
        )

        plot_mask_overlay(
            combined_mask_resized, cmap=cmap, alpha=alpha, ax=ax, vmin=vmin, vmax=vmax
        )

        return ax

    def _save_plot_overview(self, out_dir, coords_df, batch_idx):
        _, ax = plt.subplots(ncols=1, figsize=(8, 16))

        filename = coords_df.iloc[batch_idx].filename

        image_coords_df = coords_df.loc[lambda df_: df_.filename == filename]

        row, col = coords_df.iloc[batch_idx][["row", "col"]]

        self._plot_patch_mask_overview(
            image_coords_df=image_coords_df,
            ax=ax,
        )

        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{os.path.splitext(filename)[0]}-{batch_idx:>05}.png"))

    def _plot_patch(self, filename, row, col, label, ax=None):

        ax = ax or plt.gca()

        patch_loader = OpenslidePatchLoader(
            os.path.join(self.image_base_dir, filename), self.patch_size
        )

        ax.imshow(patch_loader.get_patch(row, col).swapaxes(0, -1))

        ax.set_title(f"patch ({row}, {col}) with label={label} (filename={filename})")

        return ax
