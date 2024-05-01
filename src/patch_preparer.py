import os
from copy import deepcopy
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.patch_loader import OpenslidePatchLoader

# from src.utils.masks import pad_right_and_bottom
# from src.utils.sampling import limit_sample_df, random_sample_df, top_sample_df


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

            # TODO remove this after we have fixed the mask generation
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

    # plotting #############################################################################

    def save_patch_selection_plot(
        self,
        out_dir: str,
        slide_path: str,
        img_mag_dsf: int,
        patch_size: int,
        filter_masks: Dict[str, int],
        value_masks: Optional[List[str]] = None,
        out_name: str = "patch_selection",
        **plot_kwargs,
    ):
        all_keys = list(dict(filter_masks).keys()) + (value_masks if value_masks else [])

        ncols = len(all_keys) + 1

        fig, axs = plt.subplots(figsize=(10 * ncols, 15), ncols=ncols)

        self.plot_patch_selection(
            slide_path=slide_path,
            img_mag_dsf=img_mag_dsf,
            patch_size=patch_size,
            filter_masks=filter_masks,
            value_masks=value_masks,
            ax=axs[0],
            **plot_kwargs,
        )

        for i, mask_name in enumerate(all_keys):
            self._plot_mask(mask_name, axs[i + 1])

        plt.suptitle(slide_path)
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, out_name + ".png"))

        plt.close()

    def _plot_mask(self, key: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
        ax = ax or plt.gca()
        ax.set_title(key)
        ax.imshow(self.masks[key])
        return ax

    def plot_patch_selection(
        self,
        slide_path: str,
        img_mag_dsf: int,
        patch_size: int,
        filter_masks: Dict[str, int],
        value_masks: Optional[List[str]] = None,
        sampling_strategy: str = "all",
        n_patches: int = 100,
        ax: Optional[plt.Axes] = None,
        **plot_kwargs,
    ) -> plt.Axes:
        ax = ax or plt.gca()

        patch_loader = OpenslidePatchLoader(
            filepath=slide_path,
            patch_size=patch_size,
            downsample_rate=img_mag_dsf,
        )
        ax = patch_loader.plot_patch_overview(
            tn_downsample_rate=self.mask_dsf, ax=ax, **plot_kwargs
        )

        # get patch info matrix and upsample to original size
        patch_info_df = self.get_patch_info_df(
            img_mag_dsf=img_mag_dsf,
            patch_size=patch_size,
            filter_masks=filter_masks,
            value_masks=value_masks,
        )

        if sampling_strategy == "random":
            patch_info_df = random_sample_df(patch_info_df, n_patches)
        elif sampling_strategy == "top":
            patch_info_df = top_sample_df(patch_info_df, n_patches)
        elif sampling_strategy == "limit":
            patch_info_df = limit_sample_df(patch_info_df, n_patches)
        elif sampling_strategy == "all":
            patch_info_df = patch_info_df
        else:
            raise ValueError(f"Unknown patch sampling strategy {sampling_strategy}")

        combined_mask = -np.ones([patch_loader.max_rows, patch_loader.max_cols], dtype=int)
        for i, rw in patch_info_df.iterrows():
            combined_mask[int(rw.row), int(rw.col)] = 0  # green

        downsampled_patch_size = patch_size // self.mask_dsf

        new_size = (
            combined_mask.shape[1] * downsampled_patch_size,
            combined_mask.shape[0] * downsampled_patch_size,
        )

        combined_mask_resized = cv2.resize(
            np.array(combined_mask), dsize=new_size, interpolation=cv2.INTER_NEAREST
        )

        # pad the edges that have been cut off due to non-full patches
        combined_mask_resized = pad_right_and_bottom(
            combined_mask_resized, self.masks[list(filter_masks.keys())[0]], value=-1
        )

        # create cmap
        color_dict = {
            -1: (0, 0, 0, 0),  # -1 = transparent
            0: (0, 1, 0, 1),  # 0 = green
            1: (1, 0, 0, 1),  # 1 = red
        }
        cmap = matplotlib.colors.ListedColormap(
            [v for k, v in color_dict.items() if k in np.unique(combined_mask)]
        )

        ax.set_title(f"patch size: {patch_size} \n img mag dsf: {img_mag_dsf}")
        ax.imshow(combined_mask_resized, cmap=cmap, alpha=0.5)
        return ax

    def plot_patch_outline(  # noqa: C901
        self,
        slide_path: str,
        img_mag_dsf: int,
        patch_size: int,
        filter_masks: Dict[str, int],
        value_masks: Optional[List[str]] = None,
        sampling_strategy: str = "all",
        n_patches: int = 100,
        plot_grid: bool = False,
        edge_thickness: Optional[int] = None,
        components_size_threshold: Optional[int] = None,
        inset_coords: Optional[Tuple[int]] = None,
        inset_zoom: Optional[int] = 10,
        inset_plot_grid: bool = False,
        inset_bbox_to_anchor: Tuple[float, float] = (0.5, 0.95),
        grid_color: str = "grey",
        grid_linewidth: float = 0.5,
        color_dict: Optional[Dict[int, Tuple[float]]] = None,
        heatmap_show: bool = False,
        heatmap_cmap: str = "viridis",
        heatmap_alpha: float = 1.0,
        heatmap_sigma: Optional[float] = None,
        patches_show_on_full_image: bool = True,
        show_colorbar: bool = True,
        return_hm: bool = False,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        ax = ax or plt.gca()

        patch_loader = OpenslidePatchLoader(
            filepath=slide_path,
            patch_size=patch_size,
            downsample_rate=img_mag_dsf,
        )
        image = patch_loader.get_image(tn_downsample_rate=self.mask_dsf * img_mag_dsf)
        ax.imshow(image)

        # get patch info matrix and upsample to original size
        patch_info_df = self.get_patch_info_df(
            img_mag_dsf=img_mag_dsf,
            patch_size=patch_size,
            filter_masks=filter_masks,
            value_masks=value_masks,
        )

        if sampling_strategy == "random":
            patch_info_df = random_sample_df(patch_info_df, n_patches)
        elif sampling_strategy == "top":
            patch_info_df = top_sample_df(patch_info_df, n_patches)
        elif sampling_strategy == "limit":
            patch_info_df = limit_sample_df(patch_info_df, n_patches)
        elif sampling_strategy == "all":
            patch_info_df = patch_info_df
        else:
            raise ValueError(f"Unknown patch sampling strategy {sampling_strategy}")

        combined_mask = -np.ones([patch_loader.max_rows, patch_loader.max_cols], dtype=int)
        for i, rw in patch_info_df.iterrows():
            combined_mask[int(rw.row), int(rw.col)] = 0  # green

        downsampled_patch_size = patch_size // self.mask_dsf

        new_size = (
            combined_mask.shape[1] * downsampled_patch_size,
            combined_mask.shape[0] * downsampled_patch_size,
        )

        combined_mask_resized = cv2.resize(
            np.array(combined_mask), dsize=new_size, interpolation=cv2.INTER_NEAREST
        )

        # pad the edges that have been cut off due to non-full patches
        combined_mask_resized = pad_right_and_bottom(combined_mask_resized, image, value=-1)

        if components_size_threshold is not None:
            combined_mask_resized = self._remove_small_components(combined_mask_resized)

        if heatmap_show:

            tissue_mask = self.masks[[k for k in filter_masks.keys()][0]]
            cancer_mask = self.masks[value_masks[0]]

            if heatmap_sigma is not None:
                from scipy.ndimage import gaussian_filter

                cm_orig_shape = cancer_mask.shape
                cancer_mask = self._downsample(cancer_mask, patch_size=downsampled_patch_size)
                cancer_mask = gaussian_filter(cancer_mask, sigma=heatmap_sigma)
                cancer_mask = cv2.resize(
                    cancer_mask, cm_orig_shape[::-1], interpolation=cv2.INTER_CUBIC
                )

            cancer_mask[tissue_mask == 0] = np.nan

            maskbar = ax.imshow(
                cancer_mask / 255,
                cmap=heatmap_cmap,
                alpha=heatmap_alpha,
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )

            if show_colorbar:
                plt.colorbar(maskbar, ax=ax, fraction=0.02, pad=0.04)

        if plot_grid:
            ax = self._plot_grid(
                combined_mask_resized,
                downsampled_patch_size,
                ax,
                color=grid_color,
                linewidth=grid_linewidth,
            )

        if edge_thickness is not None:
            combined_mask_resized_edges = self._draw_edges(
                combined_mask_resized, edge_thickness=edge_thickness
            )
        else:
            combined_mask_resized_edges = combined_mask_resized

        # create cmap
        if color_dict is None:
            color_dict = {
                -1: (0, 0, 0, 0),  # -1 = transparent
                0: (0, 0.5, 0, 0.5),  # 2 = dark green, more transparent
                1: (0, 0.5, 0, 1),  # 2 = dark green
            }
        cmap = matplotlib.colors.ListedColormap(
            [v for k, v in color_dict.items() if k in np.unique(combined_mask_resized_edges)]
        )

        if inset_coords is not None:
            (
                y1,
                y2,
                x1,
                x2,
            ) = inset_coords

            coords = (slice(y1, y2, None), slice(x1, x2, None))

            import mpl_toolkits.axes_grid1.inset_locator as il

            with plt.rc_context({"axes.edgecolor": "tab:gray", "axes.linewidth": 0.5}):
                axin = il.inset_axes(
                    ax,
                    inset_zoom,
                    inset_zoom,
                    bbox_to_anchor=inset_bbox_to_anchor,
                    loc="upper left",
                    bbox_transform=ax.transAxes,
                )

                axin.imshow(image[coords])
                axin.imshow(
                    combined_mask_resized_edges[coords], cmap=cmap, interpolation="nearest"
                )

                if inset_plot_grid:
                    axin = self._plot_grid(
                        combined_mask_resized[coords],
                        downsampled_patch_size,
                        axin,
                        color=grid_color,
                        linewidth=grid_linewidth,
                    )

                self._draw_custom_inset_locators(
                    x1=x1,
                    x2=x2,
                    y2=y1,
                    y1=y2,
                    ax=ax,
                    axin=axin,
                )

                axin.spines["top"].set_visible(True)
                axin.spines["bottom"].set_visible(True)
                axin.spines["right"].set_visible(True)
                axin.spines["left"].set_visible(True)

            for ax_ in [ax, axin]:
                ax_.set_xticks([])
                ax_.set_yticks([])
                ax_.set_xticklabels([])
                ax_.set_yticklabels([])

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_title(f"patch size: {patch_size} \n img mag dsf: {img_mag_dsf}")
        if patches_show_on_full_image:
            ax.imshow(combined_mask_resized, cmap=cmap, zorder=2)

        if return_hm:
            return (cancer_mask / 255), ax

        return ax

    @staticmethod
    def _draw_edges(mask, edge_thickness: int = 5, edge_value: int = 1):
        mask_8U = cv2.convertScaleAbs(mask, alpha=(255.0))
        edges = cv2.Canny(mask_8U, 100, 200)
        thick_edges = cv2.dilate(edges, None, iterations=edge_thickness)
        mask[thick_edges > 0] = edge_value
        return mask

    @staticmethod
    def _remove_small_components(
        mask: np.array, size_threshold: int = 1000, detect_value: int = 0
    ):
        from scipy.ndimage import label, sum

        # Label connected components
        labeled_array, num_features = label(mask == detect_value)
        # Compute the size of each component (number of pixels)
        component_sizes = sum(mask == 0, labeled_array, range(num_features + 1))
        # Create a mask of components that are larger than the threshold
        _mask = np.isin(labeled_array, np.where(component_sizes > size_threshold))
        # Update the combined_mask_resized using the mask
        mask[~_mask] = -1
        # Setting the value for the "not-selected" or background areas. Adjust if needed.
        return mask

    @staticmethod
    def _draw_custom_inset_locators(x1, x2, y1, y2, ax, axin):
        # Draw rectangle around area in ax
        rect = mpatches.Rectangle(
            (x1, y1),
            width=x2 - x1,
            height=y2 - y1,
            facecolor="None",
            edgecolor="black",
            alpha=0.5,
            linewidth=0.8,
        )
        ax.add_patch(rect)
        # Draw lines from corners of axin to corners of ax
        for i, x in enumerate([x1, x2]):
            for j, y in enumerate([y1, y2]):
                p = mpatches.ConnectionPatch(
                    xyA=(i, j),
                    xyB=(x, y),
                    color="black",
                    alpha=0.5,
                    coordsA="axes fraction",
                    coordsB="data",
                    axesA=axin,
                    axesB=ax,
                )
                ax.add_patch(p)
        return ax

    @staticmethod
    def _plot_grid(mask, downsampled_patch_size, ax, color="grey", linewidth=0.5):  # noqa: C901
        from scipy.ndimage import find_objects, label

        labeled_mask, _ = label(mask == 0)
        slices = find_objects(labeled_mask)

        for slice_ in slices:
            y_start, y_end = slice_[0].start, slice_[0].stop
            x_start, x_end = slice_[1].start, slice_[1].stop

            # Draw horizontal grid lines within the bounding box of the component
            for y in range(y_start, y_end, downsampled_patch_size):
                x_values_inside = []
                for x in range(x_start, x_end):
                    if labeled_mask[y, x] > 0:  # If within the component
                        x_values_inside.append(x)
                    elif x_values_inside:  # If outside the component and there are points inside
                        ax.plot(
                            [min(x_values_inside), max(x_values_inside)],
                            [y, y],
                            color=color,
                            linewidth=linewidth,
                            zorder=1,
                        )
                        x_values_inside = []
                # In case the line segment ends inside the mask
                if x_values_inside:
                    ax.plot(
                        [min(x_values_inside), max(x_values_inside)],
                        [y, y],
                        color=color,
                        linewidth=linewidth,
                        zorder=1,
                    )

            # Draw vertical grid lines within the bounding box of the component
            for x in range(x_start, x_end, downsampled_patch_size):
                y_values_inside = []
                for y in range(y_start, y_end):
                    if labeled_mask[y, x] > 0:  # If within the component
                        y_values_inside.append(y)
                    elif y_values_inside:  # If outside the component and there are points inside
                        ax.plot(
                            [x, x],
                            [min(y_values_inside), max(y_values_inside)],
                            color=color,
                            linewidth=linewidth,
                            zorder=1,
                        )
                        y_values_inside = []
                # In case the line segment ends inside the mask
                if y_values_inside:
                    ax.plot(
                        [x, x],
                        [min(y_values_inside), max(y_values_inside)],
                        color=color,
                        linewidth=linewidth,
                        zorder=1,
                    )

        return ax

    @staticmethod
    def _downsample(array, patch_size):
        """Downsample a numpy array by taking the mean of patches of given size.

        :param array: Input numpy array to downsample.
        :param patch_size: Size of the patch to compute the mean over.
        :return: Downsampled array.
        """

        # Get the shape of the input array
        rows, cols = array.shape

        # Compute the dimensions of the downsampled array
        new_rows = (rows + patch_size - 1) // patch_size
        new_cols = (cols + patch_size - 1) // patch_size

        # Initialize the downsampled array with zeros
        downsampled = np.zeros((new_rows, new_cols))

        for i in range(0, rows, patch_size):
            for j in range(0, cols, patch_size):
                # Determine the end indices for each patch
                end_i = min(i + patch_size, rows)
                end_j = min(j + patch_size, cols)

                # Compute the mean for the patch
                downsampled[i // patch_size, j // patch_size] = array[i:end_i, j:end_j].mean()

        return downsampled
