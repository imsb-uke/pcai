import os
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# import mpl_toolkits.axes_grid1.inset_locator as il
import numpy as np
import openslide
from openslide import deepzoom
# from tqdm import tqdm
# from src.utils.image import downsample_slide


class OpenslidePatchLoader:
    def __init__(
        self,
        filepath: str,
        patch_size: int,
        channel_is_first_axis: bool = True,
        downsample_rate: int = 1,
        # show_progress: bool = False,
    ):

        self.filepath = filepath
        self.patch_size = patch_size
        self.channel_is_first_axis = channel_is_first_axis
        self.downsample_rate = downsample_rate
        self._validate()

        def is_power_of_2(in_):
            return (2 ** int(np.log2(in_))) == in_

        if not is_power_of_2(self.downsample_rate):
            raise ValueError(
                f"Downsample rate must be a power of 2, but is {self.downsample_rate}"
            )

        # also save filename as attribute
        self.filepath = os.path.split(self.filepath)[1]

        self.slide = openslide.OpenSlide(self.filepath)
        self.dzg = deepzoom.DeepZoomGenerator(
            osr=self.slide,
            tile_size=self.patch_size,
            overlap=0,
            limit_bounds=False,
        )

        self.dzg_level = self.dzg.level_count - 1 - int(np.log2(self.downsample_rate))
        self.n_channels = np.array(self.dzg.get_tile(self.dzg_level, (0, 0))).shape[2]

        self.max_cols, self.max_rows = self.dzg.level_tiles[self.dzg_level]

        if self.patch_size * self.max_cols > self.dzg.level_dimensions[self.dzg_level][0]:
            self.max_cols -= 1
        if self.patch_size * self.max_rows > self.dzg.level_dimensions[self.dzg_level][1]:
            self.max_rows -= 1

        self.width, self.height = self.slide.dimensions

    # def no_patches(self):
    #     return self.max_cols * self.max_rows

    def _validate(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File '{self.filepath}' not found")

    # def __repr__(self):
    #     return f"OpenslidePatchLoader('{self.filepath}', [{self.patch_size}x{self.patch_size}])"

    # def get_patch(self, row, col, dtype=np.uint8):
    #     return self.get_patches([(row, col)], dtype=dtype)[0]

    # def get_patches(self, coords, dtype=np.uint8):

    #     patches = np.zeros(
    #         (len(coords), self.patch_size, self.patch_size, self.n_channels),
    #         dtype=dtype,
    #     )

    #     # suppress sys.stdout of openslide
    #     for i, (row, col) in enumerate(tqdm(coords, disable=not self.show_progress)):
    #         with pipes(bufsize=0):
    #             patch = self.dzg.get_tile(self.dzg_level, (col, row))
    #         patches[i, ...] = np.array(patch)

    #     # move channel axis to 1st position
    #     if self.channel_is_first_axis:
    #         patches = np.moveaxis(patches, -1, 1)
    #     return patches

    # def get_image(self, tn_downsample_rate=1):
    #     img, _ = downsample_slide(self.slide, tn_downsample_rate)
    #     return img

    # def plot_patch_overview(
    #     self,
    #     tn_downsample_rate=16,
    #     every_k_coordinates=2,
    #     ax=None,
    #     axin=None,
    #     patch_coords=None,
    #     zoom_factor=2,
    #     labeltop=True,
    #     labelright=True,
    #     draw_grid=True,
    #     **lines_kwargs,
    # ):
    #     ax = ax or plt.gca()

    #     image = self.get_image(tn_downsample_rate=tn_downsample_rate)
    #     ax.imshow(image)

    #     img_size = image.shape[:2]

    #     ds_patch_size = self.patch_size // tn_downsample_rate
    #     top_patch_coords = np.arange(0.5, img_size[0] - 1, ds_patch_size)
    #     left_patch_coords = np.arange(0.5, img_size[1] - 1, ds_patch_size)

    #     lines_default_kwargs = dict(
    #         color="tab:gray",
    #         linewidth=1,
    #         alpha=0.5,
    #     )
    #     lines_kwargs = lines_default_kwargs | lines_kwargs

    #     if draw_grid:
    #         ax.hlines(top_patch_coords, 0, img_size[1], **lines_kwargs)
    #         ax.vlines(left_patch_coords, 0, img_size[0], **lines_kwargs)

    #     ax.set_ylim(img_size[0], 0)
    #     ax.set_xlim(0, img_size[1])

    #     ax.set_xlabel("column")
    #     ax.set_ylabel("row")

    #     ax.tick_params(labelbottom=True, labeltop=labeltop, labelleft=True, labelright=labelright)

    #     tickdist = ds_patch_size * every_k_coordinates
    #     ax.set_xticks(
    #         np.arange(ds_patch_size / 2, ax.get_xlim()[1], tickdist, dtype=int),
    #     )
    #     ax.set_xticklabels([int(tick) // ds_patch_size for tick in ax.get_xticks()])

    #     ax.set_yticks(
    #         np.arange(ds_patch_size / 2, ax.get_ylim()[0], tickdist, dtype=int),
    #     )
    #     ax.set_yticklabels([int(tick) // ds_patch_size for tick in ax.get_yticks()])

    #     if patch_coords is not None:
    #         patch_row, patch_col = patch_coords

    #         if axin is None:
    #             axin = il.inset_axes(
    #                 ax,
    #                 zoom_factor,
    #                 zoom_factor,
    #                 bbox_to_anchor=(2, 1),
    #                 bbox_transform=ax.transAxes,
    #             )

    #         patch = self.get_patch(patch_row, patch_col)

    #         if patch.shape[0] == 3:
    #             patch = patch.swapaxes(0, 1)
    #             patch = patch.swapaxes(1, 2)

    #         axin.imshow(patch)
    #         axin.set_xlabel("pixel")
    #         axin.set_ylabel("pixel")

    #         title = f"patch ({patch_row}, {patch_col})"
    #         axin.set_title(title)

    #         self._draw_custom_patch_inset_locators(
    #             patch_row, patch_col, ds_patch_size, ax=ax, axin=axin
    #         )

    #         return ax, axin

    #     else:
    #         return ax

    # def _draw_custom_patch_inset_locators(
    #     self, row, col, small_patch_size, ax, axin, lw=0.8, color="gray"
    # ):
    #     x1 = 0.5 + col * small_patch_size
    #     x2 = 0.5 + (col + 1) * small_patch_size
    #     y1 = 0.5 + (row + 1) * small_patch_size
    #     y2 = 0.5 + row * small_patch_size

    #     # if x1 > x2:
    #     #     x1, x2 = x2, x1

    #     # if y1 > y2:
    #     #     y1, y2 = y2, y1

    #     # Draw rectangle around area in ax
    #     rect = mpatches.Rectangle(
    #         (x1, y1),
    #         width=x2 - x1,
    #         height=y2 - y1,
    #         facecolor="None",
    #         edgecolor=color,
    #         alpha=0.5,
    #         linewidth=lw,
    #     )

    #     ax.add_patch(rect)

    #     for i, x in enumerate([x1, x2]):
    #         for j, y in enumerate([y1, y2]):
    #             p = mpatches.ConnectionPatch(
    #                 xyA=(i, j),
    #                 xyB=(x, y),
    #                 color=color,
    #                 alpha=0.5,
    #                 coordsA="axes fraction",
    #                 coordsB="data",
    #                 axesA=axin,
    #                 axesB=ax,
    #             )
    #             ax.add_patch(p)

    #     return ax
