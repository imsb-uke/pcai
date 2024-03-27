import os
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide
from openslide import deepzoom


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
        self.filename = os.path.split(self.filepath)[1]
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

    def no_patches(self):
        return self.max_cols * self.max_rows

    def _validate(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File '{self.filepath}' not found")

    def __repr__(self):
        return f"OpenslidePatchLoader('{self.filepath}', [{self.patch_size}x{self.patch_size}])"

    def get_patch(self, row, col, dtype=np.uint8):
        return self.get_patches([(row, col)], dtype=dtype)[0]

    def get_patches(self, coords, dtype=np.uint8):

        patches = np.zeros(
            (len(coords), self.patch_size, self.patch_size, self.n_channels),
            dtype=dtype,
        )

        # suppress sys.stdout of openslide
        for i, (row, col) in enumerate(coords):
            patch = self.dzg.get_tile(self.dzg_level, (col, row))
            patches[i, ...] = np.array(patch)

        # move channel axis to 1st position
        if self.channel_is_first_axis:
            patches = np.moveaxis(patches, -1, 1)
        return patches

    def get_all_patches(self, dtype=np.uint8):
        coords = [(row, col) for row in range(self.max_rows) for col in range(self.max_cols)]
        return self.get_patches(coords, dtype=dtype)

    def plot_patch(self, row, col, ax=None, show_title=True, **imshow_kwargs):
        ax = ax or plt.gca()
        patch = self.get_patch(row, col)
        ax.imshow(patch, **imshow_kwargs)

        ax.set_xticks([])
        ax.set_yticks([])

        if show_title:
            ax.set_title(f"Patch ({row}, {col})")

        return ax

    def get_image(self, tn_downsample_rate=1):
        img, _ = _downsample_slide(self.slide, tn_downsample_rate)
        return img

    def plot_patch_overview(
        self,
        tn_downsample_rate=16,
        every_k_coordinates=2,
        ax=None,
        axin=None,
        labeltop=True,
        labelright=True,
        draw_grid=True,
        **lines_kwargs,
    ):
        ax = ax or plt.gca()

        image = self.get_image(tn_downsample_rate=tn_downsample_rate)
        ax.imshow(image)

        img_size = image.shape[:2]

        ds_patch_size = self.patch_size // tn_downsample_rate
        top_patch_coords = np.arange(0.5, img_size[0] - 1, ds_patch_size)
        left_patch_coords = np.arange(0.5, img_size[1] - 1, ds_patch_size)

        lines_default_kwargs = dict(
            color="tab:gray",
            linewidth=1,
            alpha=0.5,
        )
        lines_kwargs = lines_default_kwargs | lines_kwargs

        if draw_grid:
            ax.hlines(top_patch_coords, 0, img_size[1], **lines_kwargs)
            ax.vlines(left_patch_coords, 0, img_size[0], **lines_kwargs)

        ax.set_ylim(img_size[0], 0)
        ax.set_xlim(0, img_size[1])

        ax.set_xlabel("column")
        ax.set_ylabel("row")

        ax.tick_params(labelbottom=True, labeltop=labeltop, labelleft=True, labelright=labelright)

        tickdist = ds_patch_size * every_k_coordinates
        ax.set_xticks(
            np.arange(ds_patch_size / 2, ax.get_xlim()[1], tickdist, dtype=int),
        )
        ax.set_xticklabels([int(tick) // ds_patch_size for tick in ax.get_xticks()])

        ax.set_yticks(
            np.arange(ds_patch_size / 2, ax.get_ylim()[0], tickdist, dtype=int),
        )
        ax.set_yticklabels([int(tick) // ds_patch_size for tick in ax.get_yticks()])

        return ax

def _downsample_slide(slide, downsampling_factor, mode="numpy"):
    """Downsample an Openslide at a factor.

    Takes an OpenSlide SVS object and downsamples the original resolution
    (level 0) by the requested downsampling factor, using the most convenient
    image level. Returns an RGB numpy array or PIL image.

    Args:
        slide: An OpenSlide object.
        downsampling_factor: Power of 2 to downsample the slide.
        mode: String, either "numpy" or "PIL" to define the output type.

    Returns:
        img: An RGB numpy array or PIL image, depending on the mode,
            at the requested downsampling_factor.
        best_downsampling_level: The level determined by OpenSlide to perform the downsampling.

    Remarks:
        from: https://github.com/manuel-munoz-aguirre/PyHIST/blob
            /7c2ccb41b97559740d31d5b748f74a89d262252c/src/utility_functions.py#L66
    """

    # Get the best level to quickly downsample the image
    # Add a pseudofactor of 0.1 to ensure getting the next best level
    # (i.e. if 16x is chosen, avoid getting 4x instead of 16x)
    best_downsampling_level = slide.get_best_level_for_downsample(downsampling_factor + 0.1)

    # Get the image at the requested scale
    svs_native_levelimg = slide.read_region(
        (0, 0), best_downsampling_level, slide.level_dimensions[best_downsampling_level]
    )
    target_size = tuple(int(x // downsampling_factor) for x in slide.dimensions)
    img = svs_native_levelimg.resize(target_size)

    # By default, return a numpy array as RGB, otherwise, return PIL image
    if mode == "numpy":
        # Remove the alpha channel
        img = np.array(img.convert("RGB"))

    return img, best_downsampling_level