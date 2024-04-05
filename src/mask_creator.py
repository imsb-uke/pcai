import os
from pathlib import Path

import cv2
import numpy as np
import openslide
from wurlitzer import pipes

from src.patch_loader import _downsample_slide


class MaskCreator:
    def __init__(
        self,
        img_dir: str,
        out_dir: str,
        downsample_factor: int = 1,
        save_mask_files: bool = True,
        save_png_files: bool = True,
    ):
        self.img_dir = img_dir
        self.out_dir = out_dir

        self.mask_dir = os.path.join(self.out_dir, "masks")
        self.png_dir = os.path.join(self.out_dir, "pngs")

        self.downsample_factor = downsample_factor
        self.save_mask_files = save_mask_files
        self.save_png_files = save_png_files

    def create_masks(self, slide_path):
        masks = {}

        masks["meta"] = {}
        masks["meta"]["slide_path"] = slide_path
        masks["meta"]["mask_downsample_factor"] = self.downsample_factor
        masks["meta"]["mask_max_value"] = 255

        # load original downsampled image
        img_ds, orig_slide_shape = self.load_openslide_img(
            self._get_img_path(slide_path), self.downsample_factor
        )
        masks["thumbnail"] = img_ds.astype(np.uint8)
        masks["meta"]["original_slide_shape"] = orig_slide_shape

        # first tissue mask on unfiltered image
        tissue_mask_raw = self.create_tissue_mask(img_ds, invert=False)
        masks["tissue_raw"] = tissue_mask_raw.astype(np.uint8)

        if self.save_mask_files:
            self.save_npz(masks, slide_path)

        if self.save_png_files:
            self.save_pngs(masks, slide_path)

        return masks

    def save_npz(self, masks, slide_name):
        path = self._get_out_path(slide_name, suffix="_masks.npz")
        Path(os.path.split(path)[0]).mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez_compressed(f, **masks)

    def save_pngs(self, masks, slide_name):
        for k, m in masks.items():
            if k == "meta":
                continue
            path = self._get_png_path(slide_name, f"_{k}.png")
            Path(os.path.split(path)[0]).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(path, m)

    def _get_img_path(self, slide_path):
        return os.path.join(self.img_dir, slide_path)

    def _get_out_path(self, slide_path, suffix="_masks.npz"):
        return os.path.join(self.mask_dir, slide_path + suffix)

    def _get_png_path(self, slide_path, suffix=".png"):
        return os.path.join(self.png_dir, slide_path + suffix)
    
    @staticmethod
    def load_openslide_img(path, downsample_factor):
        with pipes(bufsize=0):
            slide = openslide.OpenSlide(path)
            orig_slide_shape = slide.level_dimensions[0][::-1]
            img, _ = _downsample_slide(slide, downsample_factor)
        return img, orig_slide_shape
    
    @staticmethod
    def create_tissue_mask(img, invert=False):
        """Create tissue mask from rgb image."""
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Remove noise using a Gaussian filter
        img = cv2.GaussianBlur(img, (5, 5), 0)
        # Otsu thresholding and mask generation
        _, tissue_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.bitwise_not(tissue_mask) if invert else tissue_mask

