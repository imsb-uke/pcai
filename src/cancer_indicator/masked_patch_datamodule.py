from functools import partial
from typing import Dict, List, Optional, Union

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import transforms

from src import utils
from src.cancer_indicator.masked_patch_dataset import MaskedPatchDataset
from src.cancer_indicator.patch_mask_preparer import PatchMaskPreparer
from src.transforms import ImageTransforms

log = utils.get_pylogger(__name__)


class MaskedPatchDataModule(LightningDataModule):
    """MaskedPatchDataModule loads masked patches and masked labels from disk.

    It also uses the database to load the experiment definition.

    The DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    label_dfs = {}

    def __init__(
        self,
        experiment_df: pd.DataFrame,
        patch_mask_preparer: PatchMaskPreparer,
        label_transforms: Optional[transforms.Compose] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        samples_per_epoch: Optional[int] = None,
        n_patches_per_split: Dict = None,
        eval_split: Optional[List[str]] = None,
        **mpd_kwargs,
    ):
        """Initialize a MaskedPatchDataModule instance.

        mpd_kwargs are passed to MaskedPatchDataset. See class definition for parameters.
        """
        super().__init__()

        self.patch_mask_preparer = patch_mask_preparer
        self.experiment_dfs = {k:v for k,v in experiment_df.groupby('split')}

        # data loader params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # kwargs to create MaskedPatchDatasets
        self.n_patches_per_split = n_patches_per_split
        self.mpd_kwargs = mpd_kwargs

        self.samples_per_epoch = samples_per_epoch

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.image_transforms_train = ImageTransforms(
            normalize_imagenet=True,
            augment=True,
        )
        
        self.image_transforms_val = ImageTransforms(
            normalize_imagenet=True,
            augment=False,
        )
        
        self.label_transforms = label_transforms

        self.eval_split = eval_split

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `train_dataset`, `val_dataset`, `test_dataset`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        if not hasattr(self, "experiment_dfs"):
            self.__load_experiment_dfs()

        if not hasattr(self, "coords_df"):
            self.__load_coords_df()

        # load MaskedPatchDatasets
        if stage == "fit":
            if not hasattr(self, "train_dataset"):
                self.train_dataset = self.__load_mpd("train")
            if not hasattr(self, "val_dataset"):
                self.val_dataset = self.__load_mpd("val")
        elif stage == "test":
            if not hasattr(self, "test_dataset"):
                self.test_dataset = self.__load_mpd("test")

        elif stage == "predict":
            if not hasattr(self, "predict_dataset"):
                self.predict_dataset = self.__load_predict_mpd(self.eval_split)
        else:
            # e.g. validation
            raise ValueError(f"Unimplemented stage: {stage}")

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def train_dataloader(self):
        if self.samples_per_epoch:
            loader_kwargs = dict(
                sampler=RandomSampler(
                    self.train_dataset,
                    num_samples=self.samples_per_epoch,
                    replacement=False,
                ),
            )
        else:
            loader_kwargs = dict(shuffle=True)

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            **loader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    # helper methods

    def __load_predict_mpd(self, eval_split: List[str]):
        combined_coords_df = pd.DataFrame()
        for stage in eval_split:
            if self.n_patches_per_split is not None:
                tmp_n_patches_per_label = self.n_patches_per_split.get(stage, None)
            else:
                tmp_n_patches_per_label = None

            tmp_df = self.coords_df.loc[lambda df_: df_["split"] == stage]

            stage_dataset = MaskedPatchDataset(
                **self.mpd_kwargs,
                coords_df=tmp_df,
                n_patches_per_label=tmp_n_patches_per_label,
                image_transforms=self.image_transforms_val,
            )

            self.label_dfs[stage] = stage_dataset.get_label_df()

            combined_coords_df = pd.concat([combined_coords_df, stage_dataset.coords_df])

        tmp_dataset = MaskedPatchDataset(
            **self.mpd_kwargs,
            coords_df=combined_coords_df,
            n_patches_per_label=None,
            image_transforms=self.image_transforms_val,
        )

        # remove unused from coords df stage
        self.coords_df = tmp_dataset.coords_df

        log.info(f"Loaded predict_dataset: {tmp_dataset}")

        return tmp_dataset

    def __load_mpd(self, stage):
        tmp_transform = (
            self.image_transforms_train if stage == "train" else self.image_transforms_val
        )

        # get n_patches_per_label based on split ratio
        if self.n_patches_per_split is not None:
            tmp_n_patches_per_label = self.n_patches_per_split.get(stage, None)
        else:
            tmp_n_patches_per_label = None

        tmp_dataset = MaskedPatchDataset(
            **self.mpd_kwargs,
            coords_df=self.coords_df.loc[lambda df_: df_["split"] == stage],
            n_patches_per_label=tmp_n_patches_per_label,
            image_transforms=tmp_transform,
        )

        # remove unused coords
        unused_coords = self.coords_df.loc[lambda df_: df_["split"] == stage].loc[
            lambda df_: ~df_.index.isin(tmp_dataset.coords_df.index)
        ]

        # remove unused from coords df stage
        self.coords_df = self.coords_df.loc[lambda df_: ~df_.index.isin(unused_coords.index)]

        if not hasattr(self, "label_dfs"):
            self.label_dfs = {}

        # get label_df from patch level for current dataset
        self.label_dfs[stage] = tmp_dataset.get_label_df()

        log.info(f"Loaded {stage}_dataset: {tmp_dataset}")
        return tmp_dataset

    def __load_coords_df(self, splits=["train", "val", "test"]):
        self.coords_df = pd.DataFrame()
        for split in splits:
            tmp_filenames = [f"{index}.tiff" for index in self.experiment_dfs[split].index]
            tmp_df = self.patch_mask_preparer.get_patch_coords(tmp_filenames).assign(split=split)
            self.coords_df = pd.concat([self.coords_df, tmp_df])

        log.info(f"Loaded coords_df: {self.coords_df.shape}")

        # make sure indices are unique
        self.coords_df.reset_index(drop=True, inplace=True)
