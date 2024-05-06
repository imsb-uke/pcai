from typing import Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.patch_dataset import PatchDataset

from src.transforms import ImageTransformsAugMix, LabelTransformsClas, LabelTransformsDomain

class PatchDataModule(LightningDataModule):
    """
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

    def __init__(
        self,
        metadata_path: str = '../data/tma_dataset/metadata/metadata.csv',
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        **dataset_kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset_kwargs = dataset_kwargs    

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None

        # data transformations
        self.image_transforms_train = ImageTransformsAugMix(augment=True)
        self.image_transforms_val = ImageTransformsAugMix(augment=False)
        self.label_transforms_clas = LabelTransformsClas(cutoff_months=60)
        self.label_transforms_domain = LabelTransformsDomain(domain_mapping={'A': 0, 'B': 1, 'C': 2})

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `train_dataset`, `val_dataset`, `test_dataset`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        self.metadata_df = (
            pd.read_csv(self.hparams.metadata_path)
            # sample id describes the bag level information
            .assign(sample_id=lambda df_: df_['patient_id'])
            .assign(filepath=lambda df_: df_['sample_id'].apply(lambda x: f"{x}.tif"))
        )

        self.metadata_df = pd.concat(
            [
                self.metadata_df.assign(split="train"),
                self.metadata_df.assign(split="val"),
                self.metadata_df.assign(split="test"),
            ]
        )

        # load PatchDatasets
        if stage == "fit" and not self.train_dataset and not self.val_dataset:
            self.train_dataset = PatchDataset(
                info_df=self.metadata_df.loc[lambda df_: df_["split"] == "train"],
                strategy="random",
                image_transforms=self.image_transforms_train,
                label_transforms_clas=self.label_transforms_clas,
                label_transforms_domain=self.label_transforms_domain,
                **self.dataset_kwargs,
            )

            self.val_dataset = PatchDataset(
                info_df=self.metadata_df.loc[lambda df_: df_["split"] == "val"],
                strategy="all",
                image_transforms=self.image_transforms_val,
                label_transforms_clas=self.label_transforms_clas,
                label_transforms_domain=self.label_transforms_domain,
                **self.dataset_kwargs,
            )

        if stage == "test" and not self.test_dataset:
            self.test_dataset = PatchDataset(
                info_df=self.metadata_df.loc[lambda df_: df_["split"] == "test"],
                strategy="all",
                image_transforms=self.image_transforms_val,
                label_transforms_clas=self.label_transforms_clas,
                label_transforms_domain=self.label_transforms_domain,       
                **self.dataset_kwargs,
            )

        if stage == "predict" and not self.predict_dataset:
            self.predict_dataset = PatchDataset(
                info_df=self.metadata_df,
                strategy="all",
                image_transforms=self.image_transforms_val,
                label_transforms_clas=self.label_transforms_clas,
                label_transforms_domain=self.label_transforms_domain,
                **self.dataset_kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
