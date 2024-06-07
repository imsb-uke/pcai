import pickle
from typing import Dict, List, Optional

import cv2
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.rich import tqdm as rich_tqdm  # noqa E402

from src.patch_dataset import PatchDataset, PatchColorAdaptDataset
from src.pcai_datamodule import PatchDataModule

from src.color_adaptation.histogram_km import HistogramKM

class PatchColorAdaptDataModule(PatchDataModule):
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
        metadata_path_fit: Optional[str] = None,
        histograms_path: str = None,
        histograms_path_fit: Optional[str] = None,
        cp_preds_path: str = None,
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        **dataset_kwargs,
    ):
        super().__init__(
            metadata_path=metadata_path,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **dataset_kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `train_dataset`, `val_dataset`, `test_dataset`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        self.metadata_df = self._load_metadata_df(self.hparams.metadata_path)

        # add CP conformal assignment to metadata_df
        self.metadata_df = self._add_cp_preds(self.metadata_df)

        # optional when using different datasets for fit and predict
        if self.hparams.metadata_path_fit is not None:
            self.metadata_df_fit = self._load_metadata_df(self.hparams.metadata_path_fit)
        else:
            self.metadata_df_fit = self.metadata_df.copy()

        # load precomputed histograms
        self.hist_data = self._get_hist_data(self.hparams.histograms_path)

        # optional when using different datasets for fit and predict
        if self.hparams.histograms_path_fit is not None:
            # add histograms from different dataset
            _hist_data_fit = self._get_hist_data(self.hparams.histograms_path_fit)
            # remove samples from metadata_df_fit that are not in _hist_data_fit (no masks during histogram creation)
            self.metadata_df_fit = self.metadata_df_fit.loc[
                lambda df_: df_["sample_id"].isin(_hist_data_fit.index)
            ]
            # concat to full hist_data
            self.hist_data = pd.concat([self.hist_data, _hist_data_fit])

        # initialize the histogram matcher
        self.hist_matcher = HistogramKM(self.hist_data)

        self.hist_matcher.fit(
            self.metadata_df_fit.loc[lambda df_: df_["split"] == "train"]["sample_id"].unique()
        )

        # load PatchDatasets
        if stage == "fit" and not self.train_dataset and not self.val_dataset:
            self.train_dataset = PatchColorAdaptDataset(
                info_df=self.metadata_df.loc[lambda df_: df_["split"] == "train"],
                strategy="random",
                image_transforms=self.image_transforms_train,
                label_transforms_clas=self.label_transforms_clas,
                label_transforms_domain=self.label_transforms_domain,
                hist_matcher=self.hist_matcher,
                **self.dataset_kwargs,
            )

            self.val_dataset = PatchColorAdaptDataset(
                info_df=self.metadata_df.loc[lambda df_: df_["split"] == "val"],
                strategy="all",
                image_transforms=self.image_transforms_val,
                label_transforms_clas=self.label_transforms_clas,
                label_transforms_domain=self.label_transforms_domain,
                hist_matcher=self.hist_matcher,
                **self.dataset_kwargs,
            )

        if stage == "test" and not self.test_dataset:
            self.test_dataset = PatchColorAdaptDataset(
                info_df=self.metadata_df.loc[lambda df_: df_["split"] == "test"],
                strategy="all",
                image_transforms=self.image_transforms_val,
                label_transforms_clas=self.label_transforms_clas,
                label_transforms_domain=self.label_transforms_domain,  
                hist_matcher=self.hist_matcher,
                **self.dataset_kwargs,
            )

        if stage == "predict" and not self.predict_dataset:
            self.predict_dataset = PatchColorAdaptDataset(
                info_df=self.metadata_df,
                strategy="all",
                image_transforms=self.image_transforms_val,
                label_transforms_clas=self.label_transforms_clas,
                label_transforms_domain=self.label_transforms_domain,
                hist_matcher=self.hist_matcher,
                **self.dataset_kwargs,
            )

    def _add_cp_preds(self, metadata_df):
        if self.hparams.cp_preds_path is not None:
            with open(self.hparams.cp_preds_path, "rb") as f:
                cp_preds = pickle.load(f)["scalar"].rename(
                    columns={"data_meta|sample_id": "sample_id"}
                )
            metadata_df = metadata_df.merge(
                cp_preds[["sample_id", "assignment"]],
                on="sample_id",
                how="left",
            )
        else:
            print("No CP predictions found, adapting all images!")
            metadata_df = metadata_df.assign(assignment=0)
        return metadata_df

    @staticmethod
    def _get_hist_data(histograms_path):
        with open(histograms_path, "rb") as f:
            histograms = pickle.load(f)
        return pd.concat(histograms, axis=1)

    def create_slide_histograms(self):
        metadata_df = self._load_metadata_df(self.hparams.metadata_path)

        histograms_dataset = PatchDataset(
            info_df=metadata_df,
            strategy="all",
            image_transforms=None,
            label_transforms_clas=None,
            label_transforms_domain=None,
            **self.dataset_kwargs,
        )

        histograms_dataloader = DataLoader(
            dataset=histograms_dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

        histograms = {}
        for c in (channels := ["H", "S", "V"]):
            histograms[c] = {}

        for patch_bag, _, _, meta_dict in (pbar := rich_tqdm(histograms_dataloader)):
            pbar.set_description(f"Processing {meta_dict['filepath'][0]}")

            # to flat HSV array
            hsv_arr = cv2.cvtColor(
                patch_bag.squeeze(0).numpy().transpose(0, 2, 3, 1).reshape(-1, 1, 3),
                cv2.COLOR_RGB2HSV,
            )

            for i, c in enumerate(channels):
                histograms[c][meta_dict["sample_id"][0]] = cv2.calcHist(
                    [hsv_arr], [i], None, [256], [0, 256]
                ).squeeze()

        for c in channels:
            histograms[c] = pd.DataFrame.from_dict(histograms[c], orient="index")

        return histograms
