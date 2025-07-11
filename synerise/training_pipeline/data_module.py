import numpy as np
import pytorch_lightning as pl
import logging

from torch.utils.data import DataLoader

from synerise.training_pipeline.dataset import (
    BehavioralDataset,
)
from synerise.training_pipeline.target_data import (
    TargetData,
)
from synerise.training_pipeline.target_calculators import (
    TargetCalculator,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class BehavioralDataModule(pl.LightningDataModule):
    """
    DataModule containing two BehavioralDatasets, one for
    training and one for validation.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        client_ids: np.ndarray,
        target_data: TargetData,
        target_calculator: TargetCalculator,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.client_ids = client_ids
        self.embeddings = embeddings
        self.target_data = target_data
        self.target_calculator = target_calculator

    def setup(self, stage) -> None:
        if stage == "fit":

            logger.info("Constructing datasets")

            self.train_data = BehavioralDataset(
                embeddings=self.embeddings,
                client_ids=self.client_ids,
                target_df=self.target_data.train_df,
                target_calculator=self.target_calculator,
            )

            self.validation_data = BehavioralDataset(
                embeddings=self.embeddings,
                client_ids=self.client_ids,
                target_df=self.target_data.validation_df,
                target_calculator=self.target_calculator,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
