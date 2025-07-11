import numpy as np
import pandas as pd
import logging

from torch.utils.data import Dataset

from synerise.training_pipeline.target_calculators import (
    TargetCalculator,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class BehavioralDataset(Dataset):
    """
    Dataset containing client embeddings, and target
    calculator that computes targets for each client.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        client_ids: np.ndarray,
        target_df: pd.DataFrame,
        target_calculator: TargetCalculator,
    ) -> None:
        super().__init__()

        self.embeddings = embeddings
        self.client_ids = client_ids
        self.target_df = target_df
        self.target_calculator = target_calculator

    def __len__(self) -> int:
        return self.client_ids.shape[0]

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        client_id = self.client_ids[idx]
        target = self.target_calculator.compute_target(
            client_id=client_id, target_df=self.target_df
        )
        return self.embeddings[idx], target
