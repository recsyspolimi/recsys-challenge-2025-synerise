import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetData:
    """
    Dataclass for storing data for training and validation.
    """

    train_df: pd.DataFrame
    validation_df: pd.DataFrame

    @classmethod
    def read_from_dir(cls, target_dir: Path):
        train_df = pd.read_parquet(target_dir / "train_target.parquet")
        validation_df = pd.read_parquet(target_dir / "validation_target.parquet")
        return cls(train_df, validation_df)
