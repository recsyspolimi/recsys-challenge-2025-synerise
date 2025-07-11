import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict
from pathlib import Path
import argparse
import logging


from synerise.data_utils.constants import (
    EventTypes,
    DAYS_IN_TARGET,
)
from synerise.data_utils.utils import join_properties
from synerise.data_utils.data_dir import DataDir

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class DataSplitter:
    def __init__(
        self,
        challenge_data_dir: DataDir,
        days_in_target: int,
        end_date: datetime,
    ):
        """
        Args:
            challenge_data_dir (DataDir): The DataDir class where Paths to raw event data, input and targte folders are stored.
            days_in_target (int): Time-window for target events.
            end_date (datetime): No events after this data are considered in any of created sets. Usually, will be equal to last event in raw data.
        """
        self.challenge_data_dir = challenge_data_dir
        self.days_in_target = days_in_target
        self.end_date = pd.to_datetime(end_date)

        self.input_events: Dict[str, pd.DataFrame] = {}
        self.target_events: Dict[str, pd.DataFrame] = {}

    def _compute_target_start_dates(self) -> Tuple[datetime, datetime]:
        """
        The method finds the first date in train and validation targets. From the end date,
        we subtract two target periods (train and validation) minus one day as we want to
        count days not as the 24 h from end date but full days from 00:00:00
        Returns:
            tuple[datetime]: Returns a tuple with two dates: first date in train target and
            first date in validation target.
        """
        train_target_start = self.end_date - timedelta(days=2 * self.days_in_target - 1)
        train_target_start = train_target_start.replace(hour=0, minute=0, second=0)
        validation_target_start = train_target_start + timedelta(self.days_in_target)
        return train_target_start, validation_target_start

    def _create_input_chunk(
        self,
        event_df: pd.DataFrame,
        train_target_start: datetime,
    ) -> pd.DataFrame:
        """
        Returns events that occured before train_target_start.
        Args:
            event_df (pd.DataFrame): A DataFrame storing all events.
            train_target_start (datetime):  The first date in train target time-window.
        Returns:
            pd.DataFrame: the function returns DataFrame with input data.
        """
        train_input = event_df.loc[event_df["timestamp"] < train_target_start]
        return train_input

    def _create_target_chunks(
        self,
        event_df: pd.DataFrame,
        properties_df: pd.DataFrame,
        train_target_start: datetime,
        validation_target_start: datetime,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns subsequent events starting form train_target_start but before validation_target_start as train target
        and the events after validation_target_start as validation target.
        Product properties are joined into target DataFrames.
        Args:
            event_df (pd.DataFrame): A DataFrame storing all events.
            properties_df (pd.DataFrame): Product properties.
            train_target_start (datetime): The first date in train target time-window.
            validation_target_start (datetime): The first date in validation target time-window.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames: first storing train target events,
            second storing validation target events.
        """
        train_target = event_df.loc[
            (event_df["timestamp"] >= train_target_start)
            & (event_df["timestamp"] < validation_target_start)
        ]
        validation_target = event_df.loc[
            (event_df["timestamp"] >= validation_target_start)
            & (event_df["timestamp"] <= self.end_date)
        ]

        train_target = join_properties(train_target, properties_df)
        validation_target = join_properties(validation_target, properties_df)

        return train_target, validation_target

    def split(self) -> None:
        """
        This function splits event data into subset of events to use to create model inputs
        and sets to create training target and validation target. Data are splitted in time:
        - input data consists of events up to the training target starting point
        - train_target consists of events from the days_in_target subsequent days after the last event of input_data
        - validation_target consists of events from the days_in_target subsequent days after train target
        """
        train_target_start, validation_target_start = self._compute_target_start_dates()

        for event_type in EventTypes:
            msg = f"Creating splits for {event_type.value} event type"
            logger.info(msg=msg)
            events = self.load_events(event_type=event_type)
            events["timestamp"] = pd.to_datetime(events.timestamp)

            train_input = self._create_input_chunk(
                event_df=events, train_target_start=train_target_start
            )
            self.input_events[event_type.value] = train_input

            if event_type == "product_buy":
                properties = pd.read_parquet(self.challenge_data_dir.properties_file)
                train_target, validation_target = self._create_target_chunks(
                    event_df=events,
                    properties_df=properties,
                    train_target_start=train_target_start,
                    validation_target_start=validation_target_start,
                )
                self.target_events["train_target"] = train_target
                self.target_events["validation_target"] = validation_target

    def save_splits(self) -> None:
        """
        Saves splitted data into input and target subdirectories of competition data folder.
        """
        for event_type, events in self.input_events.items():
            msg = f"Saving {event_type} train input"
            logger.info(msg=msg)
            events.to_parquet(
                self.challenge_data_dir.input_dir / f"{event_type}.parquet", index=False
            )

        for target_type, events in self.target_events.items():
            msg = f"Saving {target_type}"
            logger.info(msg=msg)
            events.to_parquet(
                self.challenge_data_dir.target_dir / f"{target_type}.parquet",
                index=False,
            )

    def load_events(self, event_type: EventTypes) -> pd.DataFrame:
        return pd.read_parquet(
            self.challenge_data_dir.data_dir / f"{event_type.value}.parquet"
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge-data-dir",
        type=str,
        required=True,
        help="Competition data directory which should consists of event files, product properties and two subdirectories — input and target",
    )
    return parser


def main():
    parser = get_parser()
    params = parser.parse_args()

    challenge_data_dir = DataDir(data_dir=Path(params.challenge_data_dir))

    product_buy = pd.read_parquet(
        challenge_data_dir.data_dir / f"{EventTypes.PRODUCT_BUY.value}.parquet"
    )
    end_date = pd.to_datetime(product_buy["timestamp"].max())

    splitter = DataSplitter(
        challenge_data_dir=challenge_data_dir,
        days_in_target=DAYS_IN_TARGET,
        end_date=end_date,
    )
    splitter.split()
    splitter.save_splits()


if __name__ == "__main__":
    main()
