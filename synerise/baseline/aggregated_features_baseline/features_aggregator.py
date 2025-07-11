from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from synerise.baseline.aggregated_features_baseline.calculators import (
    Calculator,
    StatsFeaturesCalculator,
    QueryFeaturesCalculator,
)
from synerise.baseline.aggregated_features_baseline.constants import (
    EventTypes,
    EVENT_TYPE_TO_COLUMNS,
    QUERY_COLUMN,
    EMBEDDINGS_DTYPE,
)


def get_top_values(
    events: pd.DataFrame, columns: List[str], top_n: int
) -> dict[str, pd.Index]:
    """
    Extracts the top_n most frequent values for specified columns in event data.

    Args:
        events (pd.DataFrame): DataFrame containing event data.
        columns (List[str]): List of column names in the DataFrame for which top values
        are to be found.
        top_n (int): Number of top values to extract for each column.

    Returns:
        dict: A dictionary with each key being a column name and the corresponding value
        being a list of top_n values for that column.
    """
    top_values = {}
    for column in columns:
        val_count = events[column].value_counts()
        top_val = val_count.index[:top_n]
        top_values[column] = top_val
    return top_values


class FeaturesAggregator:
    """
    Class for aggregating features for all event types and client ids.
    """

    def __init__(
        self,
        num_days: List[int],
        top_n: int,
        relevant_client_ids: np.ndarray,
    ):
        """
        Args:
            num_days (List[int]): A list of time windows (in days) for generating features. Each time window will produce different set of features from aggregated events from defined period.
            top_n (int): Number of columns' top values to consider for aggregating events.
        """
        self.num_days = num_days
        self.top_n = top_n
        self._aggregated_features: Dict[int, Dict[EventTypes, np.ndarray]] = {}
        self._features_sizes: Dict[EventTypes, int] = {}
        self._relevant_client_ids = relevant_client_ids

    @property
    def _total_dimension(self):
        return sum(
            (
                self._features_sizes[event_type]
                for event_type in self._features_sizes.keys()
            )
        )

    def _update_features_sizes(self, event_type: EventTypes, features_size: int):
        """
        Updates featues sizes for given event type.

        Args:
            event_type (EventTypes): type of an event
            features_sizes (int): features_size for given event type
        """

        self._features_sizes[event_type] = features_size

    def _get_features_size(self, event_type: EventTypes) -> int:
        """
        Extracts features size for given event type.

        Args:
            event_type (EventTypes): type of an event
        Returns:
            int: features size
        """
        return self._features_sizes[event_type]

    def _update_features(
        self, event_type: EventTypes, client_id: int, features: np.ndarray
    ):
        """
        Updates features for given event type and client id.

        Args:
            event_type (EventTypes): type of an event
            client_id (int): client id
            features: features for given client id and event type
        """

        self._aggregated_features.setdefault(client_id, {})[event_type] = features

    def _get_features(self, client_id: int, event_type: EventTypes) -> np.ndarray:
        """
        Accesses features for client id and event type. If there are no features corresponding to a particular client id and event type, then returns zero features of appropriate size.

        Args:
            client_id (int): client id
            event_type (str): type of an event
        Returns:
            np.array: features for given client id and event type
        """
        features_size = self._get_features_size(event_type=event_type)
        return self._aggregated_features.get(client_id, {}).get(
            event_type, np.zeros(features_size, dtype=EMBEDDINGS_DTYPE)
        )

    def get_calculator(
        self,
        event_type: EventTypes,
        df: pd.DataFrame,
        columns: List[str],
    ) -> Calculator:
        """
         Returns calculator for processing given event_type.

         Args:
             event_type (EventTypes): type of an event
             df (pd.DataFrame): dataframe corresponding to event type
             columns (List[str]): list of columns names to process
        Returns:
             Calculator: object which conforms to calculator interface
        """
        if event_type is EventTypes.SEARCH_QUERY:
            return QueryFeaturesCalculator(
                query_column=QUERY_COLUMN, single_query=df.iloc[0][QUERY_COLUMN]
            )
        else:
            max_date = df["timestamp"].max()
            unique_values = get_top_values(df, columns, self.top_n)
            return StatsFeaturesCalculator(
                num_days=self.num_days,
                max_date=max_date,
                columns=columns,
                unique_values=unique_values,
            )

    def _filter_events_to_relevant_clients(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Method for filtering a DataFrame to only contain lines where
        the contant of the "client_id" column is in `self._relevant_clients`.
        """
        return df[df["client_id"].isin(self._relevant_client_ids)]

    def generate_features(
        self,
        event_type: EventTypes,
        client_id_column: str,
        df: pd.DataFrame,
        columns: List[str],
    ):
        """
        The method generates features for all clients in client id column for given event type. Features are stored in the class instance dictionary attributes.

        Args:
            event_type (EventTypes): type of an event.
            client_id_column (str): Name of the column containing client ids.
            df (pd.DataFrame): DataFrame containing the event data.
            columns (List[str]): Columns to be used for feature generation, e.g., 'brand' to aggregate client id events with a product of a given brand.
        """

        df = self._filter_events_to_relevant_clients(df)

        calculator = self.get_calculator(
            event_type=event_type,
            df=df,
            columns=columns,
        )
        self._update_features_sizes(
            event_type=event_type, features_size=calculator.features_size
        )
        for client_id, events in tqdm(df.groupby(client_id_column)):
            assert isinstance(client_id, int)
            features = calculator.compute_features(events=events)
            self._update_features(
                event_type=event_type,
                client_id=client_id,
                features=features,
            )

    def merge_features(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merges feature arrays for different event types into embeddings. Computes client ids by collecting all keys of attribute dictionary.
        """
        client_ids = []
        embeddings = []
        for client_id in tqdm(self._aggregated_features.keys()):
            client_ids.append(client_id)
            embeddings_for_client: List[np.ndarray] = []
            for event_type in EVENT_TYPE_TO_COLUMNS.keys():
                features = self._get_features(
                    client_id=client_id, event_type=event_type
                )
                embeddings_for_client.append(features)
            embedding_for_client: np.ndarray = np.concatenate(embeddings_for_client)
            embeddings.append(embedding_for_client)
        return np.array(client_ids), np.array(embeddings)
