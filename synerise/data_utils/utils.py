import pandas as pd
import logging

from synerise.data_utils.data_dir import DataDir

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def join_properties(
    event_df: pd.DataFrame, properties_df: pd.DataFrame
) -> pd.DataFrame:
    """
    This function joins product properties for each event in event_df.
    Args:
        event_df (pd.DataFrame): DataFrame storing events to which properties are joined.
        properties_df (pd.DataFrame): DataFrame with product properties, that should be
        joined to event_df.
    Returns:
        pd.DataFrame: events DataFrame with product properties.
    """
    joined_df = event_df.join(properties_df.set_index("sku"), on="sku", validate="m:1")
    assert joined_df.notna().all().all(), "Missing sku in properties_df"
    return joined_df


def load_with_properties(data_dir: DataDir, event_type: str) -> pd.DataFrame:
    """
    This function load dataset for given event type. If event type admits sku column, then product properties are joined.
    Args:
        data_dir (DataDir): The DataDir class where Paths to raw event data, input and targte folders are stored.
        event_type (str): Name of the event.
    Returns:
        pd.DataFrame: events DataFrame with product joined properties if available.
    """
    event_df = pd.read_parquet(data_dir.input_dir / f"{event_type}.parquet")
    if event_type not in ["page_visit", "search_query"]:
        properties_df = pd.read_parquet(data_dir.properties_file)
        return join_properties(event_df=event_df, properties_df=properties_df)
    return event_df
