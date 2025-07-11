import polars as pl
import argparse
from pathlib import Path
from datetime import datetime
import polars as pl
import numpy as np

from gru_preprocessing.util_functions import scale
from gru_preprocessing.events.events_normalizer import normalize


def add_temporal_features(all_events):
    test_start_date = all_events["timestamp"].max()
    all_events = all_events.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("weekday"),
        pl.col("timestamp").dt.day().alias("day"),
        pl.col("timestamp").dt.month().alias("month"),

        ((pl.col("timestamp").dt.hour() / 23) * 2 - 1).alias("hour_norm"),

        ((pl.col("timestamp").dt.weekday() / 6) * 2 - 1).alias("weekday_norm"),

        ((pl.col("timestamp").dt.day().cast(pl.Float64) / 31) * 2 - 1).alias("week_of_month_norm"),

        ((pl.lit(test_start_date) - pl.col("timestamp"))
         .dt.total_days() / 7.0).alias("weeks_diff")
    ])

    all_events = all_events.with_columns([
        ((pl.lit(test_start_date) - pl.col("timestamp"))
         .dt.total_days() / 7).alias("weeks_since_test_start")
    ])
    return all_events

def get_slope_df(events_timestamps,scale_flag=True):
    events_timestamps = events_timestamps.with_columns([
        pl.col("timestamp").dt.week().alias("week")
    ])
    weekly_counts = (
        events_timestamps
        .group_by(["client_id", "week"])
        .agg(pl.len().alias("event_count"))
    )
    weeks_sorted = (
        weekly_counts
        .select("week")
        .unique()
        .sort("week", descending=True)
        .get_column("week")
        .to_list()
    )
    slope_df = (
        weekly_counts
        .pivot(
            values="event_count",
            index="client_id",
            columns="week",
            aggregate_function="first"
        )
        .fill_null(0)
        .select(["client_id"] + [str(w) for w in weeks_sorted])
    )
    if scale_flag:
        return scale(slope_df,columns=slope_df.drop("client_id").columns,transform="yeo")
    return slope_df



def main(args):
    result_dir = Path(args.result_dir)
    product_df_path = result_dir / "product_events.parquet"
    visit_df_path = result_dir / "visit_events.parquet"
    search_df_path = result_dir / "search_events.parquet"

    data_dir = Path(args.data_dir)
    relevant_clients = np.load(data_dir / "input/relevant_clients.npy").tolist()

    print("Loading data...")
    product = pl.read_parquet(product_df_path)
    visit = pl.read_parquet(visit_df_path)
    search = pl.read_parquet(search_df_path)
    
    print("Normalizing product...")
    product = normalize(product, "product", normalizer=args.normalizer, clusters_map_path=result_dir / "names_clusters_map.parquet")
    print("Normalizing visit...")
    visit = normalize(visit, "visit", normalizer=args.normalizer)
    print("Normalizing search...")
    search = normalize(search, "query", normalizer=args.normalizer, clusters_map_path=result_dir / "queries.parquet")
    print("Adding event types...")
    
    search = search.with_columns(pl.lit("search_query").alias("event_type"))
    visit = visit.with_columns(pl.lit("page_visit").alias("event_type"))
    
    print("Adding temporal features...")
    all_events = add_temporal_features(pl.concat([product, visit, search], how="diagonal"))
    
    print("Slope events...")
    scaled_slope=get_slope_df(all_events[["timestamp","client_id"]])
    all_events = all_events.join(scaled_slope, on="client_id", how="left")
    all_events = all_events.sort(["client_id", "timestamp"])
    
    grouped = (
        all_events
        .group_by("client_id", maintain_order=True)
        .agg([
            pl.struct(all_events.columns).alias("events")
        ])
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped.lazy().sink_parquet(output_dir / "groupedALL.parquet")
    grouped = grouped.filter(pl.col("client_id").is_in(relevant_clients)).sink_parquet(output_dir / "grouped.parquet")
    np.save(output_dir / "client_ids.npy", grouped["client_id"].to_numpy())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="final data processing script")
    parser.add_argument("--data_dir", type=str, help="Directory containing the data")
    parser.add_argument("--result_dir", type=str, help="Directory to load the results.")
    parser.add_argument("--normalizer", type=str, help="Normalizer function to use for scaling, options: 'minmax', 'yeo'")
    parser.add_argument("--output_dir", type=str, help="Directory to save the results")

    args = parser.parse_args()
    main(args)