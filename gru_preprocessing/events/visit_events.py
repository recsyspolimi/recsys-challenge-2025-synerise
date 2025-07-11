from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import polars as pl

from gru_preprocessing.util_functions import collapse, get_entropy_from_counts, add_nearby_skus, load_top_skus_and_cats


def main(args):
    data_dir = Path(args.data_dir)
    data_input_dir = Path(args.data_input_dir)
    result_dir = Path(args.result_dir)
    
    page_visit = pl.read_parquet(data_input_dir / "page_visit.parquet")

    top_skus, top_cats = load_top_skus_and_cats(result_dir)
    #adding counts of nearby sku events
    print("Computing collapse function for page_visit events... (This may take a while :) )")
    relevant_clients = np.load(data_dir / "input/relevant_clients.npy")
    filtered_visits=page_visit.filter(pl.col("client_id").is_in(relevant_clients))
    page_sessions=collapse(filtered_visits,"url")
    
    print("Adding nearby skus to page sessions...")
    
    # Loading product events
    product_events = pl.read_parquet(result_dir / "product_events.parquet")
    
    page_sessions = add_nearby_skus(page_sessions,"url",product_events[["sku","client_id","timestamp","category"]],top_skus, top_cats)
    
    print("Computing counts...")
    url_counts = (
        page_visit
        .group_by("url")
        .agg(pl.len().alias("count"))
    )
    client_visit_total = (
        page_visit
        .group_by("client_id")
        .agg(pl.len().alias("client_total_visits"))
    )
    client_url_visits = (
        page_visit
        .group_by(["client_id","url"])
        .agg(pl.len().alias("client_url_visits"))
    )
    
    
    page_sessions=(page_sessions.join(url_counts,on="url",how="left")
                .join(client_visit_total,on="client_id",how="left")
    .join(client_url_visits,on=["url","client_id"],how="left")

    )
    
    print("Computing entropy...")
    page_sessions=(page_sessions
        .join(get_entropy_from_counts(page_sessions,"url","url_entropy"),on="url",how="left")
        .join(get_entropy_from_counts(page_sessions,"url","client_entropy",user_entropy=True),on="client_id",how="left"))
    
    
    page_sessions=page_sessions.rename({"session_start":"timestamp"}).drop(["session_id","session_end","session_end_ext"])
    
    print("Saving page sessions to parquet...")
    page_sessions.write_parquet(result_dir / "page_sessions.parquet")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files.")
    parser.add_argument("--data_input_dir", type=str, required=True, help="Directory containing the input data files for processing.")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save the results.")
    args = parser.parse_args()
    
    main(args)
