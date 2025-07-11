from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

from gru_preprocessing.util_functions import collapse, get_entropy_from_counts, add_nearby_skus, parse_embedding, \
    load_top_skus_and_cats


def main(args):
    data_dir = Path(args.data_dir)
    data_input_dir = Path(args.data_input_dir)
    result_dir = Path(args.result_dir)
    
    search_query = pl.read_parquet(data_input_dir / "search_query.parquet")
    unique_queries = search_query.select("query").unique().with_row_index(name="query_id")
    search_query = search_query.join(unique_queries, on="query", how="left").drop("query")
    
    queries = unique_queries.with_columns(
    pl.col("query").map_elements(parse_embedding, return_dtype=pl.List(pl.Int32)).alias("embedding")
    )

    query_embeddings = np.array(queries["embedding"].to_list())

    k = 100
    print(f"Clustering queries into {k} clusters...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_ids = kmeans.fit_predict(query_embeddings)
    centroids = kmeans.cluster_centers_

    queries = queries.with_columns([
        pl.Series("cluster_id", cluster_ids)
    ])
    
    #scores
    print("Calculating query scores...")
    queries_global_scores=(
        search_query
        .group_by("query_id")
        .agg(pl.len().alias("query_score"))
    )
    queries_user_scores=(
        search_query
        .group_by(["client_id","query_id"])
        .agg(pl.len().alias("query_user_score"))
    )
    
    search_query=search_query.join(queries[["cluster_id","query_id"]],on="query_id",how="left")
    
    #cluster scores
    print("Calculating cluster scores...")
    queries_made_by_user=(search_query.group_by("client_id").agg(pl.len().alias("queries_made_by_user")))
    cluster_made_by_user=(search_query.group_by(["cluster_id","client_id"]).agg(pl.len().alias("cluster_made_by_user")))
    cluster_score=(search_query.group_by("cluster_id").agg(pl.len().alias("cluster_score")))
    
    search_query=collapse(search_query,"query_id")
    
    top_skus, top_cats = load_top_skus_and_cats(result_dir)
    product_events = pl.read_parquet(result_dir / "product_events.parquet")
    
    print("Adding nearby SKUs to search queries...")
    search_query =  add_nearby_skus(search_query,"query_id", product_events,top_skus, top_cats)
    
    search_query=search_query.join(queries[["cluster_id","query_id"]],on="query_id",how="left")
    #adding scores and entropies
    search_query=(search_query
                .join(queries_global_scores,on="query_id",how="left")
                .join(queries_user_scores,on=["client_id","query_id"],how="left")
                .join(queries_made_by_user,on="client_id",how="left")
                .join(cluster_made_by_user,on=["client_id","cluster_id"],how="left")
                .join(cluster_score,on="cluster_id",how="left")
             )
    
    search_query=(search_query
    .join(get_entropy_from_counts(search_query,"query_id","query_entropy"),on="query_id",how="left")
    .join(get_entropy_from_counts(search_query,"cluster_id","cluster_entropy"),on="cluster_id",how="left")
    .join(get_entropy_from_counts(search_query,"query_id","user_query_entropy",user_entropy=True),on="client_id",how="left")
    .join(get_entropy_from_counts(search_query,"cluster_id","user_cluster_entropy",user_entropy=True),on="client_id",how="left")
    )
    
    search_query=search_query.rename({"session_start":"timestamp"})
    
    queries = queries.with_columns(
    pl.col("cluster_id")
      .map_elements(lambda cid: centroids[int(cid)].tolist())
      .alias("centroid")  
    )
    
    queries.write_parquet(result_dir / "queries.parquet")
    
    search_query.write_parquet(result_dir / "search_events.parquet")
    
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files.")
    parser.add_argument("--data_input_dir", type=str, required=True, help="Directory containing the input data files for processing.")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save the results.")
    args = parser.parse_args()
    
    main(args)