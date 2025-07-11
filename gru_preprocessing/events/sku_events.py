from gru_preprocessing.util_functions import collapse, get_entropy_from_counts, add_nearby_skus, parse_embedding,fill_nulls
from argparse import ArgumentParser
from pathlib import Path
import polars as pl
from sklearn.cluster import KMeans
import numpy as np

def get_top_skus_and_categories(buy_product, product_properties, result_dir):
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)
        
    # If no top_skus.npy and top_cats.npy files exist, compute them
    if (result_dir / "top_skus.npy").exists() and (result_dir / "top_cats.npy").exists():
        print("Loading top_skus and top_cats from existing files.")
        top_skus = np.load(result_dir / "top_skus.npy")
        top_cats = np.load(result_dir / "top_cats.npy")
        return top_skus, top_cats
    
    print("NO top_skus and top_cats files found, computing them...")
    enriched_buy_product = buy_product.join(product_properties, on="sku", how="left").select(["sku", "category"])
    top_skus = (
        buy_product
        .group_by("sku")
        .agg(pl.count())
        .sort("count", descending=True)
        .head(150)
    )
    top_cats = (
        enriched_buy_product
        .group_by("category")
        .agg(pl.count())
        .sort("count", descending=True)
        .head(150)
    )
    top_skus=np.array(top_skus["sku"])
    top_cats=np.array(top_cats["category"])
    
    np.save(result_dir / "top_skus.npy", top_skus)
    np.save(result_dir / "top_cats.npy", top_cats)
    
    return top_skus, top_cats

def compute_basic_counts(product_events):
    
    print("Computing sku basic counts...")
    add_count=(
        product_events.filter(pl.col("event_type")=="add_to_cart")
        .group_by("sku")
        .agg(pl.len().alias("count_add"))
    )
    buy_count=(
        product_events.filter(pl.col("event_type")=="buy_product")
        .group_by("sku")
        .agg(pl.len().alias("count_buy"))
    )

    remove_count=(
        product_events.filter(pl.col("event_type")=="remove_from_cart")
        .group_by("sku")
        .agg(pl.len().alias("count_remove"))
    )

    add_user_count=(
        product_events.filter(pl.col("event_type")=="add_to_cart")
        .group_by(["sku", "client_id"])
        .agg(pl.len().alias("count_user_add"))
    )
    buy_user_count=(
        product_events.filter(pl.col("event_type")=="buy_product")
        .group_by(["sku", "client_id"])
        .agg(pl.len().alias("count_user_buy"))
    )

    remove_user_count=(
        product_events.filter(pl.col("event_type")=="remove_from_cart")
        .group_by(["sku", "client_id"])
        .agg(pl.len().alias("count_user_remove"))
    )

    tot_add_user_count=(
        product_events.filter(pl.col("event_type")=="add_to_cart")
        .group_by("client_id")
        .agg(pl.len().alias("tot_user_add"))
    )
    tot_buy_user_count=(
        product_events.filter(pl.col("event_type")=="buy_product")
        .group_by("client_id")
        .agg(pl.len().alias("tot_user_buy"))
    )

    tot_remove_user_count=(
        product_events.filter(pl.col("event_type")=="remove_from_cart")
        .group_by("client_id")
        .agg(pl.len().alias("tot_user_remove"))
    )
    

    product_events=(product_events
                    .join(add_count,how="left",on="sku")
                    .join(buy_count,how="left",on="sku")
                    .join(remove_count,how="left",on="sku")
                    .join(add_user_count,how="left",on=["sku", "client_id"])
                    .join(buy_user_count,how="left",on=["sku", "client_id"])
                    .join(remove_user_count,how="left",on=["sku", "client_id"])
                    .join(tot_add_user_count,how="left",on="client_id")
                    .join(tot_buy_user_count,how="left",on="client_id")
                    .join(tot_remove_user_count,how="left",on="client_id")
                )
    
    print("Computing category basic counts...")
    #category
    add_cat_count=(
        product_events.filter(pl.col("event_type")=="add_to_cart")
        .group_by("category")
        .agg(pl.len().alias("count_cat_add"))
    )
    buy_cat_count=(
        product_events.filter(pl.col("event_type")=="buy_product")
        .group_by("category")
        .agg(pl.len().alias("count_cat_buy"))
    )

    remove_cat_count=(
        product_events.filter(pl.col("event_type")=="remove_from_cart")
        .group_by("category")
        .agg(pl.len().alias("count_cat_remove"))
    )

    add_cat_user_count=(
        product_events.filter(pl.col("event_type")=="add_to_cart")
        .group_by(["category", "client_id"])
        .agg(pl.len().alias("count_cat_user_add"))
    )
    buy_cat_user_count=(
        product_events.filter(pl.col("event_type")=="buy_product")
        .group_by(["category", "client_id"])
        .agg(pl.len().alias("count_cat_user_buy"))
    )

    remove_cat_user_count=(
        product_events.filter(pl.col("event_type")=="remove_from_cart")
        .group_by(["category", "client_id"])
        .agg(pl.len().alias("count_cat_user_remove"))
    )
    
    product_events=(product_events
                .join(add_cat_count,how="left",on="category")
                .join(buy_cat_count,how="left",on="category")
                .join(remove_cat_count,how="left",on="category")
                .join(add_cat_user_count,how="left",on=["category", "client_id"])
                .join(buy_cat_user_count,how="left",on=["category", "client_id"])
                .join(remove_cat_user_count,how="left",on=["category", "client_id"])
               )
    
    return fill_nulls(product_events)

def compute_conversion_rates(product_events):
    print("Computing conversion rates...")
    product_events = product_events.with_columns([
        (pl.col("count_cat_buy") / pl.col("count_cat_add")).alias("global_cat_cr"),
        (pl.col("count_cat_user_buy") / pl.col("count_cat_user_add")).alias("local_cat_cr")
    ])

    product_events = product_events.with_columns([
        (pl.col("count_buy") / pl.col("count_add")).alias("global_cr"),
        (pl.col("tot_user_buy") / pl.col("tot_user_add")).alias("local_cr"),
        (pl.col("count_user_buy") / pl.col("count_user_add")).alias("cr")
    ])

    columns=["local_cr","cr","global_cr", "global_cat_cr", "local_cat_cr"]

    product_events = product_events.with_columns([
        pl.when(pl.col(c).is_infinite())
        .then(1.0)
        .otherwise(pl.col(c))
        .fill_nan(0.0)
        .alias(c) for c in columns
    ])
    
    return product_events

def get_clustering_names(product_properties, result_dir):
    print("Getting names clusters...")
    # Check if already computed
    if (result_dir / "names_clusters_map.parquet").exists():
        print("Loading names clusters from existing file.")
        return pl.read_parquet(result_dir / "names_clusters_map.parquet")
    
    print("No names_clusters_map.parquet file found, computing it...")
    names=product_properties[["sku","name"]]
    names = names.with_columns(
        pl.col("name").map_elements(parse_embedding, return_dtype=pl.List(pl.Int32)).alias("embedding")
    )


    name_embeddings = np.array(names["embedding"].to_list())

    k = 100 #found using elbow method but can be tuned too
    kmeans = KMeans(n_clusters=k, random_state=42)
    names_cluster_ids = kmeans.fit_predict(name_embeddings)
    centroids_names = kmeans.cluster_centers_

    names = names.with_columns([
        pl.Series("cluster_id",names_cluster_ids)
    ])

    centroid_names_dict = {i: list(map(float, centroid)) for i, centroid in enumerate(centroids_names)}
    names= names.with_columns([
        pl.col("cluster_id")
        .map_elements(lambda cid: centroid_names_dict.get(cid, []), return_dtype=pl.List(pl.Float32))
        .alias("centroid")
    ])
    
    names.write_parquet(result_dir / "names_clusters_map.parquet")
    return names

def compute_cluster_counts(product_events):
    print("Computing cluster counts...")
    
    cluster_add_count=(product_events.filter(pl.col("event_type")=="add_to_cart")
        .group_by("cluster_id")
        .agg(pl.len().alias("cluster_add_count")))

    cluster_buy_count=(product_events.filter(pl.col("event_type")=="buy_product")
        .group_by("cluster_id")
        .agg(pl.len().alias("cluster_buy_count")))

    cluster_remove_count=(product_events.filter(pl.col("event_type")=="remove_from_cart")
        .group_by("cluster_id")
        .agg(pl.len().alias("cluster_remove_count")))


    cluster_add_user_count=(product_events.filter(pl.col("event_type")=="add_to_cart")
        .group_by(["client_id","cluster_id"])
        .agg(pl.len().alias("cluster_add_user_count")))

    cluster_buy_user_count=(product_events.filter(pl.col("event_type")=="buy_product")
        .group_by(["client_id","cluster_id"])
        .agg(pl.len().alias("cluster_buy_user_count")))

    cluster_remove_user_count=(product_events.filter(pl.col("event_type")=="remove_from_cart")
        .group_by(["client_id","cluster_id"])
        .agg(pl.len().alias("cluster_remove_user_count")))
    
    product_events=(product_events
                    .join(cluster_add_count,on="cluster_id",how="left")
                    .join(cluster_buy_count,on="cluster_id",how="left")
                    .join(cluster_remove_count,on="cluster_id",how="left")
                    .join(cluster_add_user_count,on=["client_id","cluster_id"],how="left")
                    .join(cluster_buy_user_count,on=["client_id","cluster_id"],how="left")
                    .join(cluster_remove_user_count,on=["client_id","cluster_id"],how="left")
                )
    return product_events
    
def compute_entropy(product_events):
    print("Computing entropy...")
    product_events=(product_events
        .join(get_entropy_from_counts(product_events,"sku","sku_entropy"),on="sku",how="left")
        .join(get_entropy_from_counts(product_events,"cluster_id","cluster_entropy"),on="cluster_id",how="left")
        .join(get_entropy_from_counts(product_events,"category","cat_entropy"),on="category",how="left")
        .join(get_entropy_from_counts(product_events,"sku","user_sku_entropy",user_entropy=True),on="client_id",how="left")
        .join(get_entropy_from_counts(product_events,"cluster_id","user_cluster_entropy",user_entropy=True),on="client_id",how="left")
        .join(get_entropy_from_counts(product_events,"category","user_cat_entropy",user_entropy=True),on="client_id",how="left")
        )
    return product_events


def main(args):
    data_dir = Path(args.data_dir)
    data_input_dir = Path(args.data_input_dir)
    result_dir = Path(args.result_dir)

    buy_product = pl.read_parquet(data_input_dir / "product_buy.parquet")
    add_to_cart = pl.read_parquet(data_input_dir / "add_to_cart.parquet")
    remove_from_cart = pl.read_parquet(data_input_dir / "remove_from_cart.parquet")

    product_properties = pl.read_parquet(data_dir / "product_properties.parquet")

    buy_product = buy_product.with_columns(
        pl.lit("buy_product").alias("event_type")
    )

    add_to_cart = add_to_cart.with_columns(
        pl.lit("add_to_cart").alias("event_type")
    )

    remove_from_cart = remove_from_cart.with_columns(
        pl.lit("remove_from_cart").alias("event_type")
    )
    

    product_events = pl.concat([buy_product, add_to_cart, remove_from_cart])
    
    product_events=product_events.join(product_properties[["sku","category"]],on="sku",how="left")
    
    product_events = compute_basic_counts(product_events)
    
    product_events = compute_conversion_rates(product_events)
    
    if data_dir == data_input_dir:
        complete_buy_product = buy_product
    else:
        complete_buy_product = pl.read_parquet(data_dir / "product_buy.parquet")
        
    top_skus, top_cats = get_top_skus_and_categories(complete_buy_product, product_properties, result_dir)
    product_events = product_events.with_columns([
        pl.col("sku").is_in(top_skus).cast(pl.Int8).alias("top_sku"),
        pl.col("category").is_in(top_cats).cast(pl.Int8).alias("top_category")
    ])
    
    
    names_clusters = get_clustering_names(product_properties, result_dir)
    product_events = product_events.join(names_clusters[["sku","cluster_id"]],on="sku",how="left")
    product_events = compute_cluster_counts(product_events)
    
    product_events=fill_nulls(product_events)
    
    product_events = compute_entropy(product_events)

    print("Adding normalized price...")
    product_events=product_events.join(product_properties[["sku","price"]],on="sku",how="left").with_columns(((pl.col("price")+1)/100).alias("price"))
    print("Shape is :", product_events.shape)
    print("Product events head:", product_events.head(30))
    product_events.write_parquet(result_dir / "product_events.parquet")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the whole 6 months data files. It must have same structure as original 'ubc_data'.")
    parser.add_argument("--data_input_dir", type=str, required=True, help="Directory containing the input data files for processing.")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save the results.")
    args = parser.parse_args()
    
    main(args)

    

    