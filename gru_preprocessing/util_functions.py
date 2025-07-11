import polars as pl
import numpy as np
from datetime import datetime, timedelta
import math
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

def parse_embedding(x):
    if isinstance(x, str):
        try:
            return list(map(int, x.strip("[]").replace(",", " ").split()))
        except Exception as e:
            print(f"Parse error: {x} -> {e}")
            return None
    return x
def fill_nulls(df, columns=None, fill_value=0):
    numeric_cols = columns if columns else [col for col, dtype in df.schema.items() if dtype in (pl.Int64, pl.Int32, pl.UInt32, pl.Float64)]
    return df.with_columns([
        pl.col(col).fill_null(0) for col in numeric_cols
    ])


def get_entropy_from_counts(df, group_key,entropy_name,user_entropy=False):
    df_agg = (
        df.group_by([group_key, "client_id"])
          .agg(pl.count().alias("count"))
    )
    if(user_entropy):
        group_key="client_id"
    df_agg = df_agg.with_columns([
        pl.col("count").fill_null(0),
        pl.sum("count").over(group_key).alias("total")
    ])

    df_agg = df_agg.with_columns([
        pl.when(pl.col("total") > 0)
          .then(pl.when(((pl.col("count") / (pl.col("total") + 1e-9)))>0.999999).then(0.999999).otherwise(pl.col("count") / (pl.col("total") + 1e-9))
               )
          .otherwise(0.0)
          .alias("prob")
    ])



    df_agg = df_agg.with_columns([
        (-pl.col("prob") * (pl.col("prob") + 1e-9).log(base=2)).alias("entropy_contrib")
    ])
    return df_agg.group_by(group_key).agg(pl.sum("entropy_contrib").alias(entropy_name))

from datetime import timedelta
def collapse(df,key,session_threshold = timedelta(minutes=40).total_seconds()):
    df = df.sort(["client_id", "timestamp"])
    df= df.with_columns([
        pl.col("timestamp").shift(1).over("client_id").alias("prev_ts"),
    ])

    df = df.with_columns([
        pl.when(pl.col("prev_ts").is_not_null())
          .then((pl.col("timestamp") - pl.col("prev_ts")).dt.total_seconds())
          .otherwise(0.0)
          .alias("time_delta")
    ])
    df_pandas = df.to_pandas()

    group_ids = []
    group_id = 0
    cumulative_time = 0
    last_client = None

    for idx, row in df_pandas.iterrows():
        client = row["client_id"]
        delta = row["time_delta"]

        if client != last_client:
            group_id += 1
            cumulative_time = 0
        elif cumulative_time + delta > session_threshold:
            group_id += 1
            cumulative_time = 0
        else:
            cumulative_time += delta

        group_ids.append(group_id)
        last_client = client
    df = df.with_columns(pl.Series("session_id", group_ids))

    collapsed_sessions = (
        df.group_by(["client_id", "session_id"])
        .agg([
            pl.col(key).first(),
            pl.col("timestamp").min().alias("session_start"),
            pl.col("timestamp").max().alias("session_end"),
        ])
        .with_columns([
            (pl.col("session_end") - pl.col("session_start")).dt.total_seconds().alias("time_diff")
        ])
    )
    return collapsed_sessions

def add_nearby_skus(collapsed_sessions_sq,key,product_events,top_skus, top_cats):
    collapsed_sessions_sq = collapsed_sessions_sq.with_columns(
        (pl.col("session_start") + pl.duration(seconds=pl.col("time_diff")) + timedelta(minutes=5)).alias("session_end_ext")
    )

    min_ts = collapsed_sessions_sq.select(pl.col("session_start").min()).item()
    max_ts = collapsed_sessions_sq.select(pl.col("session_end_ext").max()).item()

    product_events_filtered = product_events.filter(
        (pl.col("timestamp") >= min_ts) & (pl.col("timestamp") <= max_ts)
    )

    joined = product_events_filtered.join_asof(
        collapsed_sessions_sq.sort(["client_id", "session_start"]),
        left_on="timestamp",
        right_on="session_start",
        by="client_id",
        strategy="backward"
    ).filter(
        (pl.col("timestamp") <= pl.col("session_end_ext"))
    )
    top_skus_set = set(top_skus)
    top_cats_set = set(top_cats)

    joined = joined.with_columns([
        pl.col("sku").is_in(top_skus_set).cast(pl.Int8).alias("is_top_sku"),
        pl.col("category").is_in(top_cats_set).cast(pl.Int8).alias("is_top_category"),
    ])

    event_counts = (
        joined.group_by(["client_id", key , "session_start", "session_end_ext"])
        .agg([
            pl.count().alias("num_events_nearby"),
            pl.col("is_top_sku").max().alias("has_top_sku"),
            pl.col("is_top_category").max().alias("has_top_category")
        ])
    )

    collapsed_sessions_sq = collapsed_sessions_sq.join(
        event_counts,
        on=["client_id", key , "session_start", "session_end_ext"],
        how="left"
    ).with_columns([
        pl.col("num_events_nearby").fill_null(0).cast(pl.Int32),
        pl.col("has_top_sku").fill_null(0).cast(pl.Int8),
        pl.col("has_top_category").fill_null(0).cast(pl.Int8)
    ])
    return collapsed_sessions_sq

def scale(df,columns,transform,scale=True):
    X = df.select(columns).to_numpy()
    if transform == "log":
        X = np.log1p(X)
    elif transform == "yeo":
        pt = PowerTransformer(method="yeo-johnson")
        X = pt.fit_transform(X)
    elif transform == "minmax":
        scale = True
    if(scale):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)
    for i, col in enumerate(columns):
        df = df.with_columns(pl.Series(col, X[:, i]))
    return df


def load_top_skus_and_cats(result_dir):
    if (result_dir / "top_skus.npy").exists() and (result_dir / "top_cats.npy").exists():
        print("Loading top_skus and top_cats from existing files.")
        top_skus = np.load(result_dir / "top_skus.npy")
        top_cats = np.load(result_dir / "top_cats.npy")
        return top_skus, top_cats
    else:
        raise FileNotFoundError("top_skus.npy and top_cats.npy files not found in the result directory. Check if you have put a wrong result dir or if you haven't preprocessed the product events yet.")


