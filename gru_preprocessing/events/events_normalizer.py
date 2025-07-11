from pathlib import Path

from gru_preprocessing.util_functions import *

##Products df columns

#yeo then min max in [-1,1]
product_count_columns=['count_add','count_buy','count_remove','count_user_add','count_user_buy','count_user_remove','tot_user_add',
 'tot_user_buy', 'tot_user_remove','cluster_add_count','cluster_buy_count','cluster_remove_count','cluster_add_user_count','cluster_buy_user_count',
'cluster_remove_user_count', 'count_cat_add', 'count_cat_buy', 'count_cat_remove', 'count_cat_user_add','count_cat_user_buy',
 'count_cat_user_remove']
#min max in [-1,1]
product_cr_columns=[ 'global_cat_cr','local_cat_cr','global_cr','local_cr','cr','price']
#log then min max in [-1,1]
product_entropies_columns=['sku_entropy','user_sku_entropy','cat_entropy','user_cat_entropy','cluster_entropy','user_cluster_entropy']

##Queries df columns
queries_entropies=[ 'query_entropy',
 'cluster_entropy',
 'user_query_entropy',
 'user_cluster_entropy']
queries_counts=[
 'num_events_nearby',
 'query_score',
 'query_user_score',
 'queries_made_by_user',
 'cluster_made_by_user',
 'cluster_score']

##Visist df columns
url_entropies_col=[ 'url_entropy','client_entropy']
url_counts=[ 'count','client_total_visits','client_url_visits','num_events_nearby']



def scale_events_df(events_data_frame, cr_columns, count_columns, entropies_columns, normalizer):
    events_data_frame=scale(events_data_frame,count_columns,transform=normalizer)
    events_data_frame=scale(events_data_frame,cr_columns,transform=None)
    events_data_frame=scale(events_data_frame,entropies_columns,transform="log")
    return events_data_frame

def scale_embeddings(map_clusters_path):
    map_clusters = pl.read_parquet(map_clusters_path)
    max_val= (
        map_clusters.explode("embedding")
        .select(pl.col("embedding").max()).item()
    )
    min_val = (
        map_clusters.explode("embedding")
        .select(pl.col("embedding").min()).item()
    )
    return map_clusters.with_columns([(((pl.col("embedding")-min_val)/(max_val-min_val))*2-1).alias("norm")])

def load_data(clusters_map_path=None):
    clusters_map_path = Path(clusters_map_path) if clusters_map_path else None
    return scale_embeddings(clusters_map_path) if clusters_map_path else None

def scale_data(event_type, events_data_frame, normalizer):
    if event_type == "product":
        count_columns=product_count_columns
        cr_columns=product_cr_columns
        entropies_columns= product_entropies_columns
    elif event_type == "query":
        count_columns=queries_counts
        cr_columns=["time_diff"]
        entropies_columns=queries_entropies
    else:
        count_columns=url_counts
        cr_columns=["time_diff"]
        entropies_columns=url_entropies_col
    return scale_events_df(events_data_frame, cr_columns, count_columns, entropies_columns, normalizer)

def normalize(events_data_frame, events_type, normalizer, clusters_map_path=None):
    scaled_embeddings= load_data(clusters_map_path)
    scaled_events_data_frame = scale_data(events_type, events_data_frame, normalizer)
    key= "sku" if events_type == "product" else "query_id" if events_type == "query" else None
    if scaled_embeddings is not None:
        scaled_events_data_frame= scaled_events_data_frame.join(
            scaled_embeddings[[key,"norm"]],
            on=key,
            how="left"
        )

    return scaled_events_data_frame