import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

time_fields = ["hour_norm", "weekday_norm", "week_of_month_norm"]

product_df_scalars=[
    'count_user_buy', 'count_user_add', 'count_user_remove',
    'user_sku_entropy', 'user_cluster_entropy', 'user_cat_entropy',
    'cr', 'local_cr', 'global_cr',
    'tot_user_buy', 'tot_user_add', 'tot_user_remove',
    'top_sku',
    'cluster_buy_user_count', 'cluster_add_user_count', 'cluster_remove_user_count',
    'cluster_buy_count', 'cluster_add_count', 'cluster_remove_count',
    'cluster_entropy', 'sku_entropy',
    'count_buy', 'count_add', 'count_remove', 'top_category',
    'count_cat_user_buy', 'count_cat_user_add', 'count_cat_user_remove',
    'global_cat_cr', 'local_cat_cr',
    'count_cat_buy', 'count_cat_add', 'count_cat_remove',
    'cat_entropy',
    'price'
]

visit_df_columns = ['client_url_visits',
                    'client_total_visits',
                    'url_entropy',
                    'count',
                    'has_top_sku',
                    'has_top_category',
                    'num_events_nearby',
                    'time_diff',
                   ]

search_df_scalars = [
    'query_user_score', 'queries_made_by_user', 'user_query_entropy', 'user_cluster_entropy',
    'query_score', 'cluster_score', 'cluster_made_by_user',
    'query_entropy', 'cluster_entropy',
    'has_top_sku', 'has_top_category',
    'num_events_nearby',
    'time_diff'
]  # 'norm'

def time_weight(t,alpha=0.06):
    return math.exp(-alpha * t)

def encode_event_type(event_type_str):
    if event_type_str== "product_buy":
        enc=[1,0,0,0,0]
    elif event_type_str== "add_to_cart":
        enc=[0,1,0,0,0]
    elif event_type_str== "remove_from_cart":
        enc=[0,0,1,0,0]
    elif event_type_str== "page_visit":
        enc=[0,0,0,1,0]
    else:
        enc=[0,0,0,0,1]
    return np.array(enc)

def encode_product_event(row):
    return torch.cat([torch.tensor([row[field] for field in product_df_scalars], dtype=torch.float32),
                      torch.tensor(row["norm"], dtype=torch.float32)])


def encode_search(row):
    return torch.cat([torch.tensor([row[field] for field in search_df_scalars], dtype=torch.float32),
                      torch.tensor(row["norm"], dtype=torch.float32)])

def encode_visit(row, slope_columns):
    visit_df_columns_sloped = visit_df_columns + slope_columns
    return torch.tensor([row[field] for field in visit_df_columns_sloped], dtype=torch.float32)


def row_to_tensor(row, slope_columns, alpha=0.06):
    event_enc = torch.tensor(encode_event_type(row['event_type']), dtype=torch.float32)  # 5-dim

    # Handle None values in time fields
    time_values = [row[field] if row[field] is not None else 0.0 for field in time_fields]
    time_enc = torch.tensor(time_values, dtype=torch.float32)  # 3-dim

    if row["event_type"] == "page_visit":
        emb = encode_visit(row, slope_columns)
    elif row["event_type"] == "search_query":
        emb = encode_search(row)
    else:
        emb = encode_product_event(row)

    full = torch.cat([event_enc, time_enc, emb])
    padded = F.pad(full, pad=(0, 59 - full.shape[0]), value=-1.0)

    # Handle None value in weeks_since_test_start
    weeks_since = row["weeks_since_test_start"] if row["weeks_since_test_start"] is not None else 0.0
    return time_weight(weeks_since, alpha=alpha) * padded

class EventDataset(torch.utils.data.Dataset):
    def __init__(self, polars_grouped_df, encoder):
        self.data = polars_grouped_df # Must be grouped already for client_id
        self.encoder = encoder

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx):
        events = self.data[idx, "events"]
        rows = events.to_list()
        encoded = [self.encoder(r) for r in rows]
        return torch.stack(encoded)
