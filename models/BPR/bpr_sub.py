import argparse
import logging
import os
from pathlib import Path

import implicit
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sps
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_interaction_matrices_with_mapping(buy_df, add_df, remove_df, buy_weight, add_weight, remove_weight,
                                          client_mapping, item_mapping, use_log=True):
    all_interactions = []
    if buy_df is not None and len(buy_df) > 0:
        all_interactions.append(buy_df.with_columns(pl.lit(buy_weight, dtype=pl.Float32).alias("weight")))
    if add_df is not None and len(add_df) > 0:
        all_interactions.append(add_df.with_columns(pl.lit(add_weight, dtype=pl.Float32).alias("weight")))
    if remove_df is not None and len(remove_df) > 0:
        all_interactions.append(remove_df.with_columns(pl.lit(remove_weight, dtype=pl.Float32).alias("weight")))
    if not all_interactions:
        return sps.csr_matrix((len(client_mapping), len(item_mapping)), dtype=np.float32)
    interactions_df = pl.concat(all_interactions)
    interaction_data = interactions_df.select([
        pl.col("client_id").replace(client_mapping).alias("client_idx"),
        pl.col("sku").replace(item_mapping).alias("item_idx"),
        pl.col("weight")
    ]).drop_nulls()
    grouped = interaction_data.group_by(['client_idx', 'item_idx']).agg(pl.sum('weight'))
    if use_log:
        grouped = grouped.with_columns(
            pl.when(pl.col('weight') > 0).then(pl.col('weight').log1p())
            .when(pl.col('weight') < 0).then(- (pl.col('weight').abs().log1p()))
            .otherwise(pl.col('weight')).alias('weight')
        )
    return sps.csr_matrix(
        (grouped['weight'].to_numpy(), (grouped['client_idx'].to_numpy(), grouped['item_idx'].to_numpy())),
        shape=(len(client_mapping), len(item_mapping)), dtype=np.float32
    )

def load_data(product_buy_path, add_to_cart_path, remove_from_cart_path, product_properties_path, category=False):
    logger.info("Loading datasets...")
    product_buy_df = pl.read_parquet(product_buy_path)
    add_to_cart_df = pl.read_parquet(add_to_cart_path)
    remove_from_cart_df = pl.read_parquet(remove_from_cart_path)
    if category:
        product_properties_df = pl.read_parquet(product_properties_path).select(['sku', 'category'])
        product_buy_df = product_buy_df.join(product_properties_df, on='sku', how='left').drop('sku').rename({'category': 'sku'})
        add_to_cart_df = add_to_cart_df.join(product_properties_df, on='sku', how='left').drop('sku').rename({'category': 'sku'})
        remove_from_cart_df = remove_from_cart_df.join(product_properties_df, on='sku', how='left').drop('sku').rename({'category': 'sku'})
    logger.info(f"Dataset loaded: {len(product_buy_df)} buy, {len(add_to_cart_df)} add, {len(remove_from_cart_df)} re")
    return product_buy_df, add_to_cart_df, remove_from_cart_df

def generate_submission_scores_known_only(model, client_idx_map, item_idx_map, propensity_sku_path, relevant_clients_path, propensity_category_path, category=False):
    logger.info("Start generating scores for known relevant clients...")
    relevant_clients = np.load(relevant_clients_path)
    propensity_file_path = propensity_category_path if category else propensity_sku_path
    propensity_skus = np.load(propensity_file_path)
    known_relevant_clients_mask = np.array([cid in client_idx_map for cid in relevant_clients])
    final_clients = relevant_clients[known_relevant_clients_mask]
    if len(final_clients) == 0:
        logger.warning("No relevant clients found in the training set. Returning empty arrays.")
        return np.array([], dtype=np.float16), np.array([], dtype=np.int64)
    logger.info(f"{len(final_clients)} know clients {len(relevant_clients)} relevant clients.")
    if hasattr(model.user_factors, 'to_numpy'):
        user_factors = model.user_factors.to_numpy()
        item_factors = model.item_factors.to_numpy()
        xp = np
    else:
        user_factors = model.user_factors
        item_factors = model.item_factors
        xp = np
    known_client_model_indices = [client_idx_map[cid] for cid in final_clients]
    item_mask = np.array([sku in item_idx_map for sku in propensity_skus])
    known_item_idxs = np.where(item_mask)[0]
    unknown_item_idxs = np.where(~item_mask)[0]
    known_item_model_indices = [item_idx_map[sku] for sku in propensity_skus[known_item_idxs]]
    logger.info(f"Know sku: {len(known_item_idxs)}/{len(propensity_skus)}")
    scores_array = xp.zeros((len(final_clients), len(propensity_skus)), dtype=xp.float32)
    if len(known_item_idxs) > 0:
        known_user_embeddings = user_factors[known_client_model_indices]
        known_item_embeddings = item_factors[known_item_model_indices]
        known_scores_matrix = known_user_embeddings @ known_item_embeddings.T
        scores_array[:, known_item_idxs] = known_scores_matrix
        if len(unknown_item_idxs) > 0:
            mean_scores_per_user = xp.mean(known_scores_matrix, axis=1, keepdims=True)
            scores_array[:, unknown_item_idxs] = mean_scores_per_user
    else:
        logger.warning("No propensity items found in the training set. Scores for known clients will be 0.")
    scores_array = scores_array.astype(np.float16)
    client_ids_array = np.array(final_clients, dtype=np.int64)
    logger.info(f"Final shape score: {scores_array.shape}")
    return scores_array, client_ids_array

def generate_submission_scores(model, client_idx_map, item_idx_map, relevant_clients_path, propensity_category_path, propensity_sku_path, category=False):
    logger.info("Start generating scores for all relevant clients...")

    relevant_clients = np.load(relevant_clients_path)
    propensity_file_path = propensity_category_path if category else propensity_sku_path
    propensity_skus = np.load(propensity_file_path)
    logger.info(f"Load {len(relevant_clients)} clients and {len(propensity_skus)} item target.")

    if hasattr(model.user_factors, 'to_numpy'):
        user_factors = model.user_factors.to_numpy()
        item_factors = model.item_factors.to_numpy()
        xp = np
    else:
        user_factors = model.user_factors
        item_factors = model.item_factors
        xp = np

    scores_array = xp.zeros((len(relevant_clients), len(propensity_skus)), dtype=xp.float32)

    client_mask = np.array([cid in client_idx_map for cid in relevant_clients])
    known_client_indices = np.where(client_mask)[0]

    if len(known_client_indices) > 0:
        logger.info(f"{len(known_client_indices)} know clients on {len(relevant_clients)}.")
        
        known_client_model_indices = [client_idx_map[cid] for cid in relevant_clients[known_client_indices]]

        item_mask = np.array([sku in item_idx_map for sku in propensity_skus])
        known_item_indices = np.where(item_mask)[0]
        unknown_item_indices = np.where(~item_mask)[0]
        
        if len(known_item_indices) > 0:
            logger.info(f"Know item: {len(known_item_indices)}/{len(propensity_skus)}")
            
            known_item_model_indices = [item_idx_map[sku] for sku in propensity_skus[known_item_indices]]

            known_user_embeddings = user_factors[known_client_model_indices]
            known_item_embeddings = item_factors[known_item_model_indices]
            known_scores_matrix = known_user_embeddings @ known_item_embeddings.T

            scores_array[xp.ix_(known_client_indices, known_item_indices)] = known_scores_matrix

            if len(unknown_item_indices) > 0:
                mean_scores_per_user = xp.mean(known_scores_matrix, axis=1, keepdims=True)
                scores_array[xp.ix_(known_client_indices, unknown_item_indices)] = mean_scores_per_user
        else:
            logger.warning("No propensity items found in the training set. Scores for known clients will be 0.")
    else:
        logger.warning("No relevant clients found in the training set. Returning empty arrays.")

    logger.info("Converting scores to float16 for memory efficiency.")
    scores_array = scores_array.astype(np.float16)
    
    client_ids_array = np.array(relevant_clients, dtype=np.int64)

    logger.info(f"Final scores shape: {scores_array.shape}")
    logger.info(f"Final clients shape: {client_ids_array.shape}")
    
    return scores_array, client_ids_array
def get_best_params_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"No file: {csv_path}")
        raise
    
    df = df.dropna(subset=['value'])
    if df.empty:
        raise ValueError("No valid trials found in the CSV file.")

    best_trial_series = df.sort_values("value", ascending=False).iloc[0]
    
    params = {}
    for col, value in best_trial_series.items():
        if col.startswith("params_"):
            param_name = col.replace("params_", "")
            params[param_name] = value
            
    if 'factors' in params:
        params['factors'] = int(params['factors'])
    if 'iterations' in params:
        params['iterations'] = int(params['iterations'])
    print(f"Best params: {params}")
    return params

def calculate_auc_for_subset(model, test_buy_df, propensity_skus_path, client_map, item_map):
    propensity_skus = np.load(propensity_skus_path)
    
    propensity_skus_in_model = [sku for sku in propensity_skus if sku in item_map]
    if not propensity_skus_in_model:
        logger.warning("No propensity items found in the model. Returning AUC = 0.0")
        return 0.0
        
    propensity_item_indices = np.array([item_map[sku] for sku in propensity_skus_in_model])

    test_user_items_map = test_buy_df.group_by('client_id').agg(
        pl.col('sku').implode()
    ).to_pandas().set_index('client_id')['sku'].to_dict()

    test_user_items_map = {k: set(v) for k, v in test_user_items_map.items()}

    test_clients_in_model = [cid for cid in test_user_items_map.keys() if cid in client_map]

    if not test_clients_in_model:
        logger.warning("No test clients found in the model. Returning AUC = 0.0")
        return 0.0
    if hasattr(model.user_factors, 'to_numpy'):
        user_factors = model.user_factors.to_numpy()
        item_factors = model.item_factors.to_numpy()
    else:
        user_factors = model.user_factors
        item_factors = model.item_factors
    
    propensity_item_factors = item_factors[propensity_item_indices]
    
    user_aucs = []
    from tqdm.auto import tqdm
    
    for client_id in tqdm(test_clients_in_model, desc="Calculating AUC", leave=False, ncols=80):
        user_model_idx = client_map[client_id]
        user_embedding = user_factors[user_model_idx]

        scores = user_embedding @ propensity_item_factors.T
        
        positive_skus = test_user_items_map.get(client_id, set())
        y_true = np.array([sku in positive_skus for sku in propensity_skus_in_model])

        if len(np.unique(y_true)) < 2:
            continue
            
        try:
            if 'cupy' in str(type(scores)):
                scores = scores.get()
            
            user_auc = roc_auc_score(y_true, scores)
            user_aucs.append(user_auc)
        except ValueError:
            continue

    if not user_aucs:
        return 0.0
    
    return np.mean(user_aucs)

def main():
    parser = argparse.ArgumentParser(description='Trining pipeline for BPR model with Optuna hyperparameters.')
    parser.add_argument('--results_csv', type=str, help='Path to the CSV file containing Optuna results.')
    parser.add_argument('--category', action='store_true', help='Use categories instead of SKUs.')
    parser.add_argument('--data_path', type=str, help="Path to the data directory.")
    parser.add_argument('--embeddings_dir', type=str, help="Path to the directory where embeddings will be saved.")
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    data_path = Path(args.data_path)    
    product_buy_path = os.path.join(data_path, "product_buy.parquet")
    add_to_cart_path = os.path.join(data_path, "add_to_cart.parquet")
    remove_from_cart_path = os.path.join(data_path, "remove_from_cart.parquet")
    product_properties_path = os.path.join(data_path, "product_properties.parquet")
    propensity_sku_path = os.path.join(data_path, "target", "propensity_sku.npy")
    propensity_category_path = os.path.join(data_path, "target", "propensity_category.npy")
    relevant_clients_path = os.path.join(data_path, "input", "relevant_clients.npy")


    logger.info(f"Loading the best parameters from:Loading the best parameters from: {args.results_csv}")
    best_params = get_best_params_from_csv(Path(args.results_csv))
    logger.info(f"Best params: {best_params}")

    product_buy_df, add_to_cart_df, remove_from_cart_df = load_data(category=args.category, 
                                                                    product_buy_path=product_buy_path,
                                                                    add_to_cart_path=add_to_cart_path,
                                                                    remove_from_cart_path=remove_from_cart_path,
                                                                    product_properties_path=product_properties_path)

    
    train_clients = pl.concat([product_buy_df['client_id'], add_to_cart_df['client_id'], remove_from_cart_df['client_id']]).unique()
    train_items = pl.concat([product_buy_df['sku'], add_to_cart_df['sku'], remove_from_cart_df['sku']]).unique()
    
    global_client_map = {client_id: idx for idx, client_id in enumerate(train_clients)}
    global_item_map = {item_id: idx for idx, item_id in enumerate(train_items)}
    
    train_interactions = create_interaction_matrices_with_mapping(
        product_buy_df, add_to_cart_df, remove_from_cart_df,
        best_params['buy_weight'],best_params['add_weight'],best_params['remove_weight'],
        global_client_map, global_item_map, use_log=True
    )

    logger.info("Start training BPR model...")
    final_model = implicit.bpr.BayesianPersonalizedRanking(
        factors=best_params['factors'],
        regularization=best_params['regularization'],
        learning_rate=best_params['learning_rate'],
        iterations=best_params['iterations'],
        use_gpu=implicit.gpu.HAS_CUDA,
        random_state=42
    )
    final_model.fit(train_interactions, show_progress=True)
    logger.info("BPR model training completed.")

    scores, client_ids = generate_submission_scores(
        final_model, 
        global_client_map, 
        global_item_map, 
        category=args.category,
        relevant_clients_path=relevant_clients_path,
        propensity_category_path=propensity_category_path,
        propensity_sku_path=propensity_sku_path
    )

    os.makedirs(embeddings_dir, exist_ok=True)

    score_file = embeddings_dir / "embeddings.npy"
    client_id_file = embeddings_dir / "client_ids.npy"
    np.save(score_file, scores)
    np.save(client_id_file, client_ids)

if __name__ == "__main__":
    main()