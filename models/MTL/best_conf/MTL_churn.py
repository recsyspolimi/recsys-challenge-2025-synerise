import polars as pl
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from models.MTL.model import *
from models.utils.utils_gru import row_to_tensor
from functools import partial
from synerise.training_pipeline.metric_calculators import (
    ChurnMetricCalculator, PropensityMetricCalculator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
MAX_ENCODER_LAYERS = 3

submission_slope_columns = [
    '45', '44', '43', '42',
    '41', '40', '39', '38',
    '37', 
    '36', '35', '34','33', 
    '32', '31', '30','29', 
]

train_slope_columns = [
    '41', '40', '39', '38',
    '37', 
    '36', '35', '34','33', 
    '32', '31', '30','29', 
    '28', '27', '26','25',
]

def create_embeddings_to_submit(data_dir,
                                dataset_train,
                                output_dir,
                                dataset_embeddings,
                                ):

    dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)

    data_dir = Path(data_dir)
    sku_popularity_data = np.load(data_dir / "target/popularity_propensity_sku.npy")
    category_popularity_data = np.load(data_dir / "target/popularity_propensity_category.npy")
    churn_metric_calculator = ChurnMetricCalculator()
    category_metric_calculator = PropensityMetricCalculator(output_dim=100, popularity_data=category_popularity_data)
    sku_metric_calculator = PropensityMetricCalculator(output_dim=100,popularity_data=sku_popularity_data)
    model = GRUAutoencoder(
        input_dim=59, 
        hidden_dim_1=128, 
        encoder_dims=[59, 128], 
        hidden_dim_2=512, 
        hidden_dim_3=128, 
        latent_dim=32,
        churn_metric_calculator=churn_metric_calculator,
        category_metric_calculator=category_metric_calculator,
        sku_metric_calculator=sku_metric_calculator,
        use_gpu=True, 
        att_dropout=0.1, 
        fc_dropout=0.3, 
        dec_dropout=0.2,
        gru_dropout=0.0, 
        n_heads_attn=2,  
        num_attn_layers=1  
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    train_model(
        model, 
        dataset_train,  
        dataloader,
        optimizer=optimizer, 
        num_epochs=6,
        patience=5, 
        val_step=None, 
        model_dir=None,
        save_best=True,
        w_churn=1.0,
        w_category=0.0,
        w_sku=0.0
    )

    _ = get_embeddings_with_event_dataset(model, dataset_embeddings, batch_size=32, num_workers=2, embedding_path=output_dir / "embeddings.npy")
    client_ids = dataset_embeddings.client_ids.to_numpy()
    np.save(output_dir / "client_ids.npy", client_ids)

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory containing the data")
    parser.add_argument("--embeddings_folder", type=str, help="Directory to save the model and embeddings")
    parser.add_argument("--train_dir_dataset", type=str, help="Directory for train (5 months) dataset")
    parser.add_argument("--submission_dir_dataset", type=str, help="Directory for all dataset")
    
    args = parser.parse_args()

    train_dir_dataset = Path(args.train_dir_dataset)
    submission_dir_dataset = Path(args.submission_dir_dataset)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.embeddings_folder)

    train_grouped = pl.read_parquet(train_dir_dataset)
    train_client_ids = train_grouped["client_id"].unique().to_numpy()

    propensity_category_targets = np.load(data_dir / "target/propensity_category.npy")
    propensity_sku_targets = np.load(data_dir / "target/propensity_sku.npy")
    target_data, churn_calc, prop_cat_calc, prop_sku_calc = setup_target_calculators(
        data_dir,
        propensity_category_targets=propensity_category_targets,
        propensity_sku_targets=propensity_sku_targets
    )
    row_to_tensor_train = partial(
        row_to_tensor,
        slope_columns=train_slope_columns,
    )
    dataset_train = EventDataset(
        polars_grouped_df=train_grouped,
        encoder= row_to_tensor_train,
        client_ids=train_client_ids,
        target_data=target_data,
        churn_calc=churn_calc,
        prop_cat_calc=prop_cat_calc,
        prop_sku_calc=prop_sku_calc,
        mode='train'
    )

    submission_grouped = pl.read_parquet(submission_dir_dataset)
    submission_client_ids = submission_grouped["client_id"].unique().to_numpy()
    
    row_to_tensor_submission = partial(
        row_to_tensor,
        slope_columns=submission_slope_columns,
    )
    dataset_embeddings = EventDataset(
        polars_grouped_df=submission_grouped,
        encoder= row_to_tensor_submission,
        client_ids=submission_client_ids,
        target_data=target_data,
        churn_calc=churn_calc,
        prop_cat_calc=prop_cat_calc,
        prop_sku_calc=prop_sku_calc,
        mode='train'
    )
    
    create_embeddings_to_submit(data_dir, dataset_train, output_dir, dataset_embeddings)


if __name__ == "__main__":
    main()
