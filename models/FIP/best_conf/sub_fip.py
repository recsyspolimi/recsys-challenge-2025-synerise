from argparse import ArgumentParser
from pathlib import Path
import os
import polars as pl
import numpy as np
from functools import partial
import torch
from torch.utils.data import DataLoader

from models.utils.utils_gru import row_to_tensor
from models.FIP.model import GRUAutoencoder, EventDataset, EventDatasetXY, custom_collate_fn_xy, train_model, get_embeddings_with_dataloader


slope_columns_ALL=['45', '44', '43', '42', '41', '40', '39', '38', '37',
 '36', '35', '34', '33', '32', '31', '30', '29']

slope_columns=['41','40','39','38','37','36','35','34','33','32','31',
 '30','29','28','27','26','25']

slope_columns_Y =['45','44','43','42']


def main():
    parser = ArgumentParser()
    parser.add_argument("--embeddings_folder", type=str, required=True, help="Folder where embeddings are saved")
    parser.add_argument("--grouped_only_rel_submission_dir", type=str, help="Grouped parquet with only relevant clients for submission step")
    parser.add_argument("--input_dir", type=str, help="Grouped parquet with all clients and five months of data")
    parser.add_argument("--target_dir", type=str, help="Grouped parquet with all clients and only the sixth months of data")
    args = parser.parse_args()

    embeddings_folder = Path(args.embeddings_folder)
    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    grouped_only_rel_submission_dir = Path(args.grouped_only_rel_submission_dir)

    os.makedirs(embeddings_folder, exist_ok=True)

    grouped_split_5_X = pl.read_parquet(input_dir)
    grouped_onlyLastMonth_Y = pl.read_parquet(target_dir)


    grouped_onlyrel_6month = pl.read_parquet(grouped_only_rel_submission_dir)

    client_ids_X = grouped_split_5_X["client_id"].to_numpy()
    client_ids_6month = grouped_onlyrel_6month["client_id"].to_numpy()

    print(f"Total clients in X: {len(client_ids_X)}")
    print(f"Unique clients in X: {len(set(client_ids_X))}")
    print(f"Clients in Y: {grouped_onlyLastMonth_Y.height}")

    row_to_tensor_ALL = partial(row_to_tensor,
                                slope_columns=slope_columns_ALL,
                                )

    row_to_tensor_X = partial(row_to_tensor,
                              slope_columns=slope_columns,
                              )

    row_to_tensor_Y = partial(row_to_tensor,
                              slope_columns=slope_columns_Y,
                              )

    dataset_train = EventDatasetXY(grouped_split_5_X, grouped_onlyLastMonth_Y, row_to_tensor_X, row_to_tensor_Y)
    dataset_test = EventDataset(grouped_onlyrel_6month, row_to_tensor_ALL)

    HIDDEN_DIM = 256
    LATENT_DIM = 128
    BATCH_SIZE = 256
    NUM_EPOCHS = 3

    model = GRUAutoencoder(input_dim=59, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=1, use_gpu=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn_xy,
        num_workers=8
    )

    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        val_dataset=None,
        val_clients=None,
        test_dataset=None,
        test_clients=None,
        embeddings_dir=None,
        data_dir=None,
        patience=10,
    )

    print("Generating final embeddings...")
    final_embeddings = get_embeddings_with_dataloader(
        model=model,
        events_dataset=dataset_test,
        batch_size=128,
        num_workers=8,
        embedding_path=None
    )

    np.save(embeddings_folder / "embeddings.npy", final_embeddings)
    np.save(embeddings_folder / "client_ids.npy", client_ids_6month)

if __name__ == "__main__":
    main()