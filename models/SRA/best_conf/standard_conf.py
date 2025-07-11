from argparse import ArgumentParser
from pathlib import Path
import polars as pl
import numpy as np
import os
from functools import partial
import torch
from torch.utils.data import DataLoader

from models.utils.utils_gru import row_to_tensor, EventDataset
from models.SRA.model import GRUAutoencoder, train_model, custom_collate_fn


# ALL
slope_columns=['45', '44', '43', '42', '41', '40', '39', '38', '37',
 '36', '35', '34', '33', '32', '31', '30', '29', '28', '27', '26', '25']
# Splitted
# slope_columns = ['41', '40', '39', '38', '37', '36', '35', '34', '33', '32', '31',
#                  '30', '29', '28', '27', '26', '25']



def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_all", type=str, required=True, help="Dataset with all clients from ubc_data")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path for creating the embeddings")
    parser.add_argument("--embeddings_folder", type=str, required=True, help="Folder where embeddings are saved")
    args = parser.parse_args()

    embeddings_folder = Path(args.embeddings_folder)
    grouped_ALL_dir = Path(args.dataset_all)
    grouped_dir = Path(args.dataset)

    os.makedirs(embeddings_folder, exist_ok=True)

    groupedAll = pl.read_parquet(grouped_ALL_dir)
    print("DEBUG: groupedAll shape:", groupedAll.shape)

    grouped = pl.read_parquet(grouped_dir)
    print("DEBUG: grouped shape:", grouped.shape)

    clients=np.array(grouped["client_id"], dtype=np.int64)
    np.save(embeddings_folder / "client_ids", clients)

    partial_row_to_tensor = partial(
        row_to_tensor,
        slope_columns=slope_columns,
    )
    dataset = EventDataset(grouped, partial_row_to_tensor)
    datasetALL = EventDataset(groupedAll, partial_row_to_tensor)
    HIDDEN_DIM = 256
    LATENT_DIM = 128
    BATCH_SIZE = 128
    MAX_NUM_EPOCHS = 1

    model = GRUAutoencoder(input_dim=59, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,num_layers=1, use_gpu=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader = DataLoader(datasetALL, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=8)
    train_model(model=model, dataloader=dataloader, optimizer=optimizer, num_epochs=MAX_NUM_EPOCHS, dataset=dataset, save_embeddings_flag=False, embeddings_folder=embeddings_folder)

if __name__ == "__main__":
    main()