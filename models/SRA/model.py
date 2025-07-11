import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1, dropout=0.2, use_gpu=False):
        super(GRUAutoencoder, self).__init__()
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.encoder_gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.decoder_gru = nn.GRU(
            latent_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        lengths = (x != -1.0).any(dim=2).sum(dim=1)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.encoder_gru(packed)
        z = self.fc_latent(h_n[-1])

        max_len = x.size(1)
        z_repeated = z.unsqueeze(1).repeat(1, max_len, 1)
        out, _ = self.decoder_gru(z_repeated)
        recon_x = self.fc_out(out)

        return recon_x, z

    def get_user_embedding(self, x):
        self.eval()
        with torch.no_grad():
            _, z = self.forward(x.to(self.device))
        return z


def reconstruction_loss(x, recon_x):
    padding_mask = (x != -1.0).any(dim=2).float().unsqueeze(2)
    loss = ((recon_x - x) ** 2) * padding_mask
    final_loss = loss.sum() / padding_mask.sum()
    return final_loss


def train_model(model, dataloader, optimizer, num_epochs=1, use_amp=True, dataset=None,
                save_embeddings_flag=False, embeddings_folder=None):
    model.to(model.device)
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp and torch.cuda.is_available() else None

    total_avg_loss_across_epochs = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0

        pbar = tqdm(dataloader,
                    desc=f"Epoch {epoch + 1}/{num_epochs}",
                    ncols=100,
                    unit="batch")

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(model.device)
            optimizer.zero_grad()

            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    recon_x, _ = model(batch)
                    loss = reconstruction_loss(batch, recon_x)

                scaler.scale(loss).backward()
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon_x, _ = model(batch)
                loss = reconstruction_loss(batch, recon_x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_total_loss += loss.item()
            current_avg_loss_in_epoch = epoch_total_loss / (batch_idx + 1)

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{current_avg_loss_in_epoch:.4f}'
            })

        pbar.close()

        final_avg_loss_for_epoch = epoch_total_loss / len(dataloader)
        total_avg_loss_across_epochs += final_avg_loss_for_epoch

        if save_embeddings_flag and dataset is not None:
            _ = get_embeddings_with_dataloader(
                model=model,
                events_dataset=dataset,
                batch_size=128,  # You can adjust this
                num_workers=8,  # Match your training dataloader
                embedding_path=embeddings_folder / f"embeddings_epoch_{epoch + 1}.npy"
            )

        print(f"Epoch {epoch + 1}/{num_epochs} completed - Final Avg Loss: {final_avg_loss_for_epoch:.4f}")
    print(f"Training completed - Total Avg Loss across all epochs: {total_avg_loss_across_epochs / num_epochs:.4f}")

    _ = get_embeddings_with_dataloader(
        model=model,
        events_dataset=dataset,
        batch_size=128,  # You can adjust this
        num_workers=8,  # Match your training dataloader
        embedding_path=embeddings_folder / "embeddings.npy"
    )
    return total_avg_loss_across_epochs / num_epochs

def get_embeddings(model, events_dataset, clients):
    all_embeddings = []
    model.eval()
    for idx in range(len(clients)):
        x = events_dataset[idx].unsqueeze(0)
        embedding = model.get_user_embedding(x.to(model.device))
        all_embeddings.append(embedding.squeeze(0).cpu())
    all_embeddings = np.vstack(all_embeddings).astype(np.float16)

    return all_embeddings


def custom_collate_fn(batch):
    # Assuming pad_sequence is imported from torch.nn.utils.rnn
    return pad_sequence(batch, batch_first=True, padding_value=-1.0)


def get_embeddings_with_dataloader(model, events_dataset, batch_size=64, num_workers=4, embedding_path=None):
    """
    Optimized version using DataLoader to parallelize data loading
    """

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    # Create DataLoader with custom collate function for variable length sequences
    dataloader = DataLoader(
        events_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,  # Improves GPU performance
        shuffle=False,  # Important: maintain original order
        collate_fn=custom_collate_fn  # Use the custom collate function
    )

    all_embeddings = []

    with torch.no_grad():
        # Progress bar
        pbar = tqdm(dataloader,
                    desc="Generating embeddings",
                    ncols=100,
                    unit="batch")

        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            batch_x = batch.to(model.device)

            z = model.get_user_embedding(batch_x)

            all_embeddings.append(z.cpu().numpy())

        pbar.close()

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings).astype(np.float16)

    if embedding_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path, all_embeddings)

    return all_embeddings