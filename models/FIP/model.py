import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.utils.utils import run_training_pipeline
from synerise.data_utils.data_dir import DataDir

logger = logging.getLogger(__name__)

class EventDataset(torch.utils.data.Dataset):
    def __init__(self, polars_grouped_df, encoder):
        self.data = polars_grouped_df
        self.encoder = encoder

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx):
        events = self.data[idx, "events"]
        rows = events.to_list()
        encoded = [self.encoder(r) for r in rows]
        return torch.stack(encoded)

slope_columns=['41','40','39','38','37','36','35','34','33','32','31',
 '30','29','28','27','26','25']

def create_default_event(grouped_y, slope_columns):
    for i in range(min(10, grouped_y.height)):  
        events_series = grouped_y[i, "events"]
        if events_series is not None and len(events_series) > 0:
            first_event = events_series[0]
            if isinstance(first_event, dict):
                sample_schema = set(first_event.keys())
                break
    else:
        raise ValueError("No valid events found in grouped_y")
    
    default_event = {
        'event_type': 'page_visit',
        'hour_norm': 0.0,
        'weekday_norm': 0.0,
        'week_of_month_norm': 0.0,
        'weeks_since_test_start': 0.0,
        'client_url_visits': 0,
        'client_total_visits': 0,
        'url_entropy': 0.0,
        'count': 0,
        'has_top_sku': 0,
        'has_top_category': 0,
        'num_events_nearby': 0,
        'time_diff': 0.0,
    }
    
    for col in slope_columns:
        default_event[col] = 0.0
    
    missing_fields = sample_schema - set(default_event.keys())
    for field in missing_fields:
        default_event[field] = 0.0
    
    return default_event

def align_x_y_data_optimized(grouped_x, grouped_y):
    
    clients_x_array = grouped_x["client_id"].to_numpy()  
    clients_y_array = grouped_y["client_id"].to_numpy()
    
    print(f"Total clients in X: {len(clients_x_array)}")
    print(f"Clients in Y: {len(clients_y_array)}")
    
    y_client_to_idx = {client_id: idx for idx, client_id in enumerate(clients_y_array)}
    
    aligned_y_indices = []
    missing_clients = []
    
    with tqdm(total=len(clients_x_array), desc="Building alligned Y", ncols=100) as pbar:
        for i, client_id in enumerate(clients_x_array):
            if client_id in y_client_to_idx:
                aligned_y_indices.append(y_client_to_idx[client_id])
            else:
                aligned_y_indices.append(-1)  
                missing_clients.append(i)
            pbar.update(1)
    
    print(f"Clients in X but not in Y: {len(missing_clients)}")
    
    default_event = create_default_event(grouped_y, slope_columns)
    
    return aligned_y_indices, missing_clients, default_event

class EventDatasetXY(torch.utils.data.Dataset):
    def __init__(self, polars_grouped_df_x, polars_grouped_df_y, encoder_X, encoder_Y):
        self.data_x = polars_grouped_df_x
        self.data_y = polars_grouped_df_y
        self.encoder_X = encoder_X  
        self.encoder_Y = encoder_Y  
        
        self.client_ids_X = self.data_x["client_id"].to_numpy()
        
        self.aligned_y_indices, self.missing_clients, self.default_event = align_x_y_data_optimized(
            polars_grouped_df_x, polars_grouped_df_y
        )
        
    def __len__(self):
        return self.data_x.height
    
    def __getitem__(self, idx):
        events_x = self.data_x[idx, "events"]
        rows_x = events_x.to_list()
        encoded_x = [self.encoder_X(r) for r in rows_x]
        
        y_idx = self.aligned_y_indices[idx]
        is_default_flag = False
        
        if y_idx == -1:  
            is_default_flag = True
            rows_y = [self.default_event]
        else:
            events_y = self.data_y[y_idx, "events"]
            rows_y = events_y.to_list()
            if not rows_y:  
                is_default_flag = True
                rows_y = [self.default_event]
        
        encoded_y = [self.encoder_Y(r) for r in rows_y]
        
        return torch.stack(encoded_x), torch.stack(encoded_y), torch.tensor(is_default_flag, dtype=torch.bool) 

class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim,num_layers=1, use_gpu=False):
        super(GRUAutoencoder, self).__init__()
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder_gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, maxlen_y):
        lengths = (x != -1.0).any(dim=2).sum(dim=1) 

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.encoder_gru(packed)
        z = self.fc_latent(h_n[-1]) 
        
        z_repeated = z.unsqueeze(1).repeat(1, maxlen_y, 1)
        out, _ = self.decoder_gru(z_repeated)
        recon_y = self.fc_out(out)

        return recon_y, z
    
    def get_user_embedding(self, x):
        self.eval()
        with torch.no_grad():
            lengths = (x != -1.0).any(dim=2).sum(dim=1)
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h_n = self.encoder_gru(packed)
            z = self.fc_latent(h_n[-1])
        return z

def reconstruction_loss(x, recon_x):
    padding_mask = (x != -1.0).any(dim=2).float().unsqueeze(2)
    loss = ((recon_x - x) ** 2) * padding_mask
    final_loss = loss.sum() / padding_mask.sum()
    return final_loss

def weighted_reconstruction_loss(y, recon_y, is_default=None):
    padding_mask = (y != -1.0).any(dim=2).float().unsqueeze(2)
    loss = ((recon_y - y) ** 2) * padding_mask
    
    if is_default is not None:
        weight = torch.where(is_default.unsqueeze(1).unsqueeze(2), 0.8, 1.0)
        loss = loss * weight
    
    final_loss = loss.sum() / padding_mask.sum()
    return final_loss

def validate_model(model, val_dataset, val_clients, embeddings_dir, data_dir, epoch, is_final_run=False):
    print(f"Running validation at epoch {epoch}...")
    
    temp_embeddings_dir = Path(embeddings_dir) / f"epoch_{epoch}_validation"
    temp_embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    _ = get_embeddings_with_dataloader(
        model=model,
        events_dataset=val_dataset,
        batch_size=128,  
        num_workers=8,   
        embedding_path=temp_embeddings_dir / "embeddings.npy"
    )
    
    np.save(temp_embeddings_dir / "client_ids.npy", val_clients)
    print(f"Validation embeddings saved to {temp_embeddings_dir}")
    data_dir = DataDir(data_dir)

    if is_final_run:
        task_names = ["churn", "propensity_sku", "propensity_category"]
    else:
        task_names = ["propensity_sku","propensity_category"]

    avg_val_score, val_scores = run_training_pipeline(
        embeddings_dir=temp_embeddings_dir,
        data_dir=data_dir,
        task_names=task_names,
    )
    
    print(f"Validation scores at epoch {epoch}:")
    for task, score in val_scores.items():
        print(f"  {task}: {score:.4f}")
    print(f"  Average: {avg_val_score:.4f}")
    
    return avg_val_score, val_scores

def generate_submission(model, test_dataset, test_clients, embeddings_dir, epoch):

    print("\nGenerating submission...")
    
    temp_embeddings_dir = embeddings_dir / f"epoch_{epoch}_submission"
    temp_embeddings_dir.mkdir(parents=True, exist_ok=True)

    np.save(temp_embeddings_dir / "client_ids.npy", test_clients)
    
    _ = get_embeddings_with_dataloader(
        model=model,
        events_dataset=test_dataset,
        batch_size=128,  
        num_workers=8,   
        embedding_path=temp_embeddings_dir / "embeddings.npy"
    )
    
    print(f"Submission embeddings saved to {temp_embeddings_dir}")


def train_model(model, dataloader, optimizer, num_epochs=1, use_amp=True, val_dataset=None, val_clients=None, test_dataset=None, test_clients=None, embeddings_dir=None, data_dir=None, patience=2, model_folder=None):
    model.to(model.device)
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp and torch.cuda.is_available() else None
    
    best_val_score = float('-inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        mse_base = 0.0
        
        pbar = tqdm(dataloader, 
                   desc=f"Epoch {epoch + 1}/{num_epochs}", 
                   ncols=100,
                   unit="batch")

        for batch_idx, (batch_x, batch_y, is_default_flags) in enumerate(pbar):  

            batch_x = batch_x.to(model.device)
            batch_y = batch_y.to(model.device)
            is_default_flags = is_default_flags.to(model.device)

            maxlen_y = batch_y.size(1)
            
            optimizer.zero_grad()

            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    recon_y, _ = model(batch_x, maxlen_y)  
                    loss = weighted_reconstruction_loss(batch_y, recon_y, is_default=is_default_flags)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon_y, _ = model(batch_x, maxlen_y)
                loss = weighted_reconstruction_loss(batch_y, recon_y, is_default=is_default_flags)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_total_loss += loss.item()
            current_avg_loss_in_epoch = epoch_total_loss / (batch_idx + 1)

            mse_base += torch.sum((batch_y - torch.mean(batch_y)) ** 2)

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{current_avg_loss_in_epoch:.4f}'
            })
            
        pbar.close()
        
        final_avg_loss_for_epoch = epoch_total_loss / len(dataloader)

        if test_dataset is not None and test_clients is not None:
            generate_submission(model=model,
                                test_dataset=test_dataset,
                                test_clients=test_clients,
                                embeddings_dir=embeddings_dir,
                                epoch=epoch + 1)
        
        if (epoch + 1) % 3 == 0 and val_dataset is not None and val_clients is not None:
            
            avg_val_score, val_scores = validate_model(
                model=model,
                val_dataset=val_dataset,
                val_clients=val_clients,
                embeddings_dir=embeddings_dir,
                data_dir=data_dir,
                epoch=epoch + 1
            )
            
            avg_propensity_score = (val_scores['propensity_sku'] + val_scores['propensity_category']) / 2
            if avg_propensity_score > best_val_score:

                epochs_since_last_improvement = 0
                best_val_score = avg_propensity_score
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_score': avg_val_score,
                    'val_score_propensity': val_scores['propensity_sku'],
                    'val_score_category': val_scores['propensity_category'],
                }, model_folder / "best_model.pt")

                print(f"New best model saved with validation score: {avg_propensity_score:.4f}")

            else:
                epochs_since_last_improvement += 1
                if epochs_since_last_improvement >= patience:
                    print(f"Early stopping triggered after {epochs_since_last_improvement} epochs without improvement.")
                    break
                else: 
                    print(f"No improvement in validation score: {avg_val_score:.4f} (best: {best_val_score:.4f})")
            
        print(f"Epoch {epoch + 1}/{num_epochs} completed - Final Avg Loss: {final_avg_loss_for_epoch:.4f}")

    return final_avg_loss_for_epoch


def custom_collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=-1.0)

def custom_collate_fn_xy(batch):
    x_batch = [item[0] for item in batch]
    y_batch = [item[1] for item in batch]
    is_default_batch = [item[2] for item in batch] 
    
    x_padded = pad_sequence(x_batch, batch_first=True, padding_value=-1.0)
    y_padded = pad_sequence(y_batch, batch_first=True, padding_value=-1.0)
    
    is_default_tensor = torch.stack(is_default_batch) 
    
    return x_padded, y_padded, is_default_tensor 

def get_embeddings_with_dataloader(model, events_dataset, batch_size=64, num_workers=4, embedding_path=None):
    
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    
    dataloader = DataLoader(
        events_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True,  
        shuffle=False,  
        collate_fn=custom_collate_fn  
    )
    
    all_embeddings = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, 
                   desc="Generating embeddings", 
                   ncols=100,
                   unit="batch")
       
        for batch_idx, batch in enumerate(pbar):
            batch_x = batch.to(model.device)
            z = model.get_user_embedding(batch_x)
            all_embeddings.append(z.cpu().numpy())
            
        pbar.close()
    
    all_embeddings = np.vstack(all_embeddings).astype(np.float16)

    if embedding_path is not None:
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path, all_embeddings)
    
    return all_embeddings