import json
import logging
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence, pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from models.utils.utils import run_training_pipeline
from synerise.training_pipeline.target_calculators import (
    ChurnTargetCalculator,
    PropensityTargetCalculator
)
from synerise.training_pipeline.target_data import TargetData
from synerise.training_pipeline.tasks import (
    PropensityTasks,
)

PADDING_VALUE = -1.0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_hidden_dims(input_dim: int, output_dim: int, num_hidden_layers: int, hidden_dim: int) -> list[int]:
   
    if num_hidden_layers % 2 == 0 or num_hidden_layers < 1:
        raise ValueError("Use an odd number of hidden layers greater than 0.")
    peak_index = num_hidden_layers // 2
    num_points_up = peak_index + 2
    dims_up = np.linspace(input_dim, hidden_dim, num=num_points_up, dtype=int).tolist()
    num_points_down = (num_hidden_layers - peak_index) + 1
    dims_down = np.linspace(hidden_dim, output_dim, num=num_points_down, dtype=int).tolist()
    final_dims = dims_up[:-1] + dims_down
    return final_dims

class DummyWorstCaseDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, max_len, input_dim, num_cat_targets, num_sku_targets):
        self.batch_size = batch_size
        self.events = torch.randn(batch_size, max_len, input_dim)
        self.churn = torch.randint(0, 1, (batch_size, 1)).float()
        self.cat = torch.randint(0, 1, (batch_size, num_cat_targets)).float()
        self.sku = torch.randint(0, 1, (batch_size, num_sku_targets)).float()
    
    def __len__(self):
        return self.batch_size
    
    def __getitem__(self, idx):
        return {
            'events': self.events[idx],
            'churn': self.churn[idx],
            'category': self.cat[idx],
            'sku': self.sku[idx]
        }
    def set_mode(self, mode):
        pass

class EventDataset(torch.utils.data.Dataset):
    def __init__(self, polars_grouped_df, encoder, client_ids, target_data, 
                 churn_calc, prop_cat_calc, prop_sku_calc, mode='train'):
                
        self.data = polars_grouped_df
        self.encoder = encoder
        self.client_ids = client_ids
        self.target_data = target_data
        self.churn_calc = churn_calc
        self.prop_cat_calc = prop_cat_calc
        self.prop_sku_calc = prop_sku_calc
        self.mode = mode
        
    def set_mode(self, mode):
        self.mode = mode
        
    def __len__(self):
        return self.data.height
        
    def __getitem__(self, idx):

        events = self.data[idx, "events"]
        rows = events.to_list()
        encoded = [self.encoder(r) for r in rows]

        client_id = self.client_ids[idx]
        target_df = self.target_data.train_df if self.mode == 'train' else self.target_data.validation_df
        
        churn_target = self.churn_calc.compute_target(client_id, target_df)
        cat_target = self.prop_cat_calc.compute_target(client_id, target_df)
        sku_target = self.prop_sku_calc.compute_target(client_id, target_df)
        
        return {
            'events': torch.stack(encoded),
            'churn': torch.tensor(churn_target, dtype=torch.float32),
            'category': torch.tensor(cat_target, dtype=torch.float32),
            'sku': torch.tensor(sku_target, dtype=torch.float32)
        }

def setup_target_calculators(data_dir, propensity_category_targets, propensity_sku_targets):
    target_data = TargetData.read_from_dir(data_dir / "target")
    
    churn_calculator = ChurnTargetCalculator()
    
    prop_cat_calculator = PropensityTargetCalculator(
        task=PropensityTasks.PROPENSITY_CATEGORY,
        propensity_targets=propensity_category_targets
    )
    
    prop_sku_calculator = PropensityTargetCalculator(
        task=PropensityTasks.PROPENSITY_SKU,
        propensity_targets=propensity_sku_targets
    )
    
    return target_data, churn_calculator, prop_cat_calculator, prop_sku_calculator


class AdvancedGruEncoder(nn.Module):
    def __init__(self, 
                 dims_sequence: List[int],
                 dropout: float = 0.2):
        super(AdvancedGruEncoder, self).__init__()
        
        self.dims_sequence = dims_sequence
        auto_skips = self._find_matching_dimensions()
        self.all_skips = list(set(auto_skips))
        # Crea layer GRU
        self.gru_layers = nn.ModuleList()
        self.skip_weights = nn.ParameterDict()
        
        for i in range(len(dims_sequence) - 1):
            input_size = dims_sequence[i]
            output_size = dims_sequence[i + 1]
            self.gru_layers.append(nn.GRU(input_size, output_size, batch_first=True))
        self.dropout_layer = nn.Dropout(dropout)
    
    def _find_matching_dimensions(self):
        skips = []
        for i in range(len(self.dims_sequence)):
            for j in range(i + 2, len(self.dims_sequence)):
                if self.dims_sequence[i] == self.dims_sequence[j]:
                    skips.append((i, j))
        return skips

    def forward(self, packed_x):
        padded_input, lengths = pad_packed_sequence(packed_x, batch_first=True)
        
        layer_outputs_padded = [padded_input]
        current_packed = packed_x
        last_hidden_state = None
        for i, gru in enumerate(self.gru_layers):
            current_packed, last_hidden_state = gru(current_packed)
            if i < len(self.gru_layers) - 1:
                packed_data = current_packed.data
                
                dropped_out_data = self.dropout_layer(packed_data)
                current_packed = PackedSequence(
                    data=dropped_out_data,
                    batch_sizes=current_packed.batch_sizes,
                    sorted_indices=current_packed.sorted_indices,
                    unsorted_indices=current_packed.unsorted_indices
                )
            skip_applied = False
            temp_padded_output = None
            
            for from_idx, to_idx in self.all_skips:
                if to_idx == i + 1:
                    if not skip_applied:
                        temp_padded_output, _ = pad_packed_sequence(current_packed, batch_first=True)
                        skip_applied = True
                    skip_input_padded = layer_outputs_padded[from_idx]
                    
                    temp_padded_output = temp_padded_output + skip_input_padded

            if skip_applied:
                current_packed = pack_padded_sequence(temp_padded_output, lengths.cpu(), batch_first=True, enforce_sorted=False)

            if any(s[0] == i + 1 for s in self.all_skips):
                final_padded_output, _ = pad_packed_sequence(current_packed, batch_first=True)
                layer_outputs_padded.append(final_padded_output)
            else:
                layer_outputs_padded.append(None) 
        return current_packed, last_hidden_state


class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, encoder_dims, hidden_dim_2, hidden_dim_3, latent_dim, churn_metric_calculator, category_metric_calculator, sku_metric_calculator, use_gpu=False, 
                 n_heads_attn=1, num_attn_layers=1 ,att_dropout=0.1, fc_dropout=0.4, dec_dropout=0.2, gru_dropout=0.2):
        
        super(GRUAutoencoder, self).__init__()
        self.num_attn_layers = num_attn_layers
        self.nonlinear = nn.GELU()
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.encoder_gru = AdvancedGruEncoder(dims_sequence=encoder_dims, dropout=gru_dropout)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim_1, 
            nhead=n_heads_attn,          
            dim_feedforward=2*hidden_dim_1, 
            dropout=att_dropout,
            batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_attn_layers
        )
        self.attention_norm = nn.LayerNorm(hidden_dim_1)
        
        self.pre_skip = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim_2),
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.GELU(),
            nn.LayerNorm(hidden_dim_3)
        )

        self.post_skip = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_dim_3, latent_dim) 
        )
        
        self.dropout_layer = nn.Dropout(dec_dropout) 

        self.fc_dec_1_churn = nn.Linear(latent_dim, 512)
        self.fc_dec_2_churn = nn.Linear(512, 256)
        self.fc_dec_3_churn = nn.Linear(256, 1)

        self.fc_dec_1_prop_category = nn.Linear(latent_dim, 512)
        self.fc_dec_2_prop_category = nn.Linear(512, 256)
        self.fc_dec_3_prop_category = nn.Linear(256, 100)

        self.fc_dec_1_prop_sku = nn.Linear(latent_dim, 512)
        self.fc_dec_2_prop_sku = nn.Linear(512, 256)
        self.fc_dec_3_prop_sku = nn.Linear(256, 100)

        self.churn_metric_calculator = churn_metric_calculator
        self.category_metric_calculator = category_metric_calculator
        self.sku_metric_calculator = sku_metric_calculator

    def forward(self, x):
        lengths = (x != PADDING_VALUE).any(dim=2).sum(dim=1) 
        max_len = x.size(1)
        padding_mask = torch.arange(max_len, device=x.device).expand(
            len(lengths), max_len
        ) >= lengths.unsqueeze(1)
        
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        output, h = self.encoder_gru(packed)
        
        if self.num_attn_layers > 0:
            unpacked_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=max_len
        )
            unpacked_output = unpacked_output.contiguous()
            attn_output = self.transformer_encoder(
                src=unpacked_output, 
                src_key_padding_mask=padding_mask
            )
            
            valid_mask = ~padding_mask.unsqueeze(-1)
            masked_attn = attn_output * valid_mask
            summed = masked_attn.sum(dim=1)  
            lengths_expanded = lengths.unsqueeze(1).float() 
            aggregated = summed / lengths_expanded 
            aggregated_norm = self.attention_norm(aggregated)
        else:
            z = (h[-1])
            aggregated_norm = self.attention_norm(z)
        
        z_transformed = self.pre_skip(aggregated_norm)
        z = z_transformed + aggregated_norm 
        z = self.post_skip(z)


        out_h1 = self.fc_dec_1_churn(z)
        out_h1 = self.nonlinear(out_h1)
        out_h1 = self.dropout_layer(out_h1) 
        out_h1 = self.fc_dec_2_churn(out_h1)
        out_h1 = self.nonlinear(out_h1)
        out_h1 = self.dropout_layer(out_h1) 
        out_h1 = self.fc_dec_3_churn(out_h1)

        out_h2 = self.fc_dec_1_prop_category(z)
        out_h2 = self.nonlinear(out_h2)
        out_h2 = self.dropout_layer(out_h2)
        out_h2 = self.fc_dec_2_prop_category(out_h2)
        out_h2 = self.nonlinear(out_h2)
        out_h2 = self.dropout_layer(out_h2)
        out_h2 = self.fc_dec_3_prop_category(out_h2)

        out_h3 = self.fc_dec_1_prop_sku(z)
        out_h3 = self.nonlinear(out_h3)
        out_h3 = self.dropout_layer(out_h3)
        out_h3 = self.fc_dec_2_prop_sku(out_h3)
        out_h3 = self.nonlinear(out_h3)
        out_h3 = self.dropout_layer(out_h3)
        out_h3 = self.fc_dec_3_prop_sku(out_h3)

        return out_h1, out_h2, out_h3, z

    def get_user_embedding(self, x):
        self.eval()
        with torch.no_grad():
            _, _, _, z = self.forward(x.to(self.device))
        return z

def calculate_loss(out_churn, out_category, out_sku, targets, w_churn, w_category, w_sku):
    churn_loss = F.binary_cross_entropy_with_logits(
        out_churn.squeeze(1), 
        targets['churn'].squeeze(1)
    )
    
    category_loss = F.binary_cross_entropy_with_logits(
        out_category, 
        targets['category']
    )

    sku_loss = F.binary_cross_entropy_with_logits(
        out_sku, 
        targets['sku']
    )
    
    total_loss = w_churn * churn_loss + w_category * category_loss + w_sku * sku_loss
    
    return total_loss, churn_loss, category_loss, sku_loss

def generate_embeddings(model, dataloader, max_batches=None, embedding_path=None):
    all_embeddings = []
    with torch.no_grad():
        pbar = tqdm(dataloader,
                    desc=f"Generating embeddings",
                    ncols=100,
                    unit="batch")

        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                print(f"  > Get embeddings stopped after {max_batches} batch to test.")
                break
            events = batch['events'].to(model.device)
            _, _, _, embeddings = model(events)
            all_embeddings.append(embeddings.cpu().numpy())
        pbar.close()
    all_embeddings = np.vstack(all_embeddings).astype(np.float16)

    if embedding_path is not None:
        # Change to path if needed
        embedding_path = Path(embedding_path)
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path, all_embeddings)

    return all_embeddings

def get_embeddings_with_dataloader(model, dataloader, embedding_path=None, max_batchs=None):
    if hasattr(model, 'module'):
        device = model.module.device
    elif hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()

    generate_embeddings(model, dataloader, max_batches=max_batchs, embedding_path=embedding_path)

def get_embeddings_with_event_dataset(model, events_dataset, batch_size=1024, num_workers=4, embedding_path=None, max_batchs=None):
    from torch.utils.data import DataLoader
    if hasattr(model, 'module'):
        device = model.module.device
    elif hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    dataloader = DataLoader(
        events_dataset, 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers
    )

    generate_embeddings(model, dataloader, max_batches=max_batchs, embedding_path=embedding_path)

def get_targets_with_dataloader(model, events_dataset, batch_size=1024,num_workers=4, embedding_path=None, max_batchs=None):
    from torch.utils.data import DataLoader
    if hasattr(model, 'module'):
        device = model.module.device
    elif hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    dataloader = DataLoader(
        events_dataset, 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers
    )
    
    all_embeddings = []
    pseudo_churn_preds = []
    pseudo_category_preds = []
    pseudo_sku_preds = []
    with torch.no_grad():
        pbar = tqdm(dataloader, 
                   desc=f"Generating embeddings", 
                   ncols=100,
                   unit="batch")
       
        for batch_idx, batch in enumerate(pbar):
            if max_batchs is not None and batch_idx >= max_batchs:
                print(f"  > Get targets stopped after {max_batchs} batch to test.")
                break
            events = batch['events'].to(model.device)
            out_churn, out_category, out_sku, embeddings = model(events)
            out_churn = torch.sigmoid(out_churn)
            out_category = torch.sigmoid(out_category)
            out_sku = torch.sigmoid(out_sku)

            all_embeddings.append(embeddings.cpu().numpy())
            pseudo_churn_preds.append(out_churn.cpu())
            pseudo_category_preds.append(out_category.cpu())
            pseudo_sku_preds.append(out_sku.cpu())
        pbar.close()
    predicted_targets = {
        'churn': torch.cat(pseudo_churn_preds, dim=0),
        'category': torch.cat(pseudo_category_preds, dim=0),
        'sku': torch.cat(pseudo_sku_preds, dim=0)
    }
    all_embeddings = np.vstack(all_embeddings).astype(np.float16)

    if embedding_path is not None:
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path, all_embeddings)
    
    return predicted_targets, all_embeddings

def validate_with_training_pipeline(model, dataloader, client_ids, embeddings_path, data_dir):
    if embeddings_path is None:
        raise ValueError("No embeddings path provided.")

    os.makedirs(embeddings_path, exist_ok=True)
    embeddings_path = Path(embeddings_path)
    np.save(embeddings_path / "client_ids.npy", client_ids)

    get_embeddings_with_dataloader(
        model=model,
        dataloader=dataloader,
        embedding_path=embeddings_path)

    avg_score, scores = run_training_pipeline(embeddings_dir=embeddings_path, data_dir=data_dir)
    avg_score = (scores['propensity_category'] + scores['propensity_sku']) / 2
    return avg_score, scores
    
def validate_model(model, dataset, dataloader, max_batches=None, w_churn=0.3, w_category=0.35, w_sku=0.35):
    dataset.set_mode('validation')
    
    total_loss = 0.0
    total_churn_loss = 0.0
    total_cat_loss = 0.0
    total_sku_loss = 0.0
    if hasattr(model, 'module'):
        churn_calc = model.module.churn_metric_calculator
        category_calc = model.module.category_metric_calculator
        sku_calc = model.module.sku_metric_calculator
        device = model.module.device
    else:
        churn_calc = model.churn_metric_calculator
        category_calc = model.category_metric_calculator
        sku_calc = model.sku_metric_calculator
        device = model.device
    model = model.to(device)
    model.eval()
    with torch.no_grad(), tqdm(dataloader,
                               desc="Validazione",
                               unit="batch",
                               total=len(dataloader),
                               leave=False) as progress:
        for batch_idx, batch in enumerate(progress):
            if max_batches is not None and batch_idx >= max_batches:
                print(f"  > Training stopped after {max_batches} batch to test.")
                break
            events = batch['events'].to(device)
            targets = {
                'churn': batch['churn'].to(device),
                'category': batch['category'].to(device),
                'sku': batch['sku'].to(device)
            }
            out_churn, out_category, out_sku, _ = model(events)
            loss, churn_l, cat_l, sku_l = calculate_loss(
                out_churn, out_category, out_sku, targets, w_churn, w_category, w_sku
            )
            
            total_loss += loss.item()
            total_churn_loss += churn_l.item()
            total_cat_loss += cat_l.item()
            total_sku_loss += sku_l.item()

            churn_calc.update(out_churn, targets['churn'].squeeze(1).long())
            category_calc.update(out_category, targets['category'].long())
            sku_calc.update(out_sku, targets['sku'].long())
            progress.set_postfix(loss=f"{loss.item():.4f}")
            
    churn_metrics = churn_calc.compute()
    category_metrics = category_calc.compute()
    sku_metrics = sku_calc.compute()

    for name, value in vars(churn_metrics).items():
        if not name.startswith('_'):
            print(f'val_churn_{name}: {value:.4f}')
    
    for name, value in vars(category_metrics).items():
        if not name.startswith('_'):
            print(f'val_category_{name}: {value:.4f}')
    
    for name, value in vars(sku_metrics).items():
        if not name.startswith('_'):
            print(f'val_sku_{name}: {value:.4f}')
    
    churn_weighted = churn_metrics.val_auroc
    category_weighted = category_metrics.val_auroc
    sku_weighted = sku_metrics.val_auroc
    
    overall_weighted = (
        w_churn * churn_weighted +
        w_category * category_weighted +
        w_sku * sku_weighted
    )
    
    print(f'val_overall_weighted: {overall_weighted:.4f}')
    dataset.set_mode('train')
    
    n_batches = len(dataloader)
    return {
        'total': total_loss / n_batches,
        'churn': total_churn_loss / n_batches,
        'category': total_cat_loss / n_batches,
        'sku': total_sku_loss / n_batches,
        'overall_weighted': overall_weighted 
    }

def train_model_with_training_pipeline(model, train_dataset, train_dataloader, val_dataloader, val_client_ids, dir_training_pipeline, optimizer, num_epochs=3, patience=3, delta=1e-4, max_batches=None, model_dir=None, val_step=1, save_best=False, w_churn=0.3, w_category=0.35, w_sku=0.35):
    if model_dir is not None:
        model_dir = Path(model_dir)

    if hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)
    if hasattr(model, 'module'):
        model.module.churn_metric_calculator.to(device)
        model.module.category_metric_calculator.to(device)
        model.module.sku_metric_calculator.to(device)
    else:
        model.churn_metric_calculator.to(device)
        model.category_metric_calculator.to(device)
        model.sku_metric_calculator.to(device)
    
    epochs_no_improve = 0
    best_state = None
    best_score = -np.inf
    best_epoch = num_epochs
    for epoch in range(num_epochs):
        model.train()
        train_dataset.set_mode('train')
        
        total_loss = 0.0
        total_churn_loss = 0.0
        total_cat_loss = 0.0
        total_sku_loss = 0.0
       
        print(f"\nEpoch {epoch + 1}/{num_epochs} - Training")
        progress = tqdm(train_dataloader,
                desc=f"Training epoch {epoch + 1}",
                leave=False)
        for batch_idx, batch in enumerate(progress):
            if max_batches is not None and batch_idx >= max_batches:
                print(f"  > Training stopped after {max_batches} batch to test.")
                break
            events = batch['events'].to(device)
            targets = {
                'churn': batch['churn'].to(device),
                'category': batch['category'].to(device),
                'sku': batch['sku'].to(device)
            }
            optimizer.zero_grad()
            out_churn, out_category, out_sku, _ = model(events)
            
            loss, churn_l, cat_l, sku_l = calculate_loss(
                out_churn, out_category, out_sku, targets, w_churn, w_category, w_sku
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if math.isnan(total_loss):
                print("Loss is NaN, training stopped.")
                return model, best_epoch, best_score
            total_churn_loss += churn_l.item()
            total_cat_loss += cat_l.item()
            total_sku_loss += sku_l.item()
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "churn": f"{churn_l.item():.3f}",
                "cat": f"{cat_l.item():.3f}",
                "sku": f"{sku_l.item():.3f}"
            })
        n_batches = len(train_dataloader)
        print(f"Training Results:")
        print(f"  Total Loss: {total_loss / n_batches:.4f}")
        print(f"  Churn Loss: {total_churn_loss / n_batches:.4f}")
        print(f"  Category Loss: {total_cat_loss / n_batches:.4f}")
        print(f"  SKU Loss: {total_sku_loss / n_batches:.4f}")
        
        if val_step is not None and (epoch + 1) % val_step == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs} - Validation")
            if model_dir is not None:
                os.makedirs(f"{model_dir}/{epoch + 1}", exist_ok=True)
            avg_score, scores = validate_with_training_pipeline(model, val_dataloader, val_client_ids, model_dir / f"{epoch + 1}", dir_training_pipeline)
            
            print(f"Validation Results:")
            print(f"  propensity_category: {scores['propensity_category']:.4f}")
            print(f"  propensity_sku: {scores['propensity_sku']:.4f}")
            print(f"  Category Loss: {avg_score:.4f}")
            current_score = avg_score
            if current_score > best_score + delta:
                best_score = current_score
                epochs_no_improve = 0
                if hasattr(model, 'module'):
                    best_state = {k: v.detach().cpu().clone()
                                for k, v in model.module.state_dict().items()}
                    best_epoch = epoch + 1
                    if save_best:
                        if model_dir is not None:
                            os.makedirs(f"{model_dir}/{epoch + 1}", exist_ok=True)
                            torch.save(model.module.state_dict(), model_dir / f"{epoch + 1}/gru_autoencoder.pth")
                else:
                    best_state = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1
                    if save_best:
                        if model_dir is not None:
                            os.makedirs(f"{model_dir}/{epoch + 1}", exist_ok=True)
                            torch.save(model.state_dict(), model_dir / f"{epoch + 1}/gru_autoencoder.pth")
                print(f"New best overall_weighted = {best_score:.4f}")
            else:
                epochs_no_improve += 1
                print(f"No improvement by {epochs_no_improve} epochs "
                    f"(best {best_score:.4f})")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs "
                    f"(patience={patience})")
                break
        if epoch % 10 == 0:
            print(f"Saving model at epoch {epoch + 1}")
            if hasattr(model, 'module'):
                if model_dir is not None:
                    os.makedirs(f"{model_dir}/{epoch + 1}", exist_ok=True)
                    torch.save(model.module.state_dict(), model_dir / f"{epoch + 1}/gru_autoencoder.pth")
            else:
                if model_dir is not None:
                    os.makedirs(f"{model_dir}/{epoch + 1}", exist_ok=True)
                    torch.save(model.state_dict(), model_dir / f"{epoch + 1}/gru_autoencoder.pth")
    if best_state is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)
        print("Loaded best model state")
    results = {
        'epoch': best_epoch,
        'score': best_score,
    }
    json_path = os.path.join(model_dir, 'scores.json')
    try:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error on saving json: {e}")
    return model, best_epoch, best_score

def train_model(model, dataset, dataloader, optimizer, num_epochs=3, patience=3, delta=1e-4, max_batches=None, model_dir=None, val_step=1, save_best=False, w_churn=0.3, w_category=0.35, w_sku=0.35, save_model_epoch_interval=10):
    if model_dir is not None:
        model_dir = Path(model_dir)

    if hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Usando {torch.cuda.device_count()} GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)
    if hasattr(model, 'module'):
        model.module.churn_metric_calculator.to(device)
        model.module.category_metric_calculator.to(device)
        model.module.sku_metric_calculator.to(device)
    else:
        model.churn_metric_calculator.to(device)
        model.category_metric_calculator.to(device)
        model.sku_metric_calculator.to(device)
    
    epochs_no_improve = 0
    best_state = None
    best_score = -np.inf 
    best_epoch = num_epochs
    for epoch in range(num_epochs):
        model.train()
        dataset.set_mode('train') 
        
        total_loss = 0.0
        total_churn_loss = 0.0
        total_cat_loss = 0.0
        total_sku_loss = 0.0
       
        print(f"\nEpoch {epoch + 1}/{num_epochs} - Training")
        progress = tqdm(dataloader,
                desc=f"Training epoch {epoch + 1}",
                leave=False)
        for batch_idx, batch in enumerate(progress):
            if max_batches is not None and batch_idx >= max_batches:
                print(f"  > Training stopped after {max_batches} batch to test.")
                break
            events = batch['events'].to(device)
            targets = {
                'churn': batch['churn'].to(device),
                'category': batch['category'].to(device),
                'sku': batch['sku'].to(device)
            }
            optimizer.zero_grad()
            out_churn, out_category, out_sku, _ = model(events)
            
            loss, churn_l, cat_l, sku_l = calculate_loss(
                out_churn, out_category, out_sku, targets, w_churn, w_category, w_sku
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if math.isnan(total_loss):
                print("Loss is NaN, training stopped.")
                return model # o break
            total_churn_loss += churn_l.item()
            total_cat_loss += cat_l.item()
            total_sku_loss += sku_l.item()
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "churn": f"{churn_l.item():.3f}",
                "cat": f"{cat_l.item():.3f}",
                "sku": f"{sku_l.item():.3f}"
            })
        n_batches = len(dataloader)
        print(f"Training Results:")
        print(f"  Total Loss: {total_loss / n_batches:.4f}")
        print(f"  Churn Loss: {total_churn_loss / n_batches:.4f}")
        print(f"  Category Loss: {total_cat_loss / n_batches:.4f}")
        print(f"  SKU Loss: {total_sku_loss / n_batches:.4f}")
        
        if val_step is not None and (epoch + 1) % val_step == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs} - Validation")
            val_losses = validate_model(model, dataset, dataloader, max_batches, w_churn, w_category, w_sku)

            print(f"Validation Results:")
            print(f"  Total Loss: {val_losses['total']:.4f}")
            print(f"  Churn Loss: {val_losses['churn']:.4f}")
            print(f"  Category Loss: {val_losses['category']:.4f}")
            print(f"  SKU Loss: {val_losses['sku']:.4f}")
            
            current_score = val_losses['overall_weighted']
            if current_score > best_score + delta:
                best_score = current_score
                epochs_no_improve = 0
                if hasattr(model, 'module'):
                    best_state = {k: v.detach().cpu().clone()
                                for k, v in model.module.state_dict().items()}
                    best_epoch = epoch + 1
                    if save_best:
                        if model_dir is not None:
                            os.makedirs(model_dir / f"{epoch + 1}", exist_ok=True)
                            torch.save(model.module.state_dict(), model_dir / f"{epoch + 1}/gru_autoencoder.pth")
                else:
                    best_state = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1
                    if save_best:
                        if model_dir is not None:
                            os.makedirs(model_dir / f"{epoch + 1}", exist_ok=True)
                            torch.save(model.state_dict(), model_dir / f"{epoch + 1}/gru_autoencoder.pth")
                print(f"New best overall_weighted = {best_score:.4f}")
            else:
                epochs_no_improve += 1
                print(f"No improvement by {epochs_no_improve} epoch "
                    f"(best {best_score:.4f})")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch+1} epoch"
                    f"(patience={patience})")
                break
        if epoch % save_model_epoch_interval == 0:
            print(f"Saving model at epoch {epoch + 1}")
            if hasattr(model, 'module'):
                if model_dir is not None:
                    os.makedirs(model_dir / f"{epoch + 1}", exist_ok=True)
                    torch.save(model.module.state_dict(), model_dir / f"{epoch + 1}/gru_autoencoder.pth")
            else:
                if model_dir is not None:
                    os.makedirs(model_dir / f"{epoch + 1}", exist_ok=True)
                    torch.save(model.state_dict(), model_dir / f"{epoch + 1}/gru_autoencoder.pth")
    if best_state is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)
        print("Loaded best model")
    return model, best_epoch

def custom_collate_fn(batch):
    events = [item['events'] for item in batch]
    events_padded = pad_sequence(events, batch_first=True, padding_value=PADDING_VALUE)
    
    return {
        'events': events_padded,
        'churn': torch.stack([item['churn'] for item in batch]),
        'category': torch.stack([item['category'] for item in batch]),
        'sku': torch.stack([item['sku'] for item in batch])
    }