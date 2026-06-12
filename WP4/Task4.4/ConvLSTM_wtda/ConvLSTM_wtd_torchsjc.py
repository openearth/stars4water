#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import os
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import gc
import torch.optim as optim

def setup_distributed():

    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://'
    )
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    return rank, local_rank, world_size, device

from ConvLSTM_utils import (
    calculate_api,
    HydrologyConvLSTM,
    SpatioTemporalDataset,
    normalize_data,
    denormalize_wtda,
    prepare_dates,
    create_mask_from_data
)

# Initialize distributed training
rank, local_rank, world_size, device = setup_distributed()

if rank == 0:
    print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")

h5_path = 'input/concatenated_WTDa_Parflow_2012_2021_0_2000_stride20.h5'

# All ranks need to load the data (HDF5 is read-only and can be accessed by all processes)
with h5py.File(h5_path, 'r') as f:
    mask = np.array(f['mask'][:])
    wtda = f['wtda'][:]
    precip = f['precip'][:]
    vwc = f['vwc'][:]
    topo = f['static'][0,:,:]
    slopex = f['static'][1,:,:]
    slopey = f['static'][2,:,:]
    clyppt=f['static'][4,:,:]
    sndppt=f['static'][5,:,:]
    dates = f['dates'][:]

# Calculate API
api = calculate_api(precip, 0.95)

# Calculate normalization statistics
wtda_mean = np.nanmean(wtda)
wtda_std = np.nanstd(wtda)
wtda_max = np.nanmax(wtda)
wtda_min = np.nanmin(wtda)

api_mean = np.nanmean(api)
api_std = np.nanstd(api)
api_max = np.nanmax(api)
api_min = np.nanmin(api)

vwc_mean = np.nanmean(vwc)
vwc_std = np.nanstd(vwc)
vwc_max = np.nanmax(vwc)
vwc_min = np.nanmin(vwc)

topo_max = np.nanmax(topo)
topo_min = np.nanmin(topo)
slopex_max = np.nanmax(slopex)
slopex_min = np.nanmin(slopex)
slopey_max = np.nanmax(slopey)
slopey_min = np.nanmin(slopey)
sndppt_max=np.nanmax(sndppt)
sndppt_min=np.nanmin(sndppt)
clyppt_max=np.nanmax(clyppt)
clyppt_min=np.nanmin(clyppt)

# Normalize data
topo_norm = (topo - topo_min) / (topo_max - topo_min + 1e-8)
slopex_norm = (slopex - slopex_min) / (slopex_max - slopex_min + 1e-8)
slopey_norm = (slopey - slopey_min) / (slopey_max - slopey_min + 1e-8)
clyppt_norm = (clyppt - clyppt_min) / (clyppt_max - clyppt_min + 1e-8)
sndppt_norm = (sndppt - sndppt_min) / (sndppt_max - sndppt_min + 1e-8)

api_norm = (api - api_mean) / (api_std + 1e-8)
vwc_norm = (vwc - vwc_mean) / (vwc_std + 1e-8)
wtda_norm = (wtda - wtda_mean) / (wtda_std + 1e-8)


# Handle NaN values after normalization
topo_norm = np.nan_to_num(topo_norm, nan=-10)
slopex_norm = np.nan_to_num(slopex_norm, nan=-10)
slopey_norm = np.nan_to_num(slopey_norm, nan=-10)
clyppt_norm=np.nan_to_num(clyppt_norm,nan=-10)
sndppt_norm=np.nan_to_num(sndppt_norm,nan=-10)

api_norm = np.nan_to_num(api_norm, nan=-10)
vwc_norm = np.nan_to_num(vwc_norm, nan=-10)

# Prepare data arrays
static_data = np.stack([topo_norm, slopex_norm, slopey_norm,sndppt_norm,clyppt_norm], axis=0)
dynamic_data = np.stack([api_norm, vwc_norm], axis=1)
target_data = wtda_norm[:, np.newaxis, :, :]

n = wtda_norm.shape[0]

dates = prepare_dates(dates)

seq_len = 30
dataset = SpatioTemporalDataset(
    static_data=static_data,
    dynamic_data=dynamic_data,
    target_data=target_data,
    dates=dates,
    seq_len=seq_len
)

# Split dataset (all ranks do the same split)
train_end = int(0.7 * n)

train_dataset = torch.utils.data.Subset(dataset, range(0, train_end))
val_dataset = torch.utils.data.Subset(dataset, range(train_end, n-seq_len))

if rank == 0:
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Total timesteps: {n}")

# Create distributed samplers
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

# Adjust batch size per GPU
batch_size_per_gpu = 4
effective_batch_size = batch_size_per_gpu * world_size

if rank == 0:
    print(f"Effective batch size: {effective_batch_size} ({batch_size_per_gpu} per GPU)")

# Create data loaders with distributed samplers
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size_per_gpu,
    sampler=train_sampler,
    shuffle=False,  # Sampler handles shuffling
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size_per_gpu,
    sampler=val_sampler,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)


learning_rate = 1e-3

# Initialize model
model = HydrologyConvLSTM(
    static_channels=5,
    dynamic_channels=2,
    hidden_dim=64,
    num_layers=2,
    kernel_size=3
)

model = model.to(device)

# Wrap model with DistributedDataParallel if using multiple GPUs
if world_size > 1:
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

# Loss function and optimizer
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate * world_size)  # Scale learning rate

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5,
    verbose=(rank == 0)  # Only print from rank 0
)

OUTPATH_FOLDER = 'saved_models'
checkpoint_path = f'{OUTPATH_FOLDER}/best_model.pth'

# Only load checkpoint on rank 0 and broadcast to other ranks
if os.path.exists(checkpoint_path):
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
    
    # All ranks load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dict
    if world_size > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state dict
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    if rank == 0:
        print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.6f}")
else:
    start_epoch = 0
    best_val_loss = float('inf')
    if rank == 0:
        print(f"Checkpoint not found at {checkpoint_path}, starting from scratch...")

# Training history (only maintained on rank 0)
if rank == 0:
    train_losses = []
    val_losses = []
    train_metrics = {'rmse': [], 'mae': [], 'r2': []}
    val_metrics = {'rmse': [], 'mae': [], 'r2': []}
    
    os.makedirs(f'{OUTPATH_FOLDER}', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Log file
    log_file = open(f'logs/training_log_{int(time.time())}.txt', 'w')
    log_file.write(f"Training started at {time.ctime()}\n")
    log_file.write(f"Number of GPUs: {world_size}\n")
    log_file.write(f"Effective batch size: {effective_batch_size}\n\n")

num_epochs = 100
start_time = time.time()

# Synchronize all processes before starting training
dist.barrier()

for epoch in range(start_epoch, num_epochs):
    epoch_start_time = time.time()
    
    # Set epoch for distributed sampler (important for shuffling)
    train_sampler.set_epoch(epoch)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
    
    # ========== TRAINING ==========
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    # Initialize metrics accumulators
    train_mse_sum = 0.0
    train_mae_sum = 0.0
    train_target_sum = 0.0
    train_pred_sum = 0.0
    train_target_sq_sum = 0.0
    train_pred_sq_sum = 0.0
    train_cross_sum = 0.0
    train_samples = 0
    
    if rank == 0:
        print("\n[Training Phase]")
        print("-" * 40)
        pbar = tqdm(train_loader, desc='Training', unit='batch', leave=False)
    else:
        pbar = train_loader
    
    for batch_idx, (static_batch, dynamic_batch, target_batch, target_idx) in enumerate(pbar):
        static_batch = static_batch.to(device, non_blocking=True)
        dynamic_batch = dynamic_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)
        
        # Forward pass
        predictions = model(static_batch, dynamic_batch)

        batch_losses = []
        for i in range(predictions.size(0)):  # batch_size = 4
            pred_i = predictions[i]  # (1, 200, 200)
            target_i = target_batch[i]  # (1, 200, 200)
            
            valid_mask_i = ~torch.isnan(target_i)
            
            if valid_mask_i.any():
                loss_i = criterion(pred_i[valid_mask_i], target_i[valid_mask_i])
                batch_losses.append(loss_i)

        loss = torch.stack(batch_losses).mean() if batch_losses else torch.tensor(0.0, device=device)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update loss
        train_loss += loss.item()
        train_batches += 1
        
        # Calculate batch metrics (detach from computation graph)
        with torch.no_grad():

            valid_mask = ~torch.isnan(target_batch)

            pred_valid = predictions[valid_mask]
            target_valid = target_batch[valid_mask]

            diff = pred_valid.detach() - target_valid.detach()

            batch_count = pred_valid.numel()
            train_samples += batch_count

            train_mse_sum += torch.sum(diff ** 2).item()
            train_mae_sum += torch.sum(torch.abs(diff)).item()

            train_target_sum += torch.sum(target_valid).item()
            train_pred_sum += torch.sum(pred_valid).item()
            train_target_sq_sum += torch.sum(target_valid ** 2).item()
            train_pred_sq_sum += torch.sum(pred_valid ** 2).item()
            train_cross_sum += torch.sum(target_valid * pred_valid).item()

        # Update progress bar on rank 0
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'avg_loss': f'{(train_loss/train_batches):.4f}',
                'lr': f'{current_lr:.2e}'
            })
    
    # Synchronize metrics across all processes
    if world_size > 1:
        # Gather all metrics
        metrics_tensor = torch.tensor([train_loss, train_mse_sum, train_mae_sum, 
                                      train_target_sum, train_pred_sum, 
                                      train_target_sq_sum, train_pred_sq_sum,
                                      train_cross_sum, train_samples], device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        train_loss = metrics_tensor[0].item()
        train_mse_sum = metrics_tensor[1].item()
        train_mae_sum = metrics_tensor[2].item()
        train_target_sum = metrics_tensor[3].item()
        train_pred_sum = metrics_tensor[4].item()
        train_target_sq_sum = metrics_tensor[5].item()
        train_pred_sq_sum = metrics_tensor[6].item()
        train_cross_sum = metrics_tensor[7].item()
        train_samples = int(metrics_tensor[8].item())
        train_batches = train_batches * world_size  # Approximate
    
    avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
    
    # Calculate final training metrics
    train_rmse = np.sqrt(train_mse_sum / train_samples) if train_samples > 0 else 0
    train_mae = train_mae_sum / train_samples if train_samples > 0 else 0
    
    # Calculate R²
    if train_samples > 0:
        ss_tot = train_target_sq_sum - (train_target_sum ** 2) / train_samples
        ss_res = train_target_sq_sum + train_pred_sq_sum - 2 * train_cross_sum
        train_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        train_r2 = 0.0
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    # Initialize metrics accumulators for validation
    val_mse_sum = 0.0
    val_mae_sum = 0.0
    val_target_sum = 0.0
    val_pred_sum = 0.0
    val_target_sq_sum = 0.0
    val_pred_sq_sum = 0.0
    val_cross_sum = 0.0
    val_samples = 0
    
    if rank == 0:
        print(f"\n[Validation Phase]")
        print("-" * 40)
        val_pbar = tqdm(val_loader, desc='Validation', unit='batch', leave=False)
    else:
        val_pbar = val_loader
            
    with torch.no_grad():
        for batch_idx, (static_batch, dynamic_batch, target_batch, target_idx) in enumerate(val_pbar):
            static_batch = static_batch.to(device, non_blocking=True)
            dynamic_batch = dynamic_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            predictions = model(static_batch, dynamic_batch)

            batch_losses = []
            for i in range(predictions.size(0)):  # batch_size = 4
                pred_i = predictions[i]  # (1, 200, 200)
                target_i = target_batch[i]  # (1, 200, 200)
                
                valid_mask_i = ~torch.isnan(target_i)
                
                if valid_mask_i.any():
                    loss_i = criterion(pred_i[valid_mask_i], target_i[valid_mask_i])
                    batch_losses.append(loss_i)

            loss = torch.stack(batch_losses).mean() if batch_losses else torch.tensor(0.0, device=device)

                
            val_loss += loss.item()
            val_batches += 1
            
            # Calculate batch metrics
            valid_mask = ~torch.isnan(target_batch)

            pred_valid = predictions[valid_mask]
            target_valid = target_batch[valid_mask]

            diff = pred_valid.detach() - target_valid.detach()

            batch_count = pred_valid.numel()
            val_samples += batch_count

            val_mse_sum += torch.sum(diff ** 2).item()
            val_mae_sum += torch.sum(torch.abs(diff)).item()

            val_target_sum += torch.sum(target_valid).item()
            val_pred_sum += torch.sum(pred_valid).item()
            val_target_sq_sum += torch.sum(target_valid ** 2).item()
            val_pred_sq_sum += torch.sum(pred_valid ** 2).item()
            val_cross_sum += torch.sum(target_valid * pred_valid).item()
    
    # Synchronize validation metrics across all processes
    if world_size > 1:
        val_metrics_tensor = torch.tensor([val_loss, val_mse_sum, val_mae_sum, 
                                          val_target_sum, val_pred_sum, 
                                          val_target_sq_sum, val_pred_sq_sum,
                                          val_cross_sum, val_samples], device=device)
        dist.all_reduce(val_metrics_tensor, op=dist.ReduceOp.SUM)
        
        val_loss = val_metrics_tensor[0].item()
        val_mse_sum = val_metrics_tensor[1].item()
        val_mae_sum = val_metrics_tensor[2].item()
        val_target_sum = val_metrics_tensor[3].item()
        val_pred_sum = val_metrics_tensor[4].item()
        val_target_sq_sum = val_metrics_tensor[5].item()
        val_pred_sq_sum = val_metrics_tensor[6].item()
        val_cross_sum = val_metrics_tensor[7].item()
        val_samples = int(val_metrics_tensor[8].item())
        val_batches = val_batches * world_size  # Approximate
    
    avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
    
    # Calculate final validation metrics
    val_rmse = np.sqrt(val_mse_sum / val_samples) if val_samples > 0 else 0
    val_mae = val_mae_sum / val_samples if val_samples > 0 else 0
    
    # Calculate R²
    if val_samples > 0:
        ss_tot_val = val_target_sq_sum - (val_target_sum ** 2) / val_samples
        ss_res_val = val_target_sq_sum + val_pred_sq_sum - 2 * val_cross_sum
        val_r2 = 1 - (ss_res_val / ss_tot_val) if ss_tot_val > 0 else 0.0
    else:
        val_r2 = 0.0
    
    # Update scheduler (only on rank 0)
    if rank == 0:
        scheduler.step(avg_val_loss)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_metrics['rmse'].append(train_rmse)
        train_metrics['mae'].append(train_mae)
        train_metrics['r2'].append(train_r2)
        val_metrics['rmse'].append(val_rmse)
        val_metrics['mae'].append(val_mae)
        val_metrics['r2'].append(val_r2)
    
    # ========== PRINT EPOCH SUMMARY ==========
    epoch_time = time.time() - epoch_start_time
    total_time = time.time() - start_time
    current_lr = optimizer.param_groups[0]['lr']
    
    # Only rank 0 prints summary
    if rank == 0:
        print(f"\n{'='*60}")
        print("EPOCH SUMMARY:")
        print(f"{'='*60}")
        print(f"Time: {epoch_time:.1f}s (Total: {total_time/60:.1f}min) | LR: {current_lr:.2e}")
        print(f"{'-'*60}")
        print(f"{'Metric':<12} {'Training':<15} {'Validation':<15}")
        print(f"{'-'*60}")
        print(f"{'Loss':<12} {avg_train_loss:<15.6f} {avg_val_loss:<15.6f}")
        print(f"{'RMSE':<12} {train_rmse:<15.6f} {val_rmse:<15.6f}")
        print(f"{'MAE':<12} {train_mae:<15.6f} {val_mae:<15.6f}")
        print(f"{'R²':<12} {train_r2:<15.6f} {val_r2:<15.6f}")
        
        # Loss trend
        if len(train_losses) > 1:
            train_loss_diff = train_losses[-1] - train_losses[-2]
            val_loss_diff = val_losses[-1] - val_losses[-2]
            print(f"{'-'*60}")
            print(f"Loss Δ:        {train_loss_diff:+.6f}          {val_loss_diff:+.6f}")
        
        print(f"{'='*60}")
        
        # ========== LOG TO FILE ==========
        log_file.write(f"Epoch {epoch+1}/{num_epochs}\n")
        log_file.write(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}\n")
        log_file.write(f"  Train RMSE: {train_rmse:.6f} | Val RMSE: {val_rmse:.6f}\n")
        log_file.write(f"  Train MAE:  {train_mae:.6f} | Val MAE:  {val_mae:.6f}\n")
        log_file.write(f"  Train R²:   {train_r2:.6f} | Val R²:   {val_r2:.6f}\n")
        log_file.write(f"  LR: {current_lr:.6f}\n")
        log_file.write(f"  Time: {epoch_time:.1f}s\n\n")
        log_file.flush()
        
        # ========== SAVE CHECKPOINTS ==========
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, f'{OUTPATH_FOLDER}/epoch_{epoch+1}.pth')
            print(f"\n✓ Saved checkpoint: {OUTPATH_FOLDER}/epoch_{epoch+1}.pth")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': best_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, f'{OUTPATH_FOLDER}/best_model.pth')
            print(f"New best model! Val Loss: {best_val_loss:.6f} (Saved)")
    
    # Early stopping check (only rank 0)
    if epoch > 60 and rank == 0:
        if len(val_losses) >= 10:
            if all(val_losses[-i] > val_losses[-i-1] for i in range(1, 4)):
                print(f"\n Early stopping triggered! Validation loss increased for 20 consecutive epochs.")
                log_file.write("Early stopping triggered!\n")
                break
    
    # Learning rate too low check (only rank 0)
    if current_lr < 1e-6 and rank == 0:
        print(f"\n Learning rate too low ({current_lr:.2e}), stopping training.")
        log_file.write("Learning rate too low, stopping training.\n")
        break
    
    # Synchronize before next epoch
    dist.barrier()

# Cleanup
if rank == 0:
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    if train_losses:
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    log_file.write(f"\nTraining completed at {time.ctime()}\n")
    log_file.write(f"Total time: {(time.time() - start_time)/60:.1f} minutes\n")
    log_file.write(f"Best validation loss: {best_val_loss:.6f}\n")
    log_file.close()

# Clean up distributed process group
dist.destroy_process_group()