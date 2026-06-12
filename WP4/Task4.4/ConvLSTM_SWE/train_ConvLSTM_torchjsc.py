#%%
#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
import os
import time
import h5py
import gc
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ConvLSTM import SweDataset, AutoregressiveConvLSTM, denormalize_swe, BiasAwareLoss, calculate_metrics


def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    return rank, local_rank, world_size, device

# Configuration
SEQ_LEN = 60
FORECAST_STEPS = 7
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
EPOCHS = 50
INPUT_FEATURES = 7
HIDDEN_SIZE = 128

def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect()

def main():
    rank, local_rank, world_size, device = setup_distributed()
    dist.barrier()

    h5_path='input/concatenated_SeNorge_Drammen_2010_2020.h5'
    
    with h5py.File(h5_path, 'r') as f:

        mask = np.array(f['mask'][:])
        swe = np.array(f['swe'][:])
        precip = np.array(f['precip'][:])
        temp = np.array(f['temp'][:])
        static = np.array(f['static'][:])
        times = np.array(f['dates'][:], dtype='datetime64')
        lons = np.array(f['lon'][:])
        lats = np.array(f['lat'][:])

        stats = {
            'swe_min': f.attrs['swe_min'],
            'swe_max': f.attrs['swe_max'],
            'precip_min': f.attrs['precip_min'],
            'precip_max': f.attrs['precip_max'],
            'temp_mean': f.attrs['temp_mean'],
            'temp_std': f.attrs['temp_std'],
            'topo_min': f.attrs['topo_min'],
            'topo_max': f.attrs['topo_max'],
            'lat_mean': f.attrs['lat_mean'],
            'lat_std': f.attrs['lat_std']
        }

    train_ds = SweDataset(swe,precip,temp,static,list(range(0, 7*365)),times,SEQ_LEN,FORECAST_STEPS,stats=stats)
    val_ds = SweDataset(swe,precip,temp,static,list(range(8*365, 10*365)),times,SEQ_LEN,FORECAST_STEPS,stats=stats)

    if rank == 0:
        np.save('training_stats_autoreg_full.npy', stats)

    ## Create distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )

    # Model setup
    model = AutoregressiveConvLSTM(
        input_dim=INPUT_FEATURES,
        hidden_dim=HIDDEN_SIZE,
        kernel_size=3,
        num_layers=2,
        static_channels=2
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False)

    criterion = BiasAwareLoss(alpha=0.6).to(device)    
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=rank==0)
    
    #epoc=8
    #MODEL_PATH = 'saved_models_Drammen/epochcd2_{}.pth'.format(epoc)
    MODEL_PATH= ''

    if os.path.exists(MODEL_PATH):
        if rank == 0:
            print(f"Loading checkpoint from {MODEL_PATH}")
        
        dist.barrier()

        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # Handle state dict keys properly for DDP
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # If checkpoint was saved from DDP model
                new_state_dict[k[7:]] = v
            else:
                # If checkpoint was saved from plain model
                new_state_dict[k] = v

        # Load state dict
        model.module.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        if rank == 0:
            print(f"No checkpoint found at {MODEL_PATH}, starting from scratch")
        
        dist.barrier()

    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    if rank == 0:
        os.makedirs('saved_models_full2', exist_ok=True)
        log_file = open('training_log_autoreg_full.txt', 'w')
        log_file.write('Epoch,Train Loss,Val Loss,Train MAE,Train RMSE,Train R2,Val MAE,Val RMSE,Val R2,Time (s),LR\n')

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        total_train_loss = 0.0
        total_train_mae = 0.0
        total_train_rmse = 0.0
        total_train_r2 = 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            dynamic = batch[0].to(device, non_blocking=True)
            static_feat = batch[1].to(device, non_blocking=True)
            future_precip = batch[2].to(device, non_blocking=True)
            future_temp = batch[3].to(device, non_blocking=True)
            future_temporal = batch[4].to(device, non_blocking=True)
            target = batch[5].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(
                    x=dynamic,
                    static=static_feat,
                    future_precip=future_precip,
                    future_temp=future_temp,
                    future_temporal=future_temporal,
                    target=target,
                    epoch=epoch
                )
                loss = criterion(output, target) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            # Calculate metrics for this batch
            with torch.no_grad():
                mae, rmse, r2 = calculate_metrics(output, target, stats)
                total_train_mae += mae.item()
                total_train_rmse += rmse.item()
                total_train_r2 += r2.item()
                total_train_loss += loss.item() * GRAD_ACCUM_STEPS  # Undo the division

            # Gradient accumulation
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if rank == 0 and batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}')

        # Reduce training metrics across all processes
        train_metrics = torch.tensor([total_train_loss, total_train_mae, total_train_rmse, total_train_r2], device=device)
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
        train_metrics /= world_size
        
        avg_train_loss = train_metrics[0].item() / len(train_loader)
        avg_train_mae = train_metrics[1].item() / len(train_loader)
        avg_train_rmse = train_metrics[2].item() / len(train_loader)
        avg_train_r2 = train_metrics[3].item() / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_mae = 0.0
        total_val_rmse = 0.0
        total_val_r2 = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                dynamic = batch[0].to(device, non_blocking=True)
                static_feat = batch[1].to(device, non_blocking=True)
                future_precip = batch[2].to(device, non_blocking=True)
                future_temp = batch[3].to(device, non_blocking=True)
                future_temporal = batch[4].to(device, non_blocking=True)
                target = batch[5].to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    output = model(
                        x=dynamic,
                        static=static_feat,
                        future_precip=future_precip,
                        future_temp=future_temp,
                        future_temporal=future_temporal
                    )
                    loss = criterion(output, target)

                # Calculate validation metrics
                mae, rmse, r2 = calculate_metrics(output, target, stats)
                total_val_loss += loss.item()
                total_val_mae += mae.item()
                total_val_rmse += rmse.item()
                total_val_r2 += r2.item()

        # Reduce validation metrics across all processes
        val_metrics = torch.tensor([total_val_loss, total_val_mae, total_val_rmse, total_val_r2], device=device)
        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        val_metrics /= world_size
        
        avg_val_loss = val_metrics[0].item() / len(val_loader)
        avg_val_mae = val_metrics[1].item() / len(val_loader)
        avg_val_rmse = val_metrics[2].item() / len(val_loader)
        avg_val_r2 = val_metrics[3].item() / len(val_loader)

        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Logging and saving
        if rank == 0:
            epoch_time = time.time() - start_time
            
            log_file.write(
                f'{epoch + 1},'
                f'{avg_train_loss:.6f},'
                f'{avg_val_loss:.6f},'
                f'{avg_train_mae:.4f},'
                f'{avg_train_rmse:.4f},'
                f'{avg_train_r2:.4f},'
                f'{avg_val_mae:.4f},'
                f'{avg_val_rmse:.4f},'
                f'{avg_val_r2:.4f},'
                f'{epoch_time:.1f},'
                f'{current_lr:.2e}\n'
            )
            log_file.flush()

            print(f'\nEpoch {epoch+1} completed in {epoch_time:.1f}s')
            print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
            print(f'Train MAE: {avg_train_mae:.4f} | Train RMSE: {avg_train_rmse:.4f} | Train R2: {avg_train_r2:.4f}')
            print(f'Val MAE: {avg_val_mae:.4f} | Val RMSE: {avg_val_rmse:.4f} | Val R2: {avg_val_r2:.4f}')
            print(f'Learning Rate: {current_lr:.2e}\n')

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'stats': stats
            }, f'saved_models_full2/epochcd3_{epoch}.pth')

        cleanup_memory()
        dist.barrier()

    if rank == 0:
        log_file.close()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()


