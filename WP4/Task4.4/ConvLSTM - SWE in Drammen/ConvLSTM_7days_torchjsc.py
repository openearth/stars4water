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


# Distributed training setup
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
LAT_DIM = 233
LON_DIM = 389
HIDDEN_SIZE = 128

def save_to_hdf5(swe, precip, temp, static, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('swe', data=swe, compression='gzip', chunks=True)
        f.create_dataset('precip', data=precip, compression='gzip', chunks=True)
        f.create_dataset('temp', data=temp, compression='gzip', chunks=True)
        f.create_dataset('static', data=static, compression='gzip', chunks=True)
        
        swe_log = np.log1p(swe)
        swe_log = np.nan_to_num(swe_log, nan=0.0)
        
        f.attrs['swe_min'] = np.min(swe_log)
        f.attrs['swe_max'] = np.max(swe_log)
        f.attrs['precip_min'] = np.nanmin(precip)
        f.attrs['precip_max'] = np.nanmax(precip)
        f.attrs['temp_mean'] = np.nanmean(temp)
        f.attrs['temp_std'] = np.nanstd(temp)
        f.attrs['topo_min'] = np.nanmin(static[0])
        f.attrs['topo_max'] = np.nanmax(static[0])
        f.attrs['lat_mean'] = np.nanmean(static[1])
        f.attrs['lat_std'] = np.nanstd(static[1])

def load_data():

    source_path='/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/region_drammen'

    swe_ds = xr.open_dataset(f'{source_path}/swe_2010_2020.nc')
    input_ds = xr.open_dataset(f'{source_path}/input_2010_2020.nc')
    topo_ds = xr.open_dataset(f'{source_path}/topo.nc')

    swe = swe_ds['snow_water_equivalent'].values.transpose(0, 2, 1)
    precip = input_ds['rr'].values.transpose(0, 2, 1)
    temp = input_ds['tg'].values.transpose(0, 2, 1)

    lat_grid = np.meshgrid(swe_ds['lat'].values, swe_ds['lon'].values)[0]
    static = np.stack([topo_ds['dem_mean'].values.T, lat_grid])

    os.makedirs("input/drammen", exist_ok=True)
    
    save_to_hdf5(swe, precip, temp, static, 'input/drammen/data.h5')
    return 'input/drammen/data.h5'

class SweDatasetHDF5(Dataset):
    def __init__(self, h5_path, time_indices, stats=None, mode='train'):
        self.h5_path = h5_path
        self.time_indices = time_indices
        self.mode = mode
        self.length = len(time_indices) - SEQ_LEN - FORECAST_STEPS
        
        if stats is None:
            with h5py.File(h5_path, 'r') as f:
                self.stats = {
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
        else:
            self.stats = stats

    def __len__(self):
        return self.length

    def _normalize_dynamic(self, data, feature):
        data = np.nan_to_num(data, nan=0.0)
        if feature == 'swe':
            return (data - self.stats['swe_min']) / (self.stats['swe_max'] - self.stats['swe_min'])
        elif feature == 'precip':
            return (data - self.stats['precip_min']) / (self.stats['precip_max'] - self.stats['precip_min'])
        elif feature == 'temp':
            return (data - self.stats['temp_mean']) / self.stats['temp_std']
        else:
            raise ValueError(f"Unknown feature: {feature}")

    def _normalize_static(self, static_data):
        static_data = np.nan_to_num(static_data, nan=0.0)
        topo = (static_data[0] - self.stats['topo_min']) / (self.stats['topo_max'] - self.stats['topo_min'])
        lat = (static_data[1] - self.stats['lat_mean']) / self.stats['lat_std']
        return np.stack([topo, lat])

    def __getitem__(self, idx):
        actual_t = self.time_indices[idx + SEQ_LEN]
        start_idx = actual_t - SEQ_LEN
        end_idx = actual_t + FORECAST_STEPS
        
        with h5py.File(self.h5_path, 'r') as f:
            chunk = {
                'swe': f['swe'][start_idx:end_idx],
                'precip': f['precip'][start_idx:end_idx],
                'temp': f['temp'][start_idx:end_idx],
                'static': f['static'][:]
            }
        
        # Process dynamic features
        swe_past = self._normalize_dynamic(np.log1p(chunk['swe'][:SEQ_LEN]), 'swe')
        precip_seq = self._normalize_dynamic(chunk['precip'], 'precip')
        temp_seq = self._normalize_dynamic(chunk['temp'], 'temp')
        
        # Process static features
        static = self._normalize_static(chunk['static'])
        
        # Temporal encoding
        all_days = np.arange(start_idx, end_idx)
        all_months = ((all_days % 365) // 30) + 1
        all_doy = (all_days % 365) + 1
        
        # Cyclical encoding
        month_sin = np.sin(2 * np.pi * all_months / 12)
        month_cos = np.cos(2 * np.pi * all_months / 12)
        day_sin = np.sin(2 * np.pi * all_doy / 365)
        day_cos = np.cos(2 * np.pi * all_doy / 365)

        # Split into historical and future
        hist_month_sin = month_sin[:SEQ_LEN]
        hist_month_cos = month_cos[:SEQ_LEN]
        hist_day_sin = day_sin[:SEQ_LEN]
        hist_day_cos = day_cos[:SEQ_LEN]
        
        fut_month_sin = month_sin[SEQ_LEN:]
        fut_month_cos = month_cos[SEQ_LEN:]
        fut_day_sin = day_sin[SEQ_LEN:]
        fut_day_cos = day_cos[SEQ_LEN:]

        # Expand temporal features spatially
        height, width = swe_past.shape[1], swe_past.shape[2]
        hist_month_sin = np.tile(hist_month_sin[:, np.newaxis, np.newaxis], (1, height, width))
        hist_month_cos = np.tile(hist_month_cos[:, np.newaxis, np.newaxis], (1, height, width))
        hist_day_sin = np.tile(hist_day_sin[:, np.newaxis, np.newaxis], (1, height, width))
        hist_day_cos = np.tile(hist_day_cos[:, np.newaxis, np.newaxis], (1, height, width))

        # Stack all historical dynamic features
        dynamic = np.stack([
            swe_past,
            precip_seq[:SEQ_LEN],
            temp_seq[:SEQ_LEN],
            hist_month_sin,
            hist_month_cos,
            hist_day_sin,
            hist_day_cos
        ], axis=1)

        # Future weather and temporal features
        future_precip = precip_seq[SEQ_LEN:]
        future_temp = temp_seq[SEQ_LEN:]
        future_temporal = np.stack([
            fut_month_sin,
            fut_month_cos,
            fut_day_sin,
            fut_day_cos
        ], axis=1)

        # Target: SWE for next 7 days
        target = self._normalize_dynamic(
            np.log1p(chunk['swe'][SEQ_LEN:SEQ_LEN+FORECAST_STEPS]),
            'swe'
        )

        return (
            torch.tensor(dynamic, dtype=torch.float32),
            torch.tensor(static, dtype=torch.float32),
            torch.tensor(future_precip, dtype=torch.float32),
            torch.tensor(future_temp, dtype=torch.float32),
            torch.tensor(future_temporal, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 
            4 * hidden_dim, 
            kernel_size=kernel_size, 
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class AutoregressiveConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, static_channels):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.input_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(hidden_dim, hidden_dim, kernel_size) for _ in range(num_layers)
        ])
        
        self.static_processor = nn.Sequential(
            nn.Conv2d(static_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        
        self.weather_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.temp_proj = nn.Linear(4, hidden_dim)
        self.step_proj = nn.Conv2d(1 + hidden_dim + hidden_dim + hidden_dim, hidden_dim, kernel_size=1)
        self.conv_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x, static, future_precip, future_temp, future_temporal, target=None, epoch=0):
        batch_size, seq_len, channels, height, width = x.shape
        
        static_feat = self.static_processor(static)
        
        h = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
            for _ in range(self.num_layers)]
        c = [torch.zeros_like(h[i]) for i in range(self.num_layers)]
        
        # Process historical sequence
        for t in range(seq_len):
            x_t = self.input_conv(x[:, t])
            for i, layer in enumerate(self.convlstm_layers):
                h[i], c[i] = checkpoint(layer, x_t, h[i], c[i], use_reentrant=False)
                x_t = h[i]
        
        predictions = []
        last_swe = x[:, -1, 0:1]  # Last SWE observation
        
        # Autoregressive forecasting
        for step in range(FORECAST_STEPS):
            # Prepare weather features
            weather_t = torch.stack([
                future_precip[:, step], 
                future_temp[:, step]
            ], dim=1)
            weather_feat = self.weather_proj(weather_t)
            
            # Prepare temporal features
            temporal_feat = self.temp_proj(future_temporal[:, step])
            temporal_feat = temporal_feat.view(batch_size, self.hidden_dim, 1, 1).expand(-1, -1, height, width)
            
            # Combine all features
            combined = torch.cat([
                last_swe,
                weather_feat,
                static_feat,
                temporal_feat
            ], dim=1)
            
            x_next = self.step_proj(combined)
            
            # Process through ConvLSTM layers
            for i, layer in enumerate(self.convlstm_layers):
                h[i], c[i] = checkpoint(layer, x_next, h[i], c[i], use_reentrant=False)
                x_next = h[i]
            
            # Predict next step
            next_swe = self.conv_out(x_next)
            predictions.append(next_swe)
            
            # Teacher forcing during training
            if self.training and target is not None:
                use_gt = torch.rand(1).item() < (0.5 * (1 - epoch/EPOCHS))  # Linear schedule
                last_swe = target[:, step:step+1] if use_gt else next_swe.detach()
            else:
                last_swe = next_swe
        
        return torch.cat(predictions, dim=1)

class BiasAwareLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        bias = torch.mean(pred - target)
        return mse_loss + self.alpha * bias**2

def calculate_metrics(output, target, stats):

    output_denorm = torch.expm1(output * (stats['swe_max'] - stats['swe_min']) + stats['swe_min'])
    target_denorm = torch.expm1(target * (stats['swe_max'] - stats['swe_min']) + stats['swe_min'])
    
    # Calculate metrics
    mae = torch.mean(torch.abs(output_denorm - target_denorm))
    mse = torch.mean((output_denorm - target_denorm)**2)
    rmse = torch.sqrt(mse)
    
    # Calculate R2 score
    target_mean = torch.mean(target_denorm)
    ss_total = torch.sum((target_denorm - target_mean)**2)
    ss_res = torch.sum((output_denorm - target_denorm)**2)
    r2 = 1 - (ss_res / ss_total)
    
    return mae, rmse, r2

def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect()

#%%
def main():
    rank, local_rank, world_size, device = setup_distributed()
    
    if rank == 0:
        if not os.path.exists('input/drammen/data.h5'):
            h5_path = load_data()
        else:
            h5_path = 'input/drammen/data.h5'
    else:
        h5_path = 'input/drammen/data.h5'
    
    # Wait for rank 0 to finish data preparation
    dist.barrier()

    # Prepare datasets
    with h5py.File(h5_path, 'r') as f:
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

    train_ds = SweDatasetHDF5(h5_path, list(range(0, 7*365)), stats=stats)
    val_ds = SweDatasetHDF5(h5_path, list(range(8*365, 10*365)), stats=stats)

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
    
    epoc=8
    MODEL_PATH = 'saved_models_full2/epochcd2_{}.pth'.format(epoc)
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
    
#%%


