#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 60  
FORECAST_STEPS = 7 
BATCH_SIZE = 2  
EPOCHS = 50  
INPUT_FEATURES = 7  
HIDDEN_SIZE = 128  


def load_data():
    
    swe_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen2/swe_2010_2020.nc')
    prec_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen2/input_rr_2010_2020.nc')
    temp_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen2/input_tg_2010_2020.nc')
    topo_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen2/topo.nc')

    swe = swe_ds['snow_water_equivalent'].values.transpose(0, 2, 1)
    precip = prec_ds['rr'].values.transpose(0, 2, 1)
    temp = temp_ds['tg'].values.transpose(0, 2, 1)

    lat_grid = np.meshgrid(swe_ds['lat'].values, swe_ds['lon'].values)[0]
    static = np.stack([topo_ds['dem_mean'].values.T, lat_grid])

    return swe, precip, temp, static

class SweDataset(Dataset):
    def __init__(self, swe, precip, temp, static, time_indices, stats=None):
        self.swe = np.nan_to_num(np.log1p(swe), nan=0.0)  # log1p transform
        self.precip = np.nan_to_num(precip, nan=0.0)
        self.temp = np.nan_to_num(temp, nan=0.0)
        self.static = np.nan_to_num(static, nan=0.0)
        self.time_indices = time_indices
        self.stats = stats if stats else self._compute_stats()

    def _compute_stats(self):
        return {
            'swe_min': np.nanmin(self.swe),
            'swe_max': np.nanmax(self.swe),
            'precip_min': np.nanmin(self.precip),
            'precip_max': np.nanmax(self.precip),
            'temp_mean': np.nanmean(self.temp),
            'temp_std': np.nanstd(self.temp),
            'topo_min': np.nanmin(self.static[0]),
            'topo_max': np.nanmax(self.static[0]),
            'lat_mean': np.nanmean(self.static[1]),
            'lat_std': np.nanstd(self.static[1])
        }

    def _normalize_dynamic(self, data, feature):
        if feature == 'swe':
            return (data - self.stats['swe_min']) / (self.stats['swe_max'] - self.stats['swe_min'])
        elif feature == 'precip':
            return (data - self.stats['precip_min']) / (self.stats['precip_max'] - self.stats['precip_min'])
        elif feature == 'temp':
            return (data - self.stats['temp_mean']) / self.stats['temp_std']
        else:
            raise ValueError(f"Unknown feature: {feature}")

    def _normalize_static(self):
        topo = (self.static[0] - self.stats['topo_min']) / (self.stats['topo_max'] - self.stats['topo_min'])
        lat = (self.static[1] - self.stats['lat_mean']) / self.stats['lat_std']
        return np.stack([topo, lat])

    def __len__(self):
        return len(self.time_indices) - SEQ_LEN - FORECAST_STEPS

    def __getitem__(self, idx):
        actual_t = self.time_indices[idx + SEQ_LEN]

        # SWE: Past 60 days (t-60 to t-1)
        swe_past = self._normalize_dynamic(
            self.swe[actual_t - SEQ_LEN : actual_t],
            'swe'
        )

        # Precipitation: Past 60 days (t-60 to t-1) + Future 7 days (t to t+6)
        precip_seq = self._normalize_dynamic(
            self.precip[actual_t - SEQ_LEN : actual_t + FORECAST_STEPS],
            'precip'
        )

        # Temperature: Past 60 days (t-60 to t-1) + Future 7 days (t to t+6)
        temp_seq = self._normalize_dynamic(
            self.temp[actual_t - SEQ_LEN : actual_t + FORECAST_STEPS],
            'temp'
        )

        # Temporal encoding for ALL timesteps (history + future)
        all_days = np.arange(actual_t - SEQ_LEN, actual_t + FORECAST_STEPS)
        all_months = ((all_days % 365) // 30) + 1  # Approximate month
        all_doy = (all_days % 365) + 1  # Day of year
        
        # Cyclical encoding
        month_sin = np.sin(2 * np.pi * all_months / 12)
        month_cos = np.cos(2 * np.pi * all_months / 12)
        day_sin = np.sin(2 * np.pi * all_doy / 365)
        day_cos = np.cos(2 * np.pi * all_doy / 365)

        # Split into historical and future parts
        hist_month_sin = month_sin[:SEQ_LEN]
        hist_month_cos = month_cos[:SEQ_LEN]
        hist_day_sin = day_sin[:SEQ_LEN]
        hist_day_cos = day_cos[:SEQ_LEN]
        
        fut_month_sin = month_sin[SEQ_LEN:]
        fut_month_cos = month_cos[SEQ_LEN:]
        fut_day_sin = day_sin[SEQ_LEN:]
        fut_day_cos = day_cos[SEQ_LEN:]

        # Ensure all historical arrays have shape (SEQ_LEN, height, width)
        height, width = swe_past.shape[1], swe_past.shape[2]
        
        # Expand temporal features to match spatial dimensions
        hist_month_sin = np.tile(hist_month_sin[:, np.newaxis, np.newaxis], (1, height, width))
        hist_month_cos = np.tile(hist_month_cos[:, np.newaxis, np.newaxis], (1, height, width))
        hist_day_sin = np.tile(hist_day_sin[:, np.newaxis, np.newaxis], (1, height, width))
        hist_day_cos = np.tile(hist_day_cos[:, np.newaxis, np.newaxis], (1, height, width))

        # Stack all historical dynamic features
        dynamic = np.stack([
            swe_past,                         # Channel 1: SWE (60 steps)
            precip_seq[:SEQ_LEN],             # Channel 2: Past precip (60 steps)
            temp_seq[:SEQ_LEN],               # Channel 3: Past temp (60 steps)
            hist_month_sin,                   # Channel 4: Month (sin)
            hist_month_cos,                   # Channel 5: Month (cos)
            hist_day_sin,                     # Channel 6: Day of year (sin)
            hist_day_cos                      # Channel 7: Day of year (cos)
        ], axis=1)  

        # Future weather (t to t+6)
        future_precip = precip_seq[SEQ_LEN:]  
        future_temp = temp_seq[SEQ_LEN:]      

        # Future temporal features (t to t+6)
        future_temporal = np.stack([
            fut_month_sin,
            fut_month_cos,
            fut_day_sin,
            fut_day_cos
        ], axis=1)  # Shape: (7, 4)

        # Static features
        static = self._normalize_static() 

        # Target: SWE for next 7 days (t to t+6)
        target = self._normalize_dynamic(
            self.swe[actual_t : actual_t + FORECAST_STEPS],
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
        
        # Initial layer to process dynamic input
        self.input_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(
                hidden_dim,
                hidden_dim,
                kernel_size
            ) for _ in range(num_layers)
        ])
        
        # Static feature processing
        self.static_processor = nn.Sequential(
            nn.Conv2d(static_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        
        # Weather feature processing
        self.weather_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
        
        # Temporal feature projection
        self.temp_proj = nn.Linear(4, hidden_dim)
        
        # Projection for autoregressive step
        self.step_proj = nn.Conv2d(1 + hidden_dim + hidden_dim + hidden_dim, hidden_dim, kernel_size=1)
        
        # Output layer
        self.conv_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, *args, **kwargs):
        
        # Handle DataParallel input wrapping
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        
        # Unpack arguments
        if len(args) >= 5:
            x, static, future_precip, future_temp, future_temporal = args[:5]
            target = kwargs.get('target', None)
            epoch = kwargs.get('epoch', 0)
        else:
            x = kwargs['x']
            static = kwargs['static']
            future_precip = kwargs['future_precip']
            future_temp = kwargs['future_temp']
            future_temporal = kwargs['future_temporal']
            target = kwargs.get('target', None)
            epoch = kwargs.get('epoch', 0)

        batch_size, seq_len, channels, height, width = x.shape
        
        # Process static features
        static_feat = self.static_processor(static)
        
        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
            for _ in range(self.num_layers)]
        c = [torch.zeros_like(h[i]) for i in range(self.num_layers)]
        
        # Process historical sequence
        for t in range(seq_len):
            x_t = self.input_conv(x[:, t])
            for i, layer in enumerate(self.convlstm_layers):
                h[i], c[i] = layer(x_t, h[i], c[i])
                x_t = h[i]
        
        predictions = []
        last_swe = x[:, -1, 0:1]  
        
        # Autoregressive prediction
        for step in range(FORECAST_STEPS):
            # Process weather features
            weather_t = torch.stack([
                future_precip[:, step], 
                future_temp[:, step]
            ], dim=1)
            weather_feat = self.weather_proj(weather_t)
            
            # Process temporal features
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
            
            # Process through ConvLSTM
            for i, layer in enumerate(self.convlstm_layers):
                h[i], c[i] = layer(x_next, h[i], c[i])
                x_next = h[i]
            
            # Predict next SWE
            next_swe = self.conv_out(x_next)
            predictions.append(next_swe)
            
            # Scheduled sampling
            if self.training and target is not None:
                use_gt = torch.rand(1).item() < (0.5 * (1 - epoch/EPOCHS))
                last_swe = target[:, step:step+1] if use_gt else next_swe.detach()
            else:
                last_swe = next_swe
        
        return torch.cat(predictions, dim=1)

class BiasAwareLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = nn.MSELoss()(pred, target)
        bias = torch.mean(pred - target)
        return mse + self.alpha * bias**2

def calculate_metrics(output, target, stats):
    output = output.detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()

    output_denorm = np.expm1(output * (stats['swe_max'] - stats['swe_min']) + stats['swe_min'])
    target_denorm = np.expm1(target * (stats['swe_max'] - stats['swe_min']) + stats['swe_min'])
    
    return (
        mean_absolute_error(target_denorm, output_denorm),
        np.sqrt(mean_squared_error(target_denorm, output_denorm)),
        r2_score(target_denorm, output_denorm)
    )

swe, precip, temp, static = load_data()

train_ds = SweDataset(swe, precip, temp, static, list(range(0, 8*365)))
val_ds = SweDataset(swe, precip, temp, static, list(range(9*365, 10*365)), stats=train_ds.stats)

np.save('training_stats_autoreg3.npy', train_ds.stats)

model = AutoregressiveConvLSTM(
    input_dim=INPUT_FEATURES,
    hidden_dim=HIDDEN_SIZE,
    kernel_size=3,
    num_layers=2,
    static_channels=2
).to(device)

epoc=49
MODEL_PATH = 'saved_models_autoreg2/ConvLSTM2_autoreg_epoch_{}.pth'.format(epoc)

checkpoint = torch.load(MODEL_PATH, map_location=device)
state_dict = checkpoint['model_state_dict']  # Access the model weights inside the checkpoint

state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model = torch.nn.DataParallel(model)
model = model.to(device)

criterion = BiasAwareLoss(alpha=0.6).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

os.makedirs('saved_models_autoreg3', exist_ok=True)

with open('training_log_autoreg3.txt', 'w') as log_file:
    log_file.write('Epoch,Train Loss,Val Loss,Train MAE,Train RMSE,Train R2,Val MAE,Val RMSE,Val R2,Time (s),LR\n')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        model.train()
        train_loss, train_mae, train_rmse, train_r2 = 0, 0, 0, 0
        
        for batch_idx, (dynamic, static_feat, future_precip, future_temp, future_temporal, target) in enumerate(train_loader):
            dynamic = dynamic.to(device)
            static_feat = static_feat.to(device)
            future_precip = future_precip.to(device)
            future_temp = future_temp.to(device)
            future_temporal = future_temporal.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(
                x=dynamic,
                static=static_feat,
                future_precip=future_precip,
                future_temp=future_temp,
                future_temporal=future_temporal,
                target=target,
                epoch=epoch
            )
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            mae, rmse, r2 = calculate_metrics(output, target, train_ds.stats)
            train_mae += mae
            train_rmse += rmse
            train_r2 += r2
            
            print(f'Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}', end='\r')

        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        avg_train_rmse = train_rmse / len(train_loader)
        avg_train_r2 = train_r2 / len(train_loader)
        
        model.eval()
        val_loss, val_mae, val_rmse, val_r2 = 0, 0, 0, 0
        
        with torch.no_grad():
            for dynamic, static_feat, future_precip, future_temp, future_temporal, target in val_loader:
                dynamic = dynamic.to(device)
                static_feat = static_feat.to(device)
                future_precip = future_precip.to(device)
                future_temp = future_temp.to(device)
                future_temporal = future_temporal.to(device)
                target = target.to(device)
                
                output = model(
                    x=dynamic,
                    static=static_feat,
                    future_precip=future_precip,
                    future_temp=future_temp,
                    future_temporal=future_temporal
                )
                loss = criterion(output, target)
                val_loss += loss.item()
                mae, rmse, r2 = calculate_metrics(output, target, train_ds.stats)
                val_mae += mae
                val_rmse += rmse
                val_r2 += r2
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        avg_val_rmse = val_rmse / len(val_loader)
        avg_val_r2 = val_r2 / len(val_loader)
        
        epoch_time = time.time() - start_time
        
        current_lr = optimizer.param_groups[0]['lr']
        
        log_file.write(
            f'{epoch + 1},'
            f'{avg_train_loss:.6e},'
            f'{avg_val_loss:.6e},'
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
        print(f'Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f}')
        print(f'Learning Rate: {current_lr:.2e}\n')
        
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': optimizer.param_groups[0]['lr']}

        torch.save(checkpoint, f'saved_models_autoreg3/ConvLSTM2_autoreg_epoch_{epoch}.pth')
        
        scheduler.step(avg_val_loss)
