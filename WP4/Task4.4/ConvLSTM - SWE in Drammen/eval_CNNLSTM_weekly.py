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
import pandas as pd 

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 60  # Secuencia temporal
FORECAST_STEPS = 7  # Días a predecir
BATCH_SIZE = 2  # Tamaño del lote
EPOCHS = 50  # Número de épocas
INPUT_FEATURES = 7  # Número de características de entrada
LAT_DIM = 233  # Dimensiones de latitud
LON_DIM = 389  # Dimensiones de longitud
HIDDEN_SIZE = 128  # Tamaño de la capa oculta

# Cargar Datos
def load_data():
    # Load datasets
    #swe_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen/swe_2021_2024.nc')    
    
    #precip_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/ERA5/pr/ERA5_rr_2021_2023.nc')
    #temp_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/ERA5/tg/ERA5_tg_2021_2023.nc')
    
    #precip_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen/input_2021_2024.nc')
    #temp_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen/input_2021_2024.nc')
    
    #topo_ds = xr.open_dataset('input/drammen/topo.nc')
    
    swe_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen2/swe_2010_2020.nc')
    precip_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen2/input_rr_2010_2020.nc')
    temp_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen2/input_tg_2010_2020.nc')
    topo_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen2/topo.nc')

    swe = swe_ds['snow_water_equivalent'].values.transpose(0, 2, 1)
    precip = precip_ds['rr'].values.transpose(0, 2, 1)
    temp = temp_ds['tg'].values.transpose(0, 2, 1)

    #topo_ds = xr.open_dataset('input/drammen/topo.nc')

    # Convert all times to "days" (ignoring hours, minutes, etc.)
    swe_days = swe_ds['time'].values.astype('datetime64[D]')
    precip_days = precip_ds['time'].values.astype('datetime64[D]')
    temp_days = temp_ds['time'].values.astype('datetime64[D]')

    # Find common days (2022 only, since precip/temp are 2022)
    common_days = np.intersect1d(swe_days, precip_days)
    common_days = np.intersect1d(common_days, temp_days)

    if len(common_days) == 0:
        raise ValueError("No overlapping days between datasets!")

    print(f"Aligned days: {len(common_days)} (from {common_days[0]} to {common_days[-1]})")

    # Select only overlapping days in each dataset
    swe_ds = swe_ds.sel(time=np.isin(swe_days, common_days))
    precip_ds = precip_ds.sel(time=np.isin(precip_days, common_days))
    temp_ds = temp_ds.sel(time=np.isin(temp_days, common_days))

    # Verify alignment
    assert np.array_equal(
        swe_ds['time'].values.astype('datetime64[D]'),
        precip_ds['time'].values.astype('datetime64[D]')
    ), "SWE and precipitation days do not match!"
    
    assert np.array_equal(
        swe_ds['time'].values.astype('datetime64[D]'),
        temp_ds['time'].values.astype('datetime64[D]')
    ), "SWE and temperature days do not match!"

    # Extract and transpose data
    swe = swe_ds['snow_water_equivalent'].values.transpose(0, 2, 1)
    precip = precip_ds['rr'].values.transpose(0, 2, 1)
    temp = temp_ds['tg'].values.transpose(0, 2, 1)
    times = swe_ds['time'].values  # Now aligned by day

    # Prepare static data (topography + latitude grid)
    lat_grid = np.meshgrid(swe_ds['lat'].values, swe_ds['lon'].values)[0]
    static = np.stack([topo_ds['dem_mean'].values.T, lat_grid])

    return swe, precip, temp, static, times

# Clase Dataset
class SweDataset(Dataset):
    def __init__(self, swe, precip, temp, static, time_indices, times, stats=None):
        self.swe = np.nan_to_num(np.log1p(swe), nan=0.0)  # log1p transform
        self.precip = np.nan_to_num(precip, nan=0.0)
        self.temp = np.nan_to_num(temp, nan=0.0)
        self.static = np.nan_to_num(static, nan=0.0)
        self.time_indices = time_indices
        self.stats = stats if stats else self._compute_stats()
        self.times = times

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

        # 1. SWE: Past 60 days (t-60 to t-1)
        swe_past = self._normalize_dynamic(
            self.swe[actual_t - SEQ_LEN : actual_t],
            'swe'
        )

        # 2. Precipitation: Past 60 days (t-60 to t-1) + Future 7 days (t to t+6)
        precip_seq = self._normalize_dynamic(
            self.precip[actual_t - SEQ_LEN : actual_t + FORECAST_STEPS],
            'precip'
        )

        # 3. Temperature: Past 60 days (t-60 to t-1) + Future 7 days (t to t+6)
        temp_seq = self._normalize_dynamic(
            self.temp[actual_t - SEQ_LEN : actual_t + FORECAST_STEPS],
            'temp'
        )

        # 4. Temporal encoding for ALL timesteps (history + future)
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
        
        forecast_dates = self.times[actual_t : actual_t + FORECAST_STEPS]

        # 5. Stack all historical dynamic features
        dynamic = np.stack([
            swe_past,                         # Channel 0: SWE (60 steps)
            precip_seq[:SEQ_LEN],             # Channel 1: Past precip (60 steps)
            temp_seq[:SEQ_LEN],               # Channel 2: Past temp (60 steps)
            hist_month_sin,                   # Channel 3: Month (sin)
            hist_month_cos,                   # Channel 4: Month (cos)
            hist_day_sin,                     # Channel 5: Day of year (sin)
            hist_day_cos                      # Channel 6: Day of year (cos)
        ], axis=1)  # Shape: (60, 7, height, width)

        # 6. Future weather (t to t+6)
        future_precip = precip_seq[SEQ_LEN:]  # Shape: (7, height, width)
        future_temp = temp_seq[SEQ_LEN:]      # Shape: (7, height, width)

        # 7. Future temporal features (t to t+6)
        future_temporal = np.stack([
            fut_month_sin,
            fut_month_cos,
            fut_day_sin,
            fut_day_cos
        ], axis=1)  # Shape: (7, 4)

        # 8. Static features
        static = self._normalize_static()  # Shape: (2, height, width)

        # 9. Target: SWE for next 7 days (t to t+6)
        target = self._normalize_dynamic(
            self.swe[actual_t : actual_t + FORECAST_STEPS],
            'swe'
        )  # Shape: (7, height, width)

        return (
            torch.tensor(dynamic, dtype=torch.float32),    # (60, 7, H, W)
            torch.tensor(static, dtype=torch.float32),     # (2, H, W)
            torch.tensor(future_precip, dtype=torch.float32),  # (7, H, W)
            torch.tensor(future_temp, dtype=torch.float32),    # (7, H, W)
            torch.tensor(future_temporal, dtype=torch.float32), # (7, 4)
            torch.tensor(target, dtype=torch.float32),
            forecast_dates # (7, H, W)
        )

# Modelo ConvLSTM Autoregresivo con Future Forcing
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
        last_swe = x[:, -1, 0:1]  # Last SWE value (channel 0)
        
        # Autoregressive prediction
        for step in range(FORECAST_STEPS):
            # Process weather features
            weather_t = torch.stack([
                future_precip[:, step], 
                future_temp[:, step]
            ], dim=1)
            weather_feat = self.weather_proj(weather_t)
            
            # Process temporal features
            temporal_feat = self.temp_proj(future_temporal[:, step])  # [B, hidden_dim]
            temporal_feat = temporal_feat.view(batch_size, self.hidden_dim, 1, 1).expand(-1, -1, height, width)
            
            # Combine all features
            combined = torch.cat([
                last_swe,                    # 1 channel
                weather_feat,                # hidden_dim channels
                static_feat,                 # hidden_dim channels
                temporal_feat                # hidden_dim channels
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

# Cálculo de métricas
def denormalize_swe(normalized_swe, stats):
    """Convert normalized SWE back to real SWE."""
    log_swe = normalized_swe * (stats['swe_max'] - stats['swe_min']) + stats['swe_min']
    swe = np.expm1(log_swe)  # Undo log1p transformation
    return swe

#%%

# Ejecución principal
swe, precip, temp, static,times = load_data()

train_ds = SweDataset(swe, precip, temp, static, list(range(0, 8*365)),times)
#val_ds = SweDataset(swe, precip, temp, static, list(range(9*365, 10*365)), stats=train_ds.stats)


#%%

total_days = len(train_ds.time_indices)  # 6*365 in your case
max_start_idx = total_days - SEQ_LEN - FORECAST_STEPS  # Last valid starting index
num_steps= max_start_idx // 7  # Integer division to get number of 7-day steps

model = AutoregressiveConvLSTM(
    input_dim=INPUT_FEATURES,
    hidden_dim=HIDDEN_SIZE,
    kernel_size=3,
    num_layers=2,
    static_channels=2
).to(device)

epoc=8
#MODEL_PATH = 'saved_models_autoreg3/ConvLSTM2_autoreg_epoch_{}.pth'.format(epoc)
MODEL_PATH = 'saved_models_autoreg_drammen2/epochcd13_{}.pth'.format(epoc)

checkpoint = torch.load(MODEL_PATH, map_location=device)
state_dict = checkpoint['model_state_dict']  # Access the model weights inside the checkpoint
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()

#stats_dt = np.load('training_stats_autoreg_full.npy', allow_pickle=True).item()
stats_dt=train_ds.stats
#%%

targets_rec = []
predictions_rec = []
prediction_dates = []

t=0
# Initialize the first sequence with true SWE values
dynamic, static_feat, future_precip, future_temp, future_temporal, target,forecast_date = train_ds[t]

dynamic = dynamic.unsqueeze(0).to(device)
static_feat = static_feat.unsqueeze(0).to(device)#%%
future_precip = future_precip.unsqueeze(0).to(device)
future_temp = future_temp.unsqueeze(0).to(device)
future_temporal = future_temporal.unsqueeze(0).to(device)

with torch.no_grad():
    for t in  range(0,7*num_steps,7):

        target = target.unsqueeze(0).to(device)
        target_np = target.detach().cpu().numpy()
        target_denorm = denormalize_swe(target_np, stats_dt)
        #date_indices.append(date_idx)

        # Predict the next SWE
        output = model(x=dynamic,
                    static=static_feat,
                    future_precip=future_precip,
                    future_temp=future_temp,
                    future_temporal=future_temporal)

        output_np = output.detach().cpu().numpy()
        pred_denorm = denormalize_swe(output_np, stats_dt)
        pred_denorm = np.clip(pred_denorm, 0, None)

        out_log=torch.tensor(output_np)
        new_swe=torch.cat([dynamic[0,7:,0,:,:],output[0,:,:,:]], dim=0)
        new_swe = new_swe.unsqueeze(1)

        next_dynamic, _,future_precip,future_temp,future_temporal, target,forecast_date= train_ds[t]

        future_precip = future_precip.unsqueeze(0).to(device)
        future_temp = future_temp.unsqueeze(0).to(device)
        future_temporal = future_temporal.unsqueeze(0).to(device)

        dynamic = torch.cat([new_swe.to(device), next_dynamic[:, 1:7, :, :].to(device)], dim=1)
        dynamic=dynamic.unsqueeze(0).to(device)

        #dynamic=next_dynamic.unsqueeze(0).to(device)

        predictions_rec.extend(pred_denorm[0,:, :, :])
        targets_rec.extend(target_denorm[0,:, :, :])
        prediction_dates.extend(forecast_date)

        print(f"Step {t} completed")

#%%
predictions_rec= np.array(predictions_rec)
targets_rec = np.array(targets_rec)
prediction_dates=pd.to_datetime(prediction_dates)

swe_ds = xr.open_dataset('/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen2/swe_2010_2020.nc')
    
lons=swe_ds['lon'].values
lats=swe_ds['lat'].values

predictions = predictions_rec.transpose(0,2,1)  # Shape: (time, height, width)
targets = targets_rec.transpose(0, 2, 1)         # Shape: (time, height, width)

mask=swe_ds['snow_water_equivalent'].isel(time=0)/swe_ds['snow_water_equivalent'].isel(time=0)
mask=mask.values 

predictions=predictions*mask.transpose(0, 1)
targets=targets*mask.transpose(0, 1)

nc_out = xr.Dataset(
    {'prediction': (('time','lat','lon'),predictions),
     'target': (('time','lat','lon'),targets)},
    coords={
        'time':prediction_dates,
        'lon':lons,
        'lat':lats})

nc_out['prediction'].attrs= {'standard_name': 'swe','long_name':"Snow Water Equivalent", 'units':"mm"}
nc_out['target'].attrs= {'standard_name': 'swe','long_name':"Snow Water Equivalent", 'units':"mm"}

nc_out['lat'].attrs= {'standard_name': 'latitude','long_name':"Latitude", 'units':"degrees_north",'axis':'Y'}
nc_out['lon'].attrs= {'standard_name': 'longitude','long_name':"Longitude", 'units':"degrees_east",'axis':'X'}

import pathlib as pl

# Get the start and end dates from the prediction dates
start_date = prediction_dates[0].strftime('%Y')
end_date = prediction_dates[-1].strftime('%Y')

# Create the output filename with the actual simulation period
output_filename = f"netcdf/SWE_CNNLSTM_weekly_SeNorgeFULL2_{start_date}_{end_date}.nc"

# Ensure the output directory exists
import pathlib as pl
netcdf_out = pl.Path(output_filename)
netcdf_out.parent.mkdir(parents=True, exist_ok=True)

# Save to NetCDF
nc_out.to_netcdf(path=netcdf_out)

print(f"Results saved to: {output_filename}")
