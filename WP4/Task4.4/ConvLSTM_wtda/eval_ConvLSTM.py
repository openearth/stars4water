# main_script.py
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import pathlib as pl
from datetime import datetime

# Import all your functions from the separate file
from ConvLSTM_utils import (
    calculate_api,
    HydrologyConvLSTM,
    SpatioTemporalDataset,
    normalize_data,
    denormalize_wtda,
    prepare_dates,
    create_mask_from_data
)

# Load data
h5_path = '/p/scratch/cesmtst/avila2/emulator_DE06/Parflow_DE06/scripts/input/concatenated_WTDa_Parflow_2012_2021_0_2000_stride5.h5'

with h5py.File(h5_path, 'r') as f:
    mask = np.array(f['mask'][:])
    wtda = f['wtda'][:]
    precip = f['precip'][:]
    vwc = f['vwc'][:]
    topo = f['static'][0, :]
    slopex = f['static'][1, :]
    slopey = f['static'][2, :]
    wtd = f['wtd'][:, :]
    dates = f['dates'][:]
    rlon_coords = f['lon'][:]
    rlat_coords = f['lat'][:]
    clyppt = f['static'][4, :]
    sndppt = f['static'][5, :]

# Calculate API
api = calculate_api(precip, 0.95)

# Calculate normalization statistics
wtda_mean = np.nanmean(wtda)
wtda_std = np.nanstd(wtda)
api_mean = np.nanmean(api)
api_std = np.nanstd(api)
vwc_mean = np.nanmean(vwc)
vwc_std = np.nanstd(vwc)

# Get min/max for static variables
topo_max, topo_min = np.nanmax(topo), np.nanmin(topo)
slopex_max, slopex_min = np.nanmax(slopex), np.nanmin(slopex)
slopey_max, slopey_min = np.nanmax(slopey), np.nanmin(slopey)
sndppt_max, sndppt_min = np.nanmax(sndppt), np.nanmin(sndppt)
clyppt_max, clyppt_min = np.nanmax(clyppt), np.nanmin(clyppt)

# Normalize data
topo_norm = (topo - topo_min) / (topo_max - topo_min + 1e-8)
slopex_norm = (slopex - slopex_min) / (slopex_max - slopex_min + 1e-8)
slopey_norm = (slopey - slopey_min) / (slopey_max - slopey_min + 1e-8)
clyppt_norm = (clyppt - clyppt_min) / (clyppt_max - clyppt_min + 1e-8)
sndppt_norm = (sndppt - sndppt_min) / (sndppt_max - sndppt_min + 1e-8)

api_norm = (api - api_mean) / (api_std + 1e-8)
vwc_norm = (vwc - vwc_mean) / (vwc_std + 1e-8)
wtda_norm = (wtda - wtda_mean) / (wtda_std + 1e-8)

# Handle NaN values
topo_norm = np.nan_to_num(topo_norm, nan=-10)
slopex_norm = np.nan_to_num(slopex_norm, nan=-10)
slopey_norm = np.nan_to_num(slopey_norm, nan=-10)
clyppt_norm = np.nan_to_num(clyppt_norm, nan=-10)
sndppt_norm = np.nan_to_num(sndppt_norm, nan=-10)
api_norm = np.nan_to_num(api_norm, nan=-10)
vwc_norm = np.nan_to_num(vwc_norm, nan=-10)

# Prepare data arrays
static_data = np.stack([topo_norm, slopex_norm, slopey_norm, sndppt_norm, clyppt_norm], axis=0)
dynamic_data = np.stack([api_norm, vwc_norm], axis=1)
target_data = wtda_norm[:, np.newaxis, :, :]

# Prepare dates
dates = prepare_dates(dates)

# Create dataset
seq_len = 30
dataset = SpatioTemporalDataset(
    static_data=static_data,
    dynamic_data=dynamic_data,
    target_data=target_data,
    dates=dates,
    seq_len=seq_len
)

#%%
# Split data
n = wtda_norm.shape[0]
train_end = int(100 * n)  # This seems wrong - should be something like int(0.8 * n)

# Fix the split
train_end = int(0.8 * n)  # Use 80% for training
train_dataset = Subset(dataset, range(0, train_end - seq_len))
val_dataset = Subset(dataset, range(train_end - seq_len, n - seq_len))

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4
)

# Initialize model
model = HydrologyConvLSTM(
    static_channels=5,      # topo, slopex, slopey, sndppt, clyppt
    dynamic_channels=2,     # api, vwc
    hidden_dim=64,
    num_layers=2,
    kernel_size=3
)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "/p/scratch/cesmtst/avila2/emulator_DE06/Parflow_DE06/scripts/saved_models_full_static/best_model.pth"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Define denormalization function using the stats
def denormalize_wtda_local(data):
    return denormalize_wtda(data, wtda_std, wtda_mean)

# Run inference
targets_rec = []
predictions_rec = []
target_date_indices = []

i = 0
with torch.no_grad():
    for batch_idx, (static_batch, dynamic_batch, target_batch, target_idx) in enumerate(train_loader):
        static_batch = static_batch.to(device)
        dynamic_batch = dynamic_batch.to(device)
        target_batch = target_batch.to(device)
        
        predictions = model(static_batch, dynamic_batch)

        output_np = predictions.detach().cpu().numpy()
        target_np = target_batch.detach().cpu().numpy()

        output_denorm = denormalize_wtda_local(output_np[:, 0, :, :])
        target_denorm = denormalize_wtda_local(target_np[:, 0, :, :])

        predictions_rec.extend(output_denorm)
        targets_rec.extend(target_denorm)
        target_date_indices.extend(target_idx.numpy())

        i += 1
        print(i)

#%%
# Convert to arrays
target_dates = [dates[idx] for idx in target_date_indices]
target_dates = pd.to_datetime(target_dates)
predictions_rec = np.array(predictions_rec)
targets_rec = np.array(targets_rec)

# Apply mask
with h5py.File(h5_path, 'r') as f:
    mask = np.array(f['static'][:][1, :, :])

mask = np.where(np.isnan(mask), np.nan, 1)
preds = np.where(mask == 1, predictions_rec, np.nan)
obss = np.where(mask == 1, targets_rec, np.nan)

#%%
# Save to NetCDF
nc_out = xr.Dataset(
    {
        'wtda_simulated': (('time', 'rlat', 'rlon'), preds),
        'wtda_target': (('time', 'rlat', 'rlon'), obss)
    },
    coords={
        'time': target_dates,
        'rlon': rlon_coords,
        'rlat': rlat_coords
    }
)

# Add attributes
nc_out['wtda_simulated'].attrs = {
    'standard_name': 'water_table_depth',
    'long_name': "Water Table Depth", 
    'units': "m",
    'description': 'Predicted water table depth from ConvLSTM model'
}
nc_out['wtda_target'].attrs = {
    'standard_name': 'water_table_depth',
    'long_name': "Water Table Depth", 
    'units': "m",
    'description': 'Actual water table depth from Parflow simulation'
}

nc_out['rlat'].attrs = {
    'standard_name': 'rotated latitude',
    'long_name': "Rotalted Latitude", 
    'units': "degrees_north",
    'axis': 'Y'
}
nc_out['rlon'].attrs = {
    'standard_name': 'rotated longitude',
    'long_name': "Rotated Longitude", 
    'units': "degrees_east",
    'axis': 'X'
}

# Save to file
start_date = pd.Timestamp(target_dates[0]).strftime('%Y%m%d')
end_date = pd.Timestamp(target_dates[-1]).strftime('%Y%m%d')
output_filename = f"netcdf/wtda_ConvLSTM_daily_Parflow_{start_date}_{end_date}.nc"

netcdf_out = pl.Path(output_filename)
netcdf_out.parent.mkdir(parents=True, exist_ok=True)
nc_out.to_netcdf(path=netcdf_out)

print(f"Results saved to {output_filename}")
# %%
