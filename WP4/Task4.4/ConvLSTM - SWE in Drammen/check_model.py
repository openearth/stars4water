#%%
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import pandas as pd
import pathlib as pl
from snow_model import ConvLSTMCell, AutoregressiveConvLSTM
from snow_data import SweDataset, load_data, denormalize_swe

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_FEATURES = 7
HIDDEN_SIZE = 128
SEQ_LEN = 60  
FORECAST_STEPS = 7

# Load and prepare data
#swe_file = 'input/drammen/swe_2010_2020.nc'
#precip_file = 'input/drammen/input_2010_2020.nc'
#temp_file = 'input/drammen/input_2010_2020.nc'
topo_file = 'input/drammen/topo.nc'

swe_file = '/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen/swe_2018_2024.nc'   
precip_file = '/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen/input_2018_2024.nc'
temp_file = '/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/NVE_SeNorge/SeNorge_WGS84/drammen/input_2018_2024.nc'

#precip_file = '/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/ERA5/pr/ERA5_rr_2018_2024.nc'
#temp_file = '/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/ERA5/tg/ERA5_tg_2018_2024.nc'

swe, precip, temp, static, times = load_data(swe_file, precip_file, temp_file, topo_file)

#%%
# Convert times to pandas DatetimeIndex
time_index = pd.to_datetime(times)

# Find the index for our end date
END_DATE = str(times[-1].astype('datetime64[D]'))
end_idx = np.where(time_index == pd.to_datetime(END_DATE))[0][0]

# Calculate maximum simulation index
max_sim_idx = end_idx - FORECAST_STEPS + 1
num_forecasts = (max_sim_idx - SEQ_LEN) // 7
final_sim_idx = SEQ_LEN + num_forecasts * 7

# Initialize dataset with correct range
train_ds = SweDataset(
    swe, precip, temp, static,
    list(range(0, final_sim_idx + FORECAST_STEPS)),
    times
)

# Load model
model = AutoregressiveConvLSTM(
    input_dim=INPUT_FEATURES,
    hidden_dim=HIDDEN_SIZE,
    kernel_size=3,
    num_layers=2,
    static_channels=2
).to(device)

model_path = 'saved_models_autoreg_full/epoch10_8.pth'
checkpoint = torch.load(model_path, map_location=device)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Get normalization stats
stats_dt = train_ds.stats

# Initialize prediction storage
targets_rec = []
predictions_rec = []
prediction_dates = []

# Initialize first sequence
dynamic, static_feat, future_precip, future_temp, future_temporal, target, forecast_date = train_ds[0]
dynamic = dynamic.unsqueeze(0).to(device)
static_feat = static_feat.unsqueeze(0).to(device)
future_precip = future_precip.unsqueeze(0).to(device)
future_temp = future_temp.unsqueeze(0).to(device)
future_temporal = future_temporal.unsqueeze(0).to(device)

# Prediction loop
with torch.no_grad():
    for t in range(0, num_forecasts * 7, 7):
        target = target.unsqueeze(0).to(device)
        
        # Predict next SWE
        output = model(
            x=dynamic,
            static=static_feat,
            future_precip=future_precip,
            future_temp=future_temp,
            future_temporal=future_temporal
        )

        # Denormalize results
        target_np = target.detach().cpu().numpy()
        target_denorm = denormalize_swe(target_np, stats_dt)
        
        output_np = output.detach().cpu().numpy()
        pred_denorm = denormalize_swe(output_np, stats_dt)
        pred_denorm = np.clip(pred_denorm, 0, None)

        # Update dynamic input for next step
        new_swe = torch.cat([dynamic[0, 7:, 0, :, :], output[0, :, :, :]], dim=0)
        new_swe = new_swe.unsqueeze(1)
        
        next_dynamic, _, future_precip, future_temp, future_temporal, target, forecast_date = train_ds[t]
        
        future_precip = future_precip.unsqueeze(0).to(device)
        future_temp = future_temp.unsqueeze(0).to(device)
        future_temporal = future_temporal.unsqueeze(0).to(device)
        
        dynamic = torch.cat([new_swe.to(device), next_dynamic[:, 1:7, :, :].to(device)], dim=1)
        dynamic = dynamic.unsqueeze(0).to(device)

        # Store results
        predictions_rec.extend(pred_denorm[0, :, :, :])
        targets_rec.extend(target_denorm[0, :, :, :])
        prediction_dates.extend(forecast_date)

        print(f"Step {t//7 + 1}/{num_forecasts}, Date: {forecast_date[-1]}")

# Convert to numpy arrays
predictions_rec = np.array(predictions_rec)
targets_rec = np.array(targets_rec)
prediction_dates = pd.to_datetime(prediction_dates)

#%%
# Apply mask
swe_ds = xr.open_dataset(swe_file)

predictions = predictions_rec.transpose(0,2,1)  # Shape: (time, height, width)
targets = targets_rec.transpose(0, 2, 1)         # Shape: (time, height, width)

mask=swe_ds['snow_water_equivalent'].isel(time=0)/swe_ds['snow_water_equivalent'].isel(time=0)
mask=mask.values 

predictions=predictions*mask.transpose(0, 1)
targets=targets*mask.transpose(0, 1)

# Create output dataset
nc_out = xr.Dataset(
    {
        'prediction': (('time', 'lat', 'lon'), predictions),
        'target': (('time', 'lat', 'lon'), targets)
    },
    coords={
        'time': prediction_dates,
        'lon': swe_ds['lon'].values,
        'lat': swe_ds['lat'].values
    }
)

# Add metadata
nc_out['prediction'].attrs = {
    'standard_name': 'swe',
    'long_name': "Snow Water Equivalent", 
    'units': "mm"
}
nc_out['target'].attrs = {
    'standard_name': 'swe',
    'long_name': "Snow Water Equivalent", 
    'units': "mm"
}

# Create output filename with date range
start_year = prediction_dates[0].strftime('%Y-%m-%d')
end_year = prediction_dates[-1].strftime('%Y-%m-%d')
output_filename = f"netcdf/SWE_CNNLSTM_weekly_SeNorge_{start_year}_{end_year}.nc"

# Save results
netcdf_out = pl.Path(output_filename)
netcdf_out.parent.mkdir(parents=True, exist_ok=True)
nc_out.to_netcdf(path=netcdf_out)

print(f"Simulation completed. Results saved to: {output_filename}")
print(f"Time period: {prediction_dates[0]} to {prediction_dates[-1]}")
# %%


#%%
