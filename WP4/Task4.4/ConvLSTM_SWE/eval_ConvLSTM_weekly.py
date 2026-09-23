#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd 
import pathlib as pl
import h5py 
from ConvLSTM import SweDataset, AutoregressiveConvLSTM, denormalize_swe

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 60  
FORECAST_STEPS = 7 
INPUT_FEATURES = 7  
HIDDEN_SIZE = 128  
EPOCHS=50

h5_path='input/concatenated_SeNorge_Drammen_2010_2020.h5'

#%%
with h5py.File(h5_path, 'r') as f:
    mask = np.array(f['mask'][:])
    swe = np.array(f['swe'][:])
    precip = np.array(f['precip'][:])
    temp = np.array(f['temp'][:])
    static = np.array(f['static'][:])
    times = np.array(f['dates'][:], dtype='datetime64')
    lons = np.array(f['lon'][:])
    lats = np.array(f['lat'][:])

train_ds = SweDataset(
    swe, 
    precip, 
    temp, 
    static, 
    list(range(0, len(swe))),
    times,
    SEQ_LEN, 
    FORECAST_STEPS)

#%%
total_days = len(train_ds.time_indices)  
max_start_idx = total_days - SEQ_LEN - FORECAST_STEPS 
num_steps= max_start_idx // 7  

model = AutoregressiveConvLSTM(
    input_dim=INPUT_FEATURES,
    hidden_dim=HIDDEN_SIZE,
    kernel_size=3,
    num_layers=2,
    static_channels=2
).to(device)

MODEL_PATH = 'saved_models_Drammen/best_model_ConvLSTM_Drammen.pth'

checkpoint = torch.load(MODEL_PATH, map_location=device)
state_dict = checkpoint['model_state_dict']  # Access the model weights inside the checkpoint
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()
stats_dt=train_ds.stats

targets_rec = []
predictions_rec = []
prediction_dates = []

dynamic, static_feat, future_precip, future_temp, future_temporal, target,forecast_date = train_ds[0]

dynamic = dynamic.unsqueeze(0).to(device)
static_feat = static_feat.unsqueeze(0).to(device)
future_precip = future_precip.unsqueeze(0).to(device)
future_temp = future_temp.unsqueeze(0).to(device)
future_temporal = future_temporal.unsqueeze(0).to(device)

#%%
with torch.no_grad():
    for t in  range(0,7*(num_steps-1),7):

        target = target.unsqueeze(0).to(device)
        target_np = target.detach().cpu().numpy()
        target_denorm = denormalize_swe(target_np, stats_dt)

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

        predictions_rec.extend(pred_denorm[0,:, :, :])
        targets_rec.extend(target_denorm[0,:, :, :])
        prediction_dates.extend(forecast_date)

        print(f"Step {t} completed")

#%%

predictions_rec= np.array(predictions_rec)
targets_rec = np.array(targets_rec)
prediction_dates=pd.to_datetime(prediction_dates)

predictions = predictions_rec.transpose(0,2,1)  
targets = targets_rec.transpose(0, 2, 1)        

mask=mask.transpose(1, 0)

predictions=np.where(mask==1, predictions, np.nan)
targets=np.where(mask==1, targets, np.nan)

#%%

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

start_date = prediction_dates[0].strftime('%Y')
end_date = prediction_dates[-1].strftime('%Y')

output_filename = f"netcdf/SWE_ConvLSTM_SeNorge_Drammen_{start_date}_{end_date}.nc"
netcdf_out = pl.Path(output_filename)
netcdf_out.parent.mkdir(parents=True, exist_ok=True)

nc_out.to_netcdf(path=netcdf_out)
print(f"Results saved to: {output_filename}")


