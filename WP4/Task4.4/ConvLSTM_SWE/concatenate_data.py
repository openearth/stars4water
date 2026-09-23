#%%
import numpy as np
import os
import h5py
import warnings
import xarray as xr 
from datetime import datetime

source_path='/p/data1/slts/avila2/Stars4Water/WP4/SWE_Drammen/datasets/region_drammen'

topo_ds = xr.open_dataset(f'{source_path}/topo.nc')
topo_data = topo_ds['dem_mean'].values[:]

all_doy_pr = []  
all_doy_swe = []
all_doy_temp = []

lat = topo_ds['lat'].values[:]
lon = topo_ds['lon'].values[:]

lon2d, lat2d = np.meshgrid(lon, lat)


# Load datasets for each year
swe_nc = xr.open_dataset(f'{source_path}/swe_2010_2020.nc')
input_nc = xr.open_dataset(f'{source_path}/input_2010_2020.nc')

dates_pr = [np.datetime64(d.astype('datetime64[D]')) for d in input_nc['time'].values]
dates_swe = [np.datetime64(d.astype('datetime64[D]')) for d in swe_nc['time'].values]
dates_temp = [np.datetime64(d.astype('datetime64[D]')) for d in input_nc['time'].values]

pr_data = input_nc['rr'].values[:]
swe_data = swe_nc['snow_water_equivalent'].values[:]
temp_data = input_nc['tg'].values[:]

def calculate_doy(dates):
    doy = []
    for date in dates:
        dt = date.astype(datetime)
        day_of_year = dt.timetuple().tm_yday
        doy.append(day_of_year)
    return doy

all_doy_pr.extend(calculate_doy(dates_pr))
all_doy_swe.extend(calculate_doy(dates_swe))
all_doy_temp.extend(calculate_doy(dates_temp))

input_nc.close()
swe_nc.close()

# Find common time periods across all datasets
print("\nFinding common time periods...")

# Convert dates to numpy arrays for easier comparison
dates_pr_array = np.array(dates_pr)
dates_swe_array = np.array(dates_swe)
dates_temp_array = np.array(dates_temp)

# Find intersection of dates across all three datasets
common_dates = np.intersect1d(dates_pr_array, dates_swe_array)
common_dates = np.intersect1d(common_dates, dates_temp_array)

print(f"Total dates - precip: {len(dates_pr_array)}, swe: {len(dates_swe_array)}, temp: {len(dates_temp_array)}")
print(f"Common dates across all datasets: {len(common_dates)}")

if len(common_dates) == 0:
    raise ValueError("No common dates found between datasets!")

# Find indices for common dates in each dataset
indices_pr = np.where(np.isin(dates_pr_array, common_dates))[0]
indices_swe = np.where(np.isin(dates_swe_array, common_dates))[0]
indices_temp = np.where(np.isin(dates_temp_array, common_dates))[0]

print(f"Indices to extract - precip: {len(indices_pr)}, swe: {len(indices_swe)}, temp: {len(indices_temp)}")

# Extract only common time steps
concatenated_pr_common = pr_data[indices_pr, :, :]
concatenated_swe_common = swe_data[indices_swe, :, :]
concatenated_temp_common = temp_data[indices_temp, :, :]

print(f"Common shapes - precip: {concatenated_pr_common.shape}, swe: {concatenated_swe_common.shape}, temp: {concatenated_temp_common.shape}")

# Prepare static data
static_data = np.stack([topo_data, lat2d])

# Convert DOY for common dates using the same normalization
common_dates_list = [datetime.strptime(str(d), '%Y-%m-%d') for d in common_dates]
doy_array = np.array([date.timetuple().tm_yday - 1 for date in common_dates_list])  # 0-indexed

# Calculate mask based on swe.mean(axis=0) using common time steps
print("Calculating mask based on swe.mean(axis=0)...")
swe_temporal_mean = concatenated_swe_common.mean(axis=0)

# Create mask: 1 for valid pixels, 0 for invalid pixels
mask = np.ones_like(swe_temporal_mean, dtype=np.int8)
mask[np.isnan(swe_temporal_mean)] = 0
mask[np.isinf(swe_temporal_mean)] = 0

print(f"Valid pixels: {np.sum(mask)} out of {mask.size} ({np.sum(mask)/mask.size*100:.2f}%)")

#%%
os.makedirs('input', exist_ok=True)
output_path = f'input/concatenated_SeNorge_Drammen_2010_2020.h5'
print(f"\nSaving to {output_path}")

with h5py.File(output_path, 'w') as f:
    f.attrs['n_time'] = len(common_dates)
    f.attrs['n_lat'] = concatenated_pr_common.shape[1]
    f.attrs['n_lon'] = concatenated_pr_common.shape[2]
    
    f.create_dataset('swe', data=concatenated_swe_common, compression='gzip', compression_opts=9)
    f.create_dataset('precip', data=concatenated_pr_common, compression='gzip', compression_opts=9)
    f.create_dataset('temp', data=concatenated_temp_common, compression='gzip', compression_opts=9)
    f.create_dataset('static', data=static_data, compression='gzip', compression_opts=9)
    f.create_dataset('doy', data=doy_array, compression='gzip', compression_opts=9)
    f.create_dataset('mask', data=mask, compression='gzip', compression_opts=9)
    
    f.create_dataset('lat', data=lat, compression='gzip', compression_opts=9)
    f.create_dataset('lon', data=lon, compression='gzip', compression_opts=9)
    
    date_strings = [date.strftime('%Y-%m-%d') for date in common_dates_list]
    dt = h5py.string_dtype(encoding='utf-8')
    f.create_dataset('dates', data=date_strings, dtype=dt)
    
    f.create_dataset('time_dims_precip', data=[len(dates_pr)], dtype=int)
    f.create_dataset('time_dims_swe', data=[len(dates_swe)], dtype=int)
    f.create_dataset('time_dims_temp', data=[len(dates_temp)], dtype=int)

print("Concatenation and saving completed successfully!")

print(f"\nVerification:")
print(f"  swe shape: {concatenated_swe_common.shape}")
print(f"  precip shape: {concatenated_pr_common.shape}")
print(f"  temp shape: {concatenated_temp_common.shape}")
print(f"  doy shape: {doy_array.shape}")
print(f"  Number of dates: {len(common_dates_list)}")
print(f"Sample dates: {date_strings[:5]}")

topo_ds.close()

