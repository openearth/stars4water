#%%
import numpy as np
import os
import h5py
import warnings
import xarray as xr 
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt 

# Configuration
xmin = 0
xmax = 2000
YEARS = np.arange(2012, 2022)
stride = 5

SOURCE_PATH='/p/scratch/cesmtst/avila2/emulator_DE06/Parflow_DE06'

topo_ds = xr.open_dataset('../static/static_DE05.nc')

topo_data = topo_ds['topo'].values[xmin:xmax:stride, xmin:xmax:stride]
slopex_data = topo_ds['slopex'].values[xmin:xmax:stride, xmin:xmax:stride]
slopey_data = topo_ds['slopey'].values[xmin:xmax:stride, xmin:xmax:stride]
soiltype_data = topo_ds['soil'].values[xmin:xmax:stride, xmin:xmax:stride]
clyppt_data = topo_ds['CLYPPT'].values[xmin:xmax:stride, xmin:xmax:stride]
sndppt_data = topo_ds['SNDPPT'].values[xmin:xmax:stride, xmin:xmax:stride]

rlat_coords = topo_ds['rlat'].values[xmin:xmax:stride]
rlon_coords = topo_ds['rlon'].values[xmin:xmax:stride]


mean_clim = xr.open_dataset(f'{SOURCE_PATH}/target/wtd/mean_climatology_wtd.nc')
std_clim = xr.open_dataset(f'{SOURCE_PATH}/target/wtd/std_climatology_wtd.nc')

mean_var_name = list(mean_clim.data_vars)[0]
std_var_name = list(std_clim.data_vars)[0]

mean_clipped = mean_clim[mean_var_name].isel(
    rlon=slice(xmin, xmax, stride), 
    rlat=slice(xmin, xmax, stride)
)
std_clipped = std_clim[std_var_name].isel(
    rlon=slice(xmin, xmax, stride), 
    rlat=slice(xmin, xmax, stride)
)

print(f"Climatology shape: {mean_clipped.shape}")
print(f"Number of days in climatology: {mean_clipped.shape[0]}")

if 'time' in mean_clim.coords:
    clim_dates = mean_clim['time'].values
    print(len(clim_dates))
    
def get_doy_non_leap_vector(date_strings):

    doy_list = []
    for date_str in date_strings:
        # Extract month and day, use 2023 as reference (non-leap year)
        month_day = date_str[5:]  # Get 'MM-DD' part
        # Create a date in 2023 (non-leap year) with the same month and day
        date_obj = datetime.strptime(f'2023-{month_day}', '%Y-%m-%d')
        # Get day of year (tm_yday is 1-indexed)
        doy = date_obj.timetuple().tm_yday
        doy_list.append(doy)
    return doy_list

# Initialize data lists
all_pr_data = []
all_wtda_data = [] 
all_wtd_data = [] 

all_vwc_data = []

all_dates_pr = []
all_dates_wtd = []
all_dates_vwc = []
all_doy_pr = []  
all_doy_wtd = []
all_doy_vwc = []

all_wtd_data = []

# Loop through all years and load data
for year in YEARS:
    print(f"Loading data for year {year}")
    
    # Load anomaly datasets for each year
    pr_anom = xr.open_dataset(f'{SOURCE_PATH}/input/pr/pra/anom_pr_DE05_{year}.nc')
    wtda_anom = xr.open_dataset(f'{SOURCE_PATH}/target/wtd/wtda/anom_wtd_DE05_{year}.nc')
    vwc_anom = xr.open_dataset(f'{SOURCE_PATH}/target/vwc/vwca/anom_vwc_DE05_{year}.nc')
    
    wtd_nc=xr.open_dataset(f'{SOURCE_PATH}/target/wtd/wtd_DE05_{year}.nc')
    
    # Extract dates as strings
    dates_pr = [str(d)[:10] for d in pr_anom['time'].values]  # Get 'YYYY-MM-DD' part
    dates_wtd = [str(d)[:10] for d in wtda_anom['time'].values]
    dates_vwc = [str(d)[:10] for d in vwc_anom['time'].values]
    
    # Extract data with spatial subset and stride
    pr_data = pr_anom['pr'].values[:, xmin:xmax:stride, xmin:xmax:stride]
    wtda_data = wtda_anom['wtd'].values[:, xmin:xmax:stride, xmin:xmax:stride]
    vwc_data = vwc_anom['vwc'].values[:, xmin:xmax:stride, xmin:xmax:stride]
    
    wtd_data = wtd_nc['wtd'].values[:, xmin:xmax:stride, xmin:xmax:stride]

    
    # Append to lists
    all_pr_data.append(pr_data)
    all_wtda_data.append(wtda_data)
    all_wtd_data.append(wtd_data)
    all_vwc_data.append(vwc_data)
    
    all_dates_pr.extend(dates_pr)
    all_dates_wtd.extend(dates_wtd)
    all_dates_vwc.extend(dates_vwc)
    
    # Calculate day-of-year using non-leap year function
    all_doy_pr.extend(get_doy_non_leap_vector(dates_pr))
    all_doy_wtd.extend(get_doy_non_leap_vector(dates_wtd))
    all_doy_vwc.extend(get_doy_non_leap_vector(dates_vwc))
    
    # Close datasets
    pr_anom.close()
    wtda_anom.close()
    vwc_anom.close()
    wtd_nc.close()

#%%

dates_pr_strings = all_dates_pr
dates_wtd_strings = all_dates_wtd
dates_vwc_strings = all_dates_vwc

common_dates_set = set(dates_pr_strings) & set(dates_wtd_strings) & set(dates_vwc_strings) 
common_dates = sorted(list(common_dates_set))

print(f"Total dates - precip: {len(dates_pr_strings)}, wtd: {len(dates_wtd_strings)}, vwc: {len(dates_vwc_strings)}")
print(f"Common dates across all datasets: {len(common_dates)}")

if len(common_dates) == 0:
    raise ValueError("No common dates found between datasets!")

indices_pr = [i for i, d in enumerate(dates_pr_strings) if d in common_dates]
indices_wtd = [i for i, d in enumerate(dates_wtd_strings) if d in common_dates]
indices_wtda = [i for i, d in enumerate(dates_wtd_strings) if d in common_dates]
indices_vwc = [i for i, d in enumerate(dates_vwc_strings) if d in common_dates]

print(f"Indices to extract - precip: {len(indices_pr)}, wtd: {len(indices_wtd)}, vwc: {len(indices_vwc)}")

print("Concatenating data across all years...")
concatenated_pr = np.concatenate(all_pr_data, axis=0)
concatenated_wtda = np.concatenate(all_wtda_data, axis=0)
concatenated_wtd = np.concatenate(all_wtd_data, axis=0)

concatenated_vwc = np.concatenate(all_vwc_data, axis=0)

print(f"Original shapes - precip: {concatenated_pr.shape}, wtd: {concatenated_wtd.shape}, vwc: {concatenated_vwc.shape}")

# Extract common time steps
concatenated_wtda_common = concatenated_wtda[indices_wtda, :, :]
concatenated_wtd_common = concatenated_wtd[indices_wtd, :, :]
concatenated_pr_common = concatenated_pr[indices_pr, :, :]
concatenated_vwc_common = concatenated_vwc[indices_vwc, :, :]


print(f"Common shapes - precip: {concatenated_pr_common.shape}, wtd: {concatenated_wtd_common.shape}, vwc: {concatenated_vwc_common.shape}")

# Prepare static data
static_data = np.stack([topo_data, slopex_data,slopey_data,soiltype_data,clyppt_data,sndppt_data])

# Get DOY values for common dates
common_doy_pr = [all_doy_pr[i] for i in indices_pr]
common_doy_wtd = [all_doy_wtd[i] for i in indices_wtd]
common_doy_vwc = [all_doy_vwc[i] for i in indices_vwc]

# Verify DOY arrays are consistent
if not (np.array_equal(common_doy_pr, common_doy_wtd) and 
        np.array_equal(common_doy_pr, common_doy_vwc)):
    print("Warning: DOY arrays don't match exactly, using precipitation DOY")
doy_array = np.array(common_doy_pr)

# Check DOY range
print(f"\nDOY range in data: {doy_array.min()} to {doy_array.max()}")

# Process DOY for climatology indexing
print("Making DOY 0-indexed for array indexing...")

# Test July 7th to verify
july_7_dates = [d for d in common_dates if d.endswith('-07-07')]
if july_7_dates:
    july_7_doy = get_doy_non_leap_vector(july_7_dates[:1])[0]
    print(f"Test: July 7th DOY = {july_7_doy} (should be 188)")

# Convert to 0-indexed for array indexing
if mean_clipped.shape[0] == 366:
    print("Climatology has 366 days - includes leap day")
    # For 366-day climatology, simple -1 works
    doy_array_0idx = doy_array - 1
    
elif mean_clipped.shape[0] == 365:
    print("Climatology has 365 days - no leap day")
    doy_array_0idx = doy_array - 1
else:
    print(f"Climatology has {mean_clipped.shape[0]} days")
    doy_array_0idx = doy_array - 1

# Verify DOY indices are within bounds
print(f"DOY 0-indexed range: {doy_array_0idx.min()} to {doy_array_0idx.max()}")
if doy_array_0idx.min() < 0 or doy_array_0idx.max() >= mean_clipped.shape[0]:
    print(f"Warning: DOY indices out of bounds for climatology (0-{mean_clipped.shape[0]-1})")
    doy_array_0idx = np.clip(doy_array_0idx, 0, mean_clipped.shape[0]-1)
    print(f"Clipped DOY to range: {doy_array_0idx.min()} to {doy_array_0idx.max()}")

# Calculate mask based on wtd.mean(axis=0) using common time steps
print("\nCalculating mask based on wtd.mean(axis=0)...")
wtd_temporal_mean = concatenated_wtd_common.mean(axis=0)

# Create mask: 1 for valid pixels, 0 for invalid pixels
mask = np.ones_like(wtd_temporal_mean, dtype=np.int8)
mask[np.isnan(wtd_temporal_mean)] = 0
mask[np.isinf(wtd_temporal_mean)] = 0

print(f"Valid pixels: {np.sum(mask)} out of {mask.size} ({np.sum(mask)/mask.size*100:.2f}%)")

# Calculate the new grid dimensions
original_size = xmax - xmin
new_size = (original_size + stride - 1) // stride

# Save to HDF5
output_path = f'input/concatenated_WTDa_Parflow_{YEARS[0]}_{YEARS[-1]}_{xmin}_{xmax}_stride{stride}.h5'
print(f"\nSaving to {output_path}")

os.makedirs('input', exist_ok=True)

with h5py.File(output_path, 'w') as f:
    # Store dimensions as attributes
    f.attrs['n_time'] = len(common_dates)
    f.attrs['n_lat'] = concatenated_pr_common.shape[1]
    f.attrs['n_lon'] = concatenated_pr_common.shape[2]
    f.attrs['n_days_climatology'] = mean_clipped.shape[0]
    f.attrs['climatology_has_leap_day'] = mean_clipped.shape[0] == 366
    f.attrs['stride'] = stride
    f.attrs['original_grid_size'] = original_size
    f.attrs['reduced_grid_size'] = new_size
    
    # Save dynamic data
    f.create_dataset('wtd', data=concatenated_wtd_common, compression='gzip', compression_opts=9)
    f.create_dataset('wtda', data=concatenated_wtda_common, compression='gzip', compression_opts=9)

    f.create_dataset('precip', data=concatenated_pr_common, compression='gzip', compression_opts=9)
    f.create_dataset('vwc', data=concatenated_vwc_common, compression='gzip', compression_opts=9)
    
    # Save static data
    f.create_dataset('static', data=static_data, compression='gzip', compression_opts=9)
    
    # Save climatology data
    f.create_dataset('wtd_mean_climatology', data=mean_clipped.values, compression='gzip', compression_opts=9)
    f.create_dataset('wtd_std_climatology', data=std_clipped.values, compression='gzip', compression_opts=9)
    
    # Save other data
    f.create_dataset('doy', data=doy_array_0idx, compression='gzip', compression_opts=9)
    f.create_dataset('mask', data=mask, compression='gzip', compression_opts=9)
    
    f.create_dataset('lat', data=rlat_coords, compression='gzip', compression_opts=9)
    f.create_dataset('lon', data=rlon_coords, compression='gzip', compression_opts=9)
    
    # Save original DOY (1-indexed) for reference
    f.create_dataset('doy_1indexed', data=doy_array, compression='gzip', compression_opts=9)
    
    # Save dates as strings
    dt = h5py.string_dtype(encoding='utf-8')
    f.create_dataset('dates', data=common_dates, dtype=dt)
    
    # Add metadata attributes for climatology datasets
    f['wtd_mean_climatology'].attrs['description'] = 'Mean wtd climatology (long-term average for each day of year)'
    f['wtd_std_climatology'].attrs['description'] = 'Standard deviation wtd climatology (long-term std for each day of year)'
    f['wtd_mean_climatology'].attrs['units'] = 'm'
    f['wtd_std_climatology'].attrs['units'] = 'm'
    f['wtd_mean_climatology'].attrs['n_days'] = mean_clipped.shape[0]
    f['wtd_std_climatology'].attrs['n_days'] = std_clipped.shape[0]
    
    # Add DOY metadata
    f['doy'].attrs['description'] = 'Day of year (0-indexed) for indexing climatology arrays'
    f['doy'].attrs['range'] = f'0 to {mean_clipped.shape[0]-1}'
    f['doy_1indexed'].attrs['description'] = 'Original day of year (1-indexed, non-leap year calendar)'
    f['doy_1indexed'].attrs['range'] = f'1 to 365'
    f['doy_1indexed'].attrs['note'] = 'Calculated assuming non-leap year calendar for all dates'
    
    # Add variable descriptions
    f['wtd'].attrs['description'] = 'Water table depth'
    f['wtda'].attrs['description'] = 'Water table depth anomalies'
    f['precip'].attrs['description'] = 'Precipitation anomalies'
    f['vwc'].attrs['description'] = 'Volumetric water content anomalies'
    f['static'].attrs['description'] = 'Static data: [0] topography, [1] slopex [2] slopey, [3] soil type, [4] CLYPPT, [5] SNDPPT'
    f['mask'].attrs['description'] = 'Land mask: 1=valid, 0=invalid'
    

print("Concatenation and saving completed successfully!")

# Verification
print(f"\nVerification:")
print(f"  wtd shape: {concatenated_wtd_common.shape}")
print(f"  precip shape: {concatenated_pr_common.shape}")
print(f"  vwc shape: {concatenated_vwc_common.shape}")
print(f"  static shape: {static_data.shape}")
print(f"  wtd_mean_climatology shape: {mean_clipped.shape}")
print(f"  wtd_std_climatology shape: {std_clipped.shape}")
print(f"  doy shape: {doy_array_0idx.shape}")
print(f"  mask shape: {mask.shape}")
print(f"  Number of dates: {len(common_dates)}")

print(f"\nTesting climatology indexing:")
print(f"  First 5 DOY values (0-indexed): {doy_array_0idx[:5]}")
print(f"  First 5 DOY values (1-indexed): {doy_array[:5]}")
print(f"  Sample dates: {common_dates[:5]}")

# Test specific dates
print(f"\nTesting specific dates:")
test_dates = [d for d in common_dates if '-07-07' in d or '-02-29' in d or '-01-01' in d or '-12-31' in d]
for date in test_dates[:10]:  # Show first 10
    idx = common_dates.index(date)
    print(f"  {date}: DOY = {doy_array[idx]}, 0-indexed = {doy_array_0idx[idx]}")

print(f"\nGrid information:")
print(f"  Original grid size: {original_size} x {original_size}")
print(f"  Reduced grid size with stride {stride}: {new_size} x {new_size}")
print(f"  Memory reduction: {(1 - (new_size**2)/(original_size**2))*100:.1f}%")

# Close datasets
topo_ds.close()
mean_clim.close()
std_clim.close()

print(f"\nData successfully saved to {output_path}")
# %%