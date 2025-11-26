# snow_data.py
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch

SEQ_LEN = 60  
FORECAST_STEPS=7

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

def load_data(swe_path, precip_path, temp_path, topo_path):
    # Load datasets from provided file paths
    swe_ds = xr.open_dataset(swe_path)
    precip_ds = xr.open_dataset(precip_path)
    temp_ds = xr.open_dataset(temp_path)
    topo_ds = xr.open_dataset(topo_path)

    # Function to normalize time to 00:00 using replace
    def normalize_time(ds):
        # Convert to pandas DatetimeIndex
        time_index = ds['time'].to_index()
        # Replace hour and minute with 00:00
        normalized_time = time_index.map(lambda t: t.replace(hour=0, minute=0))
        ds['time'] = normalized_time
        return ds

    # Normalize time in all datasets
    swe_ds = normalize_time(swe_ds)
    precip_ds = normalize_time(precip_ds)
    temp_ds = normalize_time(temp_ds)

    # Get days (already normalized to 00:00)
    swe_days = swe_ds['time'].values.astype('datetime64[D]')
    precip_days = precip_ds['time'].values.astype('datetime64[D]')
    temp_days = temp_ds['time'].values.astype('datetime64[D]')

    # Find common days
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
    times = swe_ds['time'].values  # Now aligned by day at 00:00

    # Prepare static data (topography + latitude grid)
    lat_grid = np.meshgrid(swe_ds['lat'].values, swe_ds['lon'].values)[0]
    static = np.stack([topo_ds['dem_mean'].values.T, lat_grid])

    return swe, precip, temp, static, times

def denormalize_swe(normalized_swe, stats):
    """Convert normalized SWE back to real SWE."""
    log_swe = normalized_swe * (stats['swe_max'] - stats['swe_min']) + stats['swe_min']
    swe = np.expm1(log_swe)  # Undo log1p transformation
    return swe