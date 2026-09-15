#%%
import pathlib as pl

save_directory = pl.Path("saves/simulations")
simulation_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations")
common_directory = pl.Path("saves/common")
output_directory = pl.Path("saves/simulations")


import warnings
import datetime as dt
import pandas as pd

def convert_dates(time) -> list[dt.date]:
    if not type(time) is pd.DatetimeIndex:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            dates = [date.date() for date in time.to_datetimeindex()]
    else:
        dates = [date.date() for date in time]
    return dates



import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import warnings

regions = [dir.stem for dir in save_directory.iterdir() if dir.is_dir()]

for region in regions:
    
    
    print("Region: {}".format(region))
    
    region_directory = pl.Path("{}/{}".format(save_directory, region))
    common_region_directory = pl.Path("{}/{}".format(common_directory, region))
    
    period_file = pl.Path("{}/period.csv".format(common_region_directory))
    period = pd.read_csv(period_file, parse_dates=["start", "end"]).iloc[0]
    
    patterns = [dir.stem for dir in region_directory.iterdir() if dir.is_dir()]
    
    for pattern in patterns:
        print("\tPattern: {}".format(pattern))
                
        meta_file = pl.Path("{}/{}/{}/meta.parquet".format(save_directory, region, pattern))
        meta = pd.read_parquet(meta_file)
        
        start_date = period["start"].date()
        end_date = period["end"].date()
        lons = meta["corrected_simulated_lon"]
        lats = meta["corrected_simulated_lat"]
            
        pattern_directory = pl.Path("{}/{}".format(simulation_directory, pattern))
        discharge_files = np.array([file for file in pattern_directory.iterdir() if file.is_file() and file.stem.split("_")[2] == "q"])
        discharge_files = np.sort(discharge_files)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=xr.SerializationWarning)
            dataset = xr.open_dataset(discharge_files[0])
        lon_indices = []
        for lon in lons:
            lon_diff = np.abs(dataset.coords["lon"].values - lon)
            lon_index = np.where(lon_diff == np.min(lon_diff))[0][0]
            lon_indices.append(lon_index)
        lat_indices = []
        for lat in lats:
            lat_diff = np.abs(dataset.coords["lat"].values - lat)
            lat_index = np.where(lat_diff == np.min(lat_diff))[0][0]
            lat_indices.append(lat_index)
        dataset.close()
        
        exists = True
        for i, (index, row) in enumerate(meta.iterrows()):
            discharge_out = pl.Path("{}/{}/{}/data/discharge_{}.parquet".format(output_directory, region, pattern, index))
            if not discharge_out.exists():
                exists = False
                break
        if exists:
            print("\t- Already exists")
            continue
        
        discharges = []
        dates_list = []
        for i, discharge_file in enumerate(discharge_files):
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=xr.SerializationWarning)
                
                with xr.open_dataset(discharge_file) as dataset:
                    dates = convert_dates(dataset.indexes["time"])
            
            if max(dates) < start_date or min(dates) > end_date:
                continue
            
            print("\t> File: {} ({} out of {})".format(discharge_file.stem, i, len(discharge_files)))
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=xr.SerializationWarning)
                
                with xr.open_dataset(discharge_file) as dataset:
                    date_slice = slice(str(start_date), str(end_date))
                    dataset = dataset.sel(time=date_slice)
                    dataset = dataset.sortby("time")
                    discharge = dataset["q"]
            
            discharges.append(discharge.values[:, lat_indices, lon_indices])
            dates = convert_dates(discharge.indexes["time"])
            dates_list.append(dates)
        
        discharge = np.concatenate(discharges, axis = 0)
        dates = np.concatenate(dates_list, axis = 0)
        
        for i, (index, row) in enumerate(meta.iterrows()):
            discharge_df = {"date": dates,
                            "discharge": discharge[:, i]}
            discharge_df = pd.DataFrame(discharge_df)
            discharge_df = discharge_df.astype({"date": "object",
                                                "discharge": "float32"})
            discharge_df = discharge_df.sort_values("date")
            
            discharge_out = pl.Path("{}/{}/{}/data/discharge_{}.parquet".format(output_directory, region, pattern, index))
            discharge_out.parent.mkdir(parents=True, exist_ok=True)
            discharge_df.index=pd.to_datetime(discharge_df['date'])
            discharge_df=pd.DataFrame(discharge_df['discharge'])

            discharge_df.to_parquet(discharge_out)
            
                
        print("\t- Saved {} gauges from {} simulation files".format(meta.index.size,
                                                                  len(discharges)))