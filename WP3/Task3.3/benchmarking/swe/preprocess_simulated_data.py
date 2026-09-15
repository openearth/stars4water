#%%
import xarray as xr 
import glob 
import numpy as np 
import pandas as pd 
import datetime as dt
#import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")
import sys
import os 
import pathlib as pl

def convert_dates(time) -> list[dt.date]:
    if type(time) is not pd.DatetimeIndex:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            dates = pd.to_datetime([d for d in time.to_datetimeindex()])
    else:
        dates = pd.to_datetime(time)
    dates = pd.to_datetime(dates.date) # convert to daily
    return dates


save_directory = pl.Path("../saves/observations/")
simulation_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations")
output_directory = pl.Path("../saves/simulations/")

simulation_patterns = [dir.stem for dir in simulation_directory.iterdir() if dir.is_dir()] # GOEframe has its own benchmark and parflowclm_hres has too few simulation dates
simulation_fields = [pattern.split("_") for pattern in simulation_patterns]
simulation_patterns = pd.DataFrame(data = simulation_fields, index = simulation_patterns)
simulation_patterns.columns = ["model", "meteo", "region", "resolution"]

regions = [dir.stem for dir in save_directory.iterdir() if dir.is_dir()]

for region in regions:
    
    print(region)
    
    region_path=pl.Path('{}/{}'.format(output_directory,region))
    
    patterns = [dir.stem for dir in region_path.iterdir() if dir.is_dir()]

    for pattern in patterns:
        
        print('\t',pattern)
        
        meta_file = pl.Path("../saves/simulations/{}/{}/meta.csv".format(region,pattern))
        meta = pd.read_csv(meta_file,index_col=0)

        lons = meta["corrected_simulated_lon"].values
        lats = meta["corrected_simulated_lat"].values
        
        pattern_directory = pl.Path("{}/{}".format(simulation_directory, pattern))

        swe_files = np.array([file for file in pattern_directory.iterdir() if file.is_file() and file.stem.split("_")[2] == "swe"])
        swe_files = np.sort(swe_files)
        
        if len(swe_files) == 0:
             continue
        
        swe_file = swe_files[0]
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=xr.SerializationWarning)
            dataset = xr.open_dataset(swe_file)
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
            swe_out = pl.Path("{}/{}/{}/data/swe_{}.parquet".format(output_directory, region, pattern, index))
            if not swe_out.exists():
                exists = False
                break
            
        if exists:
            print("\t- Already exists")
            continue
        
        swes = []
        dates_list = []
        for i, swe_file in enumerate(swe_files):
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=xr.SerializationWarning)
                
                with xr.open_dataset(swe_file) as dataset:
                    dates = convert_dates(dataset.indexes["time"])
            
            #if max(dates) < start_date or min(dates) > end_date:
            #    continue
            
            print("\t> File: {} ({} out of {})".format(swe_file.stem, i, len(swe_files)))
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=xr.SerializationWarning)
                
                with xr.open_dataset(swe_file) as dataset:
                    #date_slice = slice(str(start_date), str(end_date))
                    #dataset = dataset.sortby("time")
                    #dataset = dataset.sel(time=date_slice)
                    swe = dataset["swe"]
            
            if 'lisflood_obs' in pattern:
                swes.append(swe.values[:,0, lat_indices, lon_indices])
            else:
                swes.append(swe.values[:, lat_indices, lon_indices])
            
            dates = convert_dates(swe.indexes["time"])
            dates_list.append(dates)
        
        swe = np.concatenate(swes, axis = 0)
        dates = np.concatenate(dates_list, axis = 0)
        
        for i, (index, row) in enumerate(meta.iterrows()):
            swe_df = {"date": dates,
                              "swe": swe[:, i]}
            swe_df = pd.DataFrame(swe_df)
            swe_df = swe_df.astype({"date": "datetime64[ns]",
                                                    "swe": "float32"})
            swe_df = swe_df.sort_values("date")
            
            swe_out = pl.Path("{}/{}/{}/data/swe_{}.parquet".format(output_directory, region, pattern, index))
            swe_out.parent.mkdir(parents=True, exist_ok=True)
            swe_df.to_parquet(swe_out)
            
                
        print("\t- Saved {} gauges from {} simulation files".format(meta.index.size,
                                                                  len(swes)))
        
print('PREPROCESS COMPLETED')
        


