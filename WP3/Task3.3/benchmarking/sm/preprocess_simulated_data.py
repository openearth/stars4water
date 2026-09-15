#%%
import pathlib as pl
import pandas as pd
import warnings
import datetime as dt
import pandas as pd
import numpy as np 
import xarray as xr

def convert_dates(time) -> list[dt.date]:
    if type(time) is not pd.DatetimeIndex:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            dates = pd.to_datetime([d for d in time.to_datetimeindex()])
    else:
        dates = pd.to_datetime(time)
    dates = pd.to_datetime(dates.date) # convert to daily
    return dates

#simulation_directory = pl.Path("/home/l.avila/DATA/Stars4Water/data/simulations")
simulation_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations")

output_directory = pl.Path("../saves/simulations/")

regions = [dir.stem for dir in output_directory.iterdir() if dir.is_dir()]

for region in regions:
    
    print(region)
    
    patterns_files=pl.Path('{}/{}'.format(output_directory,region))
    patterns = [dir.stem for dir in patterns_files.iterdir() if dir.is_dir()][::-1]

    for pattern in patterns:            
    
        meta=pd.read_csv("../saves/simulations/{}/{}/meta.csv".format(region,pattern),index_col=0)
        print('\t',pattern)

        pattern_directory = pl.Path("{}/{}".format(simulation_directory, pattern))

        moisture_files = np.array([file for file in pattern_directory.iterdir() if file.is_file() and file.stem.split("_")[2] == "sm"])
        moisture_files = np.sort(moisture_files)

        moisture_file = moisture_files[0]

        start_date = pd.to_datetime(meta["start-year"])
        end_date = pd.to_datetime(meta["end-year"])

        lons = meta["simulated_lon"]
        lats = meta["simulated_lat"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=xr.SerializationWarning)
            dataset = xr.open_dataset(moisture_file)
            
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
            
            moisture_out = pl.Path("../saves/simulations/{}/{}/data/moisture_{}.parquet".format(region, pattern,index))
            if not moisture_out.exists():
                exists = False
                break
        if exists:
            print("\t\t- Already exists")
            continue

        moistures = []
        start_depths_list = []
        end_depths_list = []
        dates_list = []

        for i, moisture_file in enumerate(moisture_files):
            
            print("\t> File: {} ({} out of {})".format(moisture_file.stem, i, len(moisture_files)))
            
            with xr.open_dataset(moisture_file) as dataset:
                    sm = dataset["sm"]
                            
            dates = convert_dates(sm.indexes["time"])
            
            if 'parflowclm' in pattern:
                
                start_depth = 0
                end_depth = 0.05
                
                try:
                    moistures.append(sm.values[:, lat_indices, lon_indices])
                except:
                    moistures.append(sm.values[:,0, lat_indices, lon_indices])
                    
                
                start_depths_list.append(np.repeat(start_depth, len(dates)))
                end_depths_list.append(np.repeat(end_depth, len(dates)))
                dates_list.append(dates)
                
            elif 'lisflood' in pattern:
                
                start_depth = 0
                end_depth = 0.05
                
                moistures.append(sm.values[:, 0, lat_indices, lon_indices])
                
                start_depths_list.append(np.repeat(start_depth, len(dates)))
                end_depths_list.append(np.repeat(end_depth, len(dates)))
                dates_list.append(dates)
                
            else:
                
                depths = dataset["depth"].values
                depths = np.concatenate((np.array([0.0]), np.cumsum(depths)))
                
                for d in range(depths.size - 1):
                    
                    start_depth = depths[d]
                    end_depth = depths[d + 1]
                    
                    moistures.append(sm.values[:, d, lat_indices, lon_indices])
                    
                    start_depths_list.append(np.repeat(start_depth, len(dates)))
                    end_depths_list.append(np.repeat(end_depth, len(dates)))
                    dates_list.append(dates)

        moisture = np.concatenate(moistures, axis = 0)

        start_depths = np.concatenate(start_depths_list, axis = 0)
        end_depths = np.concatenate(end_depths_list, axis = 0)
        dates = np.concatenate(dates_list, axis = 0)

        for i, (index, row) in enumerate(meta.iterrows()):            
            
                lon = row["simulated_lon"]
                lat = row["simulated_lat"]
                
                moisture_df = {"date": dates,
                                "start-depth": start_depths,
                                "end-depth": end_depths,
                                "moisture": moisture[:, i]}
                moisture_df = pd.DataFrame(moisture_df)
                moisture_df = moisture_df.astype({"date": "datetime64[ns]",
                                                    "start-depth": "float32",
                                                    "end-depth": "float32",
                                                    "moisture": "float32"})
                moisture_df = moisture_df.sort_values("date")
                
                moisture_out = pl.Path("../saves/simulations/{}/{}/data/moisture_{}.parquet".format(region, pattern,index))
                moisture_out.parent.mkdir(parents=True, exist_ok=True)
                moisture_df.to_parquet(moisture_out)
                
                    
        print("\t- Saved {} gauges from {} simulation files".format(meta.index.size,
                                                                    len(moistures)))
