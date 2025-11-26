#%%
import pathlib as pl
import pandas as pd

save_directory = pl.Path("../saves/observations/")
simulation_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations")
output_directory = pl.Path("../saves/simulations/")

simulation_patterns = [dir.stem for dir in simulation_directory.iterdir() if dir.is_dir()] # GOEframe has its own benchmark and parflowclm_hres has too few simulation dates
simulation_fields = [pattern.split("_") for pattern in simulation_patterns]
simulation_patterns = pd.DataFrame(data = simulation_fields, index = simulation_patterns)
simulation_patterns.columns = ["model", "meteo", "region", "resolution"]

import warnings
import datetime as dt
import pandas as pd

def convert_dates(time) -> list[dt.date]:
    if type(time) is not pd.DatetimeIndex:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            dates = pd.to_datetime([d for d in time.to_datetimeindex()])
    else:
        dates = pd.to_datetime(time)
    dates = pd.to_datetime(dates.date) # convert to daily
    return dates


import numpy as np
import pandas as pd
import xarray as xr
import warnings

regions = [dir.stem for dir in save_directory.iterdir() if dir.is_dir()]


for region in regions:
    
    print(region)
    
    region_path=pl.Path('{}/{}'.format(output_directory,region))
    
    patterns = [dir.stem for dir in region_path.iterdir() if dir.is_dir()]

    for pattern in patterns:
        
        print('\t',pattern)
        
        meta_file = pl.Path("../saves/simulations/{}/{}/meta.csv".format(region,pattern))
        meta = pd.read_csv(meta_file,index_col=0)
   
        lons = meta["lon"].values
        lats = meta["lat"].values
        
        pattern_directory = pl.Path("{}/{}".format(simulation_directory, pattern))

        wtd_files = np.array([file for file in pattern_directory.iterdir() if file.is_file() and file.stem.split("_")[2] == "wtd"])
        wtd_files = np.sort(wtd_files)
        
        wtd_file = wtd_files[0]

        with xr.open_dataset(wtd_file) as dataset:            
            lon_indices = []            
            for lon in lons:                
                if np.isnan(lon):
                    continue                
                lon_diff = np.abs(dataset.coords["lon"].values - lon)
                lon_index = np.where(lon_diff == np.min(lon_diff))[0][0]
                lon_indices.append(lon_index)
                
            lat_indices = []
            
            for lat in lats:
                if np.isnan(lat):
                    continue
                
                lat_diff = np.abs(dataset.coords["lat"].values - lat)
                lat_index = np.where(lat_diff == np.min(lat_diff))[0][0]
                lat_indices.append(lat_index)

        swes = []
        dates_list = []


        for i, swe_file in enumerate(wtd_files):
            
            with xr.open_dataset(swe_file) as dataset:
                dates = convert_dates(dataset.indexes["time"])
                
            print("\t> File: {} out of {}".format(i, len(wtd_files)))
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=xr.SerializationWarning)
                
                with xr.open_dataset(swe_file) as dataset:
                    dataset = dataset.sortby("time")
                    swe = dataset["wtd"]
            
            if 'parflowclm' in pattern:
                swes.append(swe.values[:, lat_indices, lon_indices])
            else:
                swes.append(swe.values[:,0, lat_indices, lon_indices])
                
            dates = convert_dates(swe.indexes["time"])
            dates_list.append(dates)
            
        swe = np.concatenate(swes, axis = 0)
        dates = np.concatenate(dates_list, axis = 0)

        meta_out=pd.DataFrame()
        
        for i, (index, row) in enumerate(meta.iterrows()):
            
            #try:
                
            lon=meta['simulated_lon'].iloc[i]
            lat=meta['simulated_lat'].iloc[i]
            
            meta_out.at[i,'id']=int(index)
            meta_out.at[i,'lon']=lon
            meta_out.at[i,'lat']=lat
            
            swe_df = {"date": dates,"wtd_sim": swe[:, i]}
            swe_df = pd.DataFrame(swe_df)
            swe_df = swe_df.astype({"date": "datetime64[ns]",
                                                "wtd_sim": "float32"})
            swe_df = swe_df.sort_values("date")
            
            if region=='East_Anglia':            
                swe_df.index=pd.to_datetime(swe_df['date'])
                swe_df=swe_df.resample('M').mean()
                swe_df.index=swe_df.index.map(lambda t: t.replace(day=1,hour=0,minute=0))
            
                print(id)
            
            swe_out = pl.Path("../saves/simulations/{}/{}/data/wtd_{}.parquet".format(region,pattern,index))
            swe_out.parent.mkdir(parents=True, exist_ok=True)
            swe_df.to_parquet(swe_out)       
                    
            #print(swe_out,':',len(swe_df))
            #except:
            #    print('{} pass'.format(meta['id'].iloc[i]))
            #    pass
            
        #meta_out.to_csv('../saves/simulations/{}/{}/meta.csv'.format(region,pattern))
            

