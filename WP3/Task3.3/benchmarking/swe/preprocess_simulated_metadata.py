#%%
import pathlib as pl
import warnings
import numpy as np
import xarray as xr
import glob 
import pandas as pd 
import sys 
import os 

simulation_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations")
mya_path=pl.Path('/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations_mya')
save_directory = pl.Path("../saves/observations/")

simulation_patterns = [dir.stem for dir in simulation_directory.iterdir() if dir.is_dir()] # GOEframe has its own benchmark and parflowclm_hres has too few simulation dates
simulation_fields = [pattern.split("_") for pattern in simulation_patterns]
simulation_patterns = pd.DataFrame(data = simulation_fields, index = simulation_patterns)
simulation_patterns.columns = ["model", "meteo", "region", "resolution"]

regions = [dir.stem for dir in save_directory.iterdir() if dir.is_dir()]

for region in regions:
    
    print(region)

    meta = pd.read_parquet("../saves/observations/{}/meta.parquet".format(region))

    patterns=simulation_patterns.index 
    
    for pattern in patterns:
        
        if 'seNorge' in pattern and region!='Drammen':
            continue
        
        pattern_directory = pl.Path("{}/{}".format(simulation_directory, pattern))
        
        swe_files = np.array([file for file in pattern_directory.iterdir() if file.is_file() and file.stem.split("_")[2] == "swe"])
        swe_files = np.sort(swe_files)
        
        if len(swe_files)==0:
            continue
        
        pattern_meta = meta.copy()
        
        swe_file = swe_files[0]
        
        dataset=xr.open_dataset(swe_files[0])['swe'].isel(time=0).drop_vars(["time"])
        resolution = dataset["lon"][1] - dataset["lon"][0]
        tol=0.2
        
        simulation_mya_directory = pl.Path("{}/{}".format(mya_path,pattern))

        average_file = glob.glob('{}/*swe*'.format(simulation_mya_directory))[0]
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=xr.SerializationWarning)

        with xr.open_dataset(average_file) as dataset:
            average = dataset.data_vars["swe"]
            
        pattern_meta = meta.copy()
        pattern_meta["simulated_lat"] = np.nan
        pattern_meta["simulated_lon"] = np.nan
        pattern_meta["simulated_average"] = np.nan
        pattern_meta["corrected_simulated_lat"] = np.nan
        pattern_meta["corrected_simulated_lon"] = np.nan
        pattern_meta["corrected_simulated_average"] = np.nan

        resolution = average["lon"].values[1] - \
                    average["lon"].values[0]
                    
        correction_distance = 0.1 # degree
        
        for index,row in meta.iterrows():   
            
            obs_average=pattern_meta['average'][index]
            
            try:
     
                uncorr_average = average.sel(lon=row["lon"],
                                                        lat=row["lat"],
                                                        method="nearest",
                                                        tolerance=resolution / 2 + 1e-10)
            
                pattern_meta.at[index, "simulated_lat"] = uncorr_average.coords["lat"]
                pattern_meta.at[index, "simulated_lon"] = uncorr_average.coords["lon"]
                pattern_meta.at[index, "simulated_average"] = uncorr_average   
            
                
                lon_slice = slice(row["lon"] - correction_distance, row["lon"] + correction_distance)
                lat_slice = slice(row["lat"] + correction_distance, row["lat"] - correction_distance)
                
                if 'lisflood_obs' in pattern:
                    nearby_values = average.sel(lon=lon_slice, lat=lat_slice)
            
                else:
                    nearby_values = average.sel(lon=lon_slice, lat=lat_slice)
                
                diff_average=np.abs(nearby_values-obs_average)
                
                if np.isnan(diff_average).all()==True:
                    continue     
                
                corr_min = nearby_values.where(diff_average == diff_average.min(), drop = True)
                
                if len(corr_min['lon'])>1 or len(corr_min['lat'])>1 :
                    corr_min=corr_min[0,0]
                
                corr_average = average.sel(lon=corr_min.coords["lon"],
                                        lat=corr_min.coords["lat"],
                                        method="nearest")    
                
                pattern_meta.at[index, "corrected_simulated_lat"] = corr_average.coords["lat"]
                pattern_meta.at[index, "corrected_simulated_lon"] = corr_average.coords["lon"]
                pattern_meta.at[index, "corrected_simulated_average"] = corr_average
            except:
                pass
                
        if 'simulated_lon' in pattern_meta:
            
            pattern_meta = pattern_meta.dropna()

            meta_out = pl.Path("../saves/simulations/{}/{}/meta.csv".format(region,pattern))
            meta_out.parent.mkdir(parents=True, exist_ok=True)
            pattern_meta.to_csv(meta_out)
            
            print('\t',pattern)
        
