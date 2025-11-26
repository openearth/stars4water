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
        
        pattern_directory = pl.Path("{}/{}".format(simulation_directory, pattern))

        wtd_files = np.array([file for file in pattern_directory.iterdir() if file.is_file() and file.stem.split("_")[2] == "wtd"])
        wtd_files = np.sort(wtd_files)
        
        if len(wtd_files)==0:
            continue 
       
        pattern_meta = meta.copy()
        
        dataset=xr.open_dataset(wtd_files[0])['wtd'].isel(time=0).drop_vars(["time"])
        resolution = dataset["lon"][1] - dataset["lon"][0]
        tol=0.2

        pattern_meta = meta.copy()
        
        for index,row in meta.iterrows():   
            
            lon=row['lon']
            lat=row['lat']         
            
            if 'parflowclm' in pattern:
                
                try:
                    nearest = dataset.where((dataset.lon<=lon+tol) &
                                        (dataset.lon>=lon-tol) &
                                        (dataset.lat<=lat+tol) &
                                        (dataset.lat>=lat-tol),drop=True)


                    distance = np.sqrt((nearest.lon - lon)**2 + (nearest.lat - lat)**2)
                    valid_distances = xr.where(~np.isnan(nearest), distance, np.inf)

                    nearest_index = np.unravel_index(np.nanargmin(valid_distances.values), nearest.shape)

                    nearest_value = nearest.values[nearest_index]
                    
                    if np.isnan(nearest_value)==False:

                        nearest_coords = {dim: nearest[dim][i].item() for dim, i in zip(nearest.dims, nearest_index)}
                        
                        pattern_meta.at[index, "simulated_lat"] = nearest_coords['lat']
                        pattern_meta.at[index, "simulated_lon"] = nearest_coords['lon']   
                except:
                    pass     
            else: 
                 try:
                
                    moisture_point = dataset.sel(lon = row["lon"],
                                                            lat = row["lat"],
                                                            method = "nearest",tolerance=resolution / 2 + 1e-10)   

                    pattern_meta.at[index, "simulated_lat"] = moisture_point.coords["lat"]
                    pattern_meta.at[index, "simulated_lon"] = moisture_point.coords["lon"]
    
                 except:
                    pass

        if 'simulated_lon' in pattern_meta:
            
            pattern_meta = pattern_meta.dropna(subset=["simulated_lat", "simulated_lon"])

            meta_out = pl.Path("../saves/simulations/{}/{}/meta.csv".format(region,pattern))
            meta_out.parent.mkdir(parents=True, exist_ok=True)
            pattern_meta.to_csv(meta_out)
            
            print('\t',pattern)
        
      
    