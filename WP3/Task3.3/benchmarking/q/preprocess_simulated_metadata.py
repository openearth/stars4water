#%%
import pathlib as pl
import pandas as pd

observation_directory = pl.Path("../saves/observations")
simulation_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations")
simulation_mya_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations_mya")
output_directory = pl.Path("../saves/simulations/")

extents = {
    "europe": [-11, 33, 42, 73],
    "rhine": [3, 46, 13, 53],
    "drammen": [7,59,11, 62],
    "duoro":[-9,40,-2,44],
    "seine":[0,47,6,51],
    "danube":[8,41,30,52],
    "crete":[23,34,27,36],
    "eastanglia":[-1,51,2,54]
}


correction_distance = 0.25 # degree

simulation_patterns = [dir.stem for dir in simulation_directory.iterdir() if dir.is_dir() and "geoframe" not in dir.stem] # GOEframe has its own benchmark
simulation_fields = [pattern.split("_") for pattern in simulation_patterns]
simulation_patterns = pd.DataFrame(data = simulation_fields, index = simulation_patterns)
simulation_patterns.columns = ["model", "meteo", "region", "resolution"]
simulation_patterns=simulation_patterns.dropna()

import pathlib as pl
import warnings
import numpy as np
import xarray as xr

simulation_upareas = {}

for region in extents:    

    print("Region: {}".format(region))
    
    region_sel = simulation_patterns["region"] == region
    region_sel = np.logical_or(region_sel, simulation_patterns["region"] == "europe")
    region_patterns = simulation_patterns.loc[region_sel]
    
    simulation_upareas[region] = {}
    
    for pattern, row in region_patterns.iterrows():
            
        pattern_directory = pl.Path("{}/{}".format(simulation_directory, pattern))
        discharge_files = np.array([file for file in pattern_directory.iterdir() if file.is_file() and file.stem.split("_")[2] == "q"])
        discharge_files = np.sort(discharge_files)
        if len(discharge_files) <= 0:
            continue
        
        print("\tPattern: {}".format(pattern))
            
        pattern_directory = pl.Path("{}/{}".format(simulation_directory, pattern))
        uparea_files = np.array([file for file in pattern_directory.iterdir() if file.is_file() and file.stem.split("_")[2] == "uparea"])
        uparea_files = np.sort(uparea_files)
        if len(uparea_files) <= 0:
            print("\t- No upstream area file, skipping...")
            continue
        uparea_file = uparea_files[0]
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=xr.SerializationWarning)
            
            with xr.open_dataset(uparea_file) as dataset:
                uparea = dataset.data_vars["uparea"]
        
        if uparea.attrs["units"] == "km2":
            uparea.attrs["units"] = "m2"
            uparea *= 1e6
        
        simulation_upareas[region][pattern] = uparea
        
        print("\t- Upstream area: {} m2".format(np.nanmean(uparea)))
        

import pathlib as pl
import warnings
import numpy as np
import xarray as xr

simulation_averages = {}

for region in extents:
    print("Region: {}".format(region))
    
    region_sel = simulation_patterns["region"] == region
    region_sel = np.logical_or(region_sel, simulation_patterns["region"] == "europe")
    region_patterns = simulation_patterns.loc[region_sel]
    
    simulation_averages[region] = {}
    
    for pattern, row in region_patterns.iterrows():
            
        pattern_directory = pl.Path("{}/{}".format(simulation_directory, pattern))
        discharge_files = np.array([file for file in pattern_directory.iterdir() if file.is_file() and file.stem.split("_")[2] == "q"])
        discharge_files = np.sort(discharge_files)
        if len(discharge_files) <= 0:
            continue
        
        print("\tPattern: {}".format(pattern))
            
        pattern_directory = pl.Path("{}/{}".format(simulation_mya_directory, pattern))
        average_files = np.array([file for file in pattern_directory.iterdir() if file.is_file() and file.stem.split("_")[2] == "q"])
        average_files = np.sort(average_files)
        average_file = average_files[0]
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=xr.SerializationWarning)
            
            with xr.open_dataset(average_file) as dataset:
                average = dataset.data_vars["q"]
        
        simulation_averages[region][pattern] = average
        
        print("\t- average: {} ms-1".format(np.nanmean(average)))
        
from typing import Optional
import xarray as xr
import numpy as np

def corrected_coordinates(simulated_values: xr.DataArray,
                          observed_lat: float,
                          observed_lon: float,
                          observed_value: float,
                          difference_range: Optional[float] = None,
                          distance: Optional[float] = None) -> xr.DataArray:
    
    nearby_values= simulated_values
    
    if distance is not None:
        lon_slice = slice(observed_lon - distance, observed_lon + distance)
        lat_slice = slice(observed_lat - distance, observed_lat + distance)
        
        if simulated_values.lon[0] > simulated_values.lon[-1]:
            lon_slice = slice(observed_lon + distance, observed_lon - distance)
        if simulated_values.lat[0] > simulated_values.lat[-1]:
            lat_slice = slice(observed_lat + distance, observed_lat - distance)
        
        nearby_values = simulated_values.sel(lon = lon_slice, lat = lat_slice)
        
    values_diff = np.abs(nearby_values - observed_value)
    
    if difference_range is None:
        values_min = values_diff.where(values_diff == values_diff.min(), drop = True)
    else:
        values_min = values_diff.where(values_diff <= observed_value * difference_range, drop = True)
    
    return simulated_values.sel(lon = values_min.coords["lon"],
                                lat = values_min.coords["lat"])


import pathlib as pl
import warnings
import numpy as np
import xarray as xr

for region in extents:
    
    if region=='crete':
        continue
    
    print("Region: {}".format(region))

    extent = extents[region]
    region_simulation_upareas = simulation_upareas[region]
    region_simulation_averages = simulation_averages[region]

    meta_file = pl.Path(
        "{}/{}/meta.parquet".format(observation_directory, region))
    meta = pd.read_parquet(meta_file)

    for pattern in region_simulation_upareas.keys():

        uparea = region_simulation_upareas[pattern]
        average = region_simulation_averages[pattern]

        pattern_directory = pl.Path(
            "{}/{}".format(simulation_directory, pattern))
        discharge_files = np.array([file for file in pattern_directory.iterdir(
        ) if file.is_file() and file.stem.split("_")[2] == "q"])
        discharge_files = np.sort(discharge_files)
        if len(discharge_files) == 0:
            continue
        discharge_file = discharge_files[0]

        print("\tPattern: {}".format(pattern))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=xr.SerializationWarning)
            with xr.open_dataset(discharge_file) as dataset:
                lat_slice = slice(extent[1], extent[3])
                lon_slice = slice(extent[0], extent[2])

                if dataset.lat.values[0] > dataset.lat.values[-1]:
                    lat_slice = slice(extent[3], extent[1])
                if dataset.lon.values[0] > dataset.lon.values[-1]:
                    lon_slice = slice(extent[2], extent[0])

                dataset = dataset.sel(lat=lat_slice,
                                      lon=lon_slice).isel(time=0)

                discharge = dataset.data_vars["q"]

        discharge_mask = np.isnan(discharge)
        uparea_mask = np.isnan(uparea)
        average_mask = np.isnan(average)

        # Mask the upstream area based on the discharge
        # Needed for wflowsbm that only supplies discharge in the main river stems
        if discharge_mask.sum() > uparea_mask.sum():
            mask = np.logical_or(np.logical_or(uparea_mask,
                                               discharge_mask),
                                 average_mask).squeeze()
            uparea = uparea.where(~mask)
            average = average.where(~mask)

        pattern_meta = meta.copy()
        pattern_meta["simulated_lat"] = np.nan
        pattern_meta["simulated_lon"] = np.nan
        pattern_meta["simulated_area"] = np.nan
        pattern_meta["simulated_average"] = np.nan
        pattern_meta["corrected_simulated_lat"] = np.nan
        pattern_meta["corrected_simulated_lon"] = np.nan
        pattern_meta["corrected_simulated_area"] = np.nan
        pattern_meta["corrected_simulated_average"] = np.nan

        resolution = uparea.coords["lon"].values[1] - \
            uparea.coords["lon"].values[0]

        for index, row in pattern_meta.iterrows():

            try:
                uncorr_uparea = uparea.sel(lon=row["lon"],
                                           lat=row["lat"],
                                           method="nearest")
            except KeyError:
                raise ValueError("Gauge not in uparea?")

            try:
                uncorr_average = average.sel(lon=row["lon"],
                                             lat=row["lat"],
                                             method="nearest")
            except KeyError:
                raise ValueError("Gauge not in average?")

            pattern_meta.at[index, "simulated_lat"] = uncorr_uparea.coords["lat"]
            pattern_meta.at[index, "simulated_lon"] = uncorr_uparea.coords["lon"]
            pattern_meta.at[index, "simulated_area"] = uncorr_uparea
            pattern_meta.at[index, "simulated_average"] = uncorr_average
            pattern_meta.at[index, "corrected_simulated_lat"] = uncorr_uparea.coords["lat"]
            pattern_meta.at[index, "corrected_simulated_lon"] = uncorr_uparea.coords["lon"]
            pattern_meta.at[index, "corrected_simulated_area"] = uncorr_uparea
            pattern_meta.at[index, "corrected_simulated_average"] = uncorr_average

            if not np.isnan(pattern_meta.at[index, "area"]):
                # Take the nearest averages where the nearest upstream areas difference is EITHER
                # 1. within 10% of the gauge upstream area OR
                # 2. the minimum difference
                
                uparea_sel = corrected_coordinates(simulated_values=uparea,
                                                   observed_lat=row["lat"],
                                                   observed_lon=row["lon"],
                                                   observed_value=row["area"] * 1e6,
                                                   difference_range=0.1,
                                                   distance=correction_distance)
                
                if uparea_sel.size <= 0:
                    uparea_sel = corrected_coordinates(simulated_values=uparea,
                                                       observed_lat=row["lat"],
                                                       observed_lon=row["lon"],
                                                       observed_value=row["area"] * 1e6,
                                                       distance=correction_distance)

                # nearby_average = average.sel(lon=uparea_sel.coords["lon"],
                #                              lat=uparea_sel.coords["lat"],
                #                              method="nearest")

            #elif not np.isnan(pattern_meta.at[index, "average"]):
                # Simply take the nearest averages
    
                lon_slice = slice(row["lon"] - correction_distance, row["lon"] + correction_distance)
                lat_slice = slice(row["lat"] - correction_distance, row["lat"] + correction_distance)
                
                if average.lon[0] > average.lon[-1]:
                    lon_slice = slice(row["lon"] + correction_distance, row["lon"] - correction_distance)
                if average.lat[0] > average.lat[-1]:
                    lat_slice = slice(row["lat"] + correction_distance, row["lat"] - correction_distance)
                
                nearby_average = average.sel(lon=lon_slice, lat=lat_slice)

            else:
                raise ValueError("No upstream area or average?")

            corr_min = corrected_coordinates(simulated_values=nearby_average,
                                             observed_lat=row["lat"],
                                             observed_lon=row["lon"],
                                             observed_value=row["average"])
            corr_min = corr_min.sel(lon = row["lon"],
                                    lat = row["lat"],
                                    method = "nearest")

            corr_uparea = uparea.sel(lon=corr_min.coords["lon"],
                                     lat=corr_min.coords["lat"],
                                     method="nearest")
            corr_average = average.sel(lon=corr_min.coords["lon"],
                                       lat=corr_min.coords["lat"],
                                       method="nearest")

            pattern_meta.at[index, "corrected_simulated_lat"] = corr_uparea.coords["lat"]
            pattern_meta.at[index, "corrected_simulated_lon"] = corr_uparea.coords["lon"]
            pattern_meta.at[index, "corrected_simulated_area"] = corr_uparea
            pattern_meta.at[index, "corrected_simulated_average"] = corr_average

        if pattern_meta.index.size == 0:
            continue

        pattern_meta = pattern_meta.astype({"simulated_lat": "float64",
                                            "simulated_lon": "float64",
                                            "simulated_area": "float32",
                                            "corrected_simulated_lat": "float64",
                                            "corrected_simulated_lon": "float64",
                                            "corrected_simulated_area": "float32", })

        meta_out = pl.Path( "{}/{}/{}/meta.parquet".format(output_directory, region, pattern))
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        pattern_meta.to_parquet(meta_out)

        meta_sel = pattern_meta["simulated_area"] != pattern_meta["corrected_simulated_area"]
        corrected_meta = pattern_meta.loc[meta_sel]

        print("\t- Saved {} ({} corrected) gauges".format(pattern_meta.index.size,
                                                          corrected_meta.index.size))
