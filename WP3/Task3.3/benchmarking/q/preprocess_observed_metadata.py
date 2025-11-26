#%%
import pathlib as pl
import pandas as pd

observation_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/observations/q")
simulation_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations")
common_directory = pl.Path("saves/common/")
output_directory = pl.Path("saves/observations/")

minimum_period_overlap = 2 # years

extents = {
    "europe": [-11, 33, 42, 73],
    "rhine": [3, 46, 13, 53],
    "drammen": [7,59,11, 62],
    "duoro":[-9,40,-2,44],
    "seine":[0,47,6,51],
    "danube":[8,41,30,53],
    "crete":[23,34,27,36],
    "eastanglia":[-1,51,2,54]
}

simulation_patterns = [dir.stem for dir in simulation_directory.iterdir() if dir.is_dir() and "geoframe" not in dir.stem] # GOEframe has its own benchmark
simulation_fields = [pattern.split("_") for pattern in simulation_patterns]
simulation_patterns = pd.DataFrame(data = simulation_fields, index = simulation_patterns)
simulation_patterns.columns = ["model", "meteo", "region", "resolution"]
simulation_patterns=simulation_patterns.dropna()

#%%
import pandas as pd
import numpy as np
import xarray as xr

meta_file = pl.Path("{}/meta.parquet".format(observation_directory))
meta = pd.read_parquet(meta_file)

for region in extents:
    print("Region: {}".format(region))
    
    period_file = common_directory / region / "period.csv"
    mask_file = common_directory / region / "mask.nc"
    
    extent = extents[region]
    period = pd.read_csv(period_file, parse_dates = ["start", "end"])
    mask = xr.open_dataset(mask_file).mask
    
    region_meta = meta.copy()
    
    meta_lon_sel = np.logical_and(region_meta["lon"] >= extent[0], region_meta["lon"] <= extent[2])
    meta_lat_sel = np.logical_and(region_meta["lat"] >= extent[1], region_meta["lat"] <= extent[3])
    meta_sel = np.logical_and(meta_lon_sel, meta_lat_sel)
    print("> Removing {} guages outside of the region extent".format(np.sum(~meta_sel)))
    region_meta = region_meta.loc[meta_sel]
    
    start_date = period.at[0, "start"]
    end_date = period.at[0, "end"]
    meta_sel = np.logical_and(region_meta["start-year"] <= end_date.year - minimum_period_overlap,
                              region_meta["end-year"] >= start_date.year + minimum_period_overlap)
    print("> Removing {} stations outside of the region period".format(np.sum(~meta_sel)))
    region_meta = region_meta.loc[meta_sel]
    
    resolution = mask["lon"].values[1] - mask["lon"].values[0]

    excludes = []
    for index, row in region_meta.iterrows():
        try:
            include = mask.sel(lon = row["lon"],
                               lat = row["lat"],
                               method = "nearest",
                               tolerance = resolution / 2 + 1e-10)
            include = include.values
        except KeyError:
            include = False
            
        if not include:
            excludes.append(index)
    
    region_meta = region_meta.drop(excludes, axis = 0)
    region_meta = region_meta.sort_values("average", ascending = False)

    print("> Removing {} guages outside of the simulation domain".format(len(excludes)))
    
    if region_meta.index.size == 0:
        continue
    
    meta_out = pl.Path("{}/{}/meta.parquet".format(output_directory, region))
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    region_meta.to_parquet(meta_out)
    
    print("- Saved {} out of {} gauges".format(region_meta.index.size,
                                                 meta.index.size))