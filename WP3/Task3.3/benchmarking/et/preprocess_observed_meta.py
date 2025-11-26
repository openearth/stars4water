#%%
import pandas as pd 
import numpy as np 
import geopandas as gpd
import matplotlib.pyplot as plt 
import pandas as pd 
from shapely.geometry import Point
from geopandas.geoseries import * 
import pathlib as pl

shapefile_path = "../../additional/watershed_shapes/Basins.shp"
shapes = gpd.read_file(shapefile_path)

meta = pd.read_parquet('/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/observations/et/meta.parquet')

extents = {"europe": [-11, 33, 42, 73]}

extent=extents["europe"]

for id in range(len(meta)):
    
    lon=meta['lon'].iloc[id]
    lat=meta['lat'].iloc[id]
    
    if lat<extent[1] or lat>extent[3] or lon<extent[0] or lon>extent[2]:
        continue
    
    p1=meta['lon'].iloc[id]
    p2=meta['lat'].iloc[id]

    p3 = Point(p1,p2)

    mask=shapes.intersects(p3)

    int=shapes[mask]['layer']

    if len(int)>0:
        meta.at[id,'region']=shapes[mask]['layer'].values[0]
    else:
        meta.at[id,'region']='Europe'
        
        
meta=meta.dropna(subset=['region'])

meta=meta.reset_index(drop=True)

meta_out = pl.Path("../saves/observations/meta_out.parquet")
meta_out.parent.mkdir(parents=True, exist_ok=True)

meta.to_parquet(meta_out)

meta_out = pl.Path("../saves/observations/meta_out.csv")
meta_out.parent.mkdir(parents=True, exist_ok=True)

meta.to_csv(meta_out)
