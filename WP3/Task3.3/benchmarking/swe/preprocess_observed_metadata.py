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

meta = pd.read_csv('../observations/meta.csv')


for id in range(len(meta)):
    
    p1=meta['lon'].iloc[id]
    p2=meta['lat'].iloc[id]

    p3 = Point(p1,p2)

    mask=shapes.intersects(p3)

    int=shapes[mask]['layer']

    if len(int)>0:
        meta.at[id,'region']=shapes[mask]['layer'].values[0]
        meta=meta.reset_index(drop=True)

    else:
        meta.at[id,'region']='Europe'
        meta=meta.reset_index(drop=True)

meta=meta.dropna(subset=['region'])

meta=meta.iloc[:,1:]

meta_out = pl.Path("../saves/observations/meta.parquet")
meta_out.parent.mkdir(parents=True, exist_ok=True)

meta.to_parquet(meta_out)

meta_out = pl.Path("../saves/observations/meta.csv")
meta_out.parent.mkdir(parents=True, exist_ok=True)

meta.to_csv(meta_out)