#%%
from typing import Optional
import pandas as pd
import geopandas as gp
import plotnine as pn
from matplotlib import cm
import warnings 

def plot_performance_map(region,performance: pd.DataFrame,
                         title: Optional[str] = None,
                         shapefile: Optional[gp.GeoDataFrame] = None,
                         extent: Optional[list] = None) -> pn.ggplot:
    
    rainbow_colors = [cm.rainbow(i) for i in range(256)][::-1]  # Generate rainbow colors
    jet_colors = [cm.jet(i) for i in range(256)][::-1]
    
    ggplt = pn.ggplot()
    
    if region=='Europe':
        minsize=1
        maxsize=3
    else:
        minsize=1
        maxsize=5
    
    #performance['kge_r']=np.where(performance['kge_r']<0,0,performance['kge_r'])
    
    performance['kge']=np.where(performance['kge']<0,0,performance['kge'])
    
    if shapefile is not None:
        ggplt += pn.geom_map(data = shapefile, mapping = pn.aes(), fill=None)
        
    ggplt += pn.geom_point(data = performance, mapping = pn.aes(x = "lon",
                                                                y = "lat",                                                                
                                                                fill = "kge_r", 
                                                                size='kge_r',                                                              
                                                                group = "aggregation"),
                           stroke=0)
    
    ggplt += pn.scale_x_continuous(name="Longitude (degrees east)")
    ggplt += pn.scale_y_continuous(name="Latitude (degrees north)")
    ggplt += pn.scale_fill_gradientn(colors=jet_colors,
                                 name="Pearson correlation",
                                 limits=(-1, 1))  # Define limits if necessary
    
    ggplt += pn.scale_size_continuous(name="size",
                                  range=(minsize, maxsize))
    
    if extent is None:
        ggplt += pn.coord_fixed(xlim = (performance["lon"].min() - 0.75,
                                        performance["lon"].max() + 0.75),
                                ylim = (performance["lat"].min() - 0.75,
                                        performance["lat"].max() + 0.75))
    else:
        ggplt += pn.coord_fixed(xlim = (extent[0],
                                        extent[2]),
                                ylim = (extent[1],
                                        extent[3]))
    
    ggplt += pn.facet_wrap("~aggregation", ncol = 1, nrow = 2, dir = "v")
    if title is not None:
        ggplt += pn.ggtitle(title)
    ggplt += pn.theme(panel_background=pn.element_blank(),
                            panel_grid_major=pn.element_blank(),
                            panel_grid_minor=pn.element_blank(),
                            figure_size=(8, 8))
    return ggplt


import pathlib as pl
import numpy as np
import geopandas as gp

additional_directory = pl.Path("../../additional")

basins_eu_file = pl.Path("{}/HydroBASINS/hybas_eu_lev05_v1c.shp".format(additional_directory))
basins_eu = gp.read_file(basins_eu_file)

coastlines_file = pl.Path("{}/Stanford/ne_50m_coastline.shp".format(additional_directory))
coastlines = gp.read_file(coastlines_file)

shapefiles = {}

region_extents = {
    "Europe": [-11, 35, 30, 70],
    "Rhine": [4, 46, 12, 52],
    "Drammen": [7,59,11, 61.5],
    "Duoro":[-9,40,-2,44],
    "Seine":[0,47,6,51],
    "Danube":[8,41,29,51],
    "Messara":[23,34.5,26.5,36],
    "East_Anglia":[-1,51.5,2,53.2]
}

save_directory = pl.Path("../saves/observations/")
simulation_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations")
output_directory = pl.Path("../saves/simulations/")

regions = [dir.stem for dir in save_directory.iterdir() if dir.is_dir()]

for region in regions:
    print(region)
    
    patterns = [dir.stem for dir in (output_directory / region).iterdir() if dir.is_dir()]

    meta = pd.read_parquet('../saves/observations/{}/meta.parquet'.format(region))

    for pattern in patterns: 
  
        performance_file = pl.Path("../saves/performance/{}/{}/performance.csv".format(region,pattern))
        performance = pd.read_csv(performance_file)
        
        if region=='Europe':
            shapefile = coastlines
            shapefiles[region] = shapefile

            performance = performance.drop(performance[(performance['basin'] == 'East_Anglia') & (performance['aggregation']=='daily')].index)
            performance = performance.drop(performance[(performance['basin'] == 'Danube') & (performance['aggregation']=='daily')].index)
            performance = performance.drop(performance[(performance['basin'] == 'Rhine') & (performance['aggregation']=='daily')].index)
            performance = performance.drop(performance[(performance['basin'] == 'Duoro') & (performance['aggregation']=='daily')].index)

        else:
            
            path='../../additional/watershed_shapes/'
            basins_file = pl.Path("{}/{}.shp".format(path,region))
            
            shapefile = gp.read_file(basins_file)
            shapefiles[region] = shapefile
                
        if region!='Europe':
            performance=performance[performance['basin']==region]
        
        extent = region_extents[region]
        
        ggplt = plot_performance_map(region=region,performance=performance,
                                            title=pattern,
                                            shapefile=shapefile,
                                            extent=extent,
                                            )
            
        if region=='Europe':
            ggplt += pn.theme(figure_size=(10, 8))
        else:
            ggplt += pn.theme(figure_size=(8, 8))
        
        plot_out = pl.Path("../saves/figures/{}/Correlation_maps_{}_{}.jpg".format(region,region,pattern))
        
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module = "plotnine\..*")
            ggplt.save(plot_out,dpi=400,bbox_inches='tight')
        
        print("- Plotted summary")


