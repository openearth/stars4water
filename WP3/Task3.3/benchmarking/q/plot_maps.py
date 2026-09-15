#%%
#   Functions to make map plots

from typing import Optional
import pandas as pd
import geopandas as gp
import plotnine as pn
from matplotlib import cm

def plot_performance_map(performance: pd.DataFrame,
                         title: Optional[str] = None,
                         shapefile: Optional[gp.GeoDataFrame] = None,
                         extent: Optional[list] = None) -> pn.ggplot:
    
    jet_colors = [cm.jet(i) for i in range(256)][::-1]

    
    ggplt = pn.ggplot()
    if shapefile is not None:
        ggplt += pn.geom_map(data = shapefile, mapping = pn.aes(), fill=None)
    ggplt += pn.geom_point(data = performance, mapping = pn.aes(x = "lon",
                                                                y = "lat",
                                                                fill = "kge",
                                                                group = "aggregation",
                                                                size = "average"),stroke=0)
    
    ggplt += pn.scale_x_continuous(name="Longitude (degrees east)")
    ggplt += pn.scale_y_continuous(name="Latitude (degrees north)")
    ggplt += pn.scale_size_continuous(name="Average discharge (m3 s-1)")
    ggplt += pn.scale_fill_gradientn(colors=jet_colors,
                                     name="KGE",
                                     limits=(0, 1))  # Define limits if necessary
        
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

#%%

import pathlib as pl
import pathlib as pl
import numpy as np
import geopandas as gp

additional_directory = pl.Path("../../additional")
performance_directory = pl.Path("../saves/performance/")
simulation_directory = pl.Path("../saves/simulations/")
output_directory = pl.Path("../saves/figures")

basins_eu_file = pl.Path("{}/HydroBASINS/hybas_eu_lev05_v1c.shp".format(additional_directory))
basins_eu = gp.read_file(basins_eu_file)

coastlines_file = pl.Path("{}/Stanford/ne_50m_coastline.shp".format(additional_directory))
coastlines = gp.read_file(coastlines_file)

shapefiles = {}

region_extents = {
    "Europe": [-11, 33, 42, 70],
    "Rhine": [4, 46, 12, 52],
    "Drammen": [7,59,11, 62],
    "Duoro":[-9,40,-2,44],
    "Seine":[0,47,6,51],
    "Danube":[8,41,29,51],
    "East_Anglia":[-1,51,2,54]
}

#%%
import warnings
import pandas as pd
import plotnine as pn

for region in region_extents:
    print("Region: {}".format(region))
    
    extent = region_extents[region]
   
    region_directory = pl.Path("{}/{}".format(performance_directory, region))
    patterns = [dir.stem for dir in region_directory.iterdir() if dir.is_dir()]
    
    if region=='Europe':
        shapefile = coastlines
        shapefiles[region] = shapefile

    else:
        
        path='{}/watershed_shapes/'.format(additional_directory)
        basins_file = pl.Path("{}/{}.shp".format(path,region))
        
        shapefile = gp.read_file(basins_file)
        shapefiles[region] = shapefile

    
    for pattern in patterns:
        print("\tPattern: {}".format(pattern))
        
        performance_file = pl.Path("{}/{}/{}/performance.csv".format(performance_directory, region, pattern))
        performance = pd.read_csv(performance_file)
        
        ggplt = plot_performance_map(performance=performance,
                                     title="Discharge performance",
                                     shapefile=shapefile,
                                     extent=extent,)
        ggplt += pn.theme(figure_size=(10, 8))
        
        plot_out = pl.Path("{}/{}/{}_maps.jpg".format(output_directory, region, pattern))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module = "plotnine\..*")
            ggplt.save(plot_out,dpi=600,bbox_inches='tight')
        
        print("\t- Plotted maps")
    
    
    
#%%

