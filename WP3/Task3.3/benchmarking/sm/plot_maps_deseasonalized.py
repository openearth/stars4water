#%%
import pathlib as pl

performance_directory = pl.Path("simulations")
output_directory = pl.Path("saves")
simulation_directory = pl.Path("simulations")


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

    performance['kge']=np.where(performance['kge']<0,0,performance['kge'])
    
    ggplt = pn.ggplot()
    if shapefile is not None:
        ggplt += pn.geom_map(data = shapefile, mapping = pn.aes(), fill=None)
    ggplt += pn.geom_point(data = performance, mapping = pn.aes(x = "lon",
                                                                y = "lat",
                                                                fill = "kge",
                                                                size='kge',
                                                                group = "aggregation"),stroke=0)
    
    ggplt += pn.scale_x_continuous(name="Longitude (degrees east)")
    ggplt += pn.scale_y_continuous(name="Latitude (degrees north)")
    ggplt += pn.scale_fill_gradientn(colors=jet_colors,
                                     name="KGE",
                                     limits=(0, 1))  # Define limits if necessa
    
    ggplt += pn.scale_size_continuous(name=" ",
                                  range=(1, 3))
    
    #ggplt += pn.scale_fill_brewer(palette = "Greys")
    ggplt += pn.scale_size_continuous(name="Average evap (m3 s-1)", breaks = [100, 200, 400, 800])
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
                        panel_grid_minor=pn.element_blank())
    
    return ggplt


import pathlib as pl
import numpy as np
import geopandas as gp

additional_directory = pl.Path("../additional")

basins_eu_file = pl.Path("{}/HydroBASINS/hybas_eu_lev05_v1c.shp".format(additional_directory))
basins_eu = gp.read_file(basins_eu_file)

coastlines_file = pl.Path("{}/Stanford/ne_50m_coastline.shp".format(additional_directory))
coastlines = gp.read_file(coastlines_file)

shapefiles = {}

region_extents = {
    "europe": [-11, 33, 42, 73],
    "drammen": [7,59,11, 62],
    
    }

region='europe'
    
if region=='europe':
    shapefile = coastlines
    shapefiles[region] = shapefile

else:
    
    path='../../additional/watershed_shapes/'
    basins_file = pl.Path("{}/{}.shp".format(path,region))
    
    shapefile = gp.read_file(basins_file)
    shapefiles[region] = shapefile
        


import warnings
import pandas as pd
import plotnine as pn


print("Region: {}".format(region))

shapefile = shapefiles[region]
extent = region_extents[region]

region_directory = pl.Path("{}".format(performance_directory))
patterns = [dir.stem for dir in region_directory.iterdir() if dir.is_dir()]

for pattern in patterns:
    print("\tPattern: {}".format(pattern))
    
    performance_file = pl.Path("{}/{}/performance_deseasonalized.csv".format(performance_directory, pattern))
    performance = pd.read_csv(performance_file)
    
    performance.groupby(["lon", "lat", "aggregation"]).aggregate({"kge": "mean"})
    
    ggplt = plot_performance_map(performance=performance,
                                 title="Soil moisture anomaly performance",
                                 shapefile=shapefile,
                                 extent=extent)
    ggplt += pn.theme(figure_size=(10, 8))
    
    plot_out = pl.Path("{}/{}_maps_deseasonalized.jpg".format(output_directory, pattern))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module = "plotnine\..*")
        ggplt.save(plot_out,dpi=400)
 
    print("\t- Plotted maps")
    
#%%
