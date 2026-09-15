import pathlib as pl

save_directory = pl.Path("simulations")
observation_directory = pl.Path("observations/saves")
simulation_directory = pl.Path("simulations")
output_directory = pl.Path("saves")

min_overlap = 365 * 2 # days

aggregations = {"daily": "%j",
                "monthly": "%m"}


from typing import Optional
import pandas as pd
import plotnine as pn

def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()

def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)

def plot_timeseries(moisture: pd.DataFrame,
                    y_label: str,
                    ymin_label: Optional[str] = None,
                    ymax_label: Optional[str] = None,
                    title: Optional[str] = None,) -> pn.ggplot:
    
    ggplt = pn.ggplot(data = moisture, mapping = pn.aes(x = "agg"))
    
    if ymin_label is not None and ymax_label is not None:
        ggplt += pn.geom_ribbon(mapping = pn.aes(ymin = ymin_label,
                                                 ymax = ymax_label,
                                                 fill = "type"),
                                alpha = 0.2, color = "#ffffff00")
        
    ggplt += pn.geom_line(mapping = pn.aes(y = y_label,
                                           color = "type"))
    ggplt += pn.scale_x_continuous(name="Date")
    ggplt += pn.scale_y_continuous(name="Volumetric soil moisture (%)")
    ggplt += pn.scale_fill_manual(name="", values={"observed": "black",
                                                    "simulated": "red"})
    ggplt += pn.scale_color_manual(name="", values={"observed": "black",
                                                    "simulated": "red"})
    ggplt += pn.facet_wrap("~aggregation", ncol = 1, nrow = 2, dir = "v", scales="free")
    if title is not None:
        ggplt += pn.ggtitle(title)
    ggplt += pn.theme(panel_background=pn.element_blank(),
                        panel_grid_major=pn.element_blank(),
                        panel_grid_minor=pn.element_blank())
    return ggplt


import datetime as dt
import warnings
import pandas as pd
import numpy as np

simulation_patterns = [dir.stem for dir in simulation_directory.iterdir() if dir.is_dir() and "geoframe" not in dir.stem and "parflowclm_hres" not in dir.stem] # GOEframe has its own benchmark and parflowclm_hres has too few simulation dates
simulation_fields = [pattern.split("_") for pattern in simulation_patterns]
simulation_patterns = pd.DataFrame(data = simulation_fields, index = simulation_patterns)
simulation_patterns.columns = ["model", "meteo", "region", "resolution"]

region='europe'

extents = {
    "europe": [-11, 33, 42, 73]}

extent=extents[region]

region_sel = simulation_patterns["region"] == region

region_patterns = simulation_patterns.loc[region_sel]

    
for pattern, fields in region_patterns.iterrows():
    print("\tPattern: {}".format(pattern))
    
    plot_out = pl.Path("{}/{}_climatologies.pdf".format(output_directory, pattern))
    if plot_out.exists():
        print("\t- Already exists")
        continue
    
    meta_file = pl.Path("{}/{}/meta.csv".format(save_directory, pattern))
    meta = pd.read_csv(meta_file,index_col=1)
    meta["depth"] = meta["start-depth"] + (meta["end-depth"] - meta["start-depth"]) / 2
    
    ggplts = []
    
    for index in meta.index:
        lat = meta.loc[index, "simulated_lat"]
        lon = meta.loc[index, "simulated_lon"]
        depth = meta.loc[index, "depth"]
        
        observed_moisture_file = pl.Path("{}/moisture_{}.parquet".format(observation_directory, index))
        observed_moisture = pd.read_parquet(observed_moisture_file)  
        observed_moisture=pd.DataFrame(observed_moisture['moisture'])

        
        if observed_moisture.index.size == 0:
            print("\t> No moisture in observed period for station {}, skipping...".format(index))
            continue
        
        if observed_moisture.index.size < min_overlap:
            print("\t> To few moisture in observed period (only {} days) for station {}, skipping...".format(observed_moisture.index.size, 
                                                                                                              index))
                 
        simulated_moisture_file = pl.Path("{}/{}/data/moisture_{:.5f}_{:.5f}.parquet".format(simulation_directory, pattern, lat, lon))
        simulated_moisture = pd.read_parquet(simulated_moisture_file)
        
        simulated_moisture.loc[simulated_moisture["start-depth"] < meta.at[index, "depth"]]
        
        simulated_moisture["delta-depth"] = simulated_moisture["end-depth"] - simulated_moisture["start-depth"]
        simulated_moisture = simulated_moisture.groupby(["date"]).apply(w_avg, "moisture", "delta-depth").to_frame()
        simulated_moisture = simulated_moisture.reset_index()
        simulated_moisture.columns = ["date", "moisture"]
        
        moisture = pd.merge(observed_moisture, simulated_moisture, how='inner',on = ["date"])
        moisture = moisture.rename(columns = {"moisture_x": "observed",
                                              "moisture_y": "simulated"})
        moisture = moisture[["date", "observed", "simulated"]]
        
        if moisture.index.size == 0:
            print("\t> No moisture in overlap period for station {}, skipping...".format(index))
            continue
        
        if moisture.index.size < min_overlap:
            print("\t> To few moisture in overlap period (only {} days) for station {}, skipping...".format(moisture.index.size, 
                                                                                                          index))
            continue
        
        if np.mean(moisture.loc[:, "observed"]) == 0:
            print("\t> No observed moisture for station {}, skipping...".format(index))
            continue
        
        moisture.loc[:, "observed"] = moisture.loc[:, "observed"] - np.mean(moisture.loc[:, "observed"])
        moisture.loc[:, "simulated"] = moisture.loc[:, "simulated"] - np.mean(moisture.loc[:, "simulated"])
        
        moisture.index=moisture.date

        moisture = pd.melt(moisture,
                            id_vars=["date"],
                            var_name = "type",
                            value_name="moisture")
                    
        aggregated_moistures = []
        for aggregation, format in aggregations.items():
            aggregated_moisture = moisture.copy()
            aggregated_moisture["agg"] = [dt.datetime.strftime(date, format) for date in aggregated_moisture["date"]]
            aggregated_moisture = aggregated_moisture.groupby(["type", "agg"]).aggregate({"date": "min",
                                                                                          "moisture": ["median", q25, q75]})
            aggregated_moisture.columns = ['_'.join(i).rstrip('_') for i in aggregated_moisture.columns.values]
            aggregated_moisture = aggregated_moisture.reset_index()
            aggregated_moisture["aggregation"] = aggregation
            if aggregated_moisture.index.size == 0:
                continue
            aggregated_moistures.append(aggregated_moisture)
        aggregated_moisture = pd.concat(aggregated_moistures, axis = 0)
                    
        aggregated_moisture = aggregated_moisture.astype({"agg": "int32"})
        ggplt = plot_timeseries(moisture = aggregated_moisture,
                                y_label = "moisture_median",
                                ymin_label="moisture_q25",
                                ymax_label="moisture_q75",
                                title="Volumetric soil moisture anomaly climatology for station {} (at {:.2f} depth)".format(index, depth))
        ggplts.append(ggplt)
    
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module = "plotnine\..*")
        pn.save_as_pdf_pages(plots = ggplts, filename= plot_out, verbose=False)
    
    print("\t- Plotted climatologies")
    
    #%%
