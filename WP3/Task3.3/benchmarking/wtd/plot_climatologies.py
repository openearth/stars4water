import pathlib as pl

from typing import Optional
import pandas as pd
import plotnine as pn

def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)

def plot_timeseries(discharge: pd.DataFrame,
                    y_label: str,
                    ymin_label: Optional[str] = None,
                    ymax_label: Optional[str] = None,
                    title: Optional[str] = None,) -> pn.ggplot:
    
    ggplt = pn.ggplot(data = discharge, mapping = pn.aes(x = "agg"))
    
    if ymin_label is not None and ymax_label is not None:
        ggplt += pn.geom_ribbon(mapping = pn.aes(ymin = ymin_label,
                                                 ymax = ymax_label,
                                                 fill = "type"),
                                alpha = 0.2, color = "#ffffff00")
        
    ggplt += pn.geom_line(mapping = pn.aes(y = y_label,
                                           color = "type"))
    ggplt += pn.scale_x_continuous(name="Date")
    ggplt += pn.scale_y_continuous(name="Snow water equivalent (mm)")
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


#%%

import warnings
import datetime as dt
import pandas as pd

models=['tsmp']

min_overlap = 365 * 2 # days

aggregations = {"daily": "%j",
                "monthly": "%m"}
    
for model in models:
    
    print("\tPattern: {}".format(model))
    
    plot_out = pl.Path("figures/{}_climatologies.pdf".format(model))
    if plot_out.exists():
        print("\t- Already exists")
        continue
    
    meta_file = pl.Path("{}/meta.parquet".format(model))
    meta = pd.read_parquet(meta_file)
    meta=meta.reset_index(drop=True)
    
    ggplts = []
        
    for index in meta.index:
        
        id=int(meta['id'].iloc[index])
        
        try:
        
            observed_swe_file = pl.Path("../observations/saves/swe_{}.parquet".format(id))
            observed_swe = pd.read_parquet(observed_swe_file)
            observed_swe['time']=observed_swe.index
            
            simulated_swe_file = pl.Path("{}/saves/swe_{}.parquet".format(model,id))
            simulated_swe = pd.read_parquet(simulated_swe_file)
            simulated_swe.index=pd.to_datetime(simulated_swe['date'])
            simulated_swe['time']=simulated_swe.index
            
            swe = pd.merge(observed_swe, simulated_swe, on = "time")
            
                  
            swe = swe.rename(columns = {"swe_obs": "observed",
                                                    "swe_sim": "simulated"})
            swe = swe[["date", "observed", "simulated"]]
            
       
            if swe.index.size == 0:
                print("\t> No discharge in overlap period for station {}, skipping...".format(index))
                continue
                
            if swe.index.size < min_overlap:
                print("\t> To few discharge in overlap period (only {} days) for station {}, skipping...".format(swe.index.size, 
                                                                                                                  index))
                
    
            swe = pd.melt(swe,
                                id_vars=["date"],
                                var_name = "type",
                                value_name="swe")
             
            aggregated_discharges = []
            
            for aggregation, format in aggregations.items():
                aggregated_discharge = swe.copy()
                aggregated_discharge["agg"] = [dt.datetime.strftime(date, format) for date in aggregated_discharge["date"]]
                aggregated_discharge = aggregated_discharge.groupby(["type", "agg"]).aggregate({"date": "min",
                                                                                                "swe": ["median", q25, q75]})
                
                aggregated_discharge.columns = ['_'.join(i).rstrip('_') for i in aggregated_discharge.columns.values]
                aggregated_discharge = aggregated_discharge.reset_index()
                aggregated_discharge["aggregation"] = aggregation
                if aggregated_discharge.index.size == 0:
                    continue
                aggregated_discharges.append(aggregated_discharge)
            aggregated_discharge = pd.concat(aggregated_discharges, axis = 0)
                        
            aggregated_discharge = aggregated_discharge.astype({"agg": "int32"})
            
       
            ggplt = plot_timeseries(discharge = aggregated_discharge,
                                    y_label = "swe_median",
                                    ymin_label="swe_q25",
                                    ymax_label="swe_q75",
                                    title="Snow water equivalent for station ID {}".format(id))
            ggplts.append(ggplt)
            
            print("\t- Plotted climatologies {}: {} out of {}".format(model,index,len(meta)))
            
        except:
            pass

    
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module = "plotnine\..*")
        pn.save_as_pdf_pages(plots = ggplts, filename= plot_out, verbose=False)
    
    
