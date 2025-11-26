from typing import Optional
import pandas as pd
import plotnine as pn

def plot_timeseries(evaporation: pd.DataFrame,
                    y_label: str,
                    title: Optional[str] = None,) -> pn.ggplot:
    
    ggplt = pn.ggplot(data = evaporation, mapping = pn.aes(x = "date"))        
    ggplt += pn.geom_line(mapping = pn.aes(y = y_label,
                                           color = "type",
                                           group = "type"))
    ggplt += pn.scale_x_date(name="Date")
    ggplt += pn.scale_y_continuous(name="Snow water equivalent (mm)")
    ggplt += pn.scale_color_manual(name="", values={"observed": "black",
                                                    "simulated": "red"})
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
import pathlib as pl

models=['clm','lisflood','tsmp','lisflood_seNorge']

min_overlap = 365 * 2 # days

aggregations = {"daily": "%j",
                "monthly": "%m"}
    
for model in models:
    
   
    print("\tPattern: {}".format(model))
    
    plot_out = pl.Path("figures/{}_timeseries.pdf".format(model))
    if plot_out.exists():
        print("\t- Already exists")
        continue
    
    meta_file = pl.Path("{}/meta.parquet".format(model))
    meta = pd.read_parquet(meta_file)
    meta=meta.reset_index(drop=True)
    
    ggplts = []
        
    for index in meta.index:
        
        id=int(meta['id'].iloc[index])
        
        observed_swe_file = pl.Path("../observations/saves/swe_{}.parquet".format(id))
        observed_swe = pd.read_parquet(observed_swe_file)
        observed_swe['time']=observed_swe.index
        
        try:
            simulated_swe_file = pl.Path("{}/saves/swe_{}.parquet".format(model,id))
            simulated_swe = pd.read_parquet(simulated_swe_file)
            simulated_swe.index=pd.to_datetime(simulated_swe['date'])
            simulated_swe['time']=simulated_swe.index
        except:
            pass
        
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
        
        
        ggplt = plot_timeseries(evaporation = swe,
                                    y_label = "swe",
                                    title="Snow water equivalent for station {}".format(id))
        ggplts.append(ggplt)
         
        print("\t- Plotted time series {}: {} out of {}".format(model,index,len(meta)))
        
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module = "plotnine\..*")
        pn.save_as_pdf_pages(plots = ggplts, filename= plot_out, verbose=False)
    