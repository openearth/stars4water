#%%
import pathlib as pl

performance_directory = pl.Path("../saves/performance/")
output_directory = pl.Path("../saves/figures/")

from typing import Optional
import pandas as pd
import numpy as np
import plotnine as pn

def plot_performance(performance: pd.DataFrame,
                     individual_performance: Optional[pd.DataFrame] = None) -> pn.ggplot:
    
    ggplt = pn.ggplot()
    ggplt += pn.geom_boxplot(data = performance,
                             mapping = pn.aes(x = "metric_name",
                                              y = "value",
                                              fill = "aggregation"),
                            outlier_shape = "", color = "black",width=0.6)
    
    ggplt += pn.geom_point(data = individual_performance,
                           mapping = pn.aes(x = "metric_name",
                                            y = "value",
                                            group = "aggregation",
                                            color = "model",
                                            shape = "meteores"),
                        position = pn.position_dodge(width=0.75),
                        size = 3,
                        alpha = 0.5)
    ggplt += pn.geom_hline(data = performance,
                           mapping = pn.aes(yintercept = "optimal"),
                           linetype = "dashed")
    ggplt += pn.scale_x_discrete(name="")
    ggplt += pn.scale_y_continuous(name="")
    ggplt += pn.scale_fill_brewer(name="Aggregation",
                                  type = "seq",
                                  palette = "Greys")
    ggplt += pn.scale_shape_discrete(name = "Meteo and resolution")
    ggplt += pn.scale_color_brewer(name = "Model",
                                   type="qual",
                                   palette="Set1")
    ggplt += pn.facet_grid("region~metric_type", scales="free")
    ggplt += pn.coord_cartesian(ylim = (-0.5, 2))
    ggplt += pn.ggtitle("Discharge performance")
    ggplt += pn.theme(panel_background=pn.element_blank(),
                      panel_grid_major=pn.element_blank(),
                      panel_grid_minor=pn.element_blank(),
                      axis_text_x=pn.element_text(rotation=45, hjust=1))
    return ggplt

    
import warnings
import pandas as pd

performances = []

regions = [dir.stem for dir in performance_directory.iterdir() if dir.is_dir()]

for region in regions:
    region_directory = pl.Path("{}/{}".format(performance_directory, region))
    patterns = [dir.stem for dir in region_directory.iterdir() if dir.is_dir()]
    
    for pattern in patterns:        
        performance_file = pl.Path("{}/{}/{}/performance.csv".format(performance_directory, region, pattern))
        performance = pd.read_csv(performance_file)
        #performance = performance[["region", "pattern", "station", "aggregation",
                                #    "nme", "nrmse", 
        #                           "kge", "kge_r", "kge_alpha", "kge_beta", 
        #                           "kge-nonpar", "kge-nonpar_r", "kge-nonpar_alpha", "kge-nonpar_beta", 
        #                           "kge-prime", "kge-prime_r", "kge-prime_alpha", "kge-prime_beta","nrmse" ]]
        
        performance = performance[["region", "pattern", "station", "aggregation",
                                   "kge", "kge_r", "kge_alpha", "kge_beta"]]
        performances.append(performance)
        
performance = pd.concat(performances, axis = 0)

performance = pd.melt(performance,
                      id_vars = ["region", "pattern", "station", "aggregation"],
                      var_name = "metric")

performance["model"] = [sim.split("_")[0] for sim in performance["pattern"]]
performance["meteores"] = ["_".join((sim.split("_")[1],
                                     sim.split("_")[3])) for sim in performance["pattern"]]

performance["metric_type"] = None
performance.loc[performance["metric"] == "kge", "metric_type"] = "kge component"
performance.loc[performance["metric"] == "kge_r", "metric_type"] = "kge component"
performance.loc[performance["metric"] == "kge_alpha", "metric_type"] = "kge component"
performance.loc[performance["metric"] == "kge_beta", "metric_type"] = "kge component"

performance.loc[performance["metric"] == "nrmse", "metric_type"] = "error"


# performance.loc[performance["metric"] == "kge-nonpar", "metric_type"] = "kge nonparametric"
# performance.loc[performance["metric"] == "kge-nonpar_r", "metric_type"] = "kge nonparametric"
# performance.loc[performance["metric"] == "kge-nonpar_alpha", "metric_type"] = "kge nonparametric"
# performance.loc[performance["metric"] == "kge-nonpar_beta", "metric_type"] = "kge nonparametric"

# performance.loc[performance["metric"] == "kge-prime", "metric_type"] = "kge prime"
# performance.loc[performance["metric"] == "kge-prime_r", "metric_type"] = "kge prime"
# performance.loc[performance["metric"] == "kge-prime_alpha", "metric_type"] = "kge prime"
# performance.loc[performance["metric"] == "kge-prime_beta", "metric_type"] = "kge prime"


# performance["metric"] = pd.Categorical(performance["metric"], ["nme", "nrmse",
#                                                                "kge", "kge_r", "kge_alpha", "kge_beta",
#                                                                "kge-nonpar", "kge-nonpar_r", "kge-nonpar_alpha", "kge-nonpar_beta",
#                                                                "kge-prime", "kge-prime_r", "kge-prime_alpha", "kge-prime_beta",])
# performance["metric_type"] = pd.Categorical(performance["metric_type"], ["error", "kge", "kge nonparametric", "kge prime"])

# performance["metric_name"] = performance["metric"]
# performance["metric_name"] = performance["metric_name"].replace({"kge": "kge",
#                                                                 "kge_r": "r",
#                                                                 "kge_alpha": "alpha",
#                                                                 "kge_beta": "beta",
#                                                                 "kge-nonpar": "combined",
#                                                                 "kge-nonpar_r": "r",
#                                                                 "kge-nonpar_alpha": "alpha",
#                                                                 "kge-nonpar_beta": "beta",
#                                                                 "kge-prime": "combined",
#                                                                 "kge-prime_r": "r",
#                                                                 "kge-prime_alpha": "alpha",
#                                                                 "kge-prime_beta": "beta",})


performance["metric"] = pd.Categorical(performance["metric"], ["kge", "kge_r", "kge_alpha", "kge_beta","nrmse"])
performance["metric_type"] = pd.Categorical(performance["metric_type"], ["kge", "kge component","error"])


performance["metric_name"] = performance["metric"]
performance["metric_name"] = performance["metric_name"].replace({"kge": "kge",
                                                                 "kge_r": "correlation",
                                                                 "kge_alpha": "variability ratio",
                                                                 "kge_beta": "bias ratio",})

performance["optimal"] = 1.0
performance.loc[performance["metric_type"] == "error", "optimal"] = 0.

pattern_performance = performance.groupby(["region", "pattern", "model", "meteores", "aggregation", "metric", "metric_type", "metric_name"]).aggregate({"value": "median"})
pattern_performance = pattern_performance.dropna()
pattern_performance = pattern_performance.reset_index()

ggplt = plot_performance(performance=performance,
                         individual_performance=pattern_performance)
ggplt += pn.theme(figure_size=(9,11))

plot_out = pl.Path("{}/performance_plot_discharge.jpg".format(output_directory))
plot_out.parent.mkdir(parents=True, exist_ok=True)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", module = "plotnine\..*")
    ggplt.save(plot_out,dpi=400)

print("- Plotted summary")
# %%


A=pattern_performance[pattern_performance['aggregation']=='monthly']

regions=A['region'].unique()

for region in regions:
    B=A[A['region']==region]

    kge=B[B['metric']=='kge']

    kge=np.mean(kge['value'])
    
    
    r=B[B['metric']=='kge_r']

    r=np.mean(r['value'])

    print(region,':','KGE',np.round(kge,3),'R',np.round(r,3))
  