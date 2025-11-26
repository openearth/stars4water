#%%
from typing import Optional
import pandas as pd
import plotnine as pn

def plot_performance(metric_type,performance: pd.DataFrame,
                     individual_performance: Optional[pd.DataFrame] = None,) -> pn.ggplot:
    
    #performance=performance[performance['metric_type']==metric_type]
    #individual_performance=individual_performance[individual_performance['metric_type']==metric_type]
        
    ggplt = pn.ggplot()
    ggplt += pn.geom_boxplot(data = performance,
                             mapping = pn.aes(x = "metric_name",
                                              y = "value",
                                              fill = "aggregation"),
                            outlier_shape = "", color = "black",width=0.5)
    
    ggplt += pn.geom_point(data = individual_performance,
                           mapping = pn.aes(x = "metric_name",
                                            y = "value",
                                            group = "aggregation",
                                            color = "pattern",
                                            shape='pattern'),
                        position = pn.position_dodge(width=0.5),
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
    ggplt += pn.scale_shape_discrete(name = "Model")
    ggplt += pn.scale_color_brewer(name = "Model",
                                   type="qual",
                                   palette="Set1")
    
    ggplt += pn.facet_grid("region~metric_type", scales="free")
    ggplt += pn.coord_cartesian(ylim = (-1, 2))
    #ggplt += pn.ggtitle("Water table depth anomalies - m")
    ggplt += pn.theme(panel_background=pn.element_blank(),
                      panel_grid_major=pn.element_blank(),
                      panel_grid_minor=pn.element_blank(),
                      axis_text_x=pn.element_text(rotation=45, hjust=1))
    return ggplt



import warnings
import pandas as pd
import pathlib as pl

performances = []

save_directory = pl.Path("../saves/observations/")
output_directory = pl.Path("../saves/simulations/")

regions = [dir.stem for dir in save_directory.iterdir() if dir.is_dir()]

for region in regions:
    print(region)
    
    patterns = [dir.stem for dir in (output_directory / region).iterdir() if dir.is_dir()]


    for pattern in patterns:
        
        try:
 
            performance_file = pl.Path("../saves/performance/{}/{}/performance.csv".format(region,pattern))
            performance = pd.read_csv(performance_file)
            
            performance = performance[["station", "aggregation",
                                    "kge_r","rmse"]]
            
            performance['region']=region
            performance['pattern']=pattern

            if region=='Europe':

                performance = performance.drop(performance[(performance['basin'] == 'East_Anglia') & (performance['aggregation']=='daily')].index)
                performance = performance.drop(performance[(performance['basin'] == 'Danube') & (performance['aggregation']=='daily')].index)
                performance = performance.drop(performance[(performance['basin'] == 'Rhine') & (performance['aggregation']=='daily')].index)
                performance = performance.drop(performance[(performance['basin'] == 'Duoro') & (performance['aggregation']=='daily')].index)

            performances.append(performance)

        except:
            pass
            
performance = pd.concat(performances, axis = 0)


performance = pd.melt(performance,
                    id_vars = ["region", "pattern", "station", "aggregation"],
                    var_name = "metric")


performance["metric_type"] = None

performance.loc[performance["metric"] == "kge", "metric_type"] = "kge component"
performance.loc[performance["metric"] == "kge_r", "metric_type"] = "kge component"
performance.loc[performance["metric"] == "kge_alpha", "metric_type"] = "kge component"
performance.loc[performance["metric"] == "kge_beta", "metric_type"] = "kge component"

performance.loc[performance["metric"] == "nme", "metric_type"] = "error"
performance.loc[performance["metric"] == "mae", "metric_type"] = "error"
performance.loc[performance["metric"] == "rmse", "metric_type"] = "error"


performance["metric"] = pd.Categorical(performance["metric"], ["kge", "kge_r", "kge_alpha", "kge_beta","nme","rmse","mae"])

performance["metric_type"] = pd.Categorical(performance["metric_type"], ["kge", "kge component","error"])

performance["metric_name"] = performance["metric"]
performance["metric_name"] = performance["metric_name"].replace({"kge_r": "correlation",
                                                                "kge_alpha": "variability ratio",
                                                                "kge_beta": "bias ratio",
                                                                "rmse": "rmse",
                                                                'mae':'mae'})


performance["optimal"] = 1.0
performance.loc[performance["metric_type"] == "error", "optimal"] = 0.

pattern_performance = performance.groupby(["pattern",'region', "aggregation", "metric", "metric_type", "metric_name"]).aggregate({"value": "median"})
pattern_performance = pattern_performance.dropna()
pattern_performance = pattern_performance.reset_index()


ggplt = plot_performance(metric_type='kge component',performance=performance,
                        individual_performance=pattern_performance)
ggplt += pn.theme(figure_size=(8, 9))

plot_out = pl.Path("../saves/figures/performance_summary.jpg".format(region))
plot_out.parent.mkdir(parents=True, exist_ok=True)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", module = "plotnine\..*")
    ggplt.save(plot_out,dpi=500)

print("- Plotted summary")


#%%

from typing import Optional
import pandas as pd
import plotnine as pn
import pathlib as pl
import numpy as np 


output_directory = pl.Path("../saves/simulations/")
save_directory = pl.Path("../saves/observations/")


regions = [dir.stem for dir in save_directory.iterdir() if dir.is_dir()]

for region in regions:

    patterns = [dir.stem for dir in (output_directory / region).iterdir() if dir.is_dir()]

    performances = []

    for pattern in patterns:

        performance_file = pl.Path("../saves/performance/{}/{}/performance.csv".format(region,pattern))
        performance = pd.read_csv(performance_file)

        performance = performance[["station", "aggregation",
                                "kge_r","rmse"]]

        performance['region']=region
        performance['pattern']=pattern
        
        performances.append(performance)
        

    performance = pd.concat(performances, axis = 0)

    performance = pd.melt(performance,
                        id_vars = ["region", "pattern", "station", "aggregation"],
                        var_name = "metric")


    performance["metric_type"] = None

    performance.loc[performance["metric"] == "kge", "metric_type"] = "kge component"
    performance.loc[performance["metric"] == "kge_r", "metric_type"] = "kge component"
    performance.loc[performance["metric"] == "kge_alpha", "metric_type"] = "kge component"
    performance.loc[performance["metric"] == "kge_beta", "metric_type"] = "kge component"

    performance.loc[performance["metric"] == "nme", "metric_type"] = "error"
    performance.loc[performance["metric"] == "mae", "metric_type"] = "error"
    performance.loc[performance["metric"] == "rmse", "metric_type"] = "error"


    performance["metric"] = pd.Categorical(performance["metric"], ["kge", "kge_r", "kge_alpha", "kge_beta","nme","rmse","mae"])

    performance["metric_type"] = pd.Categorical(performance["metric_type"], ["kge", "kge component","error"])

    performance["metric_name"] = performance["metric"]
    performance["metric_name"] = performance["metric_name"].replace({"kge_r": "correlation",
                                                                    "kge_alpha": "variability ratio",
                                                                    "kge_beta": "bias ratio",
                                                                    "rmse": "rmse",
                                                                    'mae':'mae'})


    performance["optimal"] = 1.0
    performance.loc[performance["metric_type"] == "error", "optimal"] = 0.

    pattern_performance = performance.groupby(["pattern",'region', "aggregation", "metric", "metric_type", "metric_name"]).aggregate({"value": "median"})
    pattern_performance = pattern_performance.dropna()
    pattern_performance = pattern_performance.reset_index()

    individual_performance=pattern_performance

    ggplt = pn.ggplot()
    ggplt += pn.geom_boxplot(data = performance,
                                mapping = pn.aes(x = "metric_name",
                                                y = "value",
                                                fill='aggregation'),
                            outlier_shape = "", color = "black",width=0.5)

    ggplt += pn.geom_point(data = individual_performance,
                            mapping = pn.aes(x = "metric_name",
                                            y = "value",
                                            group = "aggregation",
                                            color = "pattern",
                                            shape='pattern'),
                        position = pn.position_dodge(width=0.5),
                        size = 3,
                        alpha = 0.5)
    ggplt += pn.geom_hline(data = performance,
                            mapping = pn.aes(yintercept = "optimal"),
                            linetype = "dashed")
    ggplt += pn.scale_x_discrete(name="")
    ggplt += pn.scale_y_continuous(name="")
    #ggplt += pn.scale_fill_brewer(name="Aggregation",
    #                                type = "seq",
    #                                palette = "Greys")
    #ggplt += pn.scale_shape_discrete(name = "Model")
    #ggplt += pn.scale_color_brewer(name = "Model",
    #                               type="qual")

    ggplt += pn.facet_grid("region~metric_type", scales="free")
    ggplt += pn.coord_cartesian(ylim = (-1, 2))
    #ggplt += pn.ggtitle("Water table depth anomalies - m")
    ggplt += pn.theme(panel_background=pn.element_blank(),
                        panel_grid_major=pn.element_blank(),
                        panel_grid_minor=pn.element_blank(),
                        axis_text_x=pn.element_text(rotation=45, hjust=1))

    if 'daily' in performance['aggregation'].unique():
        ggplt += pn.scale_fill_manual(['#f0f0f0','#bdbdbd'])
        
    else:
        ggplt += pn.scale_fill_manual('#bdbdbd')

    if 'parflowclm' in performance['pattern'].unique()[0]:
        
        ggplt += pn.scale_color_manual(['blue','red'])
        ggplt += pn.scale_shape_manual(['s','o'])
    else: 
        ggplt += pn.scale_color_manual(['red'])
        ggplt += pn.scale_shape_manual(['o'])
    



    ggplt += pn.theme(figure_size=(5, 4))

    plot_out = pl.Path("../saves/figures/performance_summary_{}.jpg".format(region))

    ggplt.save(plot_out,dpi=500)
