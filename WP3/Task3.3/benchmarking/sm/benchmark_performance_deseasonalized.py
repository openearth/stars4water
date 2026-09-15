#%%
import pathlib as pl
import warnings
import datetime as dt
import pandas as pd
import numpy as np
import hydroeval as he
from typing import Literal
import warnings
import os 
warnings.filterwarnings("ignore")

observation_directory = pl.Path("observations/saves")
simulation_directory = pl.Path("simulations")
output_directory = pl.Path("simulations/")

min_overlap = 365 * 2 # days

aggregations = {"daily": "%Y%m%d",
                "monthly": "%Y%m"}

def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()

def make_deseasonalized(df: pd.DataFrame,
                        aggregation: Literal['day', 'week', 'month']) -> pd.DataFrame:
    
    additional_groups = list(df.columns.difference(['date', 'observed', 'simulated']))
    
    df = df.copy()
    if aggregation == 'day':
        df['my_agg'] = [d.timetuple().tm_yday for d in df['date']]
        df['agg'] = [f'{d.year}_{d.timetuple().tm_yday}' for d in df['date']]
    elif aggregation == 'week':
        df['my_agg'] = [d.isocalendar().week for d in df['date']]
        df['agg'] = [f'{d.year}_{d.isocalendar().week}' for d in df['date']]
    elif aggregation == 'month':
        df['my_agg'] = [d.month for d in df['date']]
        df['agg'] = [f'{d.year}_{d.month}' for d in df['date']]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    df_agg = df.groupby(['agg', 'my_agg'] + additional_groups).mean()
    df_agg = df_agg.reset_index()
    
    df_my = df.groupby(['my_agg'] + additional_groups).aggregate({'observed': ['mean', 'std'],
                                                                'simulated': ['mean', 'std']})
    df_my.columns = ['_'.join(c) for c in df_my.columns.values]
    df_my = df_my.reset_index()
    
    df = pd.merge(df_agg, df_my, on=['my_agg'] + additional_groups)
    df['observed'] = df['observed'] - df['observed_mean']
    df['simulated'] = df['simulated'] - df['simulated_mean']
    
    df.pop('observed_mean')
    df.pop('observed_std')
    df.pop('simulated_mean')
    df.pop('simulated_std')
    df.pop('my_agg')
    df.pop('agg')
    
    return df


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
    
    performance_out = pl.Path("saves/performance/{}/performance_deseasonalized.csv".format(pattern))
    
    if os.path.isfile(performance_out)==True:
        print('Performance file: {} already exists'.format(pattern))
        continue

    if fields.model=='wflowsbm':
        continue
     
    meta_file = pl.Path("{}/{}/meta.csv".format(simulation_directory, pattern))
    meta = pd.read_csv(meta_file,index_col=1)
    meta["depth"] = meta["start-depth"] + (meta["end-depth"] - meta["start-depth"]) / 2

    performance_dfs = []

    print("\tPattern: {}".format(pattern))
    
    for index in meta.index:
        lat = meta.loc[index, "simulated_lat"]
        lon = meta.loc[index, "simulated_lon"]
        depth = meta.loc[index, "depth"]
        
        observed_moisture_file = pl.Path("{}/moisture_{}.parquet".format(observation_directory, index))
        observed_moisture = pd.read_parquet(observed_moisture_file)
        observed_moisture=pd.DataFrame(observed_moisture['moisture'])
        
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
            
        if np.mean(moisture.loc[:, "observed"]) == 0:
                print("\t> No observed moisture for station {}, skipping...".format(index))
                continue
            
        if moisture.index.size < min_overlap:
                print("\t> To few moisture in overlap period (only {} days) for station {}, skipping...".format(moisture.index.size, 
                                                                                                                  index))
                continue
        
        moisture.loc[:, "observed"] = (moisture.loc[:, "observed"] - np.mean(moisture.loc[:, "observed"]))#/np.std(moisture.loc[:, "observed"])
        moisture.loc[:, "simulated"] = (moisture.loc[:, "simulated"] - np.mean(moisture.loc[:, "simulated"]))#/np.std(moisture.loc[:, "simulated"])

        min_moisture = moisture.loc[:, "observed"].min()
        moisture.loc[:, "observed"] = moisture.loc[:, "observed"] - min_moisture
        moisture.loc[:, "simulated"] = moisture.loc[:, "simulated"] - min_moisture
        
        moisture = make_deseasonalized(df=moisture, aggregation='month')
        
        for aggregation, format in aggregations.items():
               
               aggregated_moisture = moisture.copy()
               aggregated_moisture["agg"] = [dt.datetime.strftime(date, format) for date in aggregated_moisture["date"]]
               aggregated_moisture = aggregated_moisture.groupby("agg").aggregate({"date": "min",
                                                                                   "observed": "mean",
                                                                                   "simulated": "mean"})
               
               me = np.mean(aggregated_moisture["simulated"] - aggregated_moisture["observed"])
               rmse = np.sqrt(np.mean((aggregated_moisture["simulated"] - aggregated_moisture["observed"])**2))
               nme = me / np.mean(aggregated_moisture["observed"])
               nrmse = rmse / np.mean(aggregated_moisture["observed"])
               with warnings.catch_warnings():
                   warnings.simplefilter("ignore", category=RuntimeWarning)
                   kge, r, alpha, beta = he.evaluator(he.kge, aggregated_moisture["simulated"], aggregated_moisture["observed"])
                   kge_prime, r_prime, alpha_prime, beta_prime = he.evaluator(he.kgeprime, aggregated_moisture["simulated"], aggregated_moisture["observed"])
                   kge_np, r_np, alpha_np, beta_np = he.evaluator(he.kgenp, aggregated_moisture["simulated"], aggregated_moisture["observed"])
               
               performance_df = {"station": index,
                                 "aggregation": aggregation,
                                 "nme": nme,
                                 "nrmse": nrmse,
                                 "kge": kge[0],
                                 "kge_r": r[0],
                                 "kge_alpha": alpha[0],
                                 "kge_beta": beta[0],
                                 "kge-prime": kge_prime[0],
                                 "kge-prime_r": r_prime[0],
                                 "kge-prime_alpha": alpha_prime[0],
                                 "kge-prime_beta": beta_prime[0],
                                 "kge-nonpar": kge_np[0],
                                 "kge-nonpar_r": r_np[0],
                                 "kge-nonpar_alpha": alpha_np[0],
                                 "kge-nonpar_beta": beta_np[0],}
               performance_dfs.append(performance_df)
               
    performance_df = {}
    for key in performance_dfs[0].keys():
        performance_df[key] = [df[key] for df in performance_dfs]
    performance_df = pd.DataFrame(performance_df)
    performance_df["region"] = region
    performance_df["pattern"] = pattern
    
    performance_df = pd.merge(performance_df, meta, left_on = "station", right_index = True)
    performance_df = performance_df.reset_index(drop = True)
   
    performance_out.parent.mkdir(parents=True, exist_ok=True)
    performance_df.to_csv(performance_out, index = False)
    
    print("\t- Saved performance indices")

#%%
# import matplotlib.pyplot as plt 

# plt.plot(moisture['observed'])
# plt.plot(moisture['simulated'])


