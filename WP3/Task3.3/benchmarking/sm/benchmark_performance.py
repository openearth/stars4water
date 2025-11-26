#%%
import xarray as xr 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import datetime as dt
import warnings
import pathlib as pl
import hydroeval as he
import os 

aggregations = {"daily": "%Y%m%d",
                "monthly": "%Y%m"}

observation_directory = pl.Path("../saves/observations/")
simulation_directory = pl.Path("../saves/simulations/")

def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()

min_overlap = 365 * 2 # days


regions = [dir.stem for dir in observation_directory.iterdir() if dir.is_dir()]

for region in regions:
    print(region)
    
    patterns = [dir.stem for dir in (simulation_directory / region).iterdir() if dir.is_dir()]

    meta = pd.read_parquet('../saves/observations/{}/meta.parquet'.format(region))

    for pattern in patterns: 
        
        print(pattern)
        
        meta["depth"] = meta["start-depth"] + (meta["end-depth"] - meta["start-depth"]) / 2

        performance_dfs = []
        
        performance_out = pl.Path("../saves/performance/{}/{}/performance_absolute.csv".format(region,pattern))

        if os.path.isfile(performance_out)==True:
            print('Performance file: {} already exists'.format(pattern))
            continue
    
        for index in meta.index:
            
            try:
            
                observed_moisture_file = pl.Path("{}/{}/data/moisture_{}.parquet".format(observation_directory,region, index))
                observed_moisture = pd.read_parquet(observed_moisture_file)
                
                simulated_moisture_file = pl.Path("{}/{}/{}/data/moisture_{}.parquet".format(simulation_directory,region, pattern, index))
                simulated_moisture = pd.read_parquet(simulated_moisture_file)
                
                simulated_moisture.loc[simulated_moisture["start-depth"] < meta.at[index, "end-depth"]]
                
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
                
                moisture.loc[:, "observed"] = (moisture.loc[:, "observed"] - np.mean(moisture.loc[:, "observed"]))/np.std(moisture.loc[:, "observed"])
                moisture.loc[:, "simulated"] = (moisture.loc[:, "simulated"] - np.mean(moisture.loc[:, "simulated"]))/np.std(moisture.loc[:, "simulated"])

                min_moisture = moisture.loc[:, "observed"].min()
                moisture.loc[:, "observed"] = moisture.loc[:, "observed"] - min_moisture
                moisture.loc[:, "simulated"] = moisture.loc[:, "simulated"] - min_moisture        
            
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
                                        "rmse":rmse,                                 
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
            except:
                print('No station')
                
        if len(performance_dfs)>0:
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
        else:
            print('No stations')

#%%
pattern='clm_rea6_europe_0p0275deg'

id=meta.index

index=id[2]

observed_moisture_file = pl.Path("{}/{}/data/moisture_{}.parquet".format(observation_directory,region, index))
observed_moisture = pd.read_parquet(observed_moisture_file)

simulated_moisture_file = pl.Path("{}/{}/{}/data/moisture_{}.parquet".format(simulation_directory,region, pattern, index))
simulated_moisture = pd.read_parquet(simulated_moisture_file)

simulated_moisture.loc[simulated_moisture["start-depth"] < meta.at[index, "end-depth"]]

simulated_moisture["delta-depth"] = simulated_moisture["end-depth"] - simulated_moisture["start-depth"]
simulated_moisture = simulated_moisture.groupby(["date"]).apply(w_avg, "moisture", "delta-depth").to_frame()
simulated_moisture = simulated_moisture.reset_index()
simulated_moisture.columns = ["date", "moisture"]

moisture = pd.merge(observed_moisture, simulated_moisture, how='inner',on = ["date"])
moisture = moisture.rename(columns = {"moisture_x": "observed",
                                    "moisture_y": "simulated"})
moisture = moisture[["date", "observed", "simulated"]]

moisture.loc[:, "observed"] = (moisture.loc[:, "observed"] - np.mean(moisture.loc[:, "observed"]))
moisture.loc[:, "simulated"] = (moisture.loc[:, "simulated"] - np.mean(moisture.loc[:, "simulated"]))

min_moisture = moisture.loc[:, "observed"].min()
moisture.loc[:, "observed"] = moisture.loc[:, "observed"] - min_moisture
moisture.loc[:, "simulated"] = moisture.loc[:, "simulated"] - min_moisture     


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

        print(kge)