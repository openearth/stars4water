#%%
import pathlib as pl

observation_directory = pl.Path("saves/observations/")
simulation_directory = pl.Path("saves/simulations/")
output_directory = pl.Path("saves/performance/")

min_overlap = 365 * 3 # days

aggregations = {"daily": "%Y%m%d",
                "monthly": "%Y%m"}


import warnings
import datetime as dt
import pandas as pd
import numpy as np
import hydroeval as he

regions = [dir.stem for dir in simulation_directory.iterdir() if dir.is_dir()]

for region in regions:
    print("Region: {}".format(region))
        
    region_directory = pl.Path("{}/{}".format(simulation_directory, region))
    patterns = [dir.stem for dir in region_directory.iterdir() if dir.is_dir()]
    
    for pattern in patterns:
        print("\tPattern: {}".format(pattern))
        
        performance_out = pl.Path("{}/{}/{}/performance.csv".format(output_directory, region, pattern))
        if performance_out.exists():
            print("\t- Already exists")
            continue
    
        meta_file = pl.Path("{}/{}/{}/meta.parquet".format(simulation_directory, region, pattern))
        meta = pd.read_parquet(meta_file)
        
        performance_dfs = []
        
        for index, row in meta.iterrows():
            observed_discharge_file = pl.Path("{}/{}/data/discharge_{}.parquet".format(observation_directory, region, index))
            observed_discharge = pd.read_parquet(observed_discharge_file)
            simulated_discharge_file = pl.Path("{}/{}/{}/data/discharge_{}.parquet".format(simulation_directory, region, pattern, index))
            simulated_discharge = pd.read_parquet(simulated_discharge_file)
            
            discharge = pd.merge(observed_discharge, simulated_discharge, on = "date")
            discharge = discharge.rename(columns = {"discharge_x": "observed",
                                                    "discharge_y": "simulated"})
            discharge = discharge[["date", "observed", "simulated"]]
            
            if discharge.index.size == 0:
                print("\t> No discharge in overlap period for gauge {}, skipping...".format(index))
                continue
            
            if discharge.index.size < min_overlap:
                print("\t> To few discharge in overlap period (only {} days) for gauge {}, skipping...".format(discharge.index.size, 
                                                                                                                  index))
                continue
            
            for aggregation, format in aggregations.items():
                
                aggregated_discharge = discharge.copy()
                aggregated_discharge["agg"] = [dt.datetime.strftime(date, format) for date in aggregated_discharge["date"]]
                aggregated_discharge = aggregated_discharge.groupby("agg").aggregate({"date": "min",
                                                                                      "observed": "mean",
                                                                                      "simulated": "mean"})
                
                me = np.mean(aggregated_discharge["simulated"] - aggregated_discharge["observed"])
                rmse = np.sqrt(np.mean((aggregated_discharge["simulated"] - aggregated_discharge["observed"])**2))
                nme = me / np.mean(aggregated_discharge["observed"])
                nrmse = rmse / np.mean(aggregated_discharge["observed"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    kge, r, alpha, beta = he.evaluator(he.kge, aggregated_discharge["simulated"], aggregated_discharge["observed"])
                    kge_prime, r_prime, alpha_prime, beta_prime = he.evaluator(he.kgeprime, aggregated_discharge["simulated"], aggregated_discharge["observed"])
                    kge_np, r_np, alpha_np, beta_np = he.evaluator(he.kgenp, aggregated_discharge["simulated"], aggregated_discharge["observed"])
                
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
