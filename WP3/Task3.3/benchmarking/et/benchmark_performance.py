#%%
import xarray as xr 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import datetime as dt
import warnings
import pathlib as pl
import hydroeval as he

aggregations = {"daily": "%Y%m%d",
                "monthly": "%Y%m"}

save_directory = pl.Path("../saves/observations/")
simulation_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/simulations")
output_directory = pl.Path("../saves/simulations/")

regions = [dir.stem for dir in save_directory.iterdir() if dir.is_dir()]

for region in regions:
    print(region)
    
    patterns = [dir.stem for dir in (output_directory / region).iterdir() if dir.is_dir()]

    meta = pd.read_parquet('../saves/observations/{}/meta.parquet'.format(region))

    for pattern in patterns: 
        
        print(pattern)
        
        performance_dfs = []
        
        for i in range(len(meta)):
            
            id=meta['id'].index[i]
            
            lat=meta['lat'].iloc[i]
            lon=meta['lon'].iloc[i]
            
            try:
            
                sim=pd.read_parquet('../saves/simulations/{}/{}/data/evaporation_{}.parquet'.format(region,pattern,id))
                sim.index=pd.to_datetime(sim['date'])
                
                obs=pd.read_parquet('../saves/observations/{}/data/evaporation_{}.parquet'.format(region,id))
                obs.index=pd.to_datetime(obs['date'])
                
                wtd_a=sim['evaporation']
                wtd_obs=obs['evaporation']
                
                simulated=pd.DataFrame({'sim':wtd_a})
                observed=pd.DataFrame({'obs':wtd_obs})
                
                simulated['time']=simulated.index 
                observed['time']=observed.index #

                s1 = pd.merge(simulated, observed, how='inner', on=['time'])
                s1.index=pd.to_datetime(s1.time)
            
                s1 = s1[["time", "obs", "sim"]]
                s1.columns=['date','observed','simulated']
                #CORR.append(np.corrcoef(s1['sim'],s1['obs'])[0,1])
                
                for aggregation, format in aggregations.items():
                        
                    if len(s1)==0:
                        continue
                            
                    aggregated_evaporation = s1.copy()
                    aggregated_evaporation["agg"] = [dt.datetime.strftime(date, format) for date in aggregated_evaporation["date"]]
                    aggregated_evaporation = aggregated_evaporation.groupby("agg").aggregate({"date": "min",
                                                                                                        "observed": "mean",
                                                                                                        "simulated": "mean"})
                        
                    me = np.mean(aggregated_evaporation["simulated"] - aggregated_evaporation["observed"])
                    rmse = np.sqrt(np.mean((aggregated_evaporation["simulated"] - aggregated_evaporation["observed"])**2))
                    nme = me / np.mean(aggregated_evaporation["observed"])
                    nrmse = rmse / np.mean(aggregated_evaporation["observed"])
                    
                    mae=np.nansum(np.abs(aggregated_evaporation["simulated"] - aggregated_evaporation["observed"]))/len(aggregated_evaporation["observed"])

                    with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            kge, r, alpha, beta = he.evaluator(he.kge, aggregated_evaporation["simulated"], aggregated_evaporation["observed"])
                            kge_prime, r_prime, alpha_prime, beta_prime = he.evaluator(he.kgeprime, aggregated_evaporation["simulated"], aggregated_evaporation["observed"])
                            kge_np, r_np, alpha_np, beta_np = he.evaluator(he.kgenp, aggregated_evaporation["simulated"], aggregated_evaporation["observed"])
                    
                    performance_df = {"station": id,
                                        'lon':lon,
                                        'lat':lat,
                                        'basin':region,
                                        "aggregation": aggregation,
                                                    "rmse":rmse,
                                                    "nme": nme,
                                                    "nrmse": nrmse,
                                                    "mae":mae,
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
                pass 

        performance_df = {}

        for key in performance_dfs[0].keys():
            performance_df[key] = [df[key] for df in performance_dfs]
            
            
        performance_df = pd.DataFrame(performance_df)
        performance_df = performance_df.reset_index(drop = True)

        performance_out = pl.Path("../saves/performance/{}/{}/performance.csv".format(region,pattern))

        performance_out.parent.mkdir(parents=True, exist_ok=True)
        performance_df.to_csv(performance_out, index = False)