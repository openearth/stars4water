#%%
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

    for pattern in patterns: 

        meta = pd.read_csv('../saves/simulations/{}/{}/meta.csv'.format(region,pattern),index_col=0)
        print(pattern)

        
        performance_dfs = []
        
        for index,row in meta.iterrows():   
            
            id=index
            
            lat=row['lat']
            lon=row['lon']
            basin=row['basin']
  
            sim=pd.read_parquet('../saves/simulations/{}/{}/data/wtd_{}.parquet'.format(region,pattern,id))

            sim.index=pd.to_datetime(sim['date'])
            
            if region=='East_Anglia':
                sim.index=sim.index.map(lambda t: t.replace(day=1,hour=0,minute=0))

            obs=pd.read_parquet('../saves/observations/{}/data/wtd_{}.parquet'.format(region,id))        
            
            wtd_a=(sim['wtd_sim']-np.mean(sim['wtd_sim']))/np.std(sim['wtd_sim'])
            wtd_obs=(obs['wtd']-np.mean(obs['wtd']))/np.std(obs['wtd'])
            
            simulated=pd.DataFrame({'sim':wtd_a})
            observed=pd.DataFrame({'obs':wtd_obs})
            
            simulated['time']=simulated.index 
            observed['time']=observed.index #

            s1 = pd.merge(simulated, observed, how='inner', on=['time'])
            s1.index=pd.to_datetime(s1.time)
        
            s1 = s1[["time", "obs", "sim"]]
            s1.columns=['date','observed','simulated']
            
            for aggregation, format in aggregations.items():                
                
                if region=='East_Anglia' and aggregation=='daily':
                    continue
                
                if region=='Danube' and aggregation=='daily':
                    continue
                
                if region=='Rhine' and aggregation=='daily':
                    continue
                
                if region=='Duoro' and aggregation=='daily':
                    continue
                    
                if len(s1)==0:
                    continue
                        
                aggregated_discharge = s1.copy()
                aggregated_discharge["agg"] = [dt.datetime.strftime(date, format) for date in aggregated_discharge["date"]]
                aggregated_discharge = aggregated_discharge.groupby("agg").aggregate({"date": "min",
                                                                                                    "observed": "mean",
                                                                                                    "simulated": "mean"})
                    
                me = np.mean(aggregated_discharge["simulated"] - aggregated_discharge["observed"])
                rmse = np.sqrt(np.mean((aggregated_discharge["simulated"] - aggregated_discharge["observed"])**2))
                nme = me / np.mean(aggregated_discharge["observed"])
                nrmse = rmse / np.mean(aggregated_discharge["observed"])
                
                mae=np.nansum(np.abs(aggregated_discharge["simulated"] - aggregated_discharge["observed"]))/len(aggregated_discharge["observed"])

                with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        kge, r, alpha, beta = he.evaluator(he.kge, aggregated_discharge["simulated"], aggregated_discharge["observed"])
                        kge_prime, r_prime, alpha_prime, beta_prime = he.evaluator(he.kgeprime, aggregated_discharge["simulated"], aggregated_discharge["observed"])
                        kge_np, r_np, alpha_np, beta_np = he.evaluator(he.kgenp, aggregated_discharge["simulated"], aggregated_discharge["observed"])
                
                performance_df = {"station": id,
                                    'lon':lon,
                                    'lat':lat,
                                    'region':region,
                                    'basin':basin,
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



        performance_df = {}

        for key in performance_dfs[0].keys():
            performance_df[key] = [df[key] for df in performance_dfs]
            
            
        performance_df = pd.DataFrame(performance_df)
        performance_df = performance_df.reset_index(drop = True)

        performance_out = pl.Path("../saves/performance/{}/{}/performance.csv".format(region,pattern))

        performance_out.parent.mkdir(parents=True, exist_ok=True)
        performance_df.to_csv(performance_out, index = False)
        
#%%

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

region='Danube'

patterns = [dir.stem for dir in (output_directory / region).iterdir() if dir.is_dir()]
index=3
for pattern in patterns: 

    meta = pd.read_csv('../saves/simulations/{}/{}/meta.csv'.format(region,pattern),index_col=0)
    print(pattern)

    performance_dfs = []
    
    #for index,row in meta.iterrows():   
        
    id=meta.index[index]
    
    lat=meta['lat'].iloc[index]
    lon=meta['lon'].iloc[index]
    basin=meta['basin'].iloc[index]

    sim=pd.read_parquet('../saves/simulations/{}/{}/data/wtd_{}.parquet'.format(region,pattern,id))

    sim.index=pd.to_datetime(sim['date'])
    
    if region=='East_Anglia':
        sim.index=sim.index.map(lambda t: t.replace(day=1,hour=0,minute=0))

    obs=pd.read_parquet('../saves/observations/{}/data/wtd_{}.parquet'.format(region,id))        
    
    wtd_a=(sim['wtd_sim']-np.mean(sim['wtd_sim']))/np.std(sim['wtd_sim'])
    wtd_obs=(obs['wtd']-np.mean(obs['wtd']))/np.std(obs['wtd'])
    
    simulated=pd.DataFrame({'sim':wtd_a})
    observed=pd.DataFrame({'obs':wtd_obs})
    
    simulated['time']=simulated.index 
    observed['time']=observed.index #

    s1 = pd.merge(simulated, observed, how='inner', on=['time'])
    s1.index=pd.to_datetime(s1.time)

    s1 = s1[["time", "obs", "sim"]]
    s1.columns=['date','observed','simulated']
 
plt.plot(s1['observed'])
plt.plot(s1['simulated'])

np.corrcoef(s1['observed'],s1['simulated'])