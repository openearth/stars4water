#%%
import pathlib as pl
import pandas as pd
import numpy as np

observations_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/observations/et")
output_directory = pl.Path("../saves/observations/")

meta=pd.read_parquet('../saves/observations/meta_out.parquet')

regions=meta['region'].unique() 

for region in regions:
    print(region)
    
    meta=pd.read_parquet('../saves/observations/meta_out.parquet')

    if region!='Europe':
        meta=meta[meta['region']==region]
        #meta=meta.reset_index(drop=True)
        
    meta_region_file=pl.Path('../saves/observations/{}/meta.parquet'.format(region))

    meta_region_file.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(meta_region_file)
    
    save_index=[]

    for i, (index, row) in enumerate(meta.iterrows()):   
        
        evaporation_out = pl.Path("{}/{}/data/evaporation_{}.parquet".format(output_directory, region, index))
        if evaporation_out.exists():
            continue
        
        row_type = row["type"]
        row_id = row["id"]
        
        type_directory = pl.Path("{}/{}".format(observations_directory, row_type))
        data_directory = pl.Path("{}/data/extracted".format(type_directory))
        
        data_file = [file for file in data_directory.rglob("FLX_{}_*".format(row_id)) if file.is_file()][0]
        evaporation = pd.read_csv(data_file)
        
        evaporation = evaporation.loc[(evaporation['LE_F_MDS_QC'] <= 1) & (evaporation['LE_F_MDS_QC'] > 0)]
        evaporation = evaporation.loc[(evaporation['TA_F_MDS_QC'] <= 1) & (evaporation['TA_F_MDS_QC'] > 0)]
        evaporation = evaporation[["TIMESTAMP", "LE_F_MDS", "TA_F_MDS"]].copy()
        evaporation.columns = ["date", "evaporation", "tair"]
        
        evaporation["date"] = pd.to_datetime(evaporation["date"], format="%Y%m%d")
        evaporation["evaporation"] = pd.to_numeric(evaporation["evaporation"], errors = "coerce")
        evaporation["tair"] = pd.to_numeric(evaporation["tair"], errors = "coerce")
                
        lhoe = 2.501 - (2.361 * 1e-3) * evaporation["tair"] # Latent heat of vaporization MJ kg-1
        lhoe = lhoe * 1e6 # MJ kg-1 to J kg-1
        evaporation["evaporation"] = evaporation["evaporation"] / lhoe # J m-2 s-1 to kg m-2 s-1
                
        evaporation = evaporation.drop("tair", axis = 1)
        evaporation = evaporation.dropna(axis = 0)
        evaporation = evaporation.astype({"date": "datetime64[ns]",
                                          "evaporation": "float32"})
        
        # Subset overlap period
        #evaporation_sel = np.logical_and(evaporation["date"] >= period["start"],
        #                                 evaporation["date"] <= period["end"])
        #evaporation = evaporation.loc[evaporation_sel]
        
        if len(evaporation)>365*2:

            evaporation_out.parent.mkdir(parents=True, exist_ok=True)
            evaporation.to_parquet(evaporation_out)        

            save_index.append(index)
            
            print("\tProcessed station {}: Len {} ({} out of {})".format(index,len(evaporation), i, meta.index.size))
        
    meta=meta.loc[save_index]
    meta.to_parquet("{}/{}/meta.parquet".format(output_directory,region))
    meta.to_csv("{}/{}/meta.csv".format(output_directory,region))     
     
        
