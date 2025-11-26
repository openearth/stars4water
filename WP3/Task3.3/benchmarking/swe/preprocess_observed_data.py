#%%
import pathlib as pl
import pandas as pd
import numpy as np

observations_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/observations/et")
output_directory = pl.Path("../saves/observations/")

meta=pd.read_parquet('../saves/observations/meta.parquet')

regions=meta['region'].unique() 

for region in regions:
    print(region)
    
    meta=pd.read_parquet('../saves/observations/meta.parquet')

    if region!='Europe':
        meta=meta[meta['region']==region]
        #meta=meta.reset_index(drop=True)
        
    meta_region_file=pl.Path('../saves/observations/{}/meta.parquet'.format(region))
    meta_region_file.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(meta_region_file)
    
    meta_region_file=pl.Path('../saves/observations/{}/meta.csv'.format(region))
    meta_region_file.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(meta_region_file)
    
    save_index=[]
    
    for i, (index, row) in enumerate(meta.iterrows()):   
        
        id=row['id']
        
        swe_out = pl.Path("{}/{}/data/swe_{}.parquet".format(output_directory, region, index))
        #if swe_out.exists():
        #    continue
               
        swe = pd.read_parquet('../observations/saves/swe_{}.parquet'.format(id))
        
        if len(swe)>365*2:
            
            swe_df = {"date": pd.to_datetime(swe.index),
                              "swe": swe.iloc[:, 0]}
            swe_df = pd.DataFrame(swe_df)
            
            swe_out.parent.mkdir(parents=True, exist_ok=True)
            swe_df.to_parquet(swe_out)        

            save_index.append(index)
            
            print("\tProcessed station {}: Len {} ({} out of {})".format(index,len(swe), i, meta.index.size))
        
       