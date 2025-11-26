#%%
import pathlib as pl
import pandas as pd
import numpy as np

observations_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/observations/wtd")
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
    
    meta_region_file=pl.Path('../saves/observations/{}/meta.csv'.format(region))
    meta_region_file.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(meta_region_file)
    
    save_index=[]
    
    for i, (index, row) in enumerate(meta.iterrows()):   
        
        id=row['id']
        
        wtd_out = pl.Path("{}/{}/data/wtd_{}.parquet".format(output_directory, region, index))
        
        if wtd_out.exists():
            continue
    
        wtd = pd.read_parquet('../observations/saves/WTDobs_{}.parquet'.format(id))
        
        wtd_out.parent.mkdir(parents=True, exist_ok=True)
        wtd.to_parquet(wtd_out)        

        save_index.append(index)
        
        print("\tProcessed station {}: Len {} ({} out of {})".format(index,len(wtd), i, meta.index.size))
            
            
#%%+

