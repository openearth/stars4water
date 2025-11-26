#%%
import pathlib as pl
import datetime as dt
import pandas as pd
import numpy as np

observations_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/observations/sm")
output_directory = pl.Path("../saves/observations")

meta=pd.read_csv('../observations/meta_region.csv')

regions=meta['region'].unique()

for region in regions:
    
    meta = pd.read_csv('../observations/meta_region.csv')

    if region!='Europe':
        meta=meta[meta['region']==region]
        
    save_index=[]

    for i, (index, row) in enumerate(meta.iterrows()):    

        moisture_out = pl.Path("{}/{}/data/moisture_{}.parquet".format(output_directory,region, index))
            
        row_type = row["type"]
        row_id = row["id"]

        data_file='/'+row['id']

        moisture = pd.read_csv(data_file, skiprows=1, header=None, delim_whitespace = True, on_bad_lines="skip")

        moisture = moisture.rename({0: "date",
                                            1: "time",
                                            2: "soil_moisture",
                                            3: "ISMN_quality",},
                                    axis=1)

        moisture.loc[moisture["ISMN_quality"] != "G", "soil_moisture"] = np.nan
        moisture = moisture[["date", "soil_moisture"]]
        moisture.columns = ["date", "moisture"]


        moisture["date"] = pd.to_datetime([dt.datetime.strptime(date, "%Y/%m/%d").date() for date in moisture["date"]])
        moisture["moisture"] = pd.to_numeric(moisture["moisture"], errors = "coerce").to_numpy()
        moisture.loc[moisture["moisture"] < 0, "moisture"] = np.nan

        moisture = moisture.dropna(axis = 0)
        moisture = moisture.astype({"date": "datetime64[ns]",
                                    "moisture": "float32"})
        moisture["moisture"] *= 100

        if moisture["moisture"].max() <= 1.0:
                    moisture["moisture"] *= 100

        moisture = moisture.groupby("date").aggregate({"moisture": "mean"})
        moisture = moisture.reset_index()
        
        if len(moisture)>365*2:

            moisture_out.parent.mkdir(parents=True, exist_ok=True)
            moisture.to_parquet(moisture_out)        

            save_index.append(index)
            
            print("\tProcessed station {}: Len {} ({} out of {})".format(index,len(moisture), i, meta.index.size))
      

    meta=meta.loc[save_index]
    meta.to_parquet("{}/{}/meta.parquet".format(output_directory,region))
    meta.to_csv("{}/{}/meta.csv".format(output_directory,region))     
     
        

    
#%%
