#%%
import pathlib as pl

save_directory = pl.Path("saves/observations")
observations_directory = pl.Path("/p/data1/slts/avila2/Stars4Water/WP3/benchmarking/data/observations/q")
common_directory = pl.Path("saves/common/")
output_directory = pl.Path("saves/observations/")

import warnings
import datetime as dt
import re
import numpy as np
import pandas as pd

regions = [dir.stem for dir in save_directory.iterdir() if dir.is_dir() if dir.is_dir()]
#%%
for region in regions:
    
    print("Region: {}".format(region))
    
    save_region_directory = pl.Path("{}/{}".format(save_directory, region))
    common_region_directory = pl.Path("{}/{}".format(common_directory, region))
    
    meta_file = pl.Path("{}/meta.parquet".format(save_region_directory))
    meta = pd.read_parquet(meta_file)
    
    period_file = pl.Path("{}/period.csv".format(common_region_directory))
    period = pd.read_csv(period_file, parse_dates=["start", "end"]).iloc[0]
    
    exists = True
    for i, (index, row) in enumerate(meta.iterrows()):  
        discharge_out = pl.Path("{}/{}/data/discharge_{}.parquet".format(output_directory, region, index))
        if not discharge_out.exists():
            exists = False
            break
    if exists:
        print("- Already exists")
        continue
    
    for i, (index, row) in enumerate(meta.iterrows()):        
        discharge_out = pl.Path("{}/{}/data/discharge_{}.parquet".format(output_directory, region, index))
        if discharge_out.exists():
            continue
        
        row_type = row["type"]
        row_id = row["id"]
        
        type_directory = pl.Path("{}/{}".format(observations_directory, row_type))
        data_directory = pl.Path("{}/data".format(type_directory))
                
        if row_type == "GRDC":
            data_file = pl.Path("{}/{}_Q_Day.Cmd.txt".format(data_directory, row_id))
            discharge = pd.read_csv(data_file, delimiter=";", encoding = "ISO-8859-1", comment="#")
            
            discharge.columns = ["date", None, "discharge"]
            discharge = discharge[["date", "discharge"]]
            
            discharge["date"] = dates = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in discharge["date"]]
            discharge["discharge"] = pd.to_numeric(discharge["discharge"], errors = "coerce")
            discharge.loc[discharge["discharge"] < 0, "discharge"] = np.nan
            
        elif row_type == "Arpa_Emilia_Romagna":
            data_file = pl.Path("{}/{}.csv".format(data_directory, row_id))
            
            discharge = pd.read_csv(data_file, skiprows=3, on_bad_lines="skip")
            discharge = discharge.iloc[:-2]
            
            discharge.columns = [None, "date", "discharge"]
            discharge = discharge[["date", "discharge"]]
            
            discharge["date"] = dates = [dt.datetime.strptime(re.sub("\\+00:00", "", date), "%Y-%m-%d %H:%M:%S").date() for date in discharge["date"]]
            discharge["discharge"] = pd.to_numeric(discharge["discharge"], errors = "coerce")
            discharge.loc[discharge["discharge"] < 0, "discharge"] = np.nan
        
        elif row_type == "Trento":
            data_file = pl.Path("{}/discharge_po_river.csv".format(data_directory))
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
                discharge = pd.read_csv(data_file, skiprows=2, header=1)
            discharge = discharge.iloc[3:]
            discharge.columns = [c.strip() for c in discharge.columns]
            
            id_name = "value_{}".format(row_id)
            discharge = discharge[["timestamp", id_name]]
            discharge.columns = ["date", "discharge"]
            
            discharge["date"] = dates = [dt.datetime.strptime(date, "%Y-%m-%d %H:%M").date() for date in discharge["date"]]
            discharge["discharge"] = pd.to_numeric(discharge["discharge"], errors = "coerce")
            discharge.loc[discharge["discharge"] < 0, "discharge"] = np.nan
            
        elif row_type == "CAMELS-CH":
            data_file = pl.Path("{}/time_series/observation_based/CAMELS_CH_obs_based_{}.csv".format(data_directory.parent, row_id))
            
            discharge = pd.read_csv(data_file, sep = ";")
            discharge = discharge[["date", "discharge_vol(m3/s)"]]
            discharge.columns = ["date", "discharge"]
            
            discharge["date"] = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in discharge["date"]]
            discharge["discharge"] = pd.to_numeric(discharge["discharge"], errors = "coerce")
            discharge.loc[discharge["discharge"] < 0, "discharge"] = np.nan
            
        elif row_type == "CAMELS-GB":
            data_file = pl.Path("{}/timeseries/CAMELS_GB_hydromet_timeseries_{}_19701001-20150930.csv".format(data_directory, row_id))
            
            discharge = pd.read_csv(data_file)
            discharge = discharge[["date", "discharge_vol"]]
            discharge.columns = ["date", "discharge"]
            
            discharge["date"] = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in discharge["date"]]
            discharge["discharge"] = pd.to_numeric(discharge["discharge"], errors = "coerce")
            discharge.loc[discharge["discharge"] < 0, "discharge"] = np.nan
            
        elif row_type == "DWS":
            data_file = pl.Path("{}/{}.csv".format(data_directory, row_id))
            
            discharge = pd.read_csv(data_file, skiprows=4, header = None)
            discharge.columns = ["Date","Level (Metres)", "","Cor.Lev. (Metres)", "","Discharge (Cumecs) Final flow", ""]
            discharge = discharge[["Date", "Discharge (Cumecs) Final flow"]]
            discharge.columns = ["date", "discharge"]
            
            discharge["date"] = [dt.datetime.strptime(str(date), "%Y%m%d").date() for date in discharge["date"]]
            discharge["discharge"] = pd.to_numeric(discharge["discharge"], errors = "coerce")
            discharge.loc[discharge["discharge"] < 0, "discharge"] = np.nan
            
        elif row_type == "LamaH":
            data_file = pl.Path("{}/D_gauges/2_timeseries/daily/ID_{}.csv".format(data_directory.parent, row_id))
            
            discharge = pd.read_csv(data_file, sep = ";")
            discharge["date"] = [dt.date(r["YYYY"], r["MM"], r["DD"]) for _, r in discharge[["YYYY", "MM", "DD"]].iterrows()]
            discharge = discharge[["date", "qobs"]]
            discharge.columns = ["date", "discharge"]
            
            discharge["discharge"] = pd.to_numeric(discharge["discharge"], errors = "coerce")
            discharge.loc[discharge["discharge"] < 0, "discharge"] = np.nan
            
        elif row_type == 'NVE':
             data_file = pl.Path("{}/{}.csv".format(data_directory, row_id))

             discharge = pd.read_csv(data_file)
             discharge = discharge[["date", "discharge"]]
             discharge["date"] = [dt.datetime.strptime(str(date), "%Y-%m-%d").date() for date in discharge["date"]]

             discharge["discharge"] = pd.to_numeric(discharge["discharge"], errors = "coerce")
             discharge.loc[discharge["discharge"] < 0, "discharge"] = np.nan
             
        elif row_type == 'CAMELS-FR':
             data_file = pl.Path("{}/CAMELS_FR_time_series/daily/CAMELS_FR_tsd_{}.csv".format(data_directory, row_id))

             discharge = pd.read_csv(data_file, sep = ";", encoding = "ISO-8859-1", comment="#")    
             discharge = discharge[["tsd_date", "tsd_q_l"]]
             discharge["tsd_date"] = np.array([dt.datetime.strptime(str(date), "%Y%m%d").date() for date in discharge["tsd_date"]])
             discharge.columns = ["date", "discharge"]

             discharge["discharge"] = pd.to_numeric(discharge["discharge"], errors = "coerce")
             discharge.loc[discharge["discharge"] < 0, "discharge"] = np.nan
             q=discharge.discharge/1000
             discharge['discharge']=q

            
        else:
            raise ValueError("Unknown discharge meta type {}".format(row["type"]))
        
        discharge = discharge.dropna(axis = 0)
        discharge = discharge.astype({"date": "object",
                                      "discharge": "float32"})
        
        # Subset overlap period
        discharge_sel = np.logical_and(discharge["date"] >= period["start"].date(),
                                       discharge["date"] <= period["end"].date())
        discharge = discharge.loc[discharge_sel]
        discharge.date=pd.to_datetime(discharge.date)        
        
        discharge_out.parent.mkdir(parents=True, exist_ok=True)
        discharge.to_parquet(discharge_out)
        
        print("- Processed gauge {} ({} out of {})".format(index, i, meta.index.size))
    
