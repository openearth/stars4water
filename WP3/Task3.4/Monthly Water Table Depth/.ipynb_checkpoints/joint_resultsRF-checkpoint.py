import pandas as pd
import numpy as np
import glob 

#%%
METRIC=pd.DataFrame()

files=np.sort(glob.glob('saves/*.csv'))

meta=pd.read_csv('../meta/coordinates_seine_tsmp.csv',index_col=0)

for file in files:
    
    data=pd.read_csv(file)
    METRIC=pd.concat([METRIC,data],axis=0)
    
METRIC['lon']=meta['lon'].values
METRIC['lat']=meta['lat'].values


METRIC.to_csv('saves/METRIC_RF.csv')

#%%
period='test'

files_train=np.sort(glob.glob(f'saves/*{period}*'))

FILES={}

for file in files_train:

    data=np.load(file,allow_pickle=True).item()
    
    FILES.update(data)
    
np.save(f'saves/sim_{period}_seine.npy',FILES)