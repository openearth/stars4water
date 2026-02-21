import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import xarray as xr 

data=pd.read_csv('saves/METRIC_SEINE.csv')
mask_nc=xr.open_dataset('../meta/mask_tsmp_wtd_seine.nc')

save_path='figures'

lat=data.lat
lon=data.lon

columns=data.columns

data['Model']=np.nan

for i in range(len(data)):
    
    value=data['dataset'].iloc[i][-1]
    data.at[i,'Model']=int(value)

#%%
fig, ax = plt.subplots(figsize=(11,11))

mask_nc.mask.plot(add_colorbar=False,levels=3,colors=['#737373'],alpha=0.12)
scatter = ax.scatter(lon,lat, c=data.Model,cmap='RdYlBu',s=500,marker='s')

legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Model")
ax.add_artist(legend1)
plt.title('Models - Random Forest')

plt.savefig('{}/Models_Seine.jpg'.format(save_path),dpi=600,bbox_inches='tight')

#%%
# Plot KGE
data.kge_test=np.where(data.kge_test<0,0,data.kge_test)

fig, ax = plt.subplots(figsize=(13,10))

mask_nc.mask.plot(add_colorbar=False,levels=3,colors=['#737373'],alpha=0.22)
scatter = ax.scatter(lon,lat, c=data.kge_test,cmap='RdYlBu',s=400,marker='s',vmin=0, vmax=1)

cbar =plt.colorbar(scatter)
cbar.set_label('KGE')
plt.title('KGE - Test Period - Random Forest')
plt.savefig('{}/KGE_Seine.jpg'.format(save_path),dpi=600,bbox_inches='tight')


#%%
# Plot Correlation
fig, ax = plt.subplots(figsize=(13,10))

mask_nc.mask.plot(add_colorbar=False,levels=3,colors=['#737373'],alpha=0.22)
scatter = ax.scatter(lon,lat, c=data.pbias_test,cmap='RdYlBu',s=400,marker='s',vmin=-30, vmax=30)

cbar =plt.colorbar(scatter)
cbar.set_label('PBIAS %')

plt.title('PBIAS - Test Period - Random Forest')
plt.savefig('{}/PBIas_Seine.jpg'.format(save_path),dpi=600,bbox_inches='tight')


#%%

fig, ax = plt.subplots(figsize=(13,10))

mask_nc.mask.plot(add_colorbar=False,levels=3,colors=['#737373'],alpha=0.22)
scatter = ax.scatter(lon,lat, c=data.r_test,cmap='RdYlBu',s=400,marker='s',vmin=0, vmax=1)

cbar =plt.colorbar(scatter)
cbar.set_label('Correlation')

plt.title('Correlation - Test Period - Random Forest')
plt.savefig('{}/Corr_RF_Seine.jpg'.format(save_path),dpi=600,bbox_inches='tight')


