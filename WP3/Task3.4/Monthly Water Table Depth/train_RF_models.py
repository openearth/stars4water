import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
import matplotlib.pyplot as plt 
import hydroeval as he
import math 
import sys

def create_sequence(dataset,sliding_window):
    x=dataset.iloc[:,:-1]
    
    dataY=dataset.iloc[sliding_window-1:,-1]
    
    date=dataset.index[sliding_window-1:]
    
    dataX=[]
    
    for i in range(len(dataset)-sliding_window+1):
        a = x[i:(i+sliding_window)].values.reshape(-1)        
        dataX.append(a)
       
    dataX=np.array(dataX)   
    
    return dataX,dataY,date

train_size=72

data=np.load('../meta/DATA_seine.npy',allow_pickle=True).item()

#%%
METRIC=pd.DataFrame()
METRIC['model1']=np.nan
METRIC['model2']=np.nan
METRIC['model3']=np.nan
METRIC['model4']=np.nan
METRIC['model5']=np.nan


sliding_windows=[4,6,8]

dt=['model1','model2','model3','model4','model5']
 
config_points=pd.DataFrame()
config_points['dataset']=np.nan
config_points['lag']=np.nan
config_points['kge_train']=np.nan
config_points['kge_test']=np.nan
config_points['r_train']=np.nan
config_points['r_test']=np.nan
config_points['pbias_train']=np.nan
config_points['pbias_test']=np.nan
config_points['rmse_train']=np.nan
config_points['rmse_test']=np.nan

config_points['dataset']=config_points['dataset'].astype(object)

SIM_test={}
SIM_train={}

#%%
set_points=[]

for i in range(15):
    set_points.append(i*50)

set_points.append(741)

se=[]
for i in range(15):
    se.append([set_points[i],set_points[i+1]])

set_points=2

vmin=se[set_points][0]
vmax=se[set_points][1]

#%%
for point in range(vmin,vmax):
    
    kge_te=-99
    
    dataset1=data[point][['grace','wtd']]
    dataset2=data[point][['grace','pr','wtd']]
    dataset3=data[point][['grace','tmax','wtd']]
    dataset4=data[point][['grace','evap','wtd']]
    dataset5=data[point][['grace','pr','tmax','evap','wtd']]
    
    datasets=[dataset1,dataset2,dataset3,dataset4,dataset5]
    
    
    for i in range(len(datasets)):
        
        for sliding_window in sliding_windows:
    
            dataX,dataY,date=create_sequence(datasets[i],sliding_window)
            
            rf = RandomForestRegressor(n_estimators=1000, random_state=0,criterion='squared_error')
            
            trainX=dataX[train_size:,:]
            trainY=dataY[train_size:]
            
            testX=dataX[:train_size:,:]
            testY=dataY[:train_size:]
            
            rf.fit(trainX, trainY)
            
            y_pred_train = rf.predict(trainX)     
            y_pred_test = rf.predict(testX)           
            
            kge_train, r_train, alpha_train, beta_train = he.evaluator(he.kge, y_pred_train, trainY) 
            kge_test, r_test, alpha_test, beta_test = he.evaluator(he.kge, y_pred_test, testY) 
    
            pbias_train = he.evaluator(he.pbias, y_pred_train, trainY) 
            pbias_test = he.evaluator(he.pbias, y_pred_test, testY) 
            
            
            MSE_t = np.square(np.subtract(trainY,y_pred_train)).mean()             
            RMSE_t = [math.sqrt(MSE_t)]
            
            MSE_te = np.square(np.subtract(testY,y_pred_test)).mean()             
            RMSE_te = [math.sqrt(MSE_te)]
    
            if kge_test>kge_te:
                
                date_train=date[train_size:]
                date_test=date[:train_size]
    
                sim_train=y_pred_train.copy()
                sim_test=y_pred_test.copy()
                
                sim_test=pd.DataFrame(sim_test,index=date_test)
                sim_train=pd.DataFrame(sim_train,index=date_train)
    
                kge_tr=kge_train.copy()   
                kge_te=kge_test.copy()     
                
                r_tr=r_train.copy()   
                r_te=r_test.copy() 
                
                bias_t=pbias_train.copy()   
                bias_te=pbias_test.copy() 
                
                rmse_t=RMSE_t[0]
                rmse_te=RMSE_te[0]
                
                D_dat=dt[i]
                D_lag=sliding_window
           
    
    print(kge_te)
    
    config_points.at[point,'dataset']=D_dat
    config_points.at[point,'lag']=D_lag
    config_points.at[point,'kge_train']=kge_tr
    config_points.at[point,'kge_test']=kge_te
    config_points.at[point,'r_train']=r_tr
    config_points.at[point,'r_test']=r_te
    config_points.at[point,'pbias_train']=bias_t
    config_points.at[point,'pbias_test']=bias_te
    config_points.at[point,'rmse_train']=rmse_t
    config_points.at[point,'rmse_test']=rmse_te
    
    SIM_test[point]=(sim_test)
    SIM_train[point]=(sim_train)
    
    
    print(point)

config_points.to_csv(f'saves/results_seine_{vmin}_{vmax}.csv',index=False)
np.save(f'saves/sim_test_seine_{vmin}_{vmax}.npy', SIM_test)
np.save(f'saves/sim_train_seine_{vmin}_{vmax}.npy', SIM_train)


