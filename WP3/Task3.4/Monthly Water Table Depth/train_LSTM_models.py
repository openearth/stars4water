import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
#import hydroeval as he
import math
import sys 
import warnings
warnings.filterwarnings("ignore")


def kge(simulations, evaluation):
    """Original Kling-Gupta Efficiency (KGE) and its three components
    (r, α, β) as per `Gupta et al., 2009
    <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.

    Note, all four values KGE, r, α, β are returned, in this order.

    :Calculation Details:
        .. math::
           E_{\\text{KGE}} = 1 - \\sqrt{[r - 1]^2 + [\\alpha - 1]^2
           + [\\beta - 1]^2}
        .. math::
           r = \\frac{\\text{cov}(e, s)}{\\sigma({e}) \\cdot \\sigma(s)}
        .. math::
           \\alpha = \\frac{\\sigma(s)}{\\sigma(e)}
        .. math::
           \\beta = \\frac{\\mu(s)}{\\mu(e)}

        where *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, *cov* is the covariance, *σ* is the
        standard deviation, and *μ* is the arithmetic mean.

    """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = np.mean(simulations, axis=0, dtype=np.float64)
    obs_mean = np.mean(evaluation, dtype=np.float64)

    r_num = np.sum((simulations - sim_mean) * (evaluation - obs_mean),
                   axis=0, dtype=np.float64)
    r_den = np.sqrt(np.sum((simulations - sim_mean) ** 2,
                           axis=0, dtype=np.float64)
                    * np.sum((evaluation - obs_mean) ** 2,
                             dtype=np.float64))
    r = r_num / r_den
    # calculate error in spread of flow alpha
    alpha = np.std(simulations, axis=0) / np.std(evaluation, dtype=np.float64)
    # calculate error in volume beta (bias of mean discharge)
    beta = (np.sum(simulations, axis=0, dtype=np.float64)
            / np.sum(evaluation, dtype=np.float64))
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return np.vstack((kge_, r, alpha, beta))

def pbias(simulations, evaluation):
    """Percent Bias (PBias).

    :Calculation Details:
        .. math::
           E_{\\text{PBias}} = 100 × \\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})}{\\sum_{i=1}^{N}e_{i}}

        where *N* is the length of the *simulations* and *evaluation*
        periods, *e* is the *evaluation* series, and *s* is (one of)
        the *simulations* series.

    """
    pbias_ = (100 * np.sum(evaluation - simulations, axis=0, dtype=np.float64)
              / np.sum(evaluation))

    return pbias_

class LSTM(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE):
        super(LSTM, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.HIDDEN_SIZE).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.HIDDEN_SIZE).to(x.device)

        h_out, _ = self.lstm(x, (h0, c0))

        out = self.fc(h_out[:, -1, :])
        return out

def create_dataset(dataset, sliding_window,date):
    x=dataset[:,:-1]
    dataY=dataset[sliding_window-1:,-1]
    date=date[sliding_window-1:]

    dataX = []
    
    for i in range(len(dataset)-sliding_window+1):
        a = x[i:(i+sliding_window)]
        dataX.append(a)
    return np.array(dataX), np.array(dataY),date



def eval_model(lstm_model,dataloader):
    lstm_model.eval()
    predictions = []
    observaton=[]
    
    with torch.no_grad():
        for inputs, y in dataloader:
            outputs = lstm_model(inputs)
            predictions.extend(outputs.tolist())
            observaton.extend(y.tolist())
    
    sim_train=sy.inverse_transform(pd.DataFrame(predictions))
    obs_train=sy.inverse_transform(pd.DataFrame(observaton))
    
    return obs_train,sim_train


sliding_windows=[4,6,8]

dt=['model1','model2','model3','model4','model5']
 
config_points=pd.DataFrame()
config_points['lat']=np.nan
config_points['lon']=np.nan
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

TRAINING_PERIOD = 200
BATCH_SIZE=5

HIDDEN_SIZE = 3      # this is the hidden size of the LSTM. It controlls the size of c and h!
OUTPUT_SIZE = 1       # this tells our LSTM (see below) how many variables we are regressing. Just 1 is enough for now
NUM_EPOCHS = 300   # by now you know what epochs are
LEARNING_RATE = 0.01  # And the learning rate for the optimizer!

DATA=np.load('../meta/DATA_seine.npy',allow_pickle=True).item()

coordinates=pd.read_csv('../meta/coordinates_seine_tsmp.csv',index_col=0)


vmin=int(sys.argv[1])
vmax=int(sys.argv[2])

#point=2
for point in range(vmin,vmax):  
   
    kge_te=-99
    
    dataset1=DATA[point][['grace','wtd']]
    dataset2=DATA[point][['grace','pr','wtd']]
    dataset3=DATA[point][['grace','tmax','wtd']]
    dataset4=DATA[point][['grace','evap','wtd']]
    dataset5=DATA[point][['grace','pr','tmax','evap','wtd']]
    
    datasets=[dataset1,dataset2,dataset3,dataset4,dataset5]
    
    lat=coordinates['lat'].iloc[point]
    lon=coordinates['lon'].iloc[point]
    
    
    SIM_test={}
    SIM_train={}
    
    ii=0
    
    print('OK')
    
    for i in range(len(datasets)):       
        
        for sliding_window in sliding_windows:
    
                data=datasets[i]
                
                sy = MinMaxScaler()
                sx = StandardScaler()
                
                X = sx.fit_transform(data.iloc[:,:-1])
                y = sy.fit_transform(pd.DataFrame(data.iloc[:,-1]))
                
                X_train = X[0:TRAINING_PERIOD]
                y_train = y[0:TRAINING_PERIOD]
                
                X_test = X[TRAINING_PERIOD:]
                y_test = y[TRAINING_PERIOD:]   
                
                date_train=data.index[0:TRAINING_PERIOD]
                date_test=data.index[TRAINING_PERIOD:]   
    
                train = np.concatenate([X_train, y_train], axis=1)
                test = np.concatenate([X_test, y_test], axis=1)
                
                trainX, trainY, dat_train = create_dataset(train, sliding_window,date_train)
                testX, testY,dat_test   = create_dataset(test, sliding_window,date_test)
                
                dataset_train = TensorDataset(torch.tensor(trainX).float(), torch.tensor(trainY).float())
                dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)
                
                dataset_test = TensorDataset(torch.tensor(testX).float(), torch.tensor(testY).float())
                dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
                
                #initialization
                INPUT_SIZE = len(data.iloc[:,:-1].columns)
    
                lstm_model = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
                
                train_losses = []
                valid_losses = []
                
                # training
                for epoch in range(NUM_EPOCHS):
                    loss = 0.0
                    for inputs, y in dataloader_train:
                        
                        if INPUT_SIZE>1:
                            inputs=inputs.squeeze(-1)
                        optimizer.zero_grad()                    
               
                        y_hat = lstm_model(inputs)
                        loss = criterion(y_hat.flatten(), y)
                        loss.backward()
                        optimizer.step()
                        
                    valid_loss = 0.0
                    if (epoch+1) % 100 == 0:
                
                        outputs = lstm_model(inputs)
                        loss = criterion(outputs, y)
                        valid_loss += loss.item() * inputs.size(0)
                        #print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {loss:.6f}, Valid Loss: {valid_loss:.6f}")
                        
    
                obs_train, sim_train=eval_model(lstm_model,dataloader_train)
                obs_test, sim_test=eval_model(lstm_model,dataloader_test)
    
                kge_train, r_train, alpha_train, beta_train = kge(sim_train,obs_train) 
                kge_test, r_test, alpha_test, beta_test = kge(sim_test,obs_test) 
    
                pbias_train = pbias(sim_train,obs_train) 
                pbias_test = pbias(sim_test,obs_test) 
    
                MSE_t = np.square(np.subtract(obs_train,sim_train)).mean()             
                RMSE_t = [math.sqrt(MSE_t)]
    
                MSE_te = np.square(np.subtract(obs_test,sim_test)).mean()             
                RMSE_te = [math.sqrt(MSE_te)]
    
                if kge_test>kge_te:   
                    
                    sim_train=sim_train.copy()
                    sim_test=sim_test.copy()
                    
                    Dat_train=dat_train.copy()
                    Dat_test=dat_test.copy()
                        
                    sim_test=pd.DataFrame(sim_test[:,0],index=Dat_test)
                    sim_train=pd.DataFrame(sim_train[:,0],Dat_train)               
                        
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
                    
                print('\t> Model {} of 15'.format(ii))
                
                ii=ii+1
                    
    config_points.at[point,'lat']=lat
    config_points.at[point,'lon']=lon
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
    
    SIM_test[point]=sim_test[:,0]
    SIM_train[point]=sim_train[:,0]         
    
    config_points.to_csv(f'saves_pytorch/results_lstm_seine_{point}.csv',index=False)
    np.save(f'saves_pytorch/sim_test_lstm_seine_{point}.npy', SIM_test)
    np.save(f'saves_pytorch/sim_train_lstm_seine_{point}.npy', SIM_train)


#%%

# plt.plot(DATA[239].wtd)
# #%%
# point=239
# NUM_EPOCHS = 500    # by now you know what epochs are


# dataset1=DATA[point][['grace','wtd']]
# dataset2=DATA[point][['grace','pr','wtd']]
# dataset3=DATA[point][['grace','tmax','wtd']]
# dataset4=DATA[point][['grace','evap','wtd']]
# dataset5=DATA[point][['grace','pr','tmax','evap','wtd']]

# datasets=[dataset1,dataset2,dataset3,dataset4,dataset5]

# data=datasets[i]

# sy = MinMaxScaler()
# sx = StandardScaler()

# X = sx.fit_transform(data.iloc[:,:-1])
# y = sy.fit_transform(pd.DataFrame(data.iloc[:,-1]))

# X_train = X[0:TRAINING_PERIOD]
# y_train = y[0:TRAINING_PERIOD]

# X_test = X[TRAINING_PERIOD:]
# y_test = y[TRAINING_PERIOD:]   

# date_train=data.index[0:TRAINING_PERIOD]
# date_test=data.index[TRAINING_PERIOD:]   

# train = np.concatenate([X_train, y_train], axis=1)
# test = np.concatenate([X_test, y_test], axis=1)

# trainX, trainY, dat_train = create_dataset(train, sliding_window,date_train)
# testX, testY,dat_test   = create_dataset(test, sliding_window,date_test)

# dataset_train = TensorDataset(torch.tensor(trainX).float(), torch.tensor(trainY).float())
# dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)

# dataset_test = TensorDataset(torch.tensor(testX).float(), torch.tensor(testY).float())
# dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# #initialization
# INPUT_SIZE = len(data.iloc[:,:-1].columns)

# lstm_model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

# train_losses = []
# valid_losses = []

# # training
# for epoch in range(NUM_EPOCHS):
#     loss = 0.0
#     for inputs, y in dataloader_train:
        
#         if INPUT_SIZE>1:
#             inputs=inputs.squeeze(-1)
#         optimizer.zero_grad()
#         y_hat = lstm_model(inputs)
#         loss = criterion(y_hat.flatten(), y)
#         loss.backward()
#         optimizer.step()
        
#     valid_loss = 0.0
#     if (epoch+1) % 100 == 0:

#         outputs = lstm_model(inputs)
#         loss = criterion(outputs, y)
#         valid_loss += loss.item() * inputs.size(0)
#         print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {loss:.6f}, Valid Loss: {valid_loss:.6f}")
        

# obs_train, sim_train=eval_model(lstm_model,dataloader_train)
# obs_test, sim_test=eval_model(lstm_model,dataloader_test)

