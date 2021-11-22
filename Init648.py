############################################ Import Python Libraries Required ############################################ 
import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors

import datetime
import os 
from os import path
import seaborn as sns
from scipy import signal

import warnings
warnings.filterwarnings('ignore')

import pickle

import statistics
import math 
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
# %config InlineBackend.figure_format = 'retina'
from PIL import Image

# ############################# Update Actual Cases #############################################3
# import requests
# os.system("rm state_wise_daily.csv")
# url = """https://data.covid19india.org/csv/latest/state_wise_daily.csv"""
# response = requests.get(url)


# with open( "state_wise_daily.csv", 'wb') as f:
#     f.write(response.content)
# df  = pd.read_csv("state_wise_daily.csv",delimiter=",")

# listOfColumns  = ["Date","Date_YMD","Status","KA"]
# df = df[listOfColumns].copy(deep=True)
# df["Date"] = pd.to_datetime(df["Date"])

# ## Sum all the values upto Jul-1st 2020 and add that as cumuliative total to Jul-1st 2020
# ## The data fitted to the model starts only from Jul-1st 2020

# ## Sum of COnfirmed 
# df_temp = df[(df["Date"] <= '2020-07-01') & (df["Status"]=="Confirmed")]
# ConfirmVal = df_temp["KA"].cumsum().iloc[-1]
# index = df.loc[(df["Date"] =='2020-07-01') & (df["Status"] == "Confirmed")]["KA"].index
# df.loc[index,"KA"] = ConfirmVal

# ## Sum of COnfirmed  
# df_temp = df[(df["Date"] <= '2020-07-01') & (df["Status"]=="Recovered")]
# RecoverVal = df_temp["KA"].cumsum().iloc[-1]
# index = df.loc[(df["Date"] =='2020-07-01') & (df["Status"] == "Recovered")]["KA"].index
# df.loc[index,"KA"] = RecoverVal

# ## Sum of Deceased 
# df_temp = df[(df["Date"] <= '2020-07-01') & (df["Status"]=="Deceased")]
# DeceaseVal = df_temp["KA"].cumsum().iloc[-1]
# index = df.loc[(df["Date"] =='2020-07-01') & (df["Status"] == "Deceased")]["KA"].index
# ## -2 , HARDCODED for adjustment with the data used to fir the model ( The current data we have has 253 as Total Deceased as)
# df.loc[index,"KA"] = DeceaseVal - 2   


# #Copy data only from Jun-01-2020
# df = df[(df["Date"] >= '2020-07-01')].copy(deep=True)

# ##reset index of Dataframe
# df.reset_index(inplace=True,drop=True)


# ## get the KA columns from the dataframe
# totaldata_all = df.KA
# totalDays  = int(len(totaldata_all)/3)

# N_dataDays = totalDays

# ##Declare Arrays 
# actualAct = np.zeros(N_dataDays) 
# actualRecov = np.zeros(N_dataDays)  
# actualTot = np.zeros(N_dataDays) 
# actualDes = np.zeros(N_dataDays) 

# ## Get the first day data
# MarInitTot = totaldata_all[0]
# MarInitRecov = totaldata_all[1]
# MarInitDes = totaldata_all[2]

# ##initial Setup
# actualTot[0] = MarInitTot 
# actualRecov[0] = MarInitRecov 
# actualDes[0] = MarInitDes 
# actualAct[0] = actualTot[0] - actualRecov[0] - actualDes[0]


# for idx in range(1, N_dataDays):
#     actualTot[idx] = actualTot[idx-1] + totaldata_all[3*(idx)]
#     actualRecov[idx] = actualRecov[idx-1] + totaldata_all[3*(idx)+1]
#     actualDes[idx] = actualDes[idx-1] + totaldata_all[3*(idx)+2]
#     actualAct[idx] = actualTot[idx] - actualRecov[idx] - actualDes[idx]


# ActualDict = {}
# ActualDict["Actual"] = actualAct
# ActualDict["Deceased"] = actualDes
# ActualDict["Recovered"] = actualRecov
# ActualDict["Cumuliative"] = actualTot

# file_name = "Data_Files/actual_Data.pkl"

# open_file = open(file_name, "wb")
# pickle.dump(ActualDict, open_file)
# open_file.close()

###################################### Actual Cases Updated ####################################################

###################################### Read Predicted and Actual Cases #########################################
df_predicted = pd.read_pickle("Data_Files/df_active")
df_predicted = df_predicted[df_predicted['KDPwsat2']!='ImmunEscp_Jul']
with open('Data_Files/actual_Data.pkl', 'rb') as f: 
    actual_cases = pickle.load(f) #actual number of active cases 


###################################### Read Ensemble Mean And Standard Errors ###################################
ensemble_mean = []
ensemble_ub = []
ensemble_lb = []
ensemble_sem=[]
for i in range(731):
    ensemble_mean.append(np.mean(df_predicted.iloc[:,i]))
    ensemble_sem.append(scipy.stats.sem(df_predicted.iloc[:,i]))
    ensemble_ub.append(ensemble_mean[i]+ensemble_sem[i])
    ensemble_lb.append(ensemble_mean[i]-ensemble_sem[i])


###################################### Update ###################################

N = 648 #No. of realizations
SF = 5e4 #Scaling Factor 
D = 0
latestDay=0
weightr = np.zeros(N) # for storing rmse weights
weightp = np.zeros(N) # for storing percentile weights
weighte = np.zeros(N) # for sem

weightr[:] = 1.0/N #Initialize

SSE = np.zeros(N) #Sum of squared errors
MSE = np.zeros(N) #Mean Squared error
RMSE = np.zeros(N) #Root Mean Squared error

for i in range(N):
    for j in range(0,len(actual_cases['Actual'])-D):
        SSE[i] += (((df_predicted.iloc[i,j]-actual_cases['Actual'][j])/SF)**2)
        latestDay = j
        
for i in range(N):
    MSE[i] = SSE[i]/len(actual_cases['Actual'])
    RMSE[i] = math.sqrt(MSE[i])
    weightr[i] = np.exp(-1.0*RMSE[i])*weightr[i]
    

sum_weightr = sum(weightr)


for i in range(N):
    weightr[i] = (weightr[i]/sum_weightr)
    weighte[i] = weightr[i]*N




# with open('Data_Files/wpcum.pkl', 'rb') as f:
#     # weightpcum = pickle.load(f)

p25 = np.percentile(weightr,25)
p50 = np.percentile(weightr,50)
p75 = np.percentile(weightr,75)

max_list = [i for i, j in enumerate(weightr) if j == max(weightr)]


for i in range(N):
    if(weightr[i]<p25):
        weightp[i]=0
    elif(p25<=weightr[i]<p50):
        weightp[i]=0.25
    elif(p50<=weightr[i]<p75):
        weightp[i]=0.5
    else:
        weightp[i]=0.75
    
for i in max_list:
    weightp[i] = 1.0   

# weightpcum.append(weightp)



df_posterior = df_predicted.copy(deep=True)
df_posterior['WeightRMSE'] = weightr
df_posterior['WeightSEM'] = weighte
for i in range(N):
    for j in range(731):    
        df_posterior.iloc[i,j] = df_posterior.iloc[i,j]*df_posterior['WeightSEM'][i]


weighted_mean= []
weighted_sem = []
weighted_ub = []
weighted_lb = []



for i in range(731):
    
    weighted_mean.append(np.mean(df_posterior.iloc[:,i]))
    weighted_sem.append(scipy.stats.sem(df_posterior.iloc[:,i]))
    weighted_ub.append(weighted_mean[i]+weighted_sem[i])
    weighted_lb.append(weighted_mean[i]-weighted_sem[i])



###################################### Update Input Files ###################################
with open('Data_Files/sse648.pkl', 'wb') as f:
    pickle.dump(SSE, f)

with open('Data_Files/day648.pkl','wb') as f:
    pickle.dump(latestDay,f)       
        
###################################### Update Output Files ###################################
with open('Data_Files/wm648.pkl', 'wb') as f:
    pickle.dump(weighted_mean, f)

with open('Data_Files/wub648.pkl','wb') as f:
    pickle.dump(weighted_ub,f)

with open('Data_Files/wlb648.pkl','wb') as f:
    pickle.dump(weighted_lb,f)

with open('Data_Files/wr648.pkl','wb') as f:
    pickle.dump(weightr,f)

with open('Data_Files/ensemblemean648.pkl','wb') as f:
    pickle.dump(ensemble_mean,f)
with open('Data_Files/ensembleub648.pkl','wb') as f:
    pickle.dump(ensemble_ub,f)
with open('Data_Files/ensemblelb648.pkl','wb') as f:
    pickle.dump(ensemble_lb,f)