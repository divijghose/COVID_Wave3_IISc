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
%config InlineBackend.figure_format = 'retina'

############################# Update Actual Cases #############################################3
import requests
os.system("rm state_wise_daily.csv")
url = """https://data.covid19india.org/csv/latest/state_wise_daily.csv"""
response = requests.get(url)


with open( "state_wise_daily.csv", 'wb') as f:
    f.write(response.content)
df  = pd.read_csv("state_wise_daily.csv",delimiter=",")

listOfColumns  = ["Date","Date_YMD","Status","KA"]
df = df[listOfColumns].copy(deep=True)
df["Date"] = pd.to_datetime(df["Date"])

## Sum all the values upto Jul-1st 2020 and add that as cumuliative total to Jul-1st 2020
## The data fitted to the model starts only from Jul-1st 2020

## Sum of COnfirmed 
df_temp = df[(df["Date"] <= '2020-07-01') & (df["Status"]=="Confirmed")]
ConfirmVal = df_temp["KA"].cumsum().iloc[-1]
index = df.loc[(df["Date"] =='2020-07-01') & (df["Status"] == "Confirmed")]["KA"].index
df.loc[index,"KA"] = ConfirmVal

## Sum of COnfirmed  
df_temp = df[(df["Date"] <= '2020-07-01') & (df["Status"]=="Recovered")]
RecoverVal = df_temp["KA"].cumsum().iloc[-1]
index = df.loc[(df["Date"] =='2020-07-01') & (df["Status"] == "Recovered")]["KA"].index
df.loc[index,"KA"] = RecoverVal

## Sum of Deceased 
df_temp = df[(df["Date"] <= '2020-07-01') & (df["Status"]=="Deceased")]
DeceaseVal = df_temp["KA"].cumsum().iloc[-1]
index = df.loc[(df["Date"] =='2020-07-01') & (df["Status"] == "Deceased")]["KA"].index
## -2 , HARDCODED for adjustment with the data used to fir the model ( The current data we have has 253 as Total Deceased as)
df.loc[index,"KA"] = DeceaseVal - 2   


#Copy data only from Jun-01-2020
df = df[(df["Date"] >= '2020-07-01')].copy(deep=True)

##reset index of Dataframe
df.reset_index(inplace=True,drop=True)


## get the KA columns from the dataframe
totaldata_all = df.KA
totalDays  = int(len(totaldata_all)/3)

N_dataDays = totalDays

##Declare Arrays 
actualAct = np.zeros(N_dataDays) 
actualRecov = np.zeros(N_dataDays)  
actualTot = np.zeros(N_dataDays) 
actualDes = np.zeros(N_dataDays) 

## Get the first day data
MarInitTot = totaldata_all[0]
MarInitRecov = totaldata_all[1]
MarInitDes = totaldata_all[2]

##initial Setup
actualTot[0] = MarInitTot 
actualRecov[0] = MarInitRecov 
actualDes[0] = MarInitDes 
actualAct[0] = actualTot[0] - actualRecov[0] - actualDes[0]


for idx in range(1, N_dataDays):
    actualTot[idx] = actualTot[idx-1] + totaldata_all[3*(idx)]
    actualRecov[idx] = actualRecov[idx-1] + totaldata_all[3*(idx)+1]
    actualDes[idx] = actualDes[idx-1] + totaldata_all[3*(idx)+2]
    actualAct[idx] = actualTot[idx] - actualRecov[idx] - actualDes[idx]


ActualDict = {}
ActualDict["Actual"] = actualAct
ActualDict["Deceased"] = actualDes
ActualDict["Recovered"] = actualRecov
ActualDict["Cumuliative"] = actualTot

file_name = "actual_Data.pkl"

open_file = open(file_name, "wb")
pickle.dump(ActualDict, open_file)
open_file.close()

###################################### Actual Cases Updated ####################################################

###################################### Read Predicted and Actual Cases #########################################
df_active = pd.read_pickle("Data_Files/df_active")

with open('actual_Data.pkl', 'rb') as f: mynewlist = pickle.load(f) #actual number of active cases 



###################################### Read Ensemble Mean And Standard Errors ###################################
with open('ensemblemean.pkl', 'rb') as f:
    ensemble_mean = pickle.load(f)


with open('ensembleub.pkl', 'rb') as f:
    ensemble_ub = pickle.load(f)

with open('ensemblelb.pkl', 'rb') as f:
    ensemble_lb = pickle.load(f)



###################################### Update ###################################

N = 972
SF = 1e4
D = 0
weightp = np.zeros(N)

weightr = np.zeros(N)
weightr[:] = 1.0/N

with open('mse.pkl', 'rb') as f:
    MSE = pickle.load(f)

lastDay = pd.read_pickle('day.pkl')
latestDay = lastDay

RMSE = np.zeros(N)

for i in range(N):
    for j in range(lastDay+1,len(mynewlist['Actual'])-D):
        MSE[i] += (((df_active.iloc[i,j]-mynewlist['Actual'][j])/SF)**2)/(len(mynewlist['Actual'])-D)
        latestDay = j
        
for i in range(N):
    RMSE[i] = math.sqrt(MSE[i])
    weightr[i] = np.exp(-1.0*RMSE[i])*weightr[i]
    

sum_weightr = sum(weightr)


for i in range(N):
    weightr[i] = (weightr[i]/sum_weightr)*N
    

df_posterior = df_active.copy(deep=True)
df_posterior['WeightRMSE'] = weightr
for i in range(N):
    for j in range(731):    
        df_posterior.iloc[i,j] = df_posterior.iloc[i,j]*df_posterior['WeightRMSE'][i]


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
with open('mse.pkl', 'wb') as f:
    pickle.dump(MSE, f)

with open('day.pkl','wb') as f:
    pickle.dump(latestDay,f)       
        




###################################### Plotting ###################################
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.labelsize']=18

plt.figure(figsize=(11,8))
plt.rcParams["font.serif"]
SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

startDay=301
xt = [304,365,427,488,549,608,670,731]   ## 1st days of the months hardcoded 
xtl = ['May\n\'21','Jul\n\'21','Sep\n\'21','Nov\n\'21', 'Jan\n\'22','Mar\n\'22','May\n\'22','Jul\n\'22']
yt=[0,1e5,2e5,3e5,4e5,5e5,6e5]
ytl=['0','','200K','','400K','','600K']
plt.plot(mynewlist['Actual'],'k-',label='Actual Data')
plt.plot(ensemble_mean,'r-', label='Ensemble Mean')
plt.fill_between(np.arange(731),ensemble_ub,ensemble_lb,color='r',alpha=0.4)
plt.plot(weighted_mean,'b--',label='Posterior Mean')
plt.fill_between(np.arange(731),weighted_ub,weighted_lb,color='b',alpha=0.4)
# plt.plot(weightedr1_ub,'y--',label='new')
# plt.plot(weightedr1_lb,'y--',label='new')
# plt.plot(weighted_meanr,'b--',label='Posterior Mean, RMSE')

# plt.plot(weighted_means,'m--',label='Posterior Mean, SSE')
plt.xlim([290,730])
plt.xticks(ticks=xt,labels=xtl)
plt.yticks(ticks=yt,labels=ytl)
plt.xlabel('Date')
plt.ylabel('Number of Active Cases')
plt.legend()
name = "Day_" + str(latestDay)+"_.png"
plt.savefig(name,dpi=600,bbox_inches='tight')