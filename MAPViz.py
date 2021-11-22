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

start ="01-07-2020"
end  = "30-06-2022"

start = datetime.datetime.strptime(start, "%d-%m-%Y")
end = datetime.datetime.strptime(end, "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days+2)]


###################################### Read Predicted and Actual Cases #########################################
df_predicted = pd.read_pickle("Data_Files/df_active")

with open('Data_Files/actual_Data.pkl', 'rb') as f: 
    actual_cases = pickle.load(f) #actual number of active cases 


###################################### Read Ensemble Mean And Standard Errors ###################################
with open('Data_Files/ensemblemean.pkl', 'rb') as f:
    ensemble_mean = pickle.load(f)


with open('Data_Files/ensembleub.pkl', 'rb') as f:
    ensemble_ub = pickle.load(f)

with open('Data_Files/ensemblelb.pkl', 'rb') as f:
    ensemble_lb = pickle.load(f)

lastDay = pd.read_pickle('Data_Files/day.pkl')
################################## Read Posterior mean and standard errors #############################
with open('Data_Files/wm.pkl', 'rb') as f:
    weighted_mean = pickle.load(f)

with open('Data_Files/wub.pkl', 'rb') as f:
    weighted_ub = pickle.load(f)

with open('Data_Files/wlb.pkl', 'rb') as f:
    weighted_lb = pickle.load(f)

with open('Data_Files/wr.pkl', 'rb') as f:
    weightr = pickle.load(f)

max_list = [i for i, j in enumerate(weightr) if j == max(weightr)]

#################################### Visualization ################################################
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
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

startDay=301
xt = [304,365,427,488,549,608,670,731]   ## 1st days of the months hardcoded 
xtl = ['May\n\'21','Jul\n\'21','Sep\n\'21','Nov\n\'21', 'Jan\n\'22','Mar\n\'22','May\n\'22','Jul\n\'22']
yt=[0,1e5,2e5,3e5,4e5,5e5,6e5]
ytl=['0','','200K','','400K','','600K']
plt.plot(actual_cases['Actual'],'k-')#,label='Actual Data')
plt.plot(ensemble_mean,'r-')#, label='Ensemble Mean')
plt.fill_between(np.arange(731),ensemble_ub,ensemble_lb,color='r',alpha=0.4)
plt.plot(weighted_mean,'b--')#,label='Posterior Mean')
plt.fill_between(np.arange(731),weighted_ub,weighted_lb,color='b',alpha=0.4)

for k in max_list:
    plt.plot(df_predicted.iloc[k,0:731],label=df_predicted['ID'][k])

plt.xlim([290,730])
plt.xticks(ticks=xt,labels=xtl)
plt.yticks(ticks=yt,labels=ytl)
plt.xlabel('Date')
plt.ylabel('Number of Active Cases')
plt.legend(title=None,loc='lower center',bbox_to_anchor=(0.5,-0.35))
plt.tight_layout()
plt.savefig('Output_Files/Bayesian_Images/PosteriorMean_EnsembleMean_MAP'+str(date_generated[lastDay])+ '.png',dpi=600,bbox_to_inhes='tight',facecolor='w')




#########################################################################33
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
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

startDay=301
xt = [304,365,427,488,549,608,670,731]   ## 1st days of the months hardcoded 
xtl = ['May\n\'21','Jul\n\'21','Sep\n\'21','Nov\n\'21', 'Jan\n\'22','Mar\n\'22','May\n\'22','Jul\n\'22']
yt=[0,1e5,2e5,3e5,4e5,5e5,6e5]
ytl=['0','','200K','','400K','','600K']
plt.plot(actual_cases['Actual'],'k-',label='Actual Data')



plt.plot(df_predicted.iloc[max_list[0],0:731],label='Max. A Posteriori Estimate')
plt.xlim([290,730])
plt.xticks(ticks=xt,labels=xtl)
plt.yticks(ticks=yt,labels=ytl)
plt.xlabel('Date')
plt.ylabel('Number of Active Cases')
# plt.legend(title=None,loc='lower center',bbox_to_anchor=(0.5,-0.35))

plt.legend()
plt.tight_layout()
plt.savefig('Output_Files/Bayesian_Images/MAP'+str(date_generated[lastDay])+ '.png',dpi=600,bbox_to_inhes='tight',facecolor='w')