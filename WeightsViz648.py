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
df_predicted = df_predicted[df_predicted['KDPwsat2']!='ImmunEscp_Jul']
with open('Data_Files/actual_Data.pkl', 'rb') as f: 
    actual_cases = pickle.load(f) #actual number of active cases 


###################################### Read Ensemble Mean And Standard Errors ###################################
with open('Data_Files/ensemblemean648.pkl', 'rb') as f:
    ensemble_mean = pickle.load(f)


with open('Data_Files/ensembleub648.pkl', 'rb') as f:
    ensemble_ub = pickle.load(f)

with open('Data_Files/ensemblelb648.pkl', 'rb') as f:
    ensemble_lb = pickle.load(f)

lastDay = pd.read_pickle('Data_Files/day648.pkl')
################################## Read Posterior mean and standard errors #############################
with open('Data_Files/wm648.pkl', 'rb') as f:
    weighted_mean = pickle.load(f)

with open('Data_Files/wub648.pkl', 'rb') as f:
    weighted_ub = pickle.load(f)

with open('Data_Files/wlb648.pkl', 'rb') as f:
    weighted_lb = pickle.load(f)

with open('Data_Files/wr648.pkl', 'rb') as f:
    weightr = pickle.load(f)

max_list = [i for i, j in enumerate(weightr) if j == max(weightr)]



#####################Rearrangement of weights ######################################
df_posterior = df_predicted.copy(deep=True)

df_posterior['WeightN']=weightr
df_weight = df_posterior.copy(deep=True)

df2 = df_weight[df_weight['KDPwsat2']=='ImmunEscp_Sep']
df3 = df_weight[df_weight['KDPwsat2']=='ImmunEscp_Nov']
id = []
imm = []
N = len(weightr)
for i in range(N):
    id.append(i+1)

for i in range(0,int(N/2)):
    imm.append('IENV-Sep21')
for i in range(int(N/2),N):
    imm.append('IENV-Nov21')

df_final = df2.append(df3)
df_final['Ensemble Number'] = id
df_final['IENV'] = imm


###################### Visualization ######################################
from matplotlib.lines import Line2D
###################################### Plotting ###################################
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize']=16

fig, ax = plt.subplots(figsize=(11,8))
plt.rcParams["font.serif"]
SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

colors = {'IENV-Sep21':'tab:orange','IENV-Nov21':'tab:green'}
ax.scatter(df_final['Ensemble Number'],df_final['WeightN'],c=df_final['IENV'].map(colors),s=250,edgecolors='k')
plt.plot([324,324],[0,0.002],'k--',alpha=0.3)
plt.plot([648,648],[0,0.002],'k--',alpha=0.3)

handles = [Line2D([0],[0],marker='o',color='w',markerfacecolor=v,label=k,markersize=8) for k,v in colors.items()]
ax.legend(title=None, handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.08))
ax.set_xlim(-20,650)
ax.set_ylim(-0.00005,0.0016)
ax.set_ylabel('Normalized Weights')
ax.set_xlabel('Scenario Number (Rearranged)')

plt.tight_layout()

name = "Output_Files/Bayesian_Images/648_WeightsRearr_Day_" + str(date_generated[lastDay])+".png"
plt.savefig(name,dpi=600,bbox_inches='tight')

####################


plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize']=16

fig, ax = plt.subplots(figsize=(11,8))
plt.rcParams["font.serif"]
SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

colors = {'IENV-Sep21':'tab:orange','IENV-Nov21':'tab:green'}
ax.scatter(df_final['Scenario'],df_final['WeightN'],c=df_final['IENV'].map(colors),s=250,edgecolors='k')

handles = [Line2D([0],[0],marker='o',color='w',markerfacecolor=v,label=k,markersize=8) for k,v in colors.items()]
ax.legend(title=None, handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.08))
ax.set_xlim(-20,650)
ax.set_ylim(-0.00005,0.0016)
ax.set_ylabel('Normalized Weights')
ax.set_xlabel('Scenario Number')
plt.tight_layout()

name = "Output_Files/Bayesian_Images/648_Weights_Day_" + str(date_generated[lastDay])+".png"
plt.savefig(name,dpi=600,bbox_inches='tight')