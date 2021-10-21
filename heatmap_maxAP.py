import plotly.express as px
import plotly.graph_objects as go

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import os.path
from os import path
import seaborn as sns
from scipy import signal
from matplotlib.ticker import FormatStrFormatter

import warnings
warnings.filterwarnings('ignore')


with open('Data_Files/wpcum.pkl', 'rb') as f:
    weightpcum = pickle.load(f)
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.labelsize']=18

plt.figure(figsize=(3.6,8))
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
fig = plt.figure(figsize = (11, 18))

sns.heatmap(np.array(weightpcum),cmap='RdYlGn')
fig.tight_layout()
plt.savefig("heatmap_post.pdf", dpi=600,bbox_inches='tight')