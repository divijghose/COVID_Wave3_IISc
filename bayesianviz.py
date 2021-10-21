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

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)start ="2020-07-01"
end  = "2022-06-30"

start = datetime.datetime.strptime(start, "%Y-%m-%d")
end = datetime.datetime.strptime(end, "%Y-%m-%d")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days+2)]

with open('Data_Files/ensemblemean.pkl', 'rb') as f:
    ensemble_m = pickle.load(f)
with open('Data_Files/ensemblelb.pkl', 'rb') as f:
    ensemble_lb = pickle.load(f)
with open('Data_Files/ensembleub.pkl', 'rb') as f:
    ensemble_ub = pickle.load(f)
with open('Data_Files/actual_Data.pkl', 'rb') as f:
    actual = pickle.load(f)
with open('Data_Files/wm.pkl', 'rb') as f:
    w_m = pickle.load(f)
with open('Data_Files/wlb.pkl', 'rb') as f:
    w_lb = pickle.load(f)
with open('Data_Files/wub.pkl', 'rb') as f:
    w_ub = pickle.load(f)

for i in range(len(date_generated)):
    date_generated[i] = date_generated[i].strftime("%Y-%m-%d")

df_plot = pd.DataFrame(ensemble_m, columns=['ensembleMean'])
df_plot['ensembleLB'] = ensemble_lb
df_plot['ensembleUB'] = ensemble_ub
# df_plot['actual'] = actual['Actual']
df_plot['weightedMean'] = w_m
df_plot['weightedLB'] = w_lb
df_plot['weightedUB'] = w_ub
actualdf = pd.DataFrame(actual['Actual'],columns=['actual'])
df_plot = pd.concat([df_plot, actualdf], axis=1) 
df_plot=df_plot.round(0)

df_plot['date'] = date_generated



fig = go.Figure()

fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['ensembleMean'], name='Ensemble Mean',
                         line=dict(color='firebrick', width=3,dash='dot')))
fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['ensembleUB'],name='Ensemble Mean Upper Bound',showlegend=False,
    fill=None,
    mode='lines',
    line_color='rgb(244,202,228)',
    ))
fig.add_trace(go.Scatter(
    x=df_plot['date'],
    y=df_plot['ensembleLB'],name='Ensemble Mean Lower Bound', showlegend=False,
    fill='tonexty', # fill area between trace0 and trace1
    mode='lines', line_color='rgb(244,202,228)'))

fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['actual'], name='Actual Cases',
                         line=dict(color='black', width=2)))

fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['weightedMean'], name='Weighted Mean',
                         line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['weightedUB'],name='Weighted Mean Upper Bound',showlegend=False,
    fill=None,
    mode='lines',
    line_color='rgb(136,204,238)',
    ))
fig.add_trace(go.Scatter(
    x=df_plot['date'],
    y=df_plot['weightedLB'],name='Weighted Mean Lower Bound', showlegend=False,
    fill='tonexty', # fill area between trace0 and trace1
    mode='lines', line_color='rgb(136,204,238)'))

fig.update_xaxes(
        type='date')
fig.update_xaxes(
dtick="M1",
tickformat="%b\n%Y")


fig.update_layout(
    autosize=False,
    width=2000,
    height=1000,
    yaxis=dict(
        title_text="Active Cases",
        titlefont=dict(size=30)
    )
)
    

fig.show()
