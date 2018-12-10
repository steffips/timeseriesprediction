# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 18:19:22 2018

@author: ASUS
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from datetime import datetime as dt

def kpss_test(timeseriesdata):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseriesdata, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

#Dickey-Fuller Test
def adf_test(timeseriesdata):
    print('Result of Dickey-Fuller Test:')
    dftest = adfuller(timeseriesdata, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test statistics','p-value','#Lags used','Number of observations'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s) '%key] = value
    print(dfoutput)

def convert_to_stationary(dataset):
    #dataset reading
    fn = os.path.join(os.path.dirname(__file__), dataset)
    with open(fn) as csv:
        df = pd.read_csv(csv, delimiter=';')
    
    #preprocessing
#    train.timestamp = pd.to_datetime(train.Month, format='%Y-%m')
#    train.index = train.timestamp
#    train.index = train.Month
#    train.drop('Month', axis=1, inplace=True)
    
    #check few first row
    print(df.head())
    
    #GROUP by Week calculate weekly hours
    tes = df.groupby(['Week'])['Hours'].sum()
    tes.index = pd.DatetimeIndex(end=pd.datetime.today(), periods=len(tes), freq='1D')
    df['Date'] = df['Date'].apply(lambda x : dt.strptime(x, '%d/%m/%Y').strftime('%Y-%m-%d'))
    df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')
    df = df.groupby(['Week', pd.Grouper(key='Date', freq='W-FRI')])['Hours'].sum().reset_index().sort_values('Week')
    df.set_index('Week')
    
    
    #plot for visual test
#    train['Hours'].plot()
          
    #apply adf test on the series
    adf_test(df['Hours'])
    kpss_test(df['Hours'])
                    
    df['Hours_diff'] = df['Hours'] - df['Hours'].shift(1)
#    train['Hours_diff'].dropna().plot()
          
    n=7
    df['Hours_diff'] = df['Hours'] - df['Hours'].shift(n)
          
    df['Hours_log'] = np.log(df['Hours'])
    df['Hours_log_diff'] = df['Hours_log'] - df['Hours_log'].shift(1)
#    train['Hours_log_diff'].dropna().plot()
    
    return df

#reading the dataset
train = pd.read_csv('AirPassengers.csv')

#preprocessing
train.timestamp = pd.to_datetime(train.Month , format = '%Y-%m')
train.index = train.timestamp
train.drop('Month',axis = 1, inplace = True)

#looking at the first few rows
#train.head()
#train['#Passengers'].plot()

adf_test(train['#Passengers'])
kpss_test(train['#Passengers'])
    
#train['#Passengers_diff'] = train['#Passengers'] - train['#Passengers'].shift(1)
#train['#Passengers_diff'].dropna().plot()
      
n=7
train['#Passengers_diff'] = train['#Passengers'] - train['#Passengers'].shift(n)

train['#Passengers_log'] = np.log(train['#Passengers'])
train['#Passengers_log_diff'] = train['#Passengers_log'] - train['#Passengers_log'].shift(1)
train['#Passengers_log_diff'].dropna().plot()
      

