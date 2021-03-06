# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:35:27 2018

@author: ASUS
"""

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime as dt


from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os
import uuid
import time
from math import sqrt
import datetime
from sklearn import metrics
from sklearn import preprocessing

#START FUNCTION
def smape(A, F):
    return np.sum(np.abs(F - A)) / np.sum(A + F)*100

def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat

def projectionARIMA (asfrdata, p, d, q):
    X = asfrdata    
    
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    
#    print(asfrdata)
#    print(X)
    
    #train = scaler.fit_transform(train.reshape(-1, 1))
    #test = scaler.fit_transform(test.reshape(-1, 1))
    
#    print(train)
#    print(test)

    history = [x for x in train]
    predictions = list()

    test = test.reset_index()
    test = test.drop('index',axis=1)
    
    for t in range(len(test)):
        #ARIMA
        model = ARIMA(history, order=(p,d,q))
        model_fit = model.fit(trend='nc', disp=False)

        ma_coef = model_fit.maparams
        resid = model_fit.resid
        yhat = predict(ma_coef, resid)
        #yhat = model_fit.predict(ma_coef, resid, type='levels')
        predictions.append(yhat)
        
        #obs = test
        obs = test['Hours'][t]
        history.append(obs)
        print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
        
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    train= pd.DataFrame(train)

    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    return rmse, test, predictions

#END FUNCTION
    
#START MAIN
plt.style.use('fivethirtyeight')

#DATA LOAD
fn = os.path.join(os.path.dirname(__file__), 'Zoel.csv')
with open(fn) as csv:
    df = pd.read_csv(csv, delimiter=';')

#GROUP by Week calculate weekly hours
#tes = df.groupby(['Week'])['Hours'].sum()
#tes.index = pd.DatetimeIndex(end=pd.datetime.today(), periods=len(tes), freq='1D')
#df['Date'] = df['Date'].apply(lambda x : dt.strptime(x, '%d/%m/%Y').strftime('%Y-%m-%d'))
#df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')
#df = df.groupby(['Week', pd.Grouper(key='Date', freq='W-FRI')])['Hours'].sum().reset_index().sort_values('Week')
#df.set_index('Week')

#SHOW plotting
#tes.plot(figsize=(15, 6))
#plt.show()

#COUNT prediction
p, d, q= 0, 0 , 2
result, testData, predictionsData = projectionARIMA(df['Hours'], p, d, q)
smapeScore = smape(testData['Hours'],predictionsData)
print('Test SMAPE: %.3f' % smapeScore)
print(len(df['Hours']))
#print('RMSE: %.3f ' % (result))
