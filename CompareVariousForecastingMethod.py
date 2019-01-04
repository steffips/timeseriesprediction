# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:35:27 2018

@author: ASUS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import os
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
#from ConvertDatatoStationary import convert_to_stationary
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA


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
    
    print('Print ARIMA')
    #Normalization    
    values = X.values
    values = values.reshape((len(values), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    normalized = scaler.transform(values)
    X = normalized
    
    size = int(len(X) * 0.70)
    train, test = X[0:size], X[size:len(X)]
    
#    print(asfrdata)
#    print(X)
    #train = scaler.fit_transform(train.reshape(-1, 1))
    #test = scaler.fit_transform(test.reshape(-1, 1))
#    print(train)
#    print(test)
    
    predictions = list()

    #test = test.reset_index()
    #test = test.drop('index',axis=1)
    
    for t in range(len(test)):
        #ARIMA
        #model = ARIMA(history, order=(p,d,q))
        model = ARIMA(train, order=(p,d,q))
        
        model_fit = model.fit(trend='nc', disp=False, solver='lbfgs')

        ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
        resid = model_fit.resid
        yhat = predict(ma_coef, resid)
        #yhat = predict(ar_coef, history) + predict(ma_coef, resid)
        predictions.append(yhat)
        
        obs = test[t]
        #obs = test['Hours'][t]
        #history.append(obs)
        train = np.append(train, obs)
        print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
        
    train= pd.DataFrame(train)

    #DENORMALIZATION START
    predictions = np.array(predictions)
    predictions = predictions.reshape(1,-1)
    predictions = scaler.inverse_transform(predictions)
    predictions = predictions.reshape(-1,1)
    test = scaler.inverse_transform(test)
    #DENORMALIZATION STOP

    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    
    for t in range(len(predictions)):
        print('>predicted=%.3f, expected=%.3f' % (predictions[t],test[t] ))
#        predictions[t] = test[0] - predictions[t]
#        print('ALTERED >predicted=%.3f, expected=%.3f' % (predictions[t],test[t] ))
    
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    
    smapeScore = smape(test,predictions)
    print('Test SMAPE: %.3f' % smapeScore)
    print(len(df['Hours']))
    print('RMSE: %.3f ' % (rmse))

def projectionAR (asfrdata):
    X = asfrdata   
    print('Print AR') 
    #Normalization    
    values = X.values
    values = values.reshape((len(values), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    normalized = scaler.transform(values)
    X = normalized
    
    size = int(len(X) * 0.70)
    train, test = X[0:size], X[size:len(X)]
    # fit model
    model = AR(train)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    
    #DENORMALIZATION START
    yhat = np.array(yhat)
    yhat = yhat.reshape(1,-1)
    yhat = scaler.inverse_transform(yhat)
    yhat = yhat.reshape(-1,1)
    test = scaler.inverse_transform(test)
    #DENORMALIZATION STOP
    
    for i in range(len(yhat)):
    	print('predicted=%f, expected=%f' % (yhat[i], test[i]))
        
    rmse = sqrt(mean_squared_error(test, yhat))
    #plot results
    pyplot.plot(test)
    pyplot.plot(yhat, color='red')
    pyplot.show()
    
    smapeScore = smape(test,yhat)
    print('Test SMAPE: %.3f' % smapeScore)
    print(len(df['Hours']))
    print('RMSE: %.3f ' % (rmse))
    
def projectionARMA (asfrdata,p,q):
    X = asfrdata    
    print('Print ARMA')
    #Normalization    
    values = X.values
    values = values.reshape((len(values), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    normalized = scaler.transform(values)
    X = normalized
    
    size = int(len(X) * 0.70)
    train, test = X[0:size], X[size:len(X)]
    # fit model
    model = ARMA(train,order =(p,q))
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    
    #DENORMALIZATION START
    yhat = np.array(yhat)
    yhat = yhat.reshape(1,-1)
    yhat = scaler.inverse_transform(yhat)
    yhat = yhat.reshape(-1,1)
    test = scaler.inverse_transform(test)
    #DENORMALIZATION STOP
    
    for i in range(len(yhat)):
    	print('predicted=%f, expected=%f' % (yhat[i], test[i]))
        
    rmse = sqrt(mean_squared_error(test, yhat))
    #plot results
    pyplot.plot(test)
    pyplot.plot(yhat, color='red')
    pyplot.show()
    
    smapeScore = smape(test,yhat)
    print('Test SMAPE: %.3f' % smapeScore)
    print(len(df['Hours']))
    print('RMSE: %.3f ' % (rmse))
#END FUNCTION
    
#START MAIN
plt.style.use('fivethirtyeight')

datasetName = 'Zainal.csv'
p, d, q= 0,2, 1

#Open Non-Stationary
fn = os.path.join(os.path.dirname(__file__), datasetName)
with open(fn) as csv:
    df = pd.read_csv(csv, delimiter=';')
    
tes = df.groupby(['Week'])['Hours'].sum()
tes.index = pd.DatetimeIndex(end=pd.datetime.today(), periods=len(tes), freq='1D')
df['Date'] = df['Date'].apply(lambda x : dt.strptime(x, '%d/%m/%Y').strftime('%Y-%m-%d'))
df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(7, unit='d')
df = df.groupby(['Week', pd.Grouper(key='Date', freq='W-FRI')])['Hours'].sum().reset_index().sort_values('Week')
df.set_index('Week')
 
#COUNT prediction
projectionARIMA(df['Hours'], p, d, q)
projectionAR(df['Hours'])
projectionARMA(df['Hours'],2,2)
#CLOSE Non-Stationary

#SHOW plotting
#tes.plot(figsize=(15, 6))
#plt.show()

#OPEN Stationary
#stationaryDataset = convert_to_stationary(datasetName)
#df = stationaryDataset
#df = df.iloc[1:]

##COUNT prediction
#result, testData, predictionsData = projectionARIMA(df['Hours_log_diff'], p, d, q)

#CLOSE Stationary


