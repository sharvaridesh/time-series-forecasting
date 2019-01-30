#!/usr/bin/env python
# File: time series forecasting.py
# Author: Sharvari Deshpande <shdeshpa@ncsu.edu>

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from sklearn.metrics import mean_squared_error
from pandas import Series
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import HoltWintersResults
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from scipy import stats
import pylab

#Reading CSV File
df = pd.read_csv('shdeshpa.csv',header=None)
print(df)

#-----------------------------------------TASK 1---------------------------------#
#plotting original data
plt.plot(df[0])
plt.xlabel('t')
plt.ylabel('df[0]')
plt.title('Time Series')
plt.show()

plt.plot(df[0][0:100])
plt.xlabel('t')
plt.ylabel('df[0]')
plt.title('Time Series')
plt.show()

#Logarithmic Transformation
a = np.log(df[0])
print(a)
plt.plot(a)
plt.xlabel('t')
plt.ylabel("xt' = log(df[0])")
plt.title('Logarithm of Time Series')
plt.show()
print(len(df))

plt.plot(a[0:100])
plt.xlabel('t')
plt.ylabel("xt' = log(df[0])")
plt.title('Logarithm of Time Series')
plt.show()

#First-Order Differencing
diff = a.diff()
plt.plot(diff)
plt.xlabel('t')
plt.ylabel("xt'-xt-1'")
plt.title('First Order Differencing')
plt.show()

plt.plot(diff[0:100])
plt.xlabel('t')
plt.ylabel("xt'-xt-1'")
plt.title('First Order Differencing')
plt.show()

df_train = df[0:1500]
df_test = df[1500:]

print('df train', df_train)
print('df test', df_test)

#------------------------------------------------------TASK 2--------------------------------------------#
X = df_train.values
win = []
rmse_1 = []

for window in range(1,50):
# window = 100
    win.append(window)
    history = [X[i] for i in range(window)]
    df_train = [X[i] for i in range(window, len(X))]
    predictions = list()

    for t in range(len(df_train)):
        length = len(history)
        y_hat = mean([history[i] for i in range(length-window, length)])
        obs = df_train[t]
        predictions.append(y_hat)
        history.append(obs)
        print('predicted=%f, expected=%f' % (y_hat, obs))
    error = mean_squared_error(df_train, predictions)
    print('MSE: %.3f' % error)
    rmse = np.sqrt(error)
    rmse_1.append(rmse)
    print('RMSE:', rmse)

    plt.plot(df_train, label='Original')
    plt.plot(predictions, color='red',label='Predictions')
    plt.xlabel('t')
    plt.legend(loc='best')
    plt.show()
# zoom plot
    plt.plot(df_train[0:100],label='Original')
    plt.plot(predictions[0:100], color='red', label='Predictions')
    plt.xlabel('t')
    plt.legend(loc='best')
    plt.show()

    plt.plot(df_train, predictions,'+')
    plt.xlabel('Original Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Original Values')
    plt.show()
plt.plot(win, rmse_1)
plt.title('RMSE vs. k')
plt.show()
#---------TEST DATASET TASK--------------#

X = df_test.values
win_2=[]
rmse_2=[]
for window in range(1,50):
    win_2.append(window)
    history = [X[i] for i in range(window)]
    df_test = [X[i] for i in range(window, len(X))]
    predictions = list()
# walk forward over time steps in test
    for t in range(len(df_test)):
        length = len(history)
        y_hat = mean([history[i] for i in range(length-window, length)])
        obs = df_test[t]
        predictions.append(y_hat)
        history.append(obs)
        print('predicted=%f, expected=%f' % (y_hat, obs))
    error = mean_squared_error(df_test, predictions)
    print('MSE: %.3f' % error)
    rmse = np.sqrt(error)
    rmse_2.append(rmse)
    #rmse_arr.append(rmse)
    print('RMSE:', rmse)
plt.plot(win_2,rmse_2)
plt.title('RMSE vs. k')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.show()
plt.plot(df_test, label='Original')
plt.plot(predictions, color='red',label='Predictions')
plt.xlabel('t')
plt.legend(loc='best')
plt.show()
# zoom plot
plt.plot(df_test[0:100],label='Original')
plt.plot(predictions[0:100], color='red', label='Predictions')
plt.xlabel('t')
plt.legend(loc='best')
plt.show()

plt.plot(df_test, predictions,'+')
plt.xlabel('Original Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Original Values')
plt.show()
print(len(df_train))
#--------------------------------------------------TASK 3---------------------------------------------------#
a_arr = []
rmse_arr = []

for a in np.arange(0.1, 1.0, 0.1):
    a_arr.append(a)
    ema_model = SimpleExpSmoothing(df_train).fit(smoothing_level=a, optimized=False)
    result = HoltWintersResults.predict(ema_model,start=0)
    mse = mean_squared_error(df_train, result)
    rmse = np.sqrt(mean_squared_error(df_train, result))
    rmse_arr.append(rmse)
    plt.plot(df_train,label='Original')
    plt.plot(result, color='red', label='Predictions')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.show()
    plt.plot(df_train[0:100], label='Original')
    plt.plot(result[0:100], color='red', label='Predictions')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.show()
    print('a:',a)
    print("RMSE:", rmse)
    print("MSE:", mse)
    plt.plot(result, df_train, '+')
    plt.xlabel('Original Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted Values against Original Values')
    plt.show()
plt.plot(a_arr, rmse_arr)
plt.xlabel('a')
plt.ylabel('RMSE')
plt.title('RMSE vs. a')
plt.show()

#TESTING DATA
a_arr1 = []
rmse_arr1 = []

for a in np.arange(0.1, 1.0, 0.1):
    a_arr1.append(a)
    ema_model1 = SimpleExpSmoothing(df_test).fit(smoothing_level=a, optimized=False)
    result1 = HoltWintersResults.predict(ema_model1,start=0)
    mseT = mean_squared_error(df_test, result1)
    rmseT = np.sqrt(mean_squared_error(df_test, result1))
    rmse_arr1.append(rmseT)
    plt.plot(df_test,label='Original')
    plt.plot(result1, color='red', label='Predictions')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.show()
    plt.plot(df_test[0:100], label='Original')
    plt.plot(result1[0:100], color='red', label='Predictions')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.show()
    print('a:',a)
    print("RMSE:", rmseT)
    print("MSE:", mseT)
    plt.plot(result1, df_test, '+')
    plt.xlabel('Original Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted Values against Original Values')
    plt.show()
plt.plot(a_arr1, rmse_arr1)
plt.xlabel('a')
plt.ylabel('RMSE')
plt.title('RMSE vs. a')
plt.show()
#---------------------------------------TASK 4-------------------------------------------#
lag_pacf = pacf(df_train, nlags=20, method='ols')

#Plot PACF
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_train)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_train)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

model = ARIMA(df_train, order=(1,0,0))
results_AR = model.fit(disp=-1)
plt.plot(df_train, label='Original')
plt.plot(results_AR.fittedvalues, color='red',label='Predictions')
plt.title('AR Model')
plt.xlabel('t')
plt.legend(loc='best')
plt.show()
res = results_AR.fittedvalues
error = mean_squared_error(df_train, res)
print('MSE: %.3f' % error)
rmse = np.sqrt(error)
print('RMSE:', rmse)

plt.plot(df_train,res,'+')
plt.xlabel('Original Values')
plt.ylabel('Predicted Values')
plt.show()

print(results_AR.summary())
print(results_AR.fittedvalues)
error_estimate = df_train - res
print(error_estimate[0])

#Histogram of Residuals
plt.hist(error_estimate[0])
plt.ylabel('Frequency')
plt.xlabel('Residual Values')
plt.title('Histogram of Residual')
plt.show()

# Q-Q Plot
fig = sm.qqplot(error_estimate[0], stats.distributions.norm)
stats.probplot(error_estimate[0], dist="norm",plot=pylab)
pylab.show()

print(error_estimate[0].shape, res.shape)
#Scatter Plot Residuals
plt.scatter(error_estimate[0], res)
plt.xlabel('Prediction Values')
plt.ylabel('Residual Values')
plt.title('Scatter Plot of Residuals')
plt.show()

#Chi-squared Test
normal_distribution = stats.norm.pdf(error_estimate[0], np.mean(error_estimate[0]), np.std(error_estimate[0]))
error_estimate = stats.norm.rvs(size=100)
print('Chi Squared Test:', stats.normaltest(error_estimate))

#-------TEST DATASET--------------#

model = ARIMA(df_test, order=(1,0,0))
results_AR = model.fit(disp=-1)
plt.plot(df_test, label='Original')
plt.plot(results_AR.fittedvalues, color='red',label='Predictions')
plt.title('AR Model')
plt.xlabel('t')
plt.legend(loc='best')
plt.show()
res_1 = results_AR.fittedvalues
print(res_1)

error_1 = mean_squared_error(df_test, res_1)
print('MSE: %.3f' % error_1)
rmse_1 = np.sqrt(error_1)
print('RMSE:', rmse_1)

plt.plot(df_test, res_1, '+')
plt.xlabel('Original Values')
plt.ylabel('Predicted Values')
plt.show()

print(results_AR.summary())

