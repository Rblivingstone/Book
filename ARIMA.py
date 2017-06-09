#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:02:31 2017

@author: ryan
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df=pd.read_csv('salesdata.csv')

df.index=pd.to_datetime(df['Date'])
df['Sales'].plot()
plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Sales'], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Sales'], lags=40, ax=ax2)
plt.show()

print(sm.tsa.stattools.adfuller(df['Sales']))

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Sales'].diff().dropna(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Sales'].diff().dropna(), lags=40, ax=ax2)
plt.show()

model=sm.tsa.ARIMA(endog=df['Sales'],order=(0,1,6))
results=model.fit()
print(results.summary())

results.resid.plot()
plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(results.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(results.resid, lags=40, ax=ax2)
plt.show()
#%%
model2=sm.tsa.ARIMA(endog=df['Sales'],order=(7,1,0))
results2=model2.fit()
print(results2.summary())

results2.resid.plot()
plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(results2.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(results2.resid, lags=40, ax=ax2)
plt.show()

#%%
forecast,std,conf=results2.forecast(12)
plt.plot(forecast)
print(forecast)
