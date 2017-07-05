#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 06:40:31 2017

@author: ryan
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
#%%
df=pd.read_csv('~/salesdata2.csv')
print(df)
df.index=pd.to_datetime(df['Month'])
df[['Marketing','Sales']].plot()
plt.show()

print(sm.tsa.stattools.adfuller(df['Marketing']))
print(sm.tsa.stattools.adfuller(df['Sales']))

df['const']=1

model1=sm.OLS(endog=df['Sales'],exog=df[['Marketing','const']])
results1=model1.fit()
print(results1.summary())

df['diffS']=df['Sales'].diff()
df['diffM']=df['Marketing'].diff()
model2=sm.OLS(endog=df['diffS'].dropna(),exog=df[['diffM','const']].dropna())
results2=model2.fit()
print(results2.summary())

#%%
print(sm.tsa.stattools.grangercausalitytests(df[['Marketing','Sales']].dropna(),1))



df['lag']=df['diffM'].shift()
df.dropna(inplace=True)
model3=sm.tsa.ARIMA(endog=df['Sales'],exog=df[['lag']],order=[1,1,0])
results3=model3.fit()
print(results3.summary())