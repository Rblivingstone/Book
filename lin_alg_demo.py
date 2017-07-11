#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 07:44:14 2017

@author: ryan
"""

import numpy as np
import statsmodels.api as sm

x=np.matrix([range(1000),[1]*1000]).T
y=15*x[:,0]+5+np.random.normal()

model=sm.OLS(endog=y,exog=x)
results=model.fit()
print(results.summary())

print((x.T*x).I*x.T*y)

#%%
M=np.matrix([[0.25,0,.75],
           [.2,.6,.2],
           [0,.9,.1]])
def progression(n):
    for i in range(n):
        print(M**(i+1))
progression(10)