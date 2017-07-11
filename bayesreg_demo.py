#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 07:19:55 2017

@author: ryan
"""

import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt

df=pd.read_csv('/home/ryan/Documents/thads2013n.txt',sep=',')
df=df[df['BURDEN']>0]
df=df[df['AGE1']>0]

plt.scatter(df['AGE1'],df['BURDEN'])
plt.show()

with pm.Model() as model: 
    # Define priors
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pm.Normal('Intercept', 0, sd=20)
    x_coeff = pm.Normal('x', 0, sd=20)

    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + x_coeff * df['AGE1'],
                        sd=sigma, observed=df['BURDEN'])

    # Inference!
    trace = pm.sample(3000)
pm.traceplot(trace)
plt.show()
print(np.mean([1 if obj<0 else 0 for obj in trace['x']]))