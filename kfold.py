# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:35:03 2017

@author: rbarnes
"""

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([0, 1, 1, 0])
kf = KFold(n_splits=4)
kf.get_n_splits(X)

print(kf)
X = pd.DataFrame(X)
X['depvar'] = y
i=0

for train_index, test_index in kf.split(X):
   for j in test_index:
       X.loc[X.index == j,'fold'] = int(i)
   i+=1


meta_X = X.copy()
meta_X['model1']=None
meta_X['model2']=None

for idx in X['fold'].unique():
    train=X[X['fold'] != idx]
    test=X[X['fold'] == idx]
    model1 = LinearRegression()
    model2 = LogisticRegression()
    model1.fit(train[[0,1]],train['depvar'])
    model2.fit(train[[0,1]],train['depvar'])
    print(model1.coef_)
    meta_X.loc[test.index,'model1'] = model1.predict(test[[0,1]])
    meta_X.loc[test.index,'model2'] = model2.predict_proba(test[[0,1]])[:,1]

model_all = LinearRegression()
model_all.fit(meta_X[['model1']],meta_X['depvar'])
#print(confusion_matrix(meta_X['depvar'],model_all.predict(meta_X[['model1','model2']])))
print(model_all.predict(meta_X[['model1']]))