# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:38:39 2016

@author: AKUPPAM
"""

import pandas as pd
import numpy as np
import os
os.chdir('C:\\users\\akuppam\\documents\\Pythprog\\Rprog')
os.getcwd()
mck = pd.read_csv('mckdata.csv')
mck.describe()
mck.dtypes

from sklearn.linear_model import LinearRegression
mckmodel = LinearRegression()

""" Re-shape the array of data and then apply the model """
X = mck.time
X
X1 = X.reshape(len(X),1)
X1
y = mck.trips
mckmodel.fit(X.reshape(len(X),1),y)
mckmodel.coef_
mckmodel.intercept_

import matplotlib.pyplot as plt
import seaborn as sb
plt.scatter(X, y,  color='red')

