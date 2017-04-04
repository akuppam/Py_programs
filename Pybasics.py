# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 22:51:46 2016

@author: AKUPPAM
"""

"""
Python basics - numpy, pandas, scipy, matplotlib.pyplot, seaborn
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir('c:\\users\\akuppam\\documents\\pythprog')
os.getcwd()

mck = pd.read_csv("mckdata.csv")
mck.describe()
mckh = mck.head()
mck.tail()

mckh.to_csv('mckh.csv')





os.chdir("c:\\users\\akuppam\\documents\\pythprog\\rprog")
os.getcwd()
mck = pd.read_csv("mckdata.csv")
mck.head()

w = np.array([mck.wt],float)
w

mck.dtypes
""" renaming variables - as some of them had a space in front"""
mck.columns = ['rowid', 'abc', 'ht', 'wt', 'dist', 'time', 'speed', 'tons', 'trips', 'tours', 'vol', 'cap', 'vc', 'vmt', 'ftype', 'atype', 'sector', 'naics', 'lat', 'long', 'epochtime']
mck.dtypes

h = np.array([mck. ht])
h

wh = w*h
wh

len(wh)

w_wh = np.concatenate((w, wh))
w_wh
w_wh.ndim
w_wh.size
w_wh.dtype
w_wh.itemsize
w_wh.shape

