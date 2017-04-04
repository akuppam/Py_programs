# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:34:46 2016

@author: AKUPPAM

""" 
""" PFDA
""" 
""" PFDA
The following is from 'Python for Data Analysis.PDF'
""" 
""" PFDA
"""
""" PFDA

"""
"""
Chapter 5 - Pandas
"""

from pandas import Series, DataFrame
obj = Series([1,2])
obj
data4 = {'st':['tx','ca','fl'],'pop':[25,20,15]}
frame = DataFrame(data4)
frame
frameT = frame.T
frameT
frame.values

"""
"""
import os
import csv
import sys
import io


os.chdir('c:\\Users\\akuppam\\Documents\\Pythprog\\Rprog\\')
os.getcwd()
with open('mckdata.csv', 'rb') as mck:
    reader = csv.reader(mck)
mck
mck.describe()


with open('mckdata.csv', 'rb') as f:
    fio  = io.FileIO(f.fileno())
    fbuf = io.BufferedReader(fio)
    print(fbuf.read(20))
    
"""    
# the above way of reading does not have much value
# easier to do this in panadas
"""
"""
Chapter 5 & 6 - pandas
"""
import os
import csv
import sys

import pandas as pd
os.chdir('c:\\Users\\akuppam\\Documents\\Pythprog\\Rprog\\')
os.getcwd()

mck = pd.read_csv('mckdata.csv')
mck.describe()
mck.wt.mean()       # mean of 'wt' var

# output by a particular var (eg., atype)
parsed = pd.read_csv('mckdata.csv', index_col=['atype'])
parsed

"""
creating a dataframe
this helps in choosing a single var
see below - mean of 'wt' var
"""
from pandas import Series, DataFrame
mck = pd.read_csv('mckdata.csv')
dfmck = pd.DataFrame(mck)
dfmck

dfmck.mean()        # mean of all vars
dfmck.wt.mean()     # mean of just 'wt' variable
 
"""
Chapter 7
"""
import os
os.chdir('c:\\Users\\akuppam\\Documents\\Pythprog\\')
os.getcwd()
import pandas as pd
citymck = pd.read_csv('citymck.csv')
citymck.describe()
dfcitymck = pd.DataFrame(citymck)
dfcitymck

dfcity = pd.merge(dfmck, dfcitymck)
dfcity

dfcity = pd.merge(dfmck, dfcitymck, on='atype')
dfcity

dfcity.to_csv('dfcity.csv', sep='\t')   # no space between columns
dfcity.to_csv('dfcity1.csv')            # this one reads well in excel

dfcity = pd.merge(dfmck, dfcitymck, how='inner')
dfcity.to_csv('dfcity2.csv')            # this one reads well in excel
dfcity = pd.merge(dfmck, dfcitymck, how='outer')
dfcity.to_csv('dfcity3.csv')            # this one reads well in excel

"""
"""
citymck1 = pd.read_csv('citymck1.csv')
dfcitymck1 = pd.DataFrame(citymck1)

"""
concatenate
# pd.concat works only if the two dataframes are inside square brackets
"""
dfcityconcat = pd.concat([dfcitymck, dfcitymck1])
dfcityconcat.to_csv('dfcity4.csv')

# stack writes out every row of data in sequence
stackmck = dfcitymck.stack()
stackmck

stackmck = dfmck.stack()
stackmck
stackmck.to_csv('stackmck.csv')

# unstack writes out every column of data in sequence
unstackmck = dfmck.unstack()
unstackmck.to_csv('unstackmck.csv')

import numpy as np
df1 = DataFrame(np.arange(6).reshape(3,2), index=['a','b','c'],
                columns=['one','two'])
df1

df2 = DataFrame(5 + np.arange(4).reshape(2,2), index=['a','c'],
                columns=['three','four'])
df2

pd.concat([df1,df2])
pd.concat([df1,df2], axis=1, keys=['level1', 'level2'])

"""
START FROM PAGE 188
GO OVER MAG AND MARC PANDAS PROGRAMS
"""
"""
From funcclassobj.py
"""
import funclassobj as fco
import funclassobj

new1 = fco.Factory(3500)
new1

# -------------------------------
class Shipper2:
    def __init__(self, value_of_goods_usd):
        self.goodsval = value_of_goods_usd
    def printgoodsval(self):
        print ('Value of Shipper2 goods is', self.goodsval, 'in British pounds.')
        return '$', self.goodsval/0.77, 'USD'
        
goodsval = Shipper2(67895.25)
Shipper2.printgoodsval(goodsval)
# ----------------------------

fco.Shipper2(3550)
fco.Shipper2.printgoodsval(3550)

print (new2)

new2 = fco.computepay(3550,100)
print (new2)

new3 = fco.computegrade(.88)
new3

new4 = fco.Complex(1,2)
new4

import seaborn as sb

conda install -c anaconda seaborn=0.7.1
