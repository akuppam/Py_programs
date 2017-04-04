# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:53:25 2015

@author: AKUPPAM
"Reference material ' "10 minutes to pandas""
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

s = pd.Series([1,2,3,4])

print (s)

dates = pd.date_range('20130101', periods=6)
print (dates)

a = np.array([[1,2,3,4,5],[3,4,5,6,7]], float)
print (a)
print (a[0])
print (a[0,1])

df2 = pd.DataFrame({'a': 1.,
                    'b': pd.Series([40,20,10,70,50]),
                    'c': np.array([8,4,1,6,5]),
                    'd': pd.Timestamp('20150303'),
                    'e': 'test'})
print (df2)
print (df2.dtypes)
print (df2.head)
print (df2.tail)
print (df2.index)
print (df2.columns)
print (df2.values)
print (df2.describe())
print (df2.T)

# axis = 1 is columns (index)
# axis = 0 is rows (index)
print (df2.sort_index(axis=1, ascending=True))
print (df2.sort_index(axis=1, ascending=False))
print (df2.sort_index(axis=0, ascending=True))
print (df2.sort_index(axis=0, ascending=False))

#print df2.sort_values(by = 'b')

print (df2['b'])

"""
# ###############################################
# Python for Data Analysis.PDF
# 3/4/2017
# ###############################################
"""
"""
CHAPTER 9 - DATA AGGREGATION AND GROUP OPERATIONS
"""

import os
import pandas as pd

os.chdir("c:\\users\\akuppam\\documents\\Pythprog")
os.getcwd()

mck = pd.read_csv("mckdata.csv")

mck.describe()
mck.dtypes
mck.head(3)
mck.index
mck.values
mck.T

# axis = 1 - arranges columsn based on alphbetical order
# axis = 0 - arranges rows based on indices (0,1,2,3......or .....3,2,1,0)
mck1 = mck.sort_index(axis=1, ascending=True)
mck1.dtypes
mck2 = mck.sort_index(axis=1, ascending=False)
mck2.dtypes
mck3 = mck.sort_index(axis=0, ascending=True)
mck3.dtypes
mck3.head(3)
mck.head(3)
mck4 = mck.sort_index(axis=0, ascending=False)
mck4.head(3)
mck.head(3)

mck[[0]]
mck[[0,10]]
mck.dtypes
mck5 = mck.sort('vol', ascending=True)
mck5[[0,10]]
mck6 = mck.sort_values('vol', ascending=False)
mck6[[0,10]]

mck7 = pd.DataFrame(mck)
mck71 = pd.DataFrame({'vol': np.array([1])})
print (mck71)
mck71.head(3)
mck8 = pd.merge(mck, mck71)
print (mck8)

df2 = pd.DataFrame({'a': 1.,
                    'b': pd.Series([40,20,10,70,50]),
                    'c': np.array([8,4,1,6,5]),
                    'd': pd.Timestamp('20150303'),
                    'e': 'test'})

print (df2)

df3 = np.array((3,2))
print (df3)

df4 = np.zeros((3,2))
print (df4)

df5 = np.empty((3,2))
print (df5)

df6 = np.arange(10)
print (df6)

mck72 = np.arange(1000)
print(mck72)
mck72 = pd.DataFrame(mck72)
print(mck72)

mck8 = pd.concat([mck7,mck72])
print (mck8)
mck8.describe()

"""
# to create a variable and join it with an existing data file
# use 'concat' to append at the bottom of the data frame
# use 'concat w/ axis=1' to append on the side of the data frame (1 = column)
"""
mck9 = pd.DataFrame(mck)
mck9.dtypes

import numpy as np

mck91 = np.arange(1000)
mck91 = pd.DataFrame(mck91)

mck92 = pd.concat([mck9, mck91])
mck92.describe()
mck92.head(2)
mck92.tail(2)

mck93 = pd.concat([mck9, mck91], axis=1)
mck93.describe()
mck93.columns.values[21] = 'var1'   # renames 21st column (starting from '0') to 'var1'

mck93['var1']=16.50  # add a new column with all values 16.50

mck['var2']=1.25  # add a new column with all values 1.25

mck94 = mck93.drop('var1', axis=1)   # dropping or deleting a whole column
mck94.columns.values[21] = 'var1'

mck95 = mck94.sort(['vol'], ascending=True)   # sort values in column 'vol'

mck96 = np.random.randn(1000,1)    # generate ramdom numbers in a separate column (or matrix of size 1000x1)
mck96 = pd.DataFrame(mck96)
mck96.describe()

mck97 = pd.concat([mck94,mck96], axis=1)
mck97.columns.values[22] = 'rand1'

mck97.sum()    # column sums
mck97.sum(axis=1)  # row sums
mck97.sum(axis=0)  # column sums

mck97.mean()
mck97.median()
mck97.cumsum()
mck97.std()
mck97.var()
#mck97.diff()
mck97.count()
mck97.min()
mck97.max()
#mck97.pct_change()

# corr between two columns - create a new DF and do corr()
  
mck97.corr()
mck98 = pd.concat([mck97.vc, mck97.vol], axis=1)
mck98.corr()

# selecting certain rows and columns from a data frame
mck97.ix[:4,1]    # row values 0,1,2,3,4 from col 2 (starts from 0 and then 1)
mck97.ix[3:6,2]   # row values 3,4,5,6 from col 2 (that's col no. 3)

mck97.ix[2:5, 3:5]  # rows 3,4,5,6 and cols 3,4 (only two col will be picked)

mck97.index
mck96.index
mck97.unstack()
mck97.stack()

mck99 = np.random.randn(3,4)
mck99 = pd.DataFrame(mck99)
mck99.unstack()    # stacks all columns one on top another
mck99.stack()      # stacks all rows by making them into a column of values

"""
CHAPTER 9 - DATA AGGREGATION AND GROUP OPERATIONS
"""
### GROUPING

abcgroup = mck.vol.groupby(mck.abc)
abcgroup.sum()   # this one sums all the 'vol' values over each 'abc' value

atypegroup = mck.vmt.groupby(mck.atype)
atypegroup.sum()
atypegroup.mean()

ftypeatype = mck.vmt.groupby([mck.ftype, mck.atype])
ftypeatype.sum()

# add the function at the end of the command (like sum()) and then unstack for tabular format
ftypeatype = mck.vmt.groupby([mck.ftype, mck.atype]).sum()
ftypeatype.unstack()

mcksub = mck.ix[2:6, 3:7]  # output or print a subset of the dataframe
mcksub

mckvc = mck.vol/mck.cap

diffvc = mckvc - mck.vc
diffvc.describe()

import matplotlib.pyplot as plt
plt.scatter(mckvc, mck.vc)

mck.pivot_table(['vc','vol'],'atype', aggfunc=sum)   # atype is Y, while vc, vol are along X
mck.pivot_table(['vc','vol'],'atype')    # by default, aggfunc is MEAN

""" the following two operations - df.pivot_table and pd.crosstab yield the same result """

mck.pivot_table('vol', 'ftype', 'atype', aggfunc=sum) # pivot by atype and ftype, and mean values of vmt
pd.crosstab(mck.ftype, mck.atype, mck.vol, aggfunc=sum)

"""
Next to do:
Not much else in this book
1. Time series - rely on RHyndman and AV printouts (do py first; re-visit R later)
2. DataViz - rely on seaborn (do this for hgac abm)
3. Numpy - rely on printouts  
4. Scipy - rely on printouts
5. scikit-learn - go thro' it all
6. Rweka - go thro' it all
7. mckdata.xls - DSC / ML / other stats related notes
8. stats and prob printouts (books ?)
9. inter ques (quora, etc.)
10.bayesian stats
"""















