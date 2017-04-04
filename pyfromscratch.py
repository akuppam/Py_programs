# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:30:18 2016

@author: AKUPPAM
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import glob
import pandas as pd

# Lists (square brackets)

print 1
templist = [10,100,1000,10000,100000]
x = templist[1]
print "Item in '1' position is = "
print x

# Tuples - are immutable' if the positions of lists do not change,
# then use Tuples (no brackets)

print 2
templist1 = 10,100,1000, 10000, 100000
print templist1[2]
y = templist1[2]
print "Item in '2' position is = "
print y

# Dictionary (flower brackets)
# set of pairs

print 3
templist2 = {'a':10,'b':100,'c':1000, 'd':10000, 'e':100000}
z = templist2['b']
print z

print templist2.keys()

print 4
if templist1[3]<1000:   # 'if' statement - no indentation reqd
    print "yes"         # 'print' statement should be intended
else:                   # 'else' statement should be straight below 'if' statement
    print "no"          # 'print' statement should be intended


print 5
# use of np, sm and smf libraries

nobs = 100
X = np.random.random((nobs, 2))

X = sm.add_constant(X)
beta = [1, .1, .5]
e = np.random.random(nobs)
y = np.dot(X, beta) + e

# Fit regression model
results = sm.OLS(y, X).fit()

# Inspect the results
print results.summary()

print 6

# Fit different equation or model
#newresults = smf.OLS('y ~ X').fit()

# Inspect the results
#print newresults.summary()

print 7
# opening csv file
import csv
f = open('linreg.csv', 'rb')
csv_f = csv.reader(f)
headers = csv_f.next()
print headers

print '7a'
# print all rows
for row in csv_f:
  print row

print 8
#import csv
#f = open('linreg.csv')
#csv_f = csv.reader(f)

#for row in csv_f:
#  print row[10]


# load the longley dataset into a pandas data frame - first column (year) used as row labels
df = pd.read_csv('linreg.csv', 'rb')

print df.head()
print df.tail()

print '8a'

import csv

ifile  = open('linreg.csv', "rb")
reader = csv.reader(ifile)

rownum = 0
for row in reader:
    # Save header row.
    if rownum == 0:
        header = row
    else:
        colnum = 0
        for col in row:
            print '%-8s: %s' % (header[colnum], col)
            colnum += 1
            
    rownum += 1

ifile.close()

print 9
# ASSIGNING NAMES TO DIFFERENT COLUMNS
data = pd.read_csv('linreg.csv')
print data.min()
data.columns = ['v1','v2','v3']
print data.v1

# ADDING TWO COLUMNS AND STORING IT IN A NEW COLUMN
print 10
v4 = data.v2 + data.v3
print v4

print 11
# ASSIGNING EXISTING COLUMNS TO NEW VARS SO THAT IT CAN BE INCLUDED INSIDE THE OLS
x = data.v2
y = data.v3

# Fit different equation or model
# OR YOU CAN USE THE NAME OF DATA FILE INSIDE OLS
# sm should always be written as sm.OLS (upper case OLS)
newmodel = sm.OLS(data.v3, data.v2)
res1 = newmodel.fit()
print res1.summary()

# Inspect the results
#print newmodel.summary()

print 12
x = data.v2
y = data.v3
z = data.v1
# sm should always be written as smf.ols (lower case OLS)
newmodel1 = smf.ols(formula = 'y ~ x + z', data=data)
res = newmodel1.fit()
print res.summary()

print '12a'
newmodel2 = smf.ols(formula = 'y ~ x + z', data=data).fit()
#res = newmodel1.fit()
print newmodel2.summary()

print 13
# log is always log and not ln
# always use np.log (np for numpy)
newmodel3 = smf.ols(formula = 'y ~ np.log(x) + z', data=data).fit()
print newmodel3.summary()

print 14
# TODO add image and put this code into an appendix at the bottom
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = data[['v1', 'v2']]
y = data['v3']

## fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()

## Create the 3d plot -- skip reading this
# TV/Radio grid for 3d plot
xx1, xx2 = np.meshgrid(np.linspace(X.v1.min(), X.v1.max(), 100), 
                       np.linspace(X.v2.min(), X.v2.max(), 100))
# plot the hyperplane by evaluating the parameters on the grid
Z = est.params[0] + est.params[1] * xx1 + est.params[2] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, azim=-115, elev=15)

# plot hyperplane
surf = ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y - est.predict(X)
ax.scatter(X[resid >= 0].v1, X[resid >= 0].v2, y[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X[resid < 0].v1, X[resid < 0].v2, y[resid < 0], color='black', alpha=1.0)

# set axis labels
ax.set_xlabel('v1')
ax.set_ylabel('v2')
ax.set_zlabel('v3')

print 15
