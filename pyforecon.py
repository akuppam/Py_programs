# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:39:26 2016

@author: AKUPPAM
"""

# Python for Econometrics, Statistics, and Data Analysis

##############
### CHAPTER 3
##############
print ('#####################')

x = [2,5,9]
print type (x)
y = (2,5,9)
print type (y)
print '#####################'
# 3.4 Exercises
print '#####################'

a = 4
b = 3.1415
c = 1.0
d = 2+4j
e = 'Hello'
f = 'World'
print '#####################'

print type(a)
print type(b)
print type(c)
print type(d)
print type(e)
print type(f)
print '#####################'

print a + 1
print a - 2
print a * 3
print a / 2
print '#####################'

print b + 1
print b - 2
print c * 3
print c / 2
print '#####################'

print d + 1
print d - 2
print d * 3
print d / 2
print '#####################'

#print e + 1
#print e - 2
#print f * 3
#print f / 2
print '#####################'
ex = 'Python is an interesting and useful language for numerical computing!'
print ex
slice1 = ex[:6]
print slice1
slice2 = ex[-1:]
print slice2
slice3 = ex[59:68]
print slice3
slice4 = ex[13:15]
print slice4
slice5 = ex[::-1]
print slice5
slice6 = slice1[::-1]
print slice6
slice7 = ex[::2]
print slice7

name = 'aruN kumaR reddY kuppaM'
reverse_name = name[::-1]
print reverse_name

#gibb = 'C:\Users\akuppam\Documents\Pythprog\gibb.txt'
#gibb = open("C:\Users\akuppam\Documents\Pythprog\gibb.txt", "r")
##### the following works but highlighting for brevity
#gibb = 'gibb.txt'
#with open(gibb) as g:
#    data = g.readlines()
#gibbrev = data[::-1]
#print gibbrev
#gibbrev2 = data[::2]
#print gibbrev2
##### the above works but highlighting for brevity

print '#####################'

list1 = [4,3.1415, 1.0, 2+4j, 'Hello', 'World']
list1.remove(1.0)
print list1

dict1 = {'alpha': 1.0,'beta': 3.1415,'gamma': -99}
print dict1
print dict1['alpha']

##############
### CHAPTER 4
##############
import numpy as np
x = np.array([0.0, 1, 2, 3, 4])
print type (x)

#z = mat(x)

###############
### CHAPTER 17 - pandas
##############

from pandas import Series
series1 = Series([4.56, 3.45, 6.89, 0.234], index = ['q','r','t','y'])
print series1
print series1*8

series2 = Series([4.56, 3.45, 6.89, 0.234])
print series2
print series2*8

series1.drop('q')
print series1.drop('q')

# *****************
#import os
#import xlrd
#type(xlrd)

#from pandas import read_excel
#mag = read_excel('C:/Users/akuppam/Documents/MAG_EST-CV Survey/StreetLight/External_for_Client/External_for_Client/magSLnew2-SAMPLE.xlsx', 'magSLnews3')


#from xlrd import open_workbook
#wb = open_workbook('C:/Users/akuppam/Documents/MAG_EST-CV Survey/StreetLight/External_for_Client/External_for_Client/magSLnew2-SAMPLE.xlsx', 'magSLnews3')

#import openpyxl
#To be able to read csv formated files, we will first have to import the
#csv module.
import os
import csv

os.chdir('c:\\Users\\akuppam\\Documents\\MAG_EST-CV Survey\\StreetLight\\External_for_Client\\External_for_Client\\')
os.getcwd()
with open('magSLnew3.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        print row

head(f)
list(f)

import csv
with open('some.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        print row



#    for row in reader:
#        print column
f.describe()
