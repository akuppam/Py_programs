# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:09:06 2016

@author: AKUPPAM
"""

# # python refresher
# https://developers.google.com/edu/python/regular-expressions

import re
str = 'Python 3.5.2 |Anaconda 4.1.1 (64-bit)| (default, Jul  5 2016, 11:41:13) [MSCv.1900 64 bit (AMD64)]Type "copyright", "credits" or "license" for more information.IPython 4.2.0 -- An enhanced Interactive Python.?->Introduction and overview of IPythons features.%quickref -> Quick reference.help -> Pythons own help system.object?   -> Details about object, use object?? for extra details.%guiref -> A brief reference about the graphicaluser interface.'

match = re.search(r'Python', str)
if match:
    print ('found'), match.group()
else:
    print ('not found')


import re
str = 'Python 3.5.2 |Anaconda 4.1.1 (64-bit)| (default, Jul  5 2016, 11:41:13) [MSCv.1900 64 bit (AMD64)]Type "copyright", "credits" or "license" for more information.IPython 4.2.0 -- An enhanced Interactive Python.?->Introduction and overview of IPythons features.%quickref -> Quick reference.help -> Pythons own help system.object?   -> Details about object, use object?? for extra details.%guiref -> A brief reference about the graphicaluser interface.'
match = re.search(r'Python', str)
if match:
    print (match.group())

import re
str = 'Python 3.5.2 |Anaconda 4.1.1 (64-bit)| (default, Jul  5 2016, 11:41:13) [MSCv.1900 64 bit (AMD64)]Type "copyright", "credits" or "license" for more information.IPython 4.2.0 -- An enhanced Interactive Python.?->Introduction and overview of IPythons features.%quickref -> Quick reference.help -> Pythons own help system.object?   -> Details about object, use object?? for extra details.%guiref -> A brief reference about the graphicaluser interface.'
match = re.search(r'\d\d\d', str)
if match:
    print (match.group())

import re
str = 'Python 3.5.2 |Anaconda 4.1.1 (64-bit)| (default, Jul  5 2016, 11:41:13) [MSCv.1900 64 bit (AMD64)]Type "copyright", "credits" or "license" for more information.IPython 4.2.0 -- An enhanced Interactive Python.?->Introduction and overview of IPythons features.%quickref -> Quick reference.help -> Pythons own help system.object?   -> Details about object, use object?? for extra details.%guiref -> A brief reference about the graphicaluser interface.'
match = re.search(r'\d+', str)
if match:
    print (match.group())
else:
    print ('not found')

# Open file
f = open('test.txt', 'r')
# Feed the file text into findall(); it returns a list of all the found strings
strings = re.findall(r'some pattern', f.read())

baby = open('C:\\Users\\akuppam\\Documents\\Pythprog\\google-python-exercises\\babynames\\baby1990.html', 'r')
names = re.findall(r'\w\s*', baby.read())
names
baby = open('C:\\Users\\akuppam\\Documents\\Pythprog\\google-python-exercises\\babynames\\baby1990.html', 'r')
names = re.findall(r'\w\S*', baby.read())
names
baby = open('C:\\Users\\akuppam\\Documents\\Pythprog\\google-python-exercises\\babynames\\baby1990.html', 'r')
names = re.findall(r'([\w\.-]+)', baby.read())
names
baby = open('C:\\Users\\akuppam\\Documents\\Pythprog\\google-python-exercises\\babynames\\baby1990.html', 'r')
names = re.findall(r'\d\s*\w\s*\w\s*', baby.read())
names
baby = open('C:\\Users\\akuppam\\Documents\\Pythprog\\google-python-exercises\\babynames\\baby1990.html', 'r')
names = re.findall(r'\d\s\w\s\w\s', baby.read())
names
# -------------------
baby = open('C:\\Users\\akuppam\\Documents\\Pythprog\\google-python-exercises\\babynames\\baby1990.html', 'r')
names = re.findall(r'\w\S*\n', baby.read())
names
"""     'align="right"><td>955</td><td>Camden</td><td>Lee</td>\n',      """
# -------------------
baby = open('C:\\Users\\akuppam\\Documents\\Pythprog\\google-python-exercises\\babynames\\baby1990.html', 'r')
names = re.findall(r'<td>(\d+)</td><td>(\w+)</td>\<td>(\w+)</td>', baby.read())
names
"""        ('1000', 'Tate', 'Peggy')]       """
# -------------------
baby = open('C:\\Users\\akuppam\\Documents\\Pythprog\\google-python-exercises\\babynames\\baby1990.html', 'r')
names = re.findall(r'<td>\d+</td><td>(\w+)</td>\<td>(\w+)</td>', baby.read())
names
"""        (Tate', 'Peggy')]       """
# -------------------

# https://www.analyticsvidhya.com/blog/2014/11/text-data-cleaning-steps-python/

from html.parser import HTMLParser
html_parser = HTMLParser.HTMLParser()
original_tweet = 'Python 3.5.2 |Anaconda 4.1.1 (64-bit)| (default, Jul  5 2016, 11:41:13) [MSCv.1900 64 bit (AMD64)]Type "copyright", "credits" or "license" for more information.IPython 4.2.0 -- An enhanced Interactive Python.?->Introduction and overview of IPythons features.%quickref -> Quick reference.help -> Pythons own help system.object?   -> Details about object, use object?? for extra details.%guiref -> A brief reference about the graphicaluser interface.'
tweet = html_parser.unescape(original_tweet)
tweet = original_tweet.decode("utf8").encode('ascii','ignore')
tweet = _slang_loopup(tweet)

# ------------------------------

""" resume from here - http://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial.html """

import os
import csv
import pandas as pd
import numpy as np

os.chdir('c:\\Users\\akuppam\\Documents\\Pythprog\\Rprog')
os.getcwd()

mck = pd.read_csv("mckdata.csv")
mck.describe()
mck.dtypes

mckarray = np.array(mck)
mckarray
""" once converted to array, it will not describe or dtype """
""" there will be no specific variables but only rows of numbers """
mckarray.ndim
mckarray.shape
mckarray.size
mckarray.dtype
mckarray.itemsize
mckarray.data

mck1 = mck.drop(mck.columns[[1,2,3,4,5,6]], axis=1)
mck1.describe()
mck1.dtypes

mck1array = np.array(mck1)
mck1array
mck1array.ndim
mck1array.shape
mck1array.size

mckdiff = mck1array - mck1array
mckdiff
mckdiff.ndim
mckdiff.shape
mckdiff.size

mcksum = mck1array + mck1array
mcksum
mcksum.ndim
mcksum.shape
mcksum.size

# universal functions
mck1exp = np.exp(mck1array)
mck1sqrt = np.sqrt(mck1array)
mck1exp
mck1sqrt

# indexing, slicing, iterating
mck1array
mck2 = mck1array[2]         # this will be 3rd row of numbers - 0, 1, 2
mck2
mck2.ndim
mck2.shape

mck1array.ndim
mck1array
""" prints 0, 1 values of column 1 """
""" check dimensions before slicing """
m1 = mck1array[0:2, 1:2]  
m1
m1.shape

""" shows all of column 1 """
m2 = mck1array[ : , 2]
m2
m2.shape

""" 12th row from behind """
m3 = mck1array[-12]
m3
m3.shape

# shape manipulation

""" ravel() falttens the array into one line of number """
m1.ravel()
m2.ravel()
m3.ravel()

mcksum.transpose()
mcksum

mcksum.resize(500,15)
mcksum.shape
mcksum

mcksum.reshape(500,15)
mcksum.shape
mcksum
mcksum.ndim

# vertical stacking
# size, ndim, size - all should be the same for vertical stacking
mckstack = np.vstack((mck1array,mcksum))
mck1array.ndim
mck1array.shape
mcksum.ndim
mcksum.resize(1000,15)
mcksum.shape
mckstack.ndim
mckstack.shape

# horizontal stacking
mckstack1 = np.hstack((mck1array,mcksum))
mckstack1.ndim
mckstack1.shape

# splitting into several smaller arrays
mckhsplit = np.hsplit(mckstack1,3)
mckhsplit

mckvsplit = np.vsplit(mckstack1,4)
mckvsplit


