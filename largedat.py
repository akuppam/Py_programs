# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 23:31:48 2016

@author: AKUPPAM
"""

import pandas as pd
import os
import numpy as np

os.chdir('C:\\Users\\akuppam\\Documents\\Pythprog\\spark\\ml-20m\\ml-20m')
os.getcwd()

largedat = pd.read_csv("ratings.csv", sep = ",", iterator=True, chunksize=1000000)
largedat = pd.concat([chunk for chunk in largedat])   
""" took 24.79 secs """
largedat.describe()
largedat.dtypes

