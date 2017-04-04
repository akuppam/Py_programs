# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 09:38:53 2017

@author: AKUPPAM
"""
"""
mckread.py
"""
import os
import pandas as pd
import mckparam as par
import glob

configfile = '~/mckparam.py'
import configfile

def main():
    data_folder = configfile.mckfolder
    os.chdir = data_folder
    for ProcessingFile in glob.glob(".csv"):
        print ('Reading mck file', ProcessingFile)
        mck = pd.read_csv(data_folder+ProcessingFile)
        mck.describe()

if __name__ == '__main__':
    main()


par.mckfolder

configfile = '~/mckparam.py'

import os
import sys

sys.path.append(os.path.dirname(os.path.expanduser(configfile)))

import configfile

# --------------  MAG

import pymck as P
import Pybasics as pb
import os
import glob
import pandas as pd

def main():

    data_folder = 'c:/users/akuppam/documents/Pythprog/Rprog/'
    os.chdir(data_folder)
    # reads all CSV files in the folder
    for ProcessingFile in glob.glob("*.csv"):

        # Open the datafile
        print ('Mock Data File', ProcessingFile)
        # There are several csv file in 'Rprog'
        # So reads the last listed file (alphabetical order)
        atri = pd.read_csv(data_folder+ProcessingFile)

        deleted = atri.id.shape[0]
        atri.drop_duplicates(atri.columns, inplace=True)
        print (deleted-atri.id.shape[0], '   repeated records')
        
        # computes total number of rows
        permw = atri.perimeter_worst.shape[0]   
        # drops rows in 'perimeter_worst' that have same values
        atri.drop_duplicates('perimeter_worst', inplace=True)    
        # computes no of dropped rows
        print (permw - atri.perimeter_worst.shape[0], '   repeated records')  


# TO DO 3-5-2017
"""create a *.py for all the paths and parameters
create another *.py to call the 1st file, read csv file, read time vars
go thro' 'finding_stops'
# SHELVED UNTIL MODULES, FUNCTIONS, ETC
# THE SOFTWARE PART OF PYTHON

"""
# --------------  MAG
# ##########################################

# SHELVED UNTIL MODULES, FUNCTIONS, ETC
# THE SOFTWARE PART OF PYTHON
# ############################################

