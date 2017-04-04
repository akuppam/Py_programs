# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:55:51 2017

@author: AKUPPAM
"""

"""
Patil_et_al_2010 (PyMC).PDF
https://anaconda.org/anaconda/pymc
conda install -c anaconda pymc=2.3.6
"""
import pymc
import numpy as np

n = 5 * np.ones(4, dtype=int)
x = np.array([-0.86, -0.3, -0.05, 0.731])

alpha = pymc.Normal('alpha', mu=0, tau=0.01)
beta = pymc.Normal('beta', mu=0, tau=0.01)











