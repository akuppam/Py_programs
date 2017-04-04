# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from supplychainpy import model_inventory
from _decimal import Decimal

yearly_demand = {'jan': 75, 'feb': 75, 'mar': 75, 'apr': 75,
                 'may': 75, 'jun': 75, 'jul': 25, 'aug': 25,
                 'sep': 25, 'oct': 25, 'nov': 25, 'dec': 25}
summary = model_inventory.analyse_orders(data_set= yearly_demand,
                                         sku_id='RX983-90',
                                         lead_time=Decimal(3), unit_cost=Decimal(34.99),
                                         reorder_cost=Decimal(400),z_value=Decimal(1.28))
                                         
print summary


import sys
sys.path

from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension

extensions =[Extension('supplychainpy.simulations.sim_summary', ['supplychainpy/simulations/sim_summary.pyx']),
             Extension('supplychainpy.demand.eoq', ['supplychainpy/demand/eoq.pyx'])
             ]


setup(name='supplychainpy',
      version='0.0.2',
      description='A library for supply chain, operations and manufacturing, analysis, modeling and simulation.',
      url='https://github.com/KevinFasusi/supplychainpy',
      download_url='https://github.com/KevinFasusi/supplychainpy/tarball/0.0.2',
      author='Kevin Fasusi',
      author_email='kevin@supplybi.com',
      license='BSD 3',
      packages=find_packages(exclude=['docs', 'tests']),
      test_suite='supplychainpy/tests',
      install_requires=['NumPy'],
      keywords=['supply chain', 'operations research', 'operations management', 'simulation'],
      ext_modules=cythonize(extensions),
      )

