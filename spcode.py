# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 21:22:46 2016

@author: AKUPPAM
"""

"""spcode.py"""
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)

sc = SparkContext("local", "Simple App")

lines = sc.textFile("README.md")
lines.count()

