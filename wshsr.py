# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:51:57 2017

@author: AKUPPAM
"""

import numpy
import matplotlib.pylab as plt
m = [[0.0, 1.47, 2.43, 3.44, 1.08, 2.83, 1.08, 2.13, 2.11, 3.7], [1.47, 0.0, 1.5,     2.39, 2.11, 2.4, 2.11, 1.1, 1.1, 3.21], [2.43, 1.5, 0.0, 1.22, 2.69, 1.33, 3.39, 2.15, 2.12, 1.87], [3.44, 2.39, 1.22, 0.0, 3.45, 2.22, 4.34, 2.54, 3.04, 2.28], [1.08, 2.11, 2.69, 3.45, 0.0, 3.13, 1.76, 2.46, 3.02, 3.85], [2.83, 2.4, 1.33, 2.22, 3.13, 0.0, 3.83, 3.32, 2.73, 0.95], [1.08, 2.11, 3.39, 4.34, 1.76, 3.83, 0.0, 2.47, 2.44, 4.74], [2.13, 1.1, 2.15, 2.54, 2.46, 3.32, 2.47, 0.0, 1.78, 4.01], [2.11, 1.1, 2.12, 3.04, 3.02, 2.73, 2.44, 1.78, 0.0, 3.57], [3.7, 3.21, 1.87, 2.28, 3.85, 0.95, 4.74, 4.01, 3.57, 0.0]]
matrix = numpy.matrix(m)
matrix

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

m1 = [[2000, 1500, 200], [300, 500, 800], [10, 25, 35]]
matrix1 = numpy.matrix(m1)
matrix1
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(matrix1, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

# #################################

import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

os.chdir("c:\\users\\akuppam\\documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\2008Autobiz")
abiz08 = pd.read_csv("2008AutoBiz.csv")
abiz08.columns = ["origin", "destination", "trips08"]
abiz08.describe()
abiz08.trips08.sum()

#df = DataFrame({'gender': np.random.choice(['m', 'f'], size=10), 'price': poisson(100, size=10)})
#abiz08hsr = abiz08(['origin': ('53073', '53057'), 'destination': ('53073', '53057')])
#df[df['first_name'].notnull() & (df['nationality'] == "USA")]
#abiz08[(abiz08['origin'] == (53073)) & (abiz08['destination'] == (53073))]
#abiz08[(abiz08['origin'] == (53073, 53057)) & (abiz08['destination'] == (53073, 53057))]
#abiz08[(abiz08['origin'] == (53073 & 53057)) & (abiz08['destination'] == (53073 & 53057))]
#s[s.isin([2, 4, 6])]
# abiz08[(abiz08['origin'].isin([53073, 53057])) & (abiz08['destination'].isin([53073, 53057]))]
"""
http://pandas.pydata.org/pandas-docs/stable/indexing.html
Good URL for indexing, slicing, subsetting columns in pandas
"""

WA = [53073,
53057,
53061,
53033,
53035,
53045,
53053,
53067,
53041,
53015,
53011]

OR = [41051,
41067,
41005,
41009,
41047]

WAOR = [53073,
53057,
53061,
53033,
53035,
53045,
53053,
53067,
53041,
53015,
53011,
41051,
41067,
41005,
41009,
41047]

abiz08hsr = abiz08[(abiz08['origin'].isin(WAOR)) & (abiz08['destination'].isin(WAOR))]
abiz08hsr.trips08.sum()
abiz08.trips08.sum()


od = abiz08hsr.pivot(index='origin', columns='destination', values='trips08')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(od, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

# http://seaborn.pydata.org/generated/seaborn.heatmap.html

sns.heatmap(od)
sns.heatmap(od, annot=True)
sns.heatmap(od, linewidths=0.5, square=True)

# dfcity.to_csv('dfcity1.csv')            # this one reads well in excel
od.to_csv('od_abiz08hsr.csv')
             
# --------   IGNORE THE FOLLOWING LINES OF CODE FOR NOW --------------------------
abiz08hsr.origin[53033 == "King"]
abiz081 = abiz08hsr.origin

abiz08hsr['origin1'] = abiz08hsr['origin']
abiz08hsr['origin1'] = abiz08hsr.origin(['53033' == 'King'])

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000} 
abiz08hsr['origin1'] = {'53033': 'King'}

abiz08hsr['origin1'] = {'41005' : 'Clackamas County',
41009 : 'Columbia County',
41047 : 'Marion County',
41051 : 'Multnomah County',  
41067 : 'Washington County',
53011 : 'Clark',
53015 : 'Cowlitz',
53033 : 'King',
53035 : 'Kitsap',
53041 : 'Lewis',
53045 : 'Mason',
53053 : 'Pierce',
53057 : 'Skagit',
53061 : 'Snohomish',
53067 : 'Thurston',
53073 : 'Whatcom'}


# obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c']) 
# from pandas import Series, DataFrame

# abiz08hsr.origin = Series([53033], index=['King'])

# df.pivot(index='date', columns='variable', values='value')

# ----------------------------

import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

os.chdir("c:\\users\\akuppam\\documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\2008Autononbiz")
anbiz08 = pd.read_csv("2008AutoNonBiz.csv")
anbiz08.columns = ["origin", "destination", "trips08"]
anbiz08.describe()

anbiz08hsr = anbiz08[(anbiz08['origin'].isin(WAOR)) & (anbiz08['destination'].isin(WAOR))]
anbiz08hsr.trips08.sum()
anbiz08.trips08.sum()

od1 = anbiz08hsr.pivot(index='origin', columns='destination', values='trips08')

sns.heatmap(od1, linewidths=0.5, square=True)
od1.to_csv('od1_anbiz08hsr.csv')

# ----------------------------

import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

os.chdir("c:\\users\\akuppam\\documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\2008Rail")
rail08 = pd.read_csv("2008Rail.csv")
rail08.columns = ["origin", "destination", "trips08"]
rail08.describe()

rail08hsr = rail08[(rail08['origin'].isin(WAOR)) & (rail08['destination'].isin(WAOR))]
rail08hsr.trips08.sum()
rail08.trips08.sum()

od2 = rail08hsr.pivot(index='origin', columns='destination', values='trips08')

sns.heatmap(od2, linewidths=0.5, square=True)
od2.to_csv('od2_rail08hsr.csv')

# ----------------------------

import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

os.chdir("c:\\users\\akuppam\\documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\2008Bus")
bus08 = pd.read_csv("2008Bus.csv")
bus08.columns = ["origin", "destination", "trips08"]
bus08.describe()

bus08hsr = bus08[(bus08['origin'].isin(WAOR)) & (bus08['destination'].isin(WAOR))]
bus08hsr.trips08.sum()
bus08.trips08.sum()

od3 = bus08hsr.pivot(index='origin', columns='destination', values='trips08')

sns.heatmap(od3, linewidths=0.5, square=True)
od3.to_csv('od3_bus08hsr.csv')

# ----------------------------

import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

os.chdir("c:\\users\\akuppam\\documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\2008Air")
air08 = pd.read_csv("2008Air.csv")
air08.columns = ["origin", "destination", "trips08"]
air08.describe()

air08hsr = air08[(air08['origin'].isin(WAOR)) & (air08['destination'].isin(WAOR))]
air08hsr.trips08.sum()
air08.trips08.sum()

od4 = air08hsr.pivot(index='origin', columns='destination', values='trips08')

sns.heatmap(od4, linewidths=0.5, square=True)
od4.to_csv('od4_air08hsr.csv')

# WRITE TO CSV FILES

# C:\Users\akuppam\Documents\ARK\BD\WSDOT-HSR\HSR2017\TAF-Data

abiz08hsr.to_csv('C:\\Users\\akuppam\\Documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\abiz08hsr.csv')
anbiz08hsr.to_csv('C:\\Users\\akuppam\\Documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\anbiz08hsr.csv')
rail08hsr.to_csv('C:\\Users\\akuppam\\Documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\rail08hsr.csv')
bus08hsr.to_csv('C:\\Users\\akuppam\\Documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\bus08hsr.csv')
air08hsr.to_csv('C:\\Users\\akuppam\\Documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\air08hsr.csv')

abiz08hsr.sum()
anbiz08hsr.sum()
rail08hsr.sum()
bus08hsr.sum()
air08hsr.sum()


"""
41005 = Clackamas County
41009 = Columbia County
41047 = Marion County
41051 = Multnomah County  
41067 = Washington County
53011 = Clark
53015 = Cowlitz
53033 = King
53035 = Kitsap
53041 = Lewis
53045 = Mason
53053 = Pierce
53057 = Skagit
53061 = Snohomish
53067 = Thurston
53073 = Whatcom
"""

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(figsize=(20,20))         # Sample figsize in inches
#sns.heatmap(od, ax=ax)
#sns.heatmap(od, annot_kws={"size": 30})
#sns.heatmap(od, annot=True, annot_kws={"size": 20})


"""
WSDOT HSR
WA FIPS Codes (53xxx)
53073
53057
53061
53033
53035
53045
53053
53067
53041
53015
53011

whatcom
skagit
snohomish
king
kitsap
mason
pierce
thurston
lewis
cowitz
clark

OR FIPS Codes (41xxx)

41051
41067
41005
41009
41047

Multnomah County  
Washington County
Clackamas County
Columbia County
Marion County
"""
