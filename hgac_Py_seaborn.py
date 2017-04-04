# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:12:43 2017

@author: akuppam
"""
""" ################################# 
  PYTHON SEABORN PAIRPLOT
  http://seaborn.pydata.org/generated/seaborn.pairplot.html
################################# """

""" EL PASO validation """
import pandas as pd
import os
import seaborn as sns

os.chdir("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate")
elpv = pd.read_csv("elpvalidation.csv")
elpv.describe()
elpv.dtypes

elpv_scat = pd.concat([elpv.ID1, elpv.Func_Class, elpv.Scrnline, elpv.AType, elpv.COUNT, elpv.VOLUME], axis=1)
elpv_scrn = elpv_scat.dropna()
elpv_scat.describe()
elpv_scrn.describe()

elpv_scat = elpv_scat[elpv_scat.COUNT > 0]
elpv_scrn = elpv_scrn[elpv_scrn.COUNT > 0]

elpv_scat = elpv_scat.drop('ID1', axis=1)
elpv_scat = elpv_scat.drop('Scrnline', axis=1)

fig = sns.pairplot(elpv_scat, hue="AType", palette="husl")
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\pair1.jpg")

os.chdir("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate")
elpva = pd.read_csv("va.csv")
elpva.describe()

""" ---------------------------------------- """

""" ################################# 
  PYTHON SEABORN LMPLOT
################################# """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig = sns.lmplot('COUNT', 'VOLUME', 
           data=elpv_scat, 
           fit_reg=False, 
           size=20,
           hue="AType",  
           scatter_kws={"marker": "D", 
                        "s": 20})
plt.title('El Paso - Count vs. Volume')
plt.xlabel('COUNT')
plt.ylabel('VOLUME')
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\scatter1.jpg")

fig = sns.lmplot('COUNT', 'VOLUME', 
           data=elpv_scat, 
           fit_reg=True, 
           size=10,
           hue="AType",  
           scatter_kws={"marker": "D", 
                        "s": 20})
plt.title('El Paso - Count vs. Volume')
plt.xlabel('COUNT')
plt.ylabel('VOLUME')
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\scatter2.jpg")

fig = sns.lmplot('COUNT', 'VOLUME', 
           data=elpv_scat, 
           fit_reg=True, 
           size=5,
           col_wrap=3,
           hue="AType",  
           col="AType",
           scatter_kws={"marker": "D", 
                        "s": 20})
plt.xlabel('COUNT')
plt.ylabel('VOLUME')
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\scatter5.jpg")

fig = sns.lmplot('COUNT', 'VOLUME', 
           data=elpv_scat, 
           fit_reg=True, 
           size=10,
           markers=["o","x",">","<","+"],
           hue="AType",  
           scatter_kws={"marker": "D", 
                        "s": 30})
plt.title('El Paso - Count vs. Volume')
plt.xlabel('COUNT')
plt.ylabel('VOLUME')
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\scatter7.jpg")

""" ---------------------------------------- """

""" ################################# 
  PYTHON SEABORN VIOLINPLOT
################################# """

import seaborn as sns
fig = sns.violinplot(elpv['AType'], elpv['VOLUME'])
# Set the size of the graph from here
fig.figure.set_size_inches(12,7)
# Set the Title of the graph from here
fig.axes.set_title("Volume by Area Type", fontsize=20,color="b",alpha=0.3)
# Save it to a *.png or *.jpg file in a specific folder
fig1 = fig.get_figure()
fig1.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\VolATYPE-violin.jpg")


""" ################################# 
  PYTHON SEABORN JOINTPLOT
################################# """

import seaborn as sns
elpv.COUNT = elpv.COUNT[elpv.COUNT != 0]
sns.jointplot('COUNT', 'VOLUME', data=elpv, size=15)
sns.jointplot('COUNT', 'VOLUME', data=elpv, kind='reg', size=15, color='g')
sns.jointplot('COUNT', 'VOLUME', data=elpv, kind='hex', size=15, color='b')
sns.jointplot('COUNT', 'VOLUME', data=elpv, kind='kde', size=15, color='r')

""" ################################# 
  PYTHON SEABORN HEATMAP
################################# """

import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

os.chdir("c:\\users\\akuppam\\documents\\ARK\\BD\\WSDOT-HSR\\HSR2017\\TAF-Data\\2008Autobiz")
abiz08 = pd.read_csv("2008AutoBiz.csv")
abiz08.columns = ["origin", "destination", "trips08"]
abiz08.describe()
abiz08.trips08.sum()

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

sns.heatmap(od, linewidths=0.5, square=True)
od.to_csv('od_abiz08hsr.csv')

# -------------------------------
             
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

# -----------------------------------------------
# H-GAC TOURCAST 'WORKDEST' - OBSERVED vs MODELED
# -----------------------------------------------

import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

os.chdir("c:\\Users\\akuppam\\Documents\\HGAC_ABM\\Training\\R-Py\\hgac")
observed = pd.read_csv("observed.csv")
observed.describe()
observed.sum()

obs = observed.pivot(index='origin', columns='destination', values='trips')

sns.heatmap(obs, linewidths=0.5, square=True)
obs.to_csv('obs.csv')

# -----------------------------------

import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

os.chdir("c:\\Users\\akuppam\\Documents\\HGAC_ABM\\Training\\R-Py\\hgac")
modeled = pd.read_csv("modeled.csv")
modeled.describe()
modeled.sum()

mod = modeled.pivot(index='origin', columns='destination', values='trips')

sns.heatmap(mod, linewidths=0.5, square=True)
mod.to_csv('mod.csv')

# ----------------------------------------
