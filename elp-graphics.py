# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 23:27:40 2016

@author: AKUPPAM
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

elp = pd.read_csv("c:\\users\\akuppam\\documents\\el paso model\\task 3\\assign\\elp.csv")
elp.describe()
elp.head(5)

sns.set(style="whitegrid")

""" DO NOT ATTEMPT THIS - TAKES A LONG LONG TIME """
# Draw a nested barplot
g = sns.factorplot(x="COUNT", y="VOLUME", hue="SCRNLINE", data=elp,
                   size=6, kind="bar", palette="muted")

g.despine()
g.set_ylabels("Screenline")
# -------------------------------------------------------------------                  
h = sns.factorplot(x="COUNT", y="VOLUME", hue="ATYPE", data=elp,
                   size=6, kind="bar", palette="muted")
h.despine()
h.set_ylabels("Area Type")

"""
plt.savefig('Count.png')

sns.factorplot.savefig('Count.png')

import matplotlib.pyplot as plt
plt.show()

fig = h.get_figure()
fig.savefig("Count_vs_Vol_by_atype.png")

fig = h.get_figure()
fig.savefig("Count_vs_Vol_by_atype.png")


budget_plot = budget.plot(kind="bar",x=budget["detail"],
                          title="MN Capital Budget - 2014",
                          legend=False)


#This does all of the heavy lifting of creating the plot using the “detail” column as well as displaying #the title and removing the legend.

#Here is the additional code needed to save the image as a png.

fig = budget_plot.get_figure()
fig.savefig("2014-mn-capital-budget.png")

h.savefig('Count vs Vol by atype.png')

sns_plot.savefig('output.png')

swarm_plot = sns.swarmplot(...)
fig = swarm_plot.get_figure()
fig.savefig(...) 
"""

# -------------------------------------------------------------------
sns.barplot(x='COUNT', y='VOLUME', data=elp)
sns.despine()
# -------------------------------------------------------------------
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="COUNT", y="VOLUME", hue="ATYPE", data=elp, split=True)
sns.despine(left=True)

sns.scatterplot(x="COUNT", y="VOLUME", hue="ATYPE", data=elp)
sns.despine()

""" look into seaborn examples, links, etc and get as many charts for elp as possible """
# -------------------------------------------------------------------
elp.COUNT = elp.COUNT[(elp.COUNT!=0)]
elp.COUNT = elp.COUNT.dropna()
elp.head(5)
elp.describe()
# -------------------------------------------------------------------

g1 = sns.regplot(x="COUNT", y="VOLUME", data=elp, ci = False, 
    scatter_kws={"color":"darkred","alpha":0.3,"s":90},
    line_kws={"color":"g","alpha":0.5,"lw":4},marker="x")

# remove the top and right line in graph
sns.despine()

# Set the size of the graph from here
g1.figure.set_size_inches(12,8)
# Set the Title of the graph from here
g1.axes.set_title('Total Bill vs. Tip', fontsize=34,color="r",alpha=0.5)
# Set the xlabel of the graph from here
g1.set_xlabel("Tip",size = 67,color="r",alpha=0.5)
# Set the ylabel of the graph from here
g1.set_ylabel("Total Bill",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g1.tick_params(labelsize=14,labelcolor="black")

# -------------------------------------------------------------------
sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
elp.SCRNLINE = elp.SCRNLINE.dropna()

g = sns.barplot(x="SCRNLINE", y="VOLUME", hue="ATYPE", 
    palette=sns.color_palette(flatui),data=elp,ci=None)

# remove the top and right line in graph
sns.despine()

# --------------------------------------------------------------------

sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
elp.SCRNLINE = elp.SCRNLINE.dropna()
elp.COUNT = elp.COUNT[(elp.COUNT!=0)]
elp.COUNT = elp.COUNT.dropna()
elp.ATYPE = elp.ATYPE.dropna()

g = sns.barplot(x="SCRNLINE", y="COUNT", hue="ATYPE", 
    palette=sns.color_palette(flatui),data=elp,ci=None)

# remove the top and right line in graph
sns.despine()


# -----------------------------------------------------------------------
import matplotlib.plotly as plt
import seaborn as sns

# Create a Pairplot
g2 = sns.pairplot(elp,hue="FUNC_CLASS",palette="muted",size=5, 
    vars=["COUNT", "VOLUME"],kind='reg',markers=['o','x','+'])

# To change the size of the scatterpoints in graph
g2 = g2.map_offdiag(plt.scatter,  s=35,alpha=0.5)

# remove the top and right line in graph
sns.despine()
# Additional line to adjust some appearance issue
plt.subplots_adjust(top=0.9)

# Set the Title of the graph from here
g2.fig.subtitle('COUNT vs VOLUME - Func Class', 
    fontsize=34,color="b",alpha=0.3)

# ------------------------------------------------------------------------
import os
import pandas as pd
os.chdir('C:\\users\\akuppam\\documents\\Pythprog\\Rprog')
os.getcwd()
mck = pd.read_csv('mckdata.csv')
mck.describe()
mck.dtypes

# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Create a Pairplot
g3 = sns.pairplot(mck,hue="ftype",palette="muted",size=10000,
                  vars=["ht", "wt"],kind='reg',markers=['o','x','+'])

# To change the size of the scatterpoints in graph
g3 = g3.map_offdiag(plt.scatter,  s=35,alpha=0.5)

# remove the top and right line in graph
sns.despine()
# Additional line to adjust some appearance issue
plt.subplots_adjust(top=0.9)

# Set the Title of the graph from here
g3.fig.subtitle('ht vs wt - ftype', 
    fontsize=34,color="b",alpha=0.3)

# ------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

g3 = sns.FacetGrid(elp, col='ATYPE', size=10000, aspect=500)
g3 = g3.map(sns.boxplot, 'COUNT', 'VOLUME', showmeans=True, color='y')

sns.barplot(x='COUNT', y='VOLUME', data=elp)
sns.despine()

# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

g3 = sns.FacetGrid(mck, col='atype', size=100, aspect=500)
g3 = g3.map(sns.boxplot, 'dist', 'wt', showmeans=True, color='y')

sns.barplot(x='dist', y='wt', data=mck)
sns.despine()

# -------------------------------------------------------------------------

# Tell ipython to load the matplotlib environment.
#%matplotlib
 
import itertools
 
import pandas
import numpy
import seaborn
import matplotlib.pyplot
 
import os
import pandas as pd
os.chdir('C:\\users\\akuppam\\documents\\Pythprog\\Rprog')
os.getcwd()
mck = pd.read_csv('mckdata.csv')
mck.describe()
mck.dtypes

# _DATA_FILEPATH = 'datagovdatasetsviewmetrics.csv'
_ROTATION_DEGREES = 90
_BOTTOM_MARGIN = 0.35
_COLOR_THEME = 'coolwarm'
_LABEL_X = 'Organizations'
_LABEL_Y = 'Views'
_TITLE = 'Organizations with Most Views'
_ORGANIZATION_COUNT = 10
_MAX_LABEL_LENGTH = 20
 
def get_data():
    # Read the dataset.
 
#    d = pandas.read_csv(_DATA_FILEPATH)
    d = mck
 
    # Group by organization.
 
    def sum_views(df):
        return sum(df['Views per Month'])
 
    g = d.groupby('Organization Name').apply(sum_views)
 
    # Sort by views (descendingly).
 
    g.sort(ascending=False)
 
    # Grab the first N to plot.
 
    items = g.iteritems()
    s = itertools.islice(items, 0, _ORGANIZATION_COUNT)
 
    s = list(s)
 
    # Sort them in ascending order, this time, so that the larger ones are on 
    # the right (in red) in the chart. This has a side-effect of flattening the 
    # generator while we're at it.
    s = sorted(s, key=lambda (n, v): v)
 
    # Truncate the names (otherwise they're unwieldy).
 
    distilled = []
    for (name, views) in s:
        if len(name) > (_MAX_LABEL_LENGTH - 3):
            name = name[:17] + '...'
 
        distilled.append((name, views))
 
    return distilled
 
def plot_chart(distilled):
    # Split the series into separate vectors of labels and values.
 
    labels_raw = []
    values_raw = []
    for (name, views) in distilled:
        labels_raw.append(name)
        values_raw.append(views)
 
    labels = numpy.array(labels_raw)
    values = numpy.array(values_raw)
 
    # Create one plot.
 
    seaborn.set(style="white", context="talk")
 
    (f, ax) = matplotlib.pyplot.subplots(1)
 
    b = seaborn.barplot(
        labels, 
        values,
        ci=None, 
        palette=_COLOR_THEME, 
        hline=0, 
        ax=ax,
        x_order=labels)
 
    # Set labels.
 
    ax.set_title(_TITLE)
    ax.set_xlabel(_LABEL_X)
    ax.set_ylabel(_LABEL_Y)
 
    # Rotate the x-labels (otherwise they'll overlap). Seaborn also doesn't do 
    # very well with diagonal labels so we'll go vertical.
    b.set_xticklabels(labels, rotation=_ROTATION_DEGREES)
 
    # Add some margin to the bottom so the labels aren't cut-off.
    matplotlib.pyplot.subplots_adjust(bottom=_BOTTOM_MARGIN)
 
distilled = get_data()
plot_chart(distilled)

# ---------------------------------------------------------------------------
sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
mck.SCRNLINE = mck.SCRNLINE.dropna()
mck.COUNT = mck.COUNT[(elp.COUNT!=0)]
mck.COUNT = mck.COUNT.dropna()
mck.ATYPE = mck.ATYPE.dropna()

g = sns.barplot(x="trips", y="tons", hue="atype", 
    palette=sns.color_palette(flatui),data=mck,ci=None)

# remove the top and right line in graph
sns.despine()

# ---------------------------------------------------------------------------

import DeckGL from 'deck.gl/react';

npm install --save deck.gl luma.gl

# ----------------------  2/2/2017 ------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

elp = pd.read_csv("c:\\users\\akuppam\\documents\\el paso model\\task 3\\assign\\elp.csv")
elp.describe()
elp.head(5)

import seaborn as sns
sns.violinplot(elp['FUNC_CLASS'], elp['VOLUME'])
sns.despine()
sns.violinplot(elp['FUNC_CLASS'], elp['MILES'])
sns.despine()

""" unable to read elp_trips.csv"""

sns.jointplot(elp['FUNC_CLASS'], elp['VOLUME'])
sns.jointplot(elp['COUNT'], elp['VOLUME'])


# --------------------------  2/4/2017 - reading dbf files ------------------

"""
from dbfread import DBF
mandFirst = DBF('C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\TourCast\\BMC_InSITE_1PercentSample\\TC_MandatoryFirstTours_1.dbf', load=True)
"""

""" ---------------------------------------- """

""" https://pypi.python.org/pypi/simpledbf/0.2.4 """
from simpledbf import Dbf5

mF = Dbf5('C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\TourCast\\BMC_InSITE_1PercentSample\\TC_MandatoryFirstTours_1.dbf', codec='utf-8')
mF.numrec
mF.fields

mFdf = mF.to_dataframe()
mFdf.describe()

""" SAVING FIGURE TO FOLDER - THIS WORKS FINE """
import seaborn as sns
fig = sns.violinplot(mFdf['tourPurp'], mFdf['nCars'])
# Set the size of the graph from here
fig.figure.set_size_inches(12,7)
# Set the Title of the graph from here
fig.axes.set_title("tourPurp vs. nCars", fontsize=20,color="b",alpha=0.3)
# Save it to a *.png or *.jpg file in a specific folder
fig1 = fig.get_figure()
fig1.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\tourPurp_nCars_violin.jpg")

import seaborn as sns
fig = sns.factorplot(x=mFdf['personId'], y=mFdf['hhId'], hue="ctype", data=mFdf)
# Save it to a *.png or *.jpg file in a specific folder
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\tourPurp_nCars_factor.jpg")

import seaborn as sns
fig = sns.factorplot(x=elp["COUNT"], y=elp["VOLUME"], hue=elp["ATYPE"], data=elp)

"""
factorplot
facetgrid
pairplot
"""
""" ---------------------------------------- """

""" http://seaborn.pydata.org/generated/seaborn.factorplot.html """

import pandas as pd
mck = pd.read_csv("c:\\users\\akuppam\\documents\\Pythprog\\mckdata.csv")
mck.describe()
mck.dtypes

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', data=mck)
fig.figure.set_size_inches(12,7)
fig.axes.set_title("ht vs. wt", fontsize=20,color="b",alpha=0.3)
fig1 = fig.get_figure()
fig1.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck1.jpg")

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', data=mck, kind="violin")
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck2.jpg")

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', col='ftype', data=mck, kind="violin")
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck3.jpg")

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', col='ftype', data=mck)
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck4.jpg")

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', col='ftype', data=mck, size=5, aspect=.8)
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck5.jpg")

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', col='ftype', data=mck, size=5, aspect=.8, orient="h")
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck6.jpg")

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', col='ftype', data=mck, size=5, aspect=.8, orient="h", split=True)
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck7.jpg")

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', row='ftype', data=mck, size=5, aspect=.8, orient="h")
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck8.jpg")

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', row='ftype', data=mck, size=5, aspect=.8, orient="h", facet_kws=None)
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck9.jpg")

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', row='ftype', data=mck, size=5, aspect=.8, orient="h", margin_titles=True)
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck10.jpg")

import seaborn as sns
fig = sns.factorplot(x='ht', y='wt', hue='atype', row='ftype', data=mck, size=5, aspect=.8, orient="h", margin_titles=True, kind="bar")
fig.savefig("C:\\Users\\akuppam\\Documents\\BMC_ABM\\Outputs\\TourCast\\mck11.jpg")

""" ---------------------------------------- """

""" EL PASO validation """
import pandas as pd
import os

os.chdir("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate")
elpv = pd.read_csv("elpvalidation.csv")
elpv.describe()
elpv.dtypes

elpv_scat = pd.concat([elpv.ID1, elpv.Func_Class, elpv.Scrnline, elpv.AType, elpv.COUNT, elpv.VOLUME], axis=1)
""" http://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-certain-columns-is-nan """
elpv_scrn = elpv_scat.dropna()
elpv_scat.describe()
elpv_scrn.describe()

elpv_scat = elpv_scat[elpv_scat.COUNT > 0]
elpv_scrn = elpv_scrn[elpv_scrn.COUNT > 0]

elpv_scat = elpv_scat.drop('ID1', axis=1)
elpv_scat = elpv_scat.drop('Scrnline', axis=1)

fig = sns.pairplot(elpv_scat, hue="AType", palette="husl")
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\pair1.jpg")

import seaborn as sns
fig = sns.violinplot(elpv_scat['COUNT'], elpv_scat['VOLUME'])
fig.figure.set_size_inches(12,7)
fig.axes.set_title("COUNT vs. VOLUME", fontsize=20,color="b",alpha=0.3)
fig1 = fig.get_figure()
fig1.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\violin1.jpg")

import seaborn as sns
fig = sns.factorplot(x='COUNT', y='VOLUME', hue='AType', data=elpv_scat, size=5, aspect=.8, orient="h", margin_titles=True)
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\factor1.jpg")

# seems like factorplot is NOT good for scatter plots (vol vs count)
import seaborn as sns
fig = sns.factorplot(x='COUNT', y='VOLUME', hue='AType', row='Func_Class', data=elpv_scat, size=5, aspect=.8, orient="h", margin_titles=True)
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\factor2.jpg")

""" ---------------------------------------- """
""" http://chrisalbon.com/python/seaborn_scatterplot.html """
""" http://seaborn.pydata.org/generated/seaborn.lmplot.html """

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
           hue="AType",  
           col="AType",
           scatter_kws={"marker": "D", 
                        "s": 20})
plt.xlabel('COUNT')
plt.ylabel('VOLUME')
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\scatter3.jpg")

fig = sns.lmplot('COUNT', 'VOLUME', 
           data=elpv_scat, 
           fit_reg=True, 
           size=5,
           hue="AType",  
           col="AType")
plt.xlabel('COUNT')
plt.ylabel('VOLUME')
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\scatter4.jpg")

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
           size=5,
           col_wrap=3,
           markers=["x"],
           hue="AType",  
           col="AType",
           scatter_kws={"marker": "D", 
                        "s": 20})
plt.xlabel('COUNT')
plt.ylabel('VOLUME')
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\scatter6.jpg")

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

""" http://seaborn.pydata.org/generated/seaborn.heatmap.html """

flights = flights.pivot("month", "year", "passengers")
ax = sns.heatmap(flights)
ax = sns.heatmap(flights, annot=True, fmt="d")

elpv_heat = elpv_scat.pivot("Func_Class", "AType", "COUNT")
elpv_heat = elpv_scat.pivot("VOLUME", "COUNT")

""" ---------------------------------------- """

""" http://seaborn.pydata.org/examples/scatterplot_categorical.html """

""" ??? savefig does NOT work for sawrmplots  ??? """

fig = sns.swarmplot(x="AType", y="VOLUME", hue="Func_Class", data=elpv_scat)
plt.title('El Paso - Volume by AType')
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\swarmplot1.jpg")

fig = sns.swarmplot(x="Func_Class", y="VOLUME", hue="AType", data=elpv_scat)
plt.title('El Paso - Volume by Func Class')
fig.savefig("C:\\Users\\akuppam\\Documents\\El Paso Model\\NewUpdate\\swarmplot2.jpg")

""" ---------------------------------------- """

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("c:\\users\\akuppam\Documents\\Pythprog")
os.getcwd()

mck = pd.read_csv("mckdata.csv")
mck.describe()
mck.dtypes

fig = sns.swarmplot(x="atype", y="vol", hue="ftype", data=mck)
plt.title('mck - Volume by AType')fig = sns.swarmplot(x="atype", y="vol", hue="ftype", data=mck)
fig.savefig("C:\\Users\\akuppam\\Documents\\Pythprog\\swarmplot3.jpg")

fig = sns.swarmplot(x="ftype", y="vol", hue="atype", data=mck)

""" # ##########################################
CHORD DIAGRAM
https://plot.ly/python/
(this link in 'Chrome' has several tutorials)
########################################### """

#  type this in Anaconda dos prompt (to install plotly) > pip install plotly 

import plotly.plotly as py
from plotly.graph_objs import *
from plotly.tools import FigureFactory as FF
plotly.figure_factory.create_table

data = [['', 'Emma', 'Isabella', 'Ava', 'Olivia', 'Sophia', 'row-sum'],
        ['Emma', 16, 3, 28, 0, 18, 65],
        ['Isabella', 18, 0, 12, 5, 29, 64],
        ['Ava', 9, 11, 17, 27, 0, 64],
        ['Olivia', 19, 0, 31, 11, 12, 73],
        ['Sophia', 23, 17, 10, 0, 34, 84]]

table = plotly.figure_factory.create_table(data, index=True)
table = FF.create_table(data, index=True)
py.iplot(table, filename='Data-Table')

import numpy as np

matrix=np.array([[16,  3, 28,  0, 18],
                 [18,  0, 12,  5, 29],
                 [ 9, 11, 17, 27,  0],
                 [19,  0, 31, 11, 12],
                 [23, 17, 10,  0, 34]], dtype=int)

def check_data(data_matrix):
    L, M=data_matrix.shape
    if L!=M:
        raise ValueError('Data array must have (n,n) shape')
    return L

L=check_data(matrix)

def make_ideogram_arc(R, phi, a=50):
    # R is the circle radius
    # phi is the list of ends angle coordinates of an arc
    # a is a parameter that controls the number of points to be evaluated on an arc
    if not test_2PI(phi[0]) or not test_2PI(phi[1]):
        phi=[moduloAB(t, 0, 2*PI) for t in phi]
    length=(phi[1]-phi[0])% 2*PI
    nr=5 if length<=PI/4 else int(a*length/PI)

    if phi[0] < phi[1]:
        theta=np.linspace(phi[0], phi[1], nr)
    else:
        phi=[moduloAB(t, -PI, PI) for t in phi]
        theta=np.linspace(phi[0], phi[1], nr)
    return R*np.exp(1j*theta)

z=make_ideogram_arc(1.3, [11*PI/6, PI/17])
print (z)

# ############################

