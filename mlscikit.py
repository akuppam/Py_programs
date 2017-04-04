# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 21:33:14 2016

@author: AKUPPAM
"""
""" MASTERING MACHINE LEARNING WITH SCIKIT-LEARN.PDF """
# *********************
# Chapter 1 and 2
# *********************

import matplotlib.pyplot as plt
x = [[6],[7],[8],[9]]
y = [[1],[2],[4],[6]]

""" plots observations, grid lines, title, x and y labels """
plt.figure()
plt.title('gfgfg')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y,'k.')
plt.axis([0,25,0,25])
plt.grid(True)
plt.show()

""" just displays observations """
plt.plot(x,y,'k.')

""" straight line connecting observations """
plt.plot(x,y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
predictions = model.predict(x)
predictions

""" output coefficients and intercepts """
model.coef_
model.intercept_

""" for some reason print statements not executing """
print('A value of x=11: %.2f' % model.predict([11])[0])
print ('A value of x=11: %.4f' % model.predict([11])[0])
print ('A value of x=11: %.2f' % model.predict([11]))


model.predict([9])
model.predict([10])
model.predict([11])

""" r-square is model score """
model.score(x,y)

""" output ??? value """
from numpy.linalg import inv
from numpy import dot, transpose
dot(inv(dot(transpose(x), x)), dot(transpose(x), y))

""" output ??? value """
from numpy.linalg import lstsq
lstsq(x,y)[0]

""" --------------------------------- """
""" --------------------------------- """

import os
import csv

os.chdir('c:\\Users\\akuppam\\Documents\\Pythprog\\Rprog\\')
os.getcwd()
with open('mckdata.csv', 'rb') as f:
    reader = csv.reader(f)
#    for row in reader:
#        print row

import pandas as pd
df = pd.read_csv('mckdata.csv')
df.describe()

plt.scatter(df['tons'], df['vol'])
plt.scatter(df['time'], df['dist'])
plt.scatter(df['cap'], df['vol'])
plt.scatter(df['atype'], df['sector'])

# start from page 44 - fitting/eval a model

""" ############################################ """
""" ############################################ """
""" ############################################ """
import pandas as pd
import numpy as np
import os
os.chdir('C:\\users\\akuppam\\documents\\Pythprog\\Rprog')
os.getcwd()
mck = pd.read_csv('mckdata.csv')
mck.describe()
mck.dtypes

from sklearn.linear_model import LinearRegression
mckmodel = LinearRegression()
#mckmodel.fit(mck.vmt, mck.tons)
#mckmodel.fit(mck.ht, mck.wt)
#mckmodel.fit(mck.dist, mck.time)
#mckmodel.fit(mck.tours, mck.trips)

X = mck.time
X
X1 = X.reshape(len(X),1)
X1
y = mck.trips
mckmodel.fit(X.reshape(len(X),1),y)
mckmodel.coef_
mckmodel.intercept_

import matplotlib.pyplot as plt
plt.scatter(X, y,  color='red')

""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

""" I would use scikit-learn's own training_test_split, and generate it from the index """
""" The following works fine """ 
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

train, test = train_test_split(mck, test_size = 0.2)
train.describe()
test.describe()

mck.columns.tolist()

""" Pandas random sample will also work """
""" The following works fine """ 
train = mck.sample(frac=0.8, random_state=200)
test = mck.drop(train.index)

""" If your wish is to have one dataframe in and two dataframes out (not numpy arrays), 
this should do the trick """
""" but need to figure out a way to apply the function 'split_data' """
def split_data(df, train_perc = 0.8):
    df['train'] = np.random.rand(len(df)) < train_perc
    train = df[df.train == 1]
    test = df[df.train == 0]
    split_data ={'train': train, 'test': test}
    return split_data
""" ^^^^^^^^ """
def split_data(mck, train_perc = 0.8):
    mck['train'] = np.random.rand(len(mck)) < train_perc
    train = mck[mck.train == 1]
    test = mck[mck.train == 0]
    split_data ={'train': train, 'test': test}
    return split_data
    
""" ^^^^^^^^^^^^^^^^^^^^^^ """
""" re-label without extra spaces in the headings of the labels """
""" to find the exact name of the column headings """
mck.columns.tolist()
""" to make all column headings to lower case """
mck.columns = mck.columns.str.lower()
""" to remove extra white space in column headings """
mck.columns = [train.ht.strip().replace(' ', '_') for train.ht in mck.columns]


from sklearn.linear_model import LinearRegression
mckmodel = LinearRegression()


""" X1 = X.reshape(len(X),1) """
""" BTW, using 'X', or using ['ht'], or '.ht' - anything should work """
train['ht']
train.ht
train.describe()
X = train['ht']
X
X_train = X.reshape(len(X),1)
X_train = train['ht'].reshape(len(train['ht']),1)
X_train = train.ht.reshape(len(train.ht),1)
Y_train = train.wt
mckmodel.fit(X_train, Y_train)
mckmodel.score(X_train, Y_train)

#Equation coefficient and Intercept 
""" These print functions work fine """
print('Coefficient: \n', mckmodel.coef_) 
print('Intercept:', mckmodel.intercept_)
print('R-squared:', mckmodel.score(X_train, Y_train))
#Predict Output 
X_test = test['ht'].reshape(len(test['ht']),1)
predicted= mckmodel.predict(X_test)
plt.scatter(predicted, test['wt'], color="red")

Y_test = test['wt']
Y_pred = predicted

""" confusion matrix (false positives, false negatives, etc) is good for boolean or dummy vars """
""" this is not for continous variables """
""" good for logistics regression or logit models """
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_pred)

""" ############################################ """
""" #### SGD - STOCHASTIC GRADIENT DESCENT  #### """
""" ############################################ """

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_train_sgd = X_scaler.fit_transform(X_train)
Y_train_sgd = Y_scaler.fit_transform(Y_train)
X_test_sgd = X_scaler.transform(X_test)
Y_test_sgd = Y_scaler.transform(Y_test)

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train_sgd, Y_train_sgd, cv=5)
print('cross validation r-squared scores:', scores)
print('avg. cross-validation r-squared scores:', np.mean(scores))

regressor.fit_transform(X_train_sgd, Y_train_sgd)
print('test set r-squared score:', regressor.score(X_test_sgd, Y_test_sgd))

print('Coefficient - sgd: \n', regressor.coef_)
print('Intercept - sgd:', regressor.intercept_)


""" ############################################ """
""" #### extracting points of interest    ## """
""" ############################################ """

import numpy as nps
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray 
import matplotlib.pyplot as plt 
import skimage.io as io 
from skimage.exposure import equalize_hist 

def show_corners(corners, image):
    fig = plt.figure() 
    plt.gray() 
    plt.imshow(image)
    y_corner, x_corner = zip(*corners)
    plt.plot(x_corner, y_corner, 'or') 
    plt.xlim(0, image.shape[1]) 
    plt.ylim(image.shape[0], 0) 
    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5) 
    plt.show() 

mandrill = io.imread('C:\\users\\akuppam\\documents\\Pythprog\\ron-st.jpg')
mandrill = equalize_hist(rgb2gray(mandrill)) 
corners = corner_peaks(corner_harris(mandrill), min_distance=2) 
show_corners(corners, mandrill)

""" ########################################################## """
""" #### trying some 'seaborn' plots in between chapters    ## """
""" ########################################################## """

mck.columns.tolist()
mck.dtypes

import seaborn as sns
sns.set()

#Finally, to drop by column number instead of by column label, try this to delete, e.g. the 1st, 2nd and 4th columns:
mck1 = mck.drop(mck.columns[[0, 1]], axis=1)  # df.columns is zero-based pd.Index 


mck2 = sns.load_dataset("mck1")
sns.pairplot(mck1, hue="atype")

sns.pairplot(mck, hue="atype")

""" TRY GRAPHICS FOR SSM OUTPUTS
TRY GRAPHICS FOR MEDIUM TRUCK TOURS
TRY FOR ATRI AND SL DATA TOO
ASK FOR JDL'S DETAILED SCTG OUTPUTS

ASGN/QUIZ ON ML - COUR """

""" ########################################################## """
""" #### LOGISTICS REGRESSION - MULTI-CLASS CLASSIFICATION  ## """
""" ########################################################## """
"""
9/2/2016
decision trees
k-means clustering
PCA (factor analysis)

9/3/2016
after this, brush up R and sci-kit ML algorithms
read the AV article on ML algorithms

9/4/2016
get res*** done

9/5/2016
re-visit books (??)
see mckdata.xlsx for other material

9/6/2016
go through inter ques.....

"""
""" ~~~~~~~~~~~~~~~~~~~~~  """
import pandas as pd
import numpy as np
import os

os.chdir('C:\\Users\\akuppam\\Documents\\MAG_LM_TruckTour\\Model\\Medium Trucks Model Output\\')
os.getcwd()

medtour = pd.read_csv('tours_output.csv', low_memory=False)     # low_memory=False will read all vrs regardless of dtype
medtour.describe()
medtour.head(5)[medtour.columns[0:20]]
medtour.dtypes

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = medtour.tourstops
cluster2 = medtour.tourcomp

X = np.hstack((cluster1, cluster2)).T
X
X = np.vstack((x, y)).T

K = range(1,10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


cluster11 = np.random.uniform(0.5, 1.5, (2,10))
cluster22 = np.random.uniform(3.5, 4.5, (2,10))

import numpy as np
from sklearn import cluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

X = medtour

kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(X)
print(kmeans.labels_[::20])
kmeans

k_range = range(1,10)
k_means_var = [KMeans(n_clusters=k).fit(X) for k in k_range]

""" ~~~~~~~~~~~~~ """
""" ~~~~~~~~~~~~~ """
""" ~THE FOLLOWING BLOCK OF CODE DOES CLUSTERING AND PLOTS THE POINTS AND ELBOW CURVE ~~~~~~~~~~~~ """
""" ~~~~~~~~~~~~~ """
""" ~~~~~~~~~~~~~ """
"""
http://www.slideshare.net/SarahGuido/kmeans-clustering-with-scikitlearn
https://github.com/sarguido/k-means-clustering/blob/master/k-means-clustering.ipynb
"""

from sklearn.cross_validation import train_test_split

import os

os.chdir('C:\\Users\\akuppam\\Documents\\MAG_LM_TruckTour\\Model\\Medium Trucks Model Output\\')
os.getcwd()

medtour = pd.read_csv('tours_output.csv', low_memory=False)     # low_memory=False will read all vrs regardless of dtype
medtour.describe()
medtour.dtypes

import pandas as pd

medtour = pd.DataFrame(medtour)
medtour.drop(medtour.columns[[0,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]], axis=1, inplace=True)
medtour.head(5)
medtour.describe()

med_toarray = medtour.values
med_fit, med_fit1 = train_test_split(med_toarray, train_size=.01)
medtour.head()

""" +++++++++  """

#%pylab inline

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

medtour = PCA(n_components=2).fit_transform(med_fit)
k_means = KMeans()
k_means.fit(medtour)

x_min, x_max = medtour[:, 0].min() - 5, medtour[:, 0].max() - 1
y_min, y_max = medtour[:, 1].min(), medtour[:, 1].max() + 5
# original value 0.02; resulted in 'MemoryError'; so changed it to 2.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, 2), np.arange(y_min, y_max, 2))
Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=plt.cm.Paired,
          aspect='auto', origin='lower')

plt.plot(medtour[:, 0], medtour[:, 1], 'k.', markersize=4)
centroids = k_means.cluster_centers_
inert = k_means.inertia_
plt.scatter(centroids[:, 0], centroids[:, 1],
           marker='x', s=169, linewidths=3,
           color='w', zorder=8)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

""" ++++++++ """

import numpy as np

from sklearn.cluster import KMeans
kmeans = KMeans()
kmeans.fit(medtour)

""" DEFAULT OUTPUT
KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=8, n_init=10,
    n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
    verbose=0)
"""
from scipy.spatial.distance import cdist, pdist
# cdist: distance computation between sets of observations
# pdist: pairwise distances between observations in the same set

k_range = range(1,14)
k_means_var = [KMeans(n_clusters=k).fit(medtour) for k in k_range]
centroids = [X.cluster_centers_ for X in k_means_var]

k_euclid = [cdist(medtour, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke, axis=1) for ke in k_euclid]

# within cluster sum of squares
wcss = [sum(d**2) for d in dist]
# total sum of square
tss = sum(pdist(medtour)**2)/medtour.shape[0]
# between cluster sum of squares
bss = tss - wcss

from matplotlib import pyplot as plt

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, bss/tss*100, 'b*-')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('n_clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Variance Explained vs. k')

""" ++++++++ """
# run 8/random, 7/kmeans++ for silhouette
k_means = KMeans(n_clusters=7)
k_means.fit(medtour)

x_min, x_max = medtour[:, 0].min() - 5, medtour[:, 0].max() - 1
y_min, y_max = medtour[:, 1].min() + 1, medtour[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 2), np.arange(y_min, y_max, 2))
Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=plt.cm.Paired,
          aspect='auto', origin='lower')

plt.plot(medtour[:, 0], medtour[:, 1], 'k.', markersize=4)
# Plot the centroids as a white X
centroids = k_means.cluster_centers_
inert = k_means.inertia_
plt.scatter(centroids[:, 0], centroids[:, 1],
           marker='x', s=169, linewidths=3,
           color='w', zorder=8)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

""" ++++++++ """

from sklearn.metrics import silhouette_score

labels = k_means.labels_
silhouette_score(medtour, labels, metric='euclidean')

""" ++++++++ """

""" ############################## """
""" Python Machine Learning.PDF """
""" ############################## """

""" Page 25 - research 'perceptron' on 1/4/2017 (wed) """

import os
import csv
import pandas as pd
import numpy as np

os.chdir("c:\\users\\akuppam\\documents\\Pythprog\\Rprog")

va = pd.read_csv("va.csv")
va.describe()

va_01 = va[(va.VehAvailable<2)]
va_01.describe()

va_0 = va[(va.VehAvailable==0)]
va_0.describe()

va_1 = va[(va.VehAvailable==1)]
va_1.describe()

va_2 = va[(va.VehAvailable==2)]
va_2.describe()

va_12 = np.concatenate((va_1, va_2))
va_12.describe()


""" test va_01 for perceptron """

import matplotlib.pyplot as plt

y = va_01.iloc[0:4917,4].values
y = np.where(y == '0', -1, 1)
X = va_01.iloc[0:4917, [0,2]].values

from sklearn.neural_network import MLPClassifier

import numpy as np
class Perceptron(object):
    """ Perceptron classifier"""

ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X,y)

""" Page 25 - research 'perceptron' on 1/4/2017 (wed) """
""" http://www.analyticbridge.com/profiles/blogs/the-perceptron-algorithm-explained-with-python-code """

""" ########################################## """
""" ML in Py and R - Cheatsheet.xlsx """
""" ########################################## """

import os
import csv
import pandas as pd
import numpy as np

os.chdir("c:\\users\\akuppam\\documents\\Pythprog\\Rprog")
mck = pd.read_csv("mckdata.csv")
va = pd.read_csv("va.csv")

mck.describe()
va.describe()

mck.dtypes
va.dtypes

mck.columns.tolist()

from sklearn.cross_validation import train_test_split

train, test = train_test_split(mck, test_size = 0.2)
train.describe()    # to check if the means are about the same between train and test
test.describe()     # to check if the means are about the same between train and test
train.shape
test.shape

x_train = train.tours
y_train = train.vmt
x_test = test.tours
y_test = test.vmt

""" LINEAR REGRESSION """

from sklearn.linear_model import LinearRegression
mckmodel = LinearRegression()

""" X1 = X.reshape(len(X),1) """
x_train = x_train.reshape(len(x_train),1)

mckmodel.fit(x_train, y_train)
mckmodel.score(x_train, y_train)

""" Equation coefficient and Intercept  """
""" These print functions work fine """
print('Coefficient: \n', mckmodel.coef_) 
print('Intercept:', mckmodel.intercept_)
print('R-squared:', mckmodel.score(x_train, y_train))

""" Predict Output  """
x_test = x_test.reshape(len(x_test),1)
predicted = mckmodel.predict(x_test)

import matplotlib.pyplot as plt

plt.scatter(predicted, y_test, color="red")

""" va """"

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir("c:\\users\\akuppam\\documents\\Pythprog\\Rprog")
os.getcwd()

va = pd.read_csv("va.csv")

from sklearn.cross_validation import train_test_split

trainva, testva = train_test_split(va, test_size = 0.3)
trainva.describe()
trainva.shape
testva.describe()
testva.shape

y_train = trainva.VehAvailable
x_train = trainva.NumOfWorker
x_train = x_train.reshape(len(x_train),1)

y_test = testva.VehAvailable
x_test = testva.NumOfWorker
x_test = x_test.reshape(len(x_test),1)

from sklearn.linear_model import LinearRegression

vamodel = LinearRegression()
vamodel.fit(x_train, y_train)
vamodel.score(x_train, y_train)

print("Coefficient: ", vamodel.coef_)
print("Intercept: ", vamodel.intercept_)
print("R-squared: ", vamodel.score(x_train, y_train))

predictedva = vamodel.predict(x_test)

plt.scatter(predictedva, y_test, color = "red")

# -------------------------

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir("c:\\users\\akuppam\\documents\\Pythprog\\Rprog")
os.getcwd()

va = pd.read_csv("va.csv")

from sklearn.cross_validation import train_test_split

trainva, testva = train_test_split(va, test_size = 0.25)
trainva.describe()
testva.describe()

trainva.shape
testva.shape

""" Two-variables """

y_train = trainva.VehAvailable
x_train1 = trainva.HHPersons
x_train2 = trainva.NumOfWorker
x_train1 = x_train1.reshape(len(x_train1),1)
x_train2 = x_train2.reshape(len(x_train2),1)
x_train = np.concatenate((x_train1, x_train2), axis = 1)  # axis = 1 joins it by columns; axis = 0 appends all rows

y_test = testva.VehAvailable
x_test1 = testva.HHPersons
x_test2 = testva.NumOfWorker
x_test1 = x_test1.reshape(len(x_test1),1)
x_test2 = x_test2.reshape(len(x_test2),1)
x_test = np.concatenate((x_test1, x_test2), axis=1)


from sklearn.linear_model import LinearRegression

vamodel1 = LinearRegression()
vamodel1.fit(x_train, y_train)
vamodel1.score(x_train, y_train)

print("Coefficients: ", vamodel1.coef_)
print("Intercept: ", vamodel1.intercept_)
print("R-squared: ", vamodel1.score(x_train, y_train))

predictedva1 = vamodel1.predict(x_test)

plt.scatter(predictedva1, y_test)

""" LOGISTIC REGRESSION """

from sklearn.linear_model import LogisticRegression

vamodel2 = LogisticRegression()
vamodel2.fit(x_train, y_train)
vamodel2.score(x_train, y_train)

print("Coefficients: ", vamodel2.coef_)
print("Intercept: ", vamodel2.intercept_)
print("R-squared: ", vamodel2.score(x_train, y_train))

predictedva2 = vamodel2.predict(x_test)

plt.scatter(predictedva2, y_test)

""" confusion matrix (false positives, false negatives, etc) is good for boolean or dummy vars """
""" this is not for continous variables """
""" good for logistics regression or logit models """

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


confusion_matrix = confusion_matrix(y_test, predictedva2)
classification_report = classification_report(y_test, predictedva2)

print("Accuracy: ", accuracy_score(y_test, predictedva2))
print("Confusion Matrix: ", confusion_matrix(y_test, predictedva2))
print("Classification Report: ", classification_report(y_test, predictedva2))

""" Classificiation Report 
             precision    recall  f1-score   support

          0       0.00      0.00      0.00        66
          1       0.00      0.00      0.00      1205
          2       0.51      0.98      0.67      3200
          3       0.48      0.19      0.27      1617
          4       0.72      0.03      0.06       579
          5       0.00      0.00      0.00       105
          6       0.00      0.00      0.00        15
          7       0.00      0.00      0.00         5
          8       0.00      0.00      0.00        10

avg / total       0.42      0.51      0.39      6802  """

""" DECISION TREE """

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

vamodel3 = DecisionTreeClassifier(criterion='gini')
vamodel3.fit(x_train, y_train)
vamodel3.score(x_train, y_train)

predictedva3 = vamodel3.predict(x_test)

plt.scatter(predictedva3, y_test)
plt.scatter(predictedva3, predictedva2)

confusion_matrix3 = confusion_matrix(y_test, predictedva3)
classification_report3 = classification_report(y_test, predictedva3)

"""
             precision    recall  f1-score   support

          0       0.00      0.00      0.00        66
          1       0.66      0.20      0.31      1205
          2       0.54      0.95      0.69      3200
          3       0.60      0.15      0.24      1617
          4       0.48      0.28      0.35       579
          5       0.00      0.00      0.00       105
          6       1.00      0.20      0.33        15
          7       0.00      0.00      0.00         5
          8       0.00      0.00      0.00        10

avg / total       0.56      0.55      0.47      6802

"""
""" total data file frequencies
Row Labels	Count of VehAvailable 	
0	296	1.09%
1	4621	16.99%
2	12802	47.06%
3	6558	24.10%
4	2408	8.85%
5	386	1.42%
6	87	0.32%
7	24	0.09%
8	24	0.09%
Grand Total	27206	100.00%

*** Aggregate totals are VERY VERY close ***

 -   	66	0.97%
 1 	1205	17.72%
 2 	3200	47.04%
 3 	1617	23.77%
 4 	579	8.51%
 5 	105	1.54%
 6 	15	0.22%
 7 	5	0.07%
 8 	10	0.15%
	6802	100.00%

"""
""" practice """
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

vamodel1 = LinearRegression()
vamodel1.fit (x, y)

vamodel2 = LogisticRegression()
vamodel2.fit(x,y)

vamodel3 = DecisionTreeClassifier(criterion = 'gini')
vamodel3.fit(x,y)

confusion_matrix3 = confusion_matrix(y_test, predictedva3)
classification_report = classification_report(y_test, predictedva3)
plt.scatter(y_test, predictedva3)

""" ################################################### """
""" ################################################### """
""" ################################################### """
""" ################################################### """
""" ################################################### """
""" ################################################### """
""" ################################################### """
""" practice 1/5/2017 """
""" THE FOLLOWING CODE INCLUDES A COMPREHENSIVE ML ALGORITHMS IN PRACTICE """
""" THESE ARE APPLIED TO VA.CSV FILE """
""" CHECK OUT THE CLASSIFICATION REPORTS EMBEDDED IN THE CODE FOR EACH ML MODELS' RESULT """
""" ################################################### """
""" ################################################### """
""" ################################################### """
""" ################################################### """
""" ################################################### """
""" ################################################### """
""" ################################################### """
""" ################################################### """

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir("c:\\users\\akuppam\\documents\\Pythprog\\Rprog")
os.getcwd()

vav = pd.read_csv("va.csv")
vav.describe()
vav.shape
vav.dtypes

from sklearn.cross_validation import train_test_split

train_vav, test_vav = train_test_split(vav, test_size = 0.25)
train_vav.describe()
test_vav.describe()

y_train_vav = train_vav.VehAvailable
x_train_vav1 = train_vav.HHPersons
x_train_vav1 = x_train_vav1.reshape(len(x_train_vav1),1)
x_train_vav2 = train_vav.HHIncome
x_train_vav2 = x_train_vav2.reshape(len(x_train_vav2),1)
x_train_vav3 = train_vav.NumOfWorker
x_train_vav3 = x_train_vav3.reshape(len(x_train_vav3),1)
x_train_vav4 = train_vav.age
x_train_vav4 = x_train_vav4.reshape(len(x_train_vav4),1)
x_train_vav = np.concatenate((x_train_vav1, x_train_vav2, x_train_vav3, x_train_vav4), axis = 1)

y_test_vav = test_vav.VehAvailable
x_test_vav1 = test_vav.HHPersons
x_test_vav1 = x_test_vav1.reshape(len(x_test_vav1),1)
x_test_vav2 = test_vav.HHIncome
x_test_vav2 = x_test_vav2.reshape(len(x_test_vav2),1)
x_test_vav3 = test_vav.NumOfWorker
x_test_vav3 = x_test_vav3.reshape(len(x_test_vav3),1)
x_test_vav4 = test_vav.age
x_test_vav4 = x_test_vav4.reshape(len(x_test_vav4),1)
x_test_vav = np.concatenate((x_test_vav1, x_test_vav2, x_test_vav3, x_test_vav4), axis = 1)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

vav_model_lm = LinearRegression()
vav_model_log = LogisticRegression()
vav_model_tree = DecisionTreeClassifier(criterion = 'gini')

vav_model_lm.fit(x_train_vav, y_train_vav)
vav_model_log.fit(x_train_vav, y_train_vav)
vav_model_tree.fit(x_train_vav, y_train_vav)

print("LM coeffs: ", vav_model_lm.coef_)
print("LOG coeffs: ", vav_model_log.coef_)
print("LM intercept: ", vav_model_lm.intercept_)
print("LOG intercept: ", vav_model_log.intercept_)
print("LM accuracy: ", vav_model_lm.score(x_train_vav, y_train_vav))
print("LOG accuracy: ", vav_model_log.score(x_train_vav, y_train_vav))
print("DecTree accuracy: ", vav_model_tree.score(x_train_vav, y_train_vav))

predict_lm = vav_model_lm.predict(x_test_vav)
predict_log = vav_model_log.predict(x_test_vav)
predict_tree = vav_model_tree.predict(x_test_vav)

plt.scatter(predict_lm, y_test_vav, color="red")
plt.savefig("c:/users/akuppam/documents/Pythprog/Rprog/lm_scatter.png")
plt.close()

plt.scatter(predict_log, y_test_vav, color="blue")
plt.savefig("c:/users/akuppam/documents/Pythprog/Rprog/log_scatter.png")
plt.close()

plt.scatter(predict_log, y_test_vav, color="green")
plt.savefig("c:/users/akuppam/documents/Pythprog/Rprog/tree_scatter.png")
plt.close()

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

""" *** accuracy score, confusion matrix and classificiation report only for log and discrete choice models *** """
accuracy_score_log = accuracy_score(y_test_vav, predict_log)
accuracy_score_tree = accuracy_score(y_test_vav, predict_tree)

confusion_matrix_log = confusion_matrix(y_test_vav, predict_log)
confusion_matrix_tree = confusion_matrix(y_test_vav, predict_tree)

classification_report_log = classification_report(y_test_vav, predict_log)
"""
             precision    recall  f1-score   support

          0       0.00      0.00      0.00        87
          1       0.56      0.14      0.22      1141
          2       0.53      0.94      0.67      3211
          3       0.46      0.21      0.28      1619
          4       0.50      0.06      0.11       609
          5       0.00      0.00      0.00        94
          6       0.00      0.00      0.00        27
          7       0.00      0.00      0.00         6
          8       0.00      0.00      0.00         8

avg / total       0.50      0.52      0.43      6802
"""
classification_report_tree = classification_report(y_test_vav, predict_tree)
"""
             precision    recall  f1-score   support

          0       0.66      0.45      0.53        87
          1       0.71      0.74      0.73      1141
          2       0.78      0.81      0.80      3211
          3       0.71      0.69      0.70      1619
          4       0.75      0.64      0.69       609
          5       0.71      0.69      0.70        94
          6       0.78      0.67      0.72        27
          7       0.80      0.67      0.73         6
          8       1.00      0.50      0.67         8

avg / total       0.75      0.75      0.75      6802
"""

""" SUPPORT VECTOR MACHINES """
""" LOOK AT PAGES 69 AND 73 (Python Machine Learning.PDF) FOR BETTER GRAPHICAL EXPLANATIONS FOR SVM """

from sklearn.svm import SVC
vav_model_svm = SVC(kernel='linear', C=1.0, random_state=0)
vav_model_svm.fit(x_train_vav, y_train_vav)
vav_model_svm.score(x_train_vav, y_train_vav)

predict_svm = vav_model_svm.predict(x_test_vav)
plt.scatter(predict_svm, y_test_vav, color="red")
plt.savefig('c:/users/akuppam/documents/Pythprog/Rprog/SVM_scatter.png')
plt.close()

classification_report_svm = classification_report(y_test_vav, predict_svm)
"""
             precision    recall  f1-score   support

          0       0.00      0.00      0.00        72
          1       0.00      0.00      0.00      1157
          2       0.48      1.00      0.65      3211
          3       0.46      0.02      0.04      1625
          4       0.00      0.00      0.00       610
          5       0.00      0.00      0.00        98
          6       0.00      0.00      0.00        16
          7       0.00      0.00      0.00         7
          8       0.00      0.00      0.00         6

avg / total       0.33      0.48      0.31      6802
"""

''' recommended to standardize vars to a uniqfied scale, but havne't done that in any of the above ML models '''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train_vav)
x_train_vav_std = sc.transform(x_train_vav)
x_test_vav_std = sc.transform(x_test_vav)

''' ------------------------------- '''
''' None of the following, till the next ML algorithm is working '''

from matplotlib.colors import ListedColormap 
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map     
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface     
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples     
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

''' ------------------------------- '''

import matplotlib.pyplot as plt
import numpy as np
x_combined_std = np.vstack((x_train_vav_std, x_test_vav_std))
y_combined = np.hstack((y_train_vav, y_test_vav))

''' ----------------------- '''
import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

''' ------------------------ '''

ppn = Perceptron(eta = 0.1, n_iter = 10)
X = x_combined_std
y = y_combined
ppn.fit(X,y)

plot_decision_regions(X = x_combined_std,
                      y = y_combined,
                      classifier = ppn)
                      #test_idx = range(105,150))
plt.xlabel('sed vars (std)')
plt.ylabel('va (std)')
plt.legend(loc='upper left')
plt.show()

""" -----------------------------"""
""" NAIVE BAYES """
""" -----------------------------"""

from sklearn.naive_bayes import GaussianNB
vav_model_nb = GaussianNB()
vav_model_nb.fit(x_train_vav, y_train_vav)
vav_model_nb.score(x_train_vav, y_train_vav)
predict_nb = vav_model_nb.predict(x_test_vav)
plt.scatter(predict_nb, y_test_vav)
plt.savefig('c:/users/akuppam/documents/Pythprog/Rprog/NB_scatter.png')
plt.close()

classification_report_nb = classification_report(y_test_vav, predict_nb)
"""
             precision    recall  f1-score   support

          0       0.00      0.00      0.00        72
          1       0.52      0.29      0.37      1157
          2       0.55      0.78      0.64      3211
          3       0.43      0.20      0.27      1625
          4       0.46      0.18      0.26       610
          5       0.00      0.00      0.00        98
          6       0.00      0.00      0.00        16
          7       0.01      1.00      0.02         7
          8       0.00      0.00      0.00         6

avg / total       0.49      0.48      0.45      6802

"""
""" -------------------------- """
""" kNN (k-NEAREST NEIGHBORS) """
""" ------------------------- """

from sklearn.neighbors import KNeighborsClassifier
vav_model_knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
vav_model_knn.fit(x_train_vav, y_train_vav)
vav_model_knn.score(x_train_vav, y_train_vav)
predict_knn = vav_model_knn.predict(x_test_vav)
plt.scatter(predict_knn, y_test_vav)
plt.savefig('c:/users/akuppam/documents/Pythprog/Rprog/KNN_scatter.png')
classification_report_knn = classification_report(predict_knn, y_test_vav)

"""
             precision    recall  f1-score   support

          0       0.26      0.36      0.30        53
          1       0.64      0.64      0.64      1157
          2       0.80      0.72      0.76      3587
          3       0.62      0.68      0.65      1479
          4       0.50      0.70      0.58       443
          5       0.47      0.62      0.53        74
          6       0.12      0.67      0.21         3
          7       0.29      0.67      0.40         3
          8       0.50      1.00      0.67         3

avg / total       0.71      0.69      0.70      6802
"""

""" -------------------------- """
""" k-MEANS (cluster analysis) """
""" ------------------------- """

from sklearn.cluster import KMeans
vav_model_kmeans = KMeans(n_clusters=8, random_state=0)
vav_model_kmeans.fit(x_train_vav, y_train_vav)
vav_model_kmeans.score(x_train_vav, y_train_vav)
predict_kmeans = vav_model_kmeans.predict(x_test_vav)
plt.scatter(predict_kmeans, y_test_vav)
plt.savefig('c:/users/akuppam/documents/Pythprog/Rprog/KMEANS_scatter.png')
plt.close()
classification_report_kmeans = classification_report(predict_kmeans, y_test_vav)

"""
             precision    recall  f1-score   support

          0       0.12      0.01      0.02       714
          1       0.14      0.15      0.15      1070
          2       0.12      0.41      0.19       960
          3       0.05      0.16      0.08       514
          4       0.17      0.13      0.15       769
          5       0.08      0.01      0.01       978
          6       0.25      0.00      0.01       862
          7       0.00      0.00      0.00       935
          8       0.00      0.00      0.00         0

avg / total       0.12      0.11      0.08      6802
"""

# --------------------------------------------------------------
""" TO DO - RANDOM FOREST, DATA/DIMENSIONALITY REDUCTION, GRADIENT BOOSTING/ADABOOST """
""" 1-15-2017 """
# --------------------------------------------------------------

import os
import csv
import pandas as pd
import numpy as np

os.chdir('c:\\users\\akuppam\\documents\\Pythprog\\Rprog')
os.getcwd()

wine = pd.read_csv('wine.csv')
wine.describe()
wine.shape
wine.dtypes

from sklearn.cross_validation import train_test_split

train, test = train_test_split(wine, test_size = 0.25)
train.describe()
train.shape
test.describe()
test.shape

y_train = train.Class

x1_train = train.Alcohol
x1_train = x1_train.reshape(len(x1_train),1)
x2_train = train.Ash
x2_train = x2_train.reshape(len(x2_train),1)
x3_train = train.Proline
x3_train = x3_train.reshape(len(x3_train),1)

x_train = np.concatenate((x1_train, x2_train, x3_train), axis = 1)

y_test = test.Class

x1_test = test.Alcohol
x1_test = x1_test.reshape(len(x1_test),1)
x2_test = test.Ash
x2_test = x2_test.reshape(len(x2_test),1)
x3_test = test.Proline
x3_test = x3_test.reshape(len(x3_test),1)
x_test = np.concatenate((x1_test, x2_test, x3_test), axis = 1)

from sklearn.linear_model import LinearRegression

wineclass = LinearRegression()
wineclass.fit(x_train, y_train)
wineclass.score(x_train, y_train)

predictclass = wineclass.predict(x_test)

plt.scatter(predictclass, y_test, color = 'red')

from sklearn.linear_model import LogisticRegression

wineclasslog = LogisticRegression()
wineclasslog.fit(x_train, y_train)
wineclass.score(x_train, y_train)

predictclasslog = wineclasslog.predict(x_test)

from sklearn.tree import DecisionTreeClassifier
wineclasstree = DecisionTreeClassifier(criterion = 'gini')
wineclasstree.fit(x_train, y_train)
wineclasstree.score(x_train, y_train)

predictclasstree = wineclasstree.predict(x_test)

print('lin coeff', wineclass.coef_)
print('log coeff', wineclasslog.coef_)

plt.scatter(predictclasslog, y_test, color = 'blue')
plt.savefig('c:/users/akuppam/documents/Pythprog/winelog.png')
plt.close()
plt.scatter(predictclasstree, y_test, color = 'green')
plt.close()

from sklearn.metrics import confusion_matrix
confmatrix_tree = confusion_matrix(y_test, predictclasstree)

from sklearn.metrics import classification_report
classreport_tree = classification_report(y_test, predictclasstree)

"""
             precision    recall  f1-score   support

          1       0.92      0.79      0.85        14
          2       0.80      0.80      0.80        15
          3       0.72      0.81      0.76        16

avg / total       0.81      0.80      0.80        45

"""

# standardize all the variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
y_train_std = sc.fit_transform(y_train)
y_test_std = sc.fit_transform(y_test)

x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

# compute cavariance matrix - 3 vars = 3x3 cov matrix
# compute eigen values (3 values for 3 vars)
# compute eigen vectors (3x3 matrix)
import numpy as np
cov_mat = np.cov(x_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenValues \n%s' % eigen_vals)
"""
EigenValues 
[ 1.81156535  0.38005101  0.83111091]
"""
tot = sum(eigen_vals)
tot
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
var_exp


import numpy as np
cov_mat = np.cov(x_train_std.T) # Transpose of standardized vars that need to be reduced
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('Eigen Value:', eigen_vals)
tot = sum(eigen_vals)
tot
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
var_exp
cum_var_exp = np.cumsum(var_exp)
cum_var_exp

import matplotlib.pyplot as plt
plt.bar(range(1,3), var_exp, alpha = 0.5, align = 'center', label = 'variance explained')
plt.step(range(1,3), cum_var_exp, where = 'mid', label = 'cumulative variance explained')
plt.ylabel('exp var ratio')
plt.xlabel('principal components')
#plt.legend(loc='best')
plt.savefig('c:/users/akuppam/documents/Pythprog/winePCA.png')
plt.close()


# -------------------------------------------
from sklearn import decomposition
fa = decomposition.FactorAnalysis()

pca = decomposition.PCA(n_components = 3)
pca


x_train_std_reduced = pca.fit_transform(x_train_std)

from sklearn.decomposition import FactorAnalysis
fa1 = FactorAnalysis()


# ----------- practice 1-17-2017 --------------------


import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('c:\\users\\akuppam\\documents\\Pythprog\\Rprog')
os.getcwd()

mck = pd.read_csv('mckdata.csv')
mck.describe()
mck.shape
mck.dtypes

from sklearn.cross_validation import train_test_split

train, test = train_test_split(mck, test_size = 0.2)
train.describe()
test.describe()

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

y_train = mck.atype
x1_train = mck.vmt
x1_train = x1_train.reshape(len(x1_train),1)
x2_train = mck.trips
x2_train = x2_train.reshape(len(x2_train),1)
x_train = np.concatenate((x1_train, x2_train), axis = 1)

y_test = mck.atype
x1_test = mck.vmt
x1_test = x1_test.reshape(len(x1_test),1)
x2_test = mck.trips
x2_test = x2_test.reshape(len(x2_test),1)
x_test = np.concatenate((x1_test, x2_test), axis = 1)

mck_lm = LinearRegression()
mck_log = LogisticRegression()
mck_tree = DecisionTreeClassifier(criterion = "gini")

mck_lm.fit(x_train, y_train)
mck_lm.score(x_train, y_train)

mck_log.fit(x_train, y_train)
mck_log.score(x_train, y_train)

mck_tree.fit(x_train, y_train)
mck_tree.score(x_train, y_train)

print('lm coeff:', mck_lm.coef_)
print('lm intercept:', mck_lm.intercept_)

predict_lm = mck_lm.predict(x_test)
predict_log = mck_log.predict(x_test)
predict_tree = mck_tree.predict(x_test)

plt.scatter(predict_lm, y_test)
plt.scatter(predict_log, y_test)
plt.scatter(predict_tree, y_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class_report_log = classification_report(y_test, predict_log)
class_report_tree = classification_report(y_test, predict_tree)

conf_matrix_tree = confusion_matrix(y_test, predict_tree)
print(conf_matrix_tree)
plt.matshow(conf_matrix_tree)
plt.title('conf matrix tree')
plt.colorbar()
plt.ylabel('true label')
plt.xlabel('predicted label')
plt.show()

# --------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, p=2, metric = 'minkowski')
knn.fit(x_train, y_train)
knn.score(x_train, y_train)

''' ------------------------------- '''
''' None of the following, till the next ML algorithm is working '''

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx = None, resolution=0.02):

    # setup marker generator and color map     
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface     
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha = 1.0, linewidth = 1, marker='o',
                    s=55, label='test set')


''' ------------------------------- '''
plot_decision_regions(x_train, 
                      y_train, 
                      classifier = knn, 
                      test_idx = range(105,150))

plot_decision_regions(x_train, 
                      y_train, 
                      classifier = knn)

plt.xlabel('sed vars (std)')
plt.ylabel('va (std)')
plt.legend(loc='upper left')
plt.show()

# -------------------

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

plot_decision_regions(X=x_train, y=y_train, classifier = knn)
plt.xlabel('sed vars (std)')
plt.ylabel('va (std)')
plt.legend(loc='upper left')
plt.show()

"""# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# Review of scikit learn - 4/4/2017 
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------"""
""" 
LINEAR REGRESSION
LOGISTIC REGRESSION
DECISION TREE
using El Paso's va.csv file
"""

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('c:\\users\\akuppam\\documents\\pythprog')
mck = pd.read_csv("mckdata.csv")
mck.describe()
va = pd.read_csv("va.csv")
va.describe()

from sklearn.cross_validation import train_test_split
train, test = train_test_split(va, test_size=0.25)
train.describe()
test.describe()

y_train = train.VehAvailable
x1_train = train.HHPersons
x1_train = x1_train.reshape(len(x1_train),1)
x2_train = train.NumOfWorker
x2_train = x2_train.reshape(len(x2_train),1)
x3_train = train.age
x3_train = x3_train.reshape(len(x3_train),1)
x_train = np.concatenate((x1_train, x2_train, x3_train), axis=1)

y_test = test.VehAvailable
x1_test = test.HHPersons
x1_test = x1_test.reshape(len(x1_test),1)
x2_test = test.NumOfWorker
x2_test = x2_test.reshape(len(x2_test),1)
x3_test = test.age
x3_test = x3_test.reshape(len(x3_test),1)
x_test = np.concatenate((x1_test, x2_test, x3_test), axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

va_lm = LinearRegression()
va_log = LogisticRegression()
va_tree = DecisionTreeClassifier(criterion="gini")

va_lm.fit(x_train, y_train)
va_lm.score(x_train, y_train)
va_lm.coef_
va_lm.intercept_
# 1 intercept and 3 coeff
# score = 0.21

va_log.fit(x_train, y_train)
va_log.score(x_train, y_train)
va_log.coef_
va_log.intercept_
print(va_log.coef_)
# several set of coeff and intercepts
# 3 x-vars - so 3^2 intercepts and 3^2 x 3 coeff
# score = 0.51

va_tree.fit(x_train, y_train)
va_tree.score(x_train, y_train)
# no coeff or intercept for decision tree classifiers
# score = 0.66

pred_lm = va_lm.predict(x_test)
pred_log = va_log.predict(x_test)
pred_tree = va_tree.predict(x_test)

plt.scatter(pred_lm, y_test)
plt.scatter(pred_log, y_test)
plt.scatter(pred_tree, y_test)

# for logit and tree models only (discrete, not for linear models)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class_report_log = classification_report(y_test, pred_log)
class_report_tree = classification_report(y_test, pred_tree)
class1 = classification_report(y_test, y_test)

conf_log = confusion_matrix(y_test, pred_log)
conf_tree = confusion_matrix(y_test, pred_tree)

plt.matshow(conf_log)
plt.title('logistic regression model')
plt.colorbar()
plt.show()

plt.matshow(conf_tree)
plt.title('decision tree')
plt.colorbar()
plt.show()

""" 
cluster analysis
dimensionality reduction
naive bayes
kNN

hidden markov model
svm
ensemble methods
stochastic gradient descent
discriminant analysis
neural nets

using ??? data ?? file
"""












