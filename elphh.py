# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 19:29:35 2016

@author: AKUPPAM
"""

""" el paso hh survey """

import os
import csv
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pylab
from pylab import xticks

os.chdir("c:\\Users\\akuppam\\Documents\\El Paso Model\\Task 3\\HHSurvey\\processed survey data")
os.getcwd

hh = pd.read_csv("geocoded_HH_Survey_record4.csv")
hh.describe()
hh.dtypes
hh.size
hh.shape
hh.ndim

hhar = np.array(hh)
hhar
hhar.size

"""
##############################################################################
######  INTRODUCTION TO PYTHON FOR ECONOMETRICS, STATISTICS AND DATA ANALYSIS
##############################################################################
"""
hhdf = pd.DataFrame(hh)
hhdf
hhdf.head(3)
hhdf.describe()


#pd.pivot_table(hhdf, values=pd.pivot_table(hhdf, values='Adj_weight_prox_gen_age', index=['PURP_FLAG','SEX'], aggfunc='sum'), index=['PURP_FLAG','SEX'])
pd.pivot_table(hhdf, values='Adj_weight_prox_gen_age', index=['PURP_FLAG','SEX'], aggfunc='mean')
purp_sex = pd.pivot_table(hhdf, values='Adj_weight_prox_gen_age', index=['PURP_FLAG','SEX'], aggfunc='sum')
purp_sex

list(hhdf.columns.values)

pd.pivot_table(hhdf, values='Adj_weight_prox_gen_age', index=['Year', 'Month', 'Day'], aggfunc=sum)
pd.pivot_table(hhdf, values='Adj_weight_prox_gen_age', index=['Year', 'Month'], aggfunc='sum')


pd.pivot_table(hhdf, values=['Adj_weight_prox_gen_age'], index=['inc1','inc2','inc3','inc4','inc5'], aggfunc=sum)
hhdf['HHPersons'].describe()
hhdf['StartHr'].describe()
hhdf['StartMin'].describe()
hhdf['EndHr'].describe()
hhdf['EndMin'].describe()

hhdf['StartTime'] = hhdf['StartHr'] + hhdf['StartMin'] / 60
hhdf['StartTime'].describe()

hhdf['EndTime'] = hhdf['EndHr'] + hhdf['EndMin'] / 60
hhdf['EndTime'].describe()

hhdf['TravelDur'] = (hhdf['EndTime'] - hhdf['StartTime']) * 60
hhdf['TravelDur'].describe()

""" drop NAn and negative numbers """
hhdf['TravelDur'] = hhdf['TravelDur'].dropna()
hhdf = hhdf[hhdf.TravelDur >=0]
hhdf['TravelDur'].describe()

#n, bins, patches = plt.hist(hmag, 30, facecolor='gray', align='mid')
plt.hist(hhdf['TravelDur'], bins=200)
xticks(range(0,50))
pylab.rc("axes", linewidth=8.0)
pylab.rc("lines", markeredgewidth=2.0) 
plt.xlabel('Travel Duration', fontsize=14)
plt.ylabel('No of Trips', fontsize=14)

pylab.xticks(fontsize=15)
pylab.yticks(fontsize=15)
plt.grid(True)

plt.savefig('hmag_histogram.eps', facecolor='w', edgecolor='w', format='eps')
plt.show()

plt.hist(hhdf['EndTime'].dropna(), bins=20)
plt.hist(hhdf['StartTime'].dropna(), bins=20)

plt.boxplot(hhdf.TravelDur)
plt.violinplot(hhdf.TravelDur)

hhdf = hhdf[hhdf.TravelDur <=100]

plt.boxplot(hhdf.TravelDur)
plt.violinplot(hhdf.TravelDur)

""" START FROM PAGE 203 OF THE PYECON BOOK 
FINISH PYECON ON OCT 6 (AM)
FINISH PYML ON OCT 6 (PM/NT)
"""

from pandas.tools.plotting import scatter_plot
from pandas.tools.plotting import scatter_matrix

fig1 = scatter_plot(hhdf, 'HHPersons', 'HHIncome')
fig1.savefig('hhpers vs hhinc.jpg')

fig2 = hhdf[['TravelDur', 'HHIncome']].plot(subplots=True)
fig22 = fig2[0].get_figure()
fig22.savefig('dur-inc22.jpg')

fig3 = scatter_matrix(hhdf[['TravelDur', 'HHIncome']], diagonal='kde')
fig33 = fig3[0,0].get_figure()
fig33.savefig('dur-inc.jpg')

""" Chapter 18 - Custom function and modules """

def sumcubes(x,y, *args):
    return x**3 + y**3
    
x = 12
y = 1

z = sumcubes(x,y)
print ('sumcubes:', z)
z = sumcubes(9,10)
print ('sumcubes:', z)

x = np.random.randn(10)
x
y = np.random.randn(10)
y

z = sumcubes(x,y)
print ('sumcubes:', z)

""" Chapter 19 - Probability and statistics functions """

import numpy as np
np.random.rand(3,4,5)           # 3 sets of arrays, with 4 rows of 5 numbers in each row and in each array
np.random.random_sample((3,4,5))

x = np.random.randint(0,10,(100))
x

x = np.random.binomial(10,0.5)      # binomial(n,p) where n is number of draws, p is between 0 and 1
x

import numpy as np
x = np.random.binomial(10,0.5,(10,10))  # (10,10) is an array 10x10 with multiple (100) binomial draws
x
import matplotlib.pyplot as plt
plt.plot(x)
plt.ylabel('some numbers')
plt.show()

import numpy as np
x = np.random.binomial(10,0.5,(100,1))  # (10,10) is an array 10x10 with multiple (100) binomial draws
x
import matplotlib.pyplot as plt
plt.plot(x)
plt.ylabel('binomial')
plt.show()

import scipy, scipy.stats
#x = scipy.linspace(0,10,11)
pmf = scipy.stats.binom.pmf(x,10,0.1)
import pylab
pylab.plot(x,pmf)

# binomial (probaility mass function)
x = scipy.linspace(0,20,21)
x
pmf = scipy.stats.binom.pmf(x,20,0.25)
pmf
import pylab
plt.ylabel('binomial')
fig = pylab.plot(x,pmf)
fig.savefig('binomial.jpg')

# normal (probability density function)
x = scipy.linspace(50,150,101)
x
pmf = scipy.stats.norm.pdf(x,100,16)  # (x, mean, sd)
pmf
import pylab
plt.ylabel('normal')
fig = pylab.plot(x,pmf)
fig.savefig('normal')

""" ##########################################################################
######  Mastering ML with scikit-learn.PDF  ##################################
########################################################################## """

from sklearn.linear_model import LinearRegression

linearmodel = LinearRegression()

hh.HHPersons
hpers = hh.HHPersons.reshape(len(hh.HHPersons),1)
hpers

hinc = hh.HHIncome.reshape(len(hh.HHIncome),1)

linearmodel.fit(hinc, hh.Adj_weight_prox_gen_age)
linearmodel.coef_
linearmodel.intercept_

import matplotlib.pyplot as plt
plt.scatter(hinc, hh.Adj_weight_prox_gen_age)

linearmodel.predict([9])
linearmodel.predict([10])
linearmodel.predict([11])

""" r-square is model score """
linearmodel.score(hinc, hh.Adj_weight_prox_gen_age)

veh = hh.VehAvailable.reshape(len(hh.VehAvailable),1)

linearmodel.fit(veh, hh.Adj_weight_prox_gen_age)
linearmodel.coef_
linearmodel.intercept_
linearmodel.score(veh, hh.Adj_weight_prox_gen_age)


linearmodel.fit(hinc, hh.trips, sample_weight=hh.Adj_weight_prox_gen_age)
linearmodel.coef_
linearmodel.intercept_
linearmodel.score(hinc, hh.trips, sample_weight=hh.Adj_weight_prox_gen_age)

# -----
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()

hpers = hh.HHPersons.reshape(len(hh.HHPersons),1)

model1.fit(hpers, hh.NumOfWorker)
model1.coef_
model1.intercept_
model1.score(hpers, hh.NumOfWorker)

import matplotlib.pyplot as plt
plt.scatter(hpers, hh.NumOfWorker)

linearmodel.predict([10])

""" ---------------------------
Splitting data into train and test
Specifiying single and multiple linear regression models
Output coeff, intercept, R-square
Predictions
Scatter plot, boxplot, violin plot of diff, diff percentages
Replacing inf with NaN and dropping NaN
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

train, test = train_test_split(hh, test_size = 0.2)
train.describe()
test.describe()

trainveh = train.VehAvailable.reshape(len(train.VehAvailable),1)
trainpers = train.HHPersons.reshape(len(train.HHPersons),1)

from sklearn.linear_model import LinearRegression

""" single variable linear regression """
model2 = LinearRegression()
model2.fit(trainveh, train.NumOfWorker)
model2.coef_
model2.intercept_
model2.score(trainveh, train.NumOfWorker)

testveh = test.VehAvailable.reshape(len(test.VehAvailable),1)
predicted = model2.predict(testveh)
plt.scatter(predicted, test.NumOfWorker, color="red")

diff = predicted-test.NumOfWorker
diffpct = (predicted/test.NumOfWorker - 1) * 100
diffpct.describe()

""" multi-variate or multiple linear regression """
""" see page 44 for example """

train.columns
train.columns[31:35]        # print columsn 31, 32, 33, 34 (but not 35)
train_X = train[list(train.columns)[31:34]]
train_Y = train['VehAvailable']

from sklearn.linear_model import LinearRegression
model3 = LinearRegression()
model3.fit(train_X, train_Y)
model3.coef_
model3.intercept_
model3.score(train_X, train_Y)

test_X = test[list(test.columns)[31:34]]
predicted3 = model3.predict(test_X)
plt.scatter(predicted3, test.VehAvailable)

diff3 = predicted3 - test.VehAvailable
diff3pct = (predicted3 / test.VehAvailable - 1) * 100
diff3pct = diff3pct.replace(np.inf, np.nan)
diff3pct.dropna().describe()

plt.hist(diff3, bins=10)
plt.savefig('hist of diff3.jpg')

plt.hist(diff3pct.dropna(), bins=10)
plt.savefig('hist of diff3pct.jpg')

plt.violinplot(diff3pct.dropna())
plt.savefig('violin plot of diff3pct.jpg')

plt.boxplot(diff3pct.dropna())
plt.savefig('boxplot of diff3pct.jpg')

""" ------------------------------- """

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_train_sgd = X_scaler.fit_transform(trainveh)
Y_train_sgd = Y_scaler.fit_transform(train.NumOfWorker)
X_test_sgd = X_scaler.transform(testveh)
Y_test_sgd = Y_scaler.transform(test.NumOfWorker)

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train_sgd, Y_train_sgd, cv=5)
print('cross validation r-squared scores:', scores)
print('avg. cross-validation r-squared scores:', np.mean(scores))

regressor.fit_transform(X_train_sgd, Y_train_sgd)
print('test set r-squared score:', regressor.score(X_test_sgd, Y_test_sgd))

print('Coefficient - sgd: \n', regressor.coef_)
print('Intercept - sgd:', regressor.intercept_)


""" get thro' the ML scikit book - AM
look into hck...."""

""" ########################################
Logistic Regression - with multiple variables
Similar to multi-variate linear regression
#########################################"""

from sklearn.linear_model.logistic import LogisticRegression

classifier = LogisticRegression()
classifier.fit(train_X, train_Y)
classifier.coef_
predictions = classifier.predict(test_X)

plt.hist(predictions, bins=10)
plt.scatter(predictions, test.VehAvailable)
diff4 = predictions - test.VehAvailable
plt.hist(diff4, bins=10)
plt.savefig('diff4 - logistic reg of VA.jpg')
plt.violinplot(diff4)
plt.savefig('diff4 - logistic reg of VA.jpg')
plt.violinplot(diff3)
plt.savefig('diff3 - linear reg of VA.jpg')

plt.scatter(predicted3, predictions)
plt.savefig('VA - lin vs. log regression.jpg')

difflinlog = predicted3 - predictions
plt.violinplot(difflinlog)
plt.savefig('diff in lin vs log.jpg')
plt.hist(difflinlog, bins=20)
plt.savefig('hist of diff in lin vs log.jpg')
plt.boxplot(difflinlog)
plt.savefig('boxplot of diff in lin vs log.jpg')

""" Confusion Matrix """
""" NOT for linear regression
Only for Logistic regression """

from sklearn.metrics import confusion_matrix
confmatrixlog = confusion_matrix(test.VehAvailable, predictions)
print(confmatrixlog)
plt.matshow(confmatrixlog)
plt.title('logistic model based confusion matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
#plt.show()
plt.savefig('log confusion matrix.jpg')

""" Accuracy Score
Cross_val_score """

from sklearn.metrics import accuracy_score
accscore = accuracy_score(test.VehAvailable, predictions)

from sklearn.cross_validation import cross_val_score
cvscore = cross_val_score(classifier, train_X, train_Y, cv=5)

""" Precision and Recall """

from sklearn.cross_validation import cross_val_score
precisions = cross_val_score(classifier, train_X, train_Y, cv=5, scoring='precision')

from sklearn.cross_validation import cross_val_score
recalls = cross_val_score(classifier, train_X, train_Y, cv=5, scoring='recall')

from sklearn.cross_validation import cross_val_score
F1s = cross_val_score(classifier, train_X, train_Y, cv=5, scoring='f1')
np.mean(F1s)
print ('F1s', np.mean(F1s))

""" ROC and AUC NOT working """
""" Receiver operating characteristic and Area under ROC """

from sklearn.metrics import roc_curve, auc
predictions1 = classifier.predict_proba(test_X)
fpr, tpr, thresholds = roc_curve(test.VehAvailable, predictions)

false_positive_rate, recall, thresholds = roc_curve(test.VehAvailable, predictions1[:,1])
roc_curve(test.VehAvailable, predictions1[:,1])
roc_auc = auc(false_positive_rate, recall)
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)


""" ############################################# """
""" ML Mastery w/ Python Mini-Course.PDF """
""" ############################################# """

import sys
print("Python: {}". format(sys.version))
import scipy
print("scipy: {}". format(scipy.__version__))
import numpy
print("numpy: {}". format(numpy.__version__))
import matplotlib
print("matplotlib: {}". format(matplotlib.__version__))
import pandas
print("pandas: {}". format(pandas.__version__))
import sklearn
print("sklearn: {}". format(sklearn.__version__))


from sklearn.preprocessing import StandardScaler
import pandas
import numpy
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array=dataframe.values
array1 = np.array(dataframe)    # array and array1 are the same arrays
X = array[:,0:8]        # all rows & ) 0 to 8th columns
X
Y = array[:,8]          # all rows & 8th column only
Y

""" standardizing data (mean = 0, stdev = 1) """

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
print(rescaledX[0:3, :])

""" logistics reg & k-fold validation """

from sklearn.linear_model import LogisticRegression
import pandas
from sklearn import cross_validation

num_folds = 10
num_folds
num_instances = len(X)      # no of rows
num_instances
seed = 7
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
model = LogisticRegression()
model.fit(X, Y)
model.coef_
model.intercept_
model.score(X, Y)
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
results
results.mean()
results.std()
print("Accuracy: %0.6f (+/- %0.2f)" % (results.mean(), results.std() * 2))


kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
model = LogisticRegression()
scoring = 'log_loss'
results_scoring = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
results_scoring
print("Accuracy: %0.2f (+/- %0.2f)" % (results_scoring.mean(), results_scoring.std() * 2))

""" kNN regression """

from sklearn import cross_validation
from sklearn.neighbors import KNeighborsRegressor

kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
model = KNeighborsRegressor()
scoring = 'mean_squared_error'
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

""" results.mean() and 'accuracy' are the same """
print (results.mean())
print("Accuracy: %0.8f (+/- %0.2f)" % (results.mean(), results.std() * 2))

model.fit(X, Y)
model.coef_
model.intercept_
model.score(X,Y)

""" Model Comparison """

import pandas
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model1 = LogisticRegression()
model2 = LinearDiscriminantAnalysis()

scoring = 'accuracy'

kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

results1 = cross_validation.cross_val_score(model1, X, Y, cv=kfold)
results2 = cross_validation.cross_val_score(model2, X, Y, cv=kfold)

print("Accuracy: %0.6f (+/- %0.2f)" % (results1.mean(), results1.std() * 2))
print("Accuracy: %0.6f (+/- %0.2f)" % (results2.mean(), results2.std() * 2))






















