# --- Setup ---
#import the libraries
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option

# #### Load the data
# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
print(data.shape)

# #### Desrciptive Stats
filename = "pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Data Types for Each Attribute
types = data.dtypes
print(types)

# Statistical Summary
data.describe()
# set_option('precision', 3)
# description = data.describe()
# print(description)

# Pairwise Pearson correlations
correlations = data.corr(method='pearson')
print(correlations)

# Class proportion
class_counts = data.groupby('class').size()
print(class_counts)

# #### Data Visualization
data.hist(figsize=(10,8))
pyplot.show()

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(10,8))
pyplot.show()

data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))
pyplot.show()

# Correlation Matrix Plot (with annotations)
import numpy # This import was implicitly used but not explicitly in a code cell earlier, so it's added here.
fig = pyplot.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# #### Data Preprocessing
# Rescale data (between 0 and 1)
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from numpy import array

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]

# scaler = MinMaxScaler(feature_range=(0, 1))
scaler = MinMaxScaler()
rescaledX = scaler.fit_transform(X)

# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])

# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import array

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])

# #### Feature Selection
# Feature Selection with RFE
from pandas import read_csv
from numpy import array
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]

# feature extraction
model = LogisticRegression(max_iter=200)
rfe = RFE(model, n_features_to_select=3)
fit = rfe.fit(X, Y)

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

# Feature Extraction with PCA
from pandas import read_csv
from numpy import array
from sklearn.decomposition import PCA

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]

# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)

# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

# Feature Extraction with Univariate Statistical Selection (Chi-squared)
from pandas import read_csv
from numpy import array
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]

# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)

# summarize selected features
print(features[0:5,:])

# #### Algorithm Evaluation
# Evaluate Algorithms
from pandas import read_csv
from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]

num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
model = LogisticRegression(max_iter=200)
results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))

# Compare Algorithms
from pandas import read_csv
from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]

# prepare models
models = []
models.append(('LR', LogisticRegression(max_iter=200)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %.3f (%.3f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure(figsize=(10,8))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results, labels=names)
pyplot.show()

# #### Model Tuning
# KNN Algorithm Tuning
from pandas import read_csv
from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler # Added import for StandardScaler used below

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

# prepare the cross-validation procedure
kfold = KFold(n_splits=10, random_state=7, shuffle=True)

# define the grid of values to search
n_neighbors = array(range(1,21))
param_grid = dict(n_neighbors=n_neighbors)

# prepare the model
model = KNeighborsClassifier()

# search the grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(rescaledX, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# SVM Algorithm Tuning
from pandas import read_csv
from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler # Added import for StandardScaler used below

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]

# rescale data for SVM
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

# prepare the cross-validation procedure
kfold = KFold(n_splits=10, random_state=7, shuffle=True)

# define the grid of values to search
param_grid = [
    {'kernel': ['linear'], 'C': [1.0, 0.1, 0.001]},
    {'kernel': ['rbf'], 'gamma': [0.1, 0.01, 0.001], 'C': [1.0, 0.1, 0.001]},
]

# prepare the model
model = SVC()

# search the grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(rescaledX, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# #### Finalize Model
# Finalize Model with best parameters
from pandas import read_csv
from numpy import array
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pickle import dump
from pickle import load

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Separate array into input and output components
array = data.values
X = array[:,0:8]
Y = array[:,8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

# Fit the model on 33%
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# save the model to disk
filename = 'finalized_model.sav'
dump(model, open('filename', 'wb'))

# some time later...

# load the model from disk
loaded_model = load(open('filename', 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
