from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
# from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, f1_score, precision_score, confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

#import data
df = pd.read_csv("Project_1_Data.csv")

# print(df.isna().any(axis=0).sum())  #how many columns have missing values?
# print(df.isna().any(axis=1).sum()) #how many rows have missing values?
#df = df.dropna()
#df = df.reset_index(drop=True)

print("First few rows of the DataFrame:")
print(df.head())

# Getting basic information about the DataFrame
print("\nDataFrame info:")
print(df.info())

print('\nDecribing data')
print(df.describe())

df.hist(bins=50, figsize=(20, 15))
plt.show()
print(df.isna().any(axis=0).sum())  # how many columns have missing values?
print(df.isna().any(axis=1).sum())  # how ma

df["attributes"] = pd.cut(
    df["Z"], bins=[0, 1, 2, 3, np.inf], right=False, labels=[0.5, 1.5, 2.5, 3.5])
df['attributes'].hist()

nan_rows = df['attributes'].isna().any()
print(nan_rows)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["attributes"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

df = strat_train_set.drop('Step', axis=1)
df_labels = strat_train_set['Step'].copy()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'])
plt.show()


df.hist(bins=50, figsize=(20, 15))
plt.show()


corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
plt.matshow(corr_matrix)
sns.heatmap(np.abs(corr_matrix))

'''
scaler = MinMaxScaler()
df['X'] = scaler.fit_transform(df['X'],[0,1])
print('norm df')
print(df['X'])

'''
# SGD classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(df,df_labels)

sgd_pred = sgd_clf.predict(df)
scores = cross_val_score(sgd_clf, df, df_labels,scoring="neg_mean_squared_error", cv=5)
clf_scores = np.sqrt(-scores)
print('scores')
print(scores)
print(scores.mean())
print(scores.std())
print('accuracy')
acc = cross_val_score(sgd_clf, df, df_labels, cv=10,scoring="accuracy")
print(acc)

param_grid = {'loss':['log_loss','modified_huber'],'penalty':['elasticnet'],'alpha':np.logspace(0, 10, 10),'l1_ratio':[0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.13,0.14,0.15,0.2]}
grid_search = GridSearchCV(sgd_clf, param_grid, cv=5,scoring='accuracy',return_train_score=True)
sgd_clf = grid_search.fit(df, df_labels)
print(grid_search.best_params_)
sgd_pred = sgd_clf.predict(df)
scores = cross_val_score(sgd_clf, df, df_labels,scoring="neg_mean_squared_error", cv=5)
#clf_scores = np.sqrt(-scores)
print('scores')
print(scores)
print(scores.mean())
print(scores.std())


# precsion, confusion matix, accuracy, F1 score
print('accuracy')
acc = cross_val_score(sgd_clf, df, df_labels, cv=10,scoring="accuracy")
print(acc)

dfpred = cross_val_predict(sgd_clf, df, df_labels, cv=5)
#print('confusion matrix')

#print(confusion_matrix(df_labels, dfpred))

prec = precision_score(df_labels, sgd_pred, average = 'micro')
print(prec)

f1 = f1_score(df_labels, dfpred, average = 'micro')
print(f1)

sgd = ['sgd', acc, prec, f1]




# random forest classifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(df,df_labels)
forest_pred = forest_clf.predict(df)
forest_mse = mean_squared_error(df_labels,forest_pred)
forest_rmse = np.sqrt(forest_mse)
print('forest')
print(forest_rmse)
scores = cross_val_score(forest_clf, df, df_labels,scoring="neg_mean_squared_error", cv=10)
clf_scores = np.sqrt(-scores)
print('scores')
print(scores)
print(scores.mean())
print(scores.std())

param_grid = [{'n_estimators': [3, 10, 30, 35], 'max_features': [3, 6, 8, 12]},{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3,4]},]
grid_search = GridSearchCV(forest_clf, param_grid, cv=10,scoring='accuracy',return_train_score=True)
grid_search.fit(df, df_labels)
print(grid_search.best_params_)

forest_clf=RandomForestClassifier(bootstrap=True, max_depth=None,
max_features=6, max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,
n_estimators=3, n_jobs=None, oob_score=False, random_state=42,verbose=0, warm_start=False)
forest_clf.fit(df,df_labels)
forest_pred = forest_clf.predict(df)
forest_mse = mean_squared_error(df_labels,forest_pred)
forest_rmse = np.sqrt(forest_mse)
print('forest')
print(forest_rmse)
scores = cross_val_score(forest_clf, df, df_labels,scoring="neg_mean_squared_error", cv=10)
clf_scores = np.sqrt(-scores)
print('scores')
print(scores)
print(scores.mean())
print(scores.std())

# precsion, confusion matix, accuracy, F1 score
print('accuracy')
acc = cross_val_score(forest_clf, df, df_labels, cv=10,scoring="accuracy")

forestpred = cross_val_predict(forest_clf, df, df_labels, cv=10)
print('confusion matrix')

print(confusion_matrix(df_labels, forestpred))

prec = precision_score(df_labels, forest_pred, average = 'micro')
print(prec)

f1 = f1_score(df_labels, forestpred, average = 'micro')
print(f1)

random = ['random', acc, prec, f1]



# GaussianNB
gnb_clf = GaussianNB()
gnb_clf.fit(df,df_labels)
gnb_pred = gnb_clf.predict(df)
scores = cross_val_score(gnb_clf, df, df_labels,scoring="neg_mean_squared_error", cv=10)
clf_scores = np.sqrt(-scores)
print('scores')
print(scores)
print(scores.mean())
print(scores.std())



# hyperparamter fine tuning 
param_nb = {'var_smoothing':np.logspace(0,-5, num=10)}

grid_search = GridSearchCV(gnb_clf, param_nb, cv=10,scoring='accuracy', verbose=1)
grid_search.fit(df,df_labels)
print(grid_search.best_params_)

# apply new best hyperparsmters
gnb_clf = grid_search.fit(df,df_labels)
gnb_pred = gnb_clf.predict(df)
scores = cross_val_score(gnb_clf, df, df_labels,scoring="neg_mean_squared_error", cv=10)
clf_scores = np.sqrt(-scores)
print('scores')
print(scores)
print(scores.mean())
print(scores.std())

# precsion, confusion matix, accuracy, F1 score
print('accuracy')
acc = cross_val_score(gnb_clf, df, df_labels, cv=10,scoring="accuracy")

gnbpred = cross_val_predict(gnb_clf, df, df_labels, cv=10)
print('confusion matrix')

print(confusion_matrix(df_labels, gnbpred))

prec = precision_score(df_labels, gnb_pred, average = 'micro')
print(prec)

f1 = f1_score(df_labels, gnbpred, average = 'micro')
print(f1)

gauss = ['gauss', acc, prec, f1]

compare = [sgd, random, gauss]




