# all stuff needed to be imported for later use 
import pandas as pd
# from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib


#DATA PROCESSING  
#import data
df = pd.read_csv("Project_1_Data.csv")


# DATA VISULIZATION 

# checking for missing values 
print(df.isna().any(axis=0).sum())
print(df.isna().any(axis=1).sum()) 
# There are no missing values so the next two lines are commented out but\n
# is how you would fix missing values 
#df = df.dropna()
#df = df.reset_index(drop=True)

# Getting basic information about the DataFrame

print("First few rows of the DataFrame:")
print(df.head())

print("\nDataFrame info:")
print(df.info())

print('\nDecribing data')
print(df.describe())

# Plot histograms for the data see if there are any obvious relations 
df.hist(bins=50, figsize=(20, 15))
plt.show()

# 3d plot to shpw the corrdintates locations might be helpful to visualize

fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'])
ax.axes.set_xlim3d(left=0, right=9) 
ax.axes.set_ylim3d(bottom=0, top=8) 
ax.axes.set_zlim3d(bottom=0, top=8) 
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# corelation matrix 
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
plt.matshow(corr_matrix)
sns.heatmap(np.abs(corr_matrix), fmt = 'f', annot = True , cmap = 'RdPu')



df["attributes"] = pd.cut(
    df["Z"], bins=[0, 1, 2, 3, np.inf], right=False, labels=[0.5, 1.5, 2.5, 3.5])
df['attributes'].hist()

nan_rows = df['attributes'].isna().any()
print(nan_rows)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["attributes"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

df = strat_train_set.drop(['Step','attributes'], axis=1)
df_labels = strat_train_set['Step'].copy()



df_test = strat_test_set.drop(['Step','attributes'], axis =1)
df_labels_test = strat_test_set['Step'].copy()







# MODEL DEVELOPMENT & ANALYSIS 
# tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(df,df_labels)

tree_pred = tree_clf.predict(df)


param_grid = {'criterion':['gini', 'entropy'],'max_depth' :[2,4,6,8,10,12], 'min_samples_split':[2,4,6,8]}
grid_search = GridSearchCV(tree_clf, param_grid, cv=5,scoring='accuracy',return_train_score=True)
tree_clf = grid_search.fit(df, df_labels)
tree_pred = tree_clf.predict(df)

# precsion, confusion matix, accuracy, F1 score

acc = cross_val_score(tree_clf, df, df_labels, cv=10,scoring="accuracy")
acc = np.array(acc)
acc = np.mean(acc)


dfpred = cross_val_predict(tree_clf, df, df_labels, cv=5)

prec = precision_score(df_labels, tree_pred, average = 'micro')


f1 = f1_score(df_labels, dfpred, average = 'micro')


tree = ['tree', acc, prec, f1]


# random forest classifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(df,df_labels)
forest_pred = forest_clf.predict(df)


param_grid = [{'n_estimators': [3, 10, 30, 35], 'max_features': [3, 6, 8, 12]},{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3,4]},]
grid_search = GridSearchCV(forest_clf, param_grid, cv=10,scoring='accuracy',return_train_score=True)
forest_clf = grid_search.fit(df, df_labels)


forest_pred = forest_clf.predict(df)

# precsion, confusion matix, accuracy, F1 score

acc = cross_val_score(forest_clf, df, df_labels, cv=10,scoring="accuracy")
acc = np.array(acc)
acc = np.mean(acc)

forestpred = cross_val_predict(forest_clf, df, df_labels, cv=10)


print('confusion matrix')
cm = confusion_matrix(df_labels, forestpred)
print(cm)
sns.heatmap(np.abs(cm), annot = True, fmt = 'd',cmap = 'RdPu', xticklabels=[1,2,3,4,5,6,7,8,9,10,11,12,13],  yticklabels=[1,2,3,4,5,6,7,8,9,10,11,12,13])


prec = precision_score(df_labels, forest_pred, average = 'micro')


f1 = f1_score(df_labels, forestpred, average = 'micro')

random = ['random', acc, prec, f1]



# GaussianNB
gnb_clf = GaussianNB()
gnb_clf.fit(df,df_labels)
gnb_pred = gnb_clf.predict(df)


# hyperparamter fine tuning 
param_nb = {'var_smoothing':np.logspace(0,-5, num=10)}

grid_search = GridSearchCV(gnb_clf, param_nb, cv=10,scoring='accuracy', verbose=1)
grid_search.fit(df,df_labels)

# apply new best hyperparsmters
gnb_clf = grid_search.fit(df,df_labels)
gnb_pred = gnb_clf.predict(df)

# precsion, confusion matix, accuracy, F1 score

acc = cross_val_score(gnb_clf, df, df_labels, cv=10,scoring="accuracy")
acc = np.array(acc)
acc = np.mean(acc)

gnbpred = cross_val_predict(gnb_clf, df, df_labels, cv=10)


prec = precision_score(df_labels, gnb_pred, average = 'micro')


f1 = f1_score(df_labels, gnbpred, average = 'micro')

gauss = ['gauss', acc, prec, f1]

compare = np.array([tree, random, gauss],dtype = object)
print(compare)

# FINAL MODEL SAVE AND TEST
# save model in joblib format 
joblib.dump(forest_clf,'class_model')

forest_clf = joblib.load('class_model')

test = forest_clf.predict(df_test)
test = np.array(test)
labels = np.array(df_labels_test)
print(len(test))


# find out if test set and predicted values are the same if not where and what was predicted
different_indices = np.where(test != labels)[0]
print(different_indices)
print(test[different_indices],labels[different_indices])


# predict the following 
df2 = pd.DataFrame(np.array([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]]),
                   columns=['X', 'Y', 'Z'])

predictions = forest_clf.predict(df2)
print('The predicted maintenace steps for the given coordinates are:\n', predictions)


