import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
data = pd.read_csv('heart.csv')
data.head()
data.info()
data.isnull().sum()
data.describe()

corr = data.corr()
plt.figure(figsize = (15,15))
sns.heatmap(corr, annot = True)
corr
sns.set_style('whitegrid')
sns.countplot(x = 'target', data = data)
dataset = data.copy()
dataset.head()

X = dataset.drop(['target'], axis = 1)
y = dataset['target']
X.columns
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

pred = model.predict(X_test)
pred[:10]

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)

from sklearn.metrics import accuracy_score
print(f"Accuracy of model is {round(accuracy_score(y_test, pred)*100, 2)}%")
classifier = RandomForestClassifier(n_jobs = -1)
from scipy.stats import randint
param_dist={'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200,300,400,500],
              'max_features':randint(1,31),
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,31),
              }
search_clfr = RandomizedSearchCV(classifier, param_distributions = param_dist, n_jobs=-1, n_iter = 40, cv = 9)
search_clfr.fit(X_train, y_train)

params = search_clfr.best_params_
score = search_clfr.best_score_
print(params)
print(score)
claasifier=RandomForestClassifier(n_jobs=-1, n_estimators=400,bootstrap= False,criterion='gini',max_depth=5,max_features=3,min_samples_leaf= 7)
classifier.fit(X_train, y_train)
confusion_matrix(y_test, classifier.predict(X_test))
print(f"Accuracy is {round(accuracy_score(y_test, classifier.predict(X_test))*100,2)}%")
import pickle
pickle.dump(classifier, open('heart.pkl', 'wb'))