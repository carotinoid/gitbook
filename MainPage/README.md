# Welcome!

```python
## Module
# General
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Model(classification) (regression)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier,XGBRFClassifier, XGBRegressor, XGBRFRegressor
from sklearn.svm import SVC

# Model(clustering)
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from kmodes.kprototypes import KPrototypes

# init
sample_dataframe = pd.DataFrame()

# cheat sheet
data_frame_name = pd.read_csv('data_frame_name.csv')
sample_dataframe = sample_dataframe.dropna(inplace=True)
sample_dataframe = sample_dataframe.fillna(0)
sample_dataframe = sample_dataframe.fillna(sample_dataframe.mean())
z_score = np.abs(stats.zscore(sample_dataframe))
sample_dataframe = sample_dataframe[(z_score < 3).all(axis=1)]
target = 'target'
X, y = sample_dataframe.drop(target, axis=1), sample_dataframe[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stdscalar = StandardScaler()
stdscalar.fit_transform(sample_dataframe)
minmaxscalar = MinMaxScaler()
minmaxscalar.fit_transform(sample_dataframe)
sample_dataframe[['column1', 'column2']] = minmaxscalar.fit_transform(sample_dataframe[['column1', 'column2']])
labelencoder = LabelEncoder()
labelencoder.fit_transform(sample_dataframe['column'])
labelencoder.inverse_transform(sample_dataframe['column'])
imputer = KNNImputer(n_neighbors=2)
imputer.fit_transform(sample_dataframe)
model = DecisionTreeClassifier() # or any model
f1_score(y_val, model.predict(X_val), average='macro')
silhouette_score(X, model.predict(X))
y_pred = model.predict(X_test)
make_submission = pd.DataFrame({'ID': np.arange(1, len(y_pred)+1), 'target': y_pred}) # rename target

# params
decision_tree_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
random_forest_params = {
    'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
xgboost_params = {
    'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'colsample_bylevel': [0.5, 0.7, 1.0],
    'colsample_bynode': [0.5, 0.7, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [0, 1, 5]
}
svc_params = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [1, 2, 3, 4, 5],
    'gamma': ['scale', 'auto']
}
knn_params = {
    'n_neighbors': [int(x) for x in np.linspace(start=1, stop=10, num=10)],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [int(x) for x in np.linspace(start=1, stop=50, num=10)]
}

grid = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=1)
randomized = RandomizedSearchCV(model, params, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
grid.best_estimator_
grid.best_params_
grid.best_score_

k_prototypes = KPrototypes(n_clusters=3, init='Cao', n_init=10, verbose=2)
# init: {'Huang', 'Cao', 'random'}, default='Huang'
k_prototypes.fit(X, categorical=[0, 1])
k_prototypes.labels_
k_means = KMeans(n_clusters=3, random_state=42)
k_means.labels_


import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# 1. 데이터 로드
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

X_df = pd.DataFrame(X, columns=feature_names)
y_df = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df)

# 2. 모델 구성 (XGBoost 4개, RandomForest 8개)

# XGBoost 모델들
xgb_models = [
    ('xgb1', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, use_label_encoder=False, eval_metric='logloss')),
    ('xgb2', XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=1, use_label_encoder=False, eval_metric='logloss')),
    ('xgb3', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=2, use_label_encoder=False, eval_metric='logloss')),
    ('xgb4', XGBClassifier(n_estimators=100, learning_rate=0.2, max_depth=3, random_state=3, use_label_encoder=False, eval_metric='logloss'))
]

# RandomForest 모델들
rf_models = [
    ('rf1', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=10)),
    ('rf2', RandomForestClassifier(n_estimators=150, max_depth=6, random_state=11)),
    ('rf3', RandomForestClassifier(n_estimators=200, max_depth=4, random_state=12)),
    ('rf4', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=13)),
    ('rf5', RandomForestClassifier(n_estimators=150, max_depth=5, random_state=14)),
    ('rf6', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=15)),
    ('rf7', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=16)),
    ('rf8', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=17))
]

# 전체 모델 조합
estimators = xgb_models + rf_models

# 3. VotingClassifier 구성 (soft voting)
voting_clf = VotingClassifier(estimators=estimators, voting='soft')

# 학습
voting_clf.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = voting_clf.predict(X_test)

# macro F1 score 계산
f1 = f1_score(y_test, y_pred, average='macro')
print("Macro F1 Score:", f1)


```

