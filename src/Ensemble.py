# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 14:36:28 2016

@author: X93
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from xgboost.sklearn import XGBClassifier
import gc

#实现Stacking Ensemble
class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, y_true, sample_weight, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        y_true = np.array(y_true)
        sample_weight = np.array(sample_weight)

        folds = list(StratifiedKFold(y_true, n_folds=self.n_folds, shuffle=True, random_state=0))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            gc.collect()
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                W_train = sample_weight[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                if isinstance(clf, XGBClassifier):
                    clf.fit(X_train, y_train, eval_metric = 'auc', sample_weight = W_train)
                else:
                    clf.fit(X_train, y_train, sample_weight = W_train)
                y_pred = clf.predict_proba(X_holdout)[:,1]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]

            S_test[:, i] = S_test_i.mean(1)
        if isinstance(self.stacker, XGBClassifier):
            self.stacker.fit(S_train, y, eval_metric = 'auc')
        else:
            self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict_proba(S_test)[:,1]
        return y_pred
        
"""
def markTarget(df):
    #正例反例标记
    df[11] = 0
    #df[11][ (df[6].notnull() & df[2].notnull() )& (df[7].astype('timedelta64[D]').fillna(200).astype('int')<16)] = 1
    #df[11][ df[6].notnull() & df[2].notnull()] =1
    #df[11][ ( (df[6].notnull()) & (df[2]>0) )] =1
    df[11][ df[6].notnull()] =1
    
    return df
    
        
df = pd.read_csv("..\\ccf_data_revised\\ccf_offline_stage1_train.csv",header=None).replace("null",np.nan)
y = markTarget(df).head(10000)[11].values
X = df.head(10000)[[0,1,2,4]].values
clf = xgb.XGBClassifier()
clf.fit(X,y)
res = clf.predict_proba(X[:1000,:])
"""


