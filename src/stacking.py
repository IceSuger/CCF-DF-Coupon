# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 14:34:04 2016

@author: X93
"""

import pandas as pd
import numpy as np
from Ensemble import Ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#import xgboost as xgb
from xgboost.sklearn import XGBClassifier
#from util import *
import util
import ManipulatingFeatures as mf
import warnings
warnings.filterwarnings("ignore")

df = util.readAsChunks_hashead("..\\ccf_data_revised\\offline16.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float,'18':float,'19':float, '20':float,'21':float,'22':float, '15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float}).replace("null",np.nan)
df.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df[5] = pd.to_datetime(df[5])
df[6] = pd.to_datetime(df[6])
sample_weight = util.readAsChunks_nohead("..\\ccf_data_revised\\sample_weight.csv",{0:float})[0]

    
#读预处理过的测试集。
df_test = util.readAsChunks_hashead("..\\ccf_data_revised\\test16.csv",{'0':int, '1':int, '4':float, '8':float, '9':float,'10':float,  '17':float,'18':float,'19':float, '20':float,'21':float,'22':float,'15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float})
df_test.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df_test[4] = df_test[4].fillna(df_test[4].mean())
    

#打上分类标记
df = mf.markTarget(df)

#选出原始特征
X = mf.chooseFeatures(df)
y = df[11]
y_true = df[31]
T = mf.chooseFeatures(df_test)

#准备好base model的参数
params_xgb = {
        "objective": "binary:logistic",
        #"scale_pos_weight": weight,#设为正例反例比例
        #"booster" : "gbtree",
        #"eval_metric": "auc", #这个参数在fit时设置，见Ensemble.py
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "silent": 0,
        "nthread":4,
        "seed": 27,
        "n_estimators": 96#96
    }
#初始化base model
#初始化XGB
base_xgbs = []
for i in range(5):
    base_xgbs.append( XGBClassifier() )
    base_xgbs[i].set_params(**params_xgb)
    base_xgbs[i].set_params(seed = i)

#for i in range(5):
#   print i,base_xgbs[i]

#xgb0 = XGBClassifier()
#xgb0.set_params(**params_xgb)

#初始化RF
rf = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 0, n_jobs=4) 

#初始化stacker
params_xgb2 = {
        "objective": "binary:logistic",
        #"scale_pos_weight": weight,#设为正例反例比例
        #"booster" : "gbtree",
        #"eval_metric": "auc", #这个参数在fit时设置，见Ensemble.py
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "silent": 0,
        "nthread":4,
        "seed": 27,
        "n_estimators": 96#96
    }
xgb2 = XGBClassifier()
xgb2.set_params(**params_xgb2)

#初始化lr，作为stacker
lr = LogisticRegression( n_jobs=4)
#GO ensemble!
base_models = [xgb2]
#en = Ensemble(n_folds=5, stacker=lr, base_models=base_models)#[xgb0,rf,xgb0,rf,xgb0])
en = Ensemble(n_folds=1, stacker=None, base_models=base_models)#[xgb0,rf,xgb0,rf,xgb0])

target_test = en.fit_predict(X, y, y_true, sample_weight, T)


#读测试数据。这里是题目给的原始数据，读它是为了保证提交结果的前三列格式不出问题
test = util.readAsChunks_nohead("..\\ccf_data_revised\\ccf_offline_stage1_test_revised.csv",{0:int, 1:int}).replace("null",np.nan)
df_res = test[[0,2,5]]
#保存结果
df_res[4] = pd.DataFrame(target_test)
#不想要科学计数法的结果
df_res[4] = df_res[4].apply(lambda x: '%.15f' % x)

df_res.to_csv("..\\result\\v6_2 stacking_5xgbs_to_lr.csv",header=None,index=False)


#rf = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1) 
#rf.fit(X_train,y_train)


#rf_whole = RandomForestClassifier(class_weight = 'auto', max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1) 
#rf_whole.fit(X,y)
#ab_whole = AdaBoostClassifier(n_estimators = 7)
#ab_whole.fit(X,y)
#lr = LogisticRegression(class_weight = 'auto', n_jobs=-1)
#lr = LogisticRegression( n_jobs=-1)
#lr.fit(X,y)
#ar = AdaBoostRegressor()
#ar.fit(X,y)
#gbdt = GradientBoostingClassifier(n_estimators=100)

