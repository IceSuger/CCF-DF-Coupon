# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 14:34:04 2016

@author: X93
"""

import pandas as pd
import numpy as np
import Ensemble as En
#import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from util import *
import ManipulatingFeatures as mf
import warnings
warnings.filterwarnings("ignore")

df = readAsChunks_hashead("..\\ccf_data_revised\\offline16.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float,'18':float,'19':float, '20':float,'21':float,'22':float, '15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float}).replace("null",np.nan)
df.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df[5] = pd.to_datetime(df[5])
df[6] = pd.to_datetime(df[6])

#打上分类标记
df = mf.markTarget(df)

#选出原始特征
X = mf.chooseFeatures(df)
y = df[11]

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
        "n_estimators": 1#96
    }
#初始化base model
clf0 = XGBClassifier()
clf0.set_params(**params_xgb)
clf0.booster()