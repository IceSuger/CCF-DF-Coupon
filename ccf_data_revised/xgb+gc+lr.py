# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 17:30:56 2016

@author: X93
"""

from __future__ import division #为了让整数相除得到小数，否则只能得到整数
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split  
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import cross_val_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import sys
import gc
from util import *

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 5

#pd.set_option('display.float_format', lambda x: '%.13f' % x) #为了直观的显示数字，且为了输出提交文件的格式不出问题，不采用科学计数法

def markTarget(df):
    #正例反例标记
    df[11] = 0
    #df[11][ (df[6].notnull() & df[2].notnull() )& (df[7].astype('timedelta64[D]').fillna(200).astype('int')<16)] = 1
    #df[11][ df[6].notnull() & df[2].notnull()] =1
    #df[11][ ( (df[6].notnull()) & (df[2]>0) )] =1
    df[11][ df[6].notnull()] =1
    
    df[7] = df[6]-df[5]
    df[31] = 0
    df[31][( ((df[6].notnull()) & (df[2]>0) )& (df[7].astype('timedelta64[D]').fillna(200).astype('int')<16))] = 1
    return df

#统一为一个函数，在这里面统一选择作为特征的列
def chooseFeatures(df):
    #先把列顺序排好
    cols = list(df)
    cols.sort()
    df = df.ix[:,cols]
    #然后选出这些列作为特征，具体含义见FeatureExplaination.txt
    #return df[[0,1]],df[[3,4,8,9,10,12,14,17,20]] #.fillna(0).values
    
    #return df[[1,3,8,9,10,12,14,15,16,17,20,23,24,25]] #根据fscore，从下面这行里选出的比较重要的特征
    #v3.8的特征：    
    #return df[[0,1,3,4,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
    #v1.11的特征：
    #return df[[0,1,3,4,8,9,10,12,14,15,16,17,20,23,24,25]]
    #v3.18的特征：    
    return df[[3,4,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
    

df = readAsChunks_hashead("offline16.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float,'18':float,'19':float, '20':float,'21':float,'22':float, '15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float}).replace("null",np.nan)
df.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df[5] = pd.to_datetime(df[5])
df[6] = pd.to_datetime(df[6])
#为了提高表现，有必要把字段14的空值填上了，毕竟他妈的在训练集里缺了将近一半
#df[14] = df[14].fillna(35)

#打上分类标记
df = markTarget(df)

#选出原始特征
X = chooseFeatures(df)
y = df[11]

#训练模型
#XGB
params = {
        "objective": "binary:logistic",
        #"scale_pos_weight": weight,#设为正例反例比例
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 9,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "silent": 0,
        "nthread":4,
        "seed": 27,
    }
num_boost_round = 100
#X = X.values
y = y.values
dtrain = xgb.DMatrix(X,label = y)
#训练XGB
gbm = xgb.train(params, dtrain, num_boost_round, verbose_eval=True)
print 'xgb train ok'

#下面对训练集跑predict，得到“新特征”
X_leaf0 = pd.DataFrame(gbm.predict(xgb.DMatrix(X), pred_leaf=True))

#下面释放一下内存
del df, X, params, dtrain
gc.collect()

#下面对新特征哑编码
#enc = OneHotEncoder(categorical_features='all', sparse=True, dtype=np.int) #categorical_features = np.array(range(num_boost_round)))
#enc.fit(X_leaf0)
#X_leaf = enc.transform(X_leaf0)

#尝试分别对每列做onehot，然后再sparse.hstack，也许不会死机？

def fitEncoders(col):
    #print col.name
    col = pd.DataFrame(col)
    enc = OneHotEncoder(categorical_features='all', sparse=True, dtype=np.int)
    enc.fit(col)
    #encs.append(enc)
    gc.collect()
    return enc
def sparsifyMid(col):
    enc = encs[col.name]
    col = pd.DataFrame(col)
    #print col
    #mid表示中间结果，每个mid代表一棵树的结果独热编码后
    mid = enc.transform(col)
    #print pd.DataFrame( mid.toarray() )
    
    #mids.append(mid)
    gc.collect()
    return mid
#对X_leaf0做哑编码
#encs = X_leaf0.apply((lambda x:fitEncoders(x)),axis=0) #这个要保留，为处理测试集用。
#v4.9 这里改成手动for循环，避免死机
encs = []
for i in range(num_boost_round):
    encs.append( fitEncoders(X_leaf0.iloc[:,i]) )
encs = pd.Series(encs)
print 'encoders fit ok'
#释放点内存
gc.collect()
#每棵树的哑编码结果，都放在mids这个Series中
mids = X_leaf0.apply((lambda x:sparsifyMid(x)),axis=0)
#释放点内存
del X_leaf0
gc.collect()

from scipy import sparse
X_leaf = sparse.hstack(list(mids))

print 'onehot ok', X_leaf.shape

#送进LR训练
lr = LogisticRegression(n_jobs=4)
lr.fit(X_leaf, y)

#释放内存
del X_leaf, mids
gc.collect()

print 'LR train ok'
def giveResultOnTestset_XGBLR():
     #读测试数据。这里是题目给的原始数据，读它是为了保证提交结果的前三列格式不出问题
    df_test = readAsChunks_nohead("ccf_offline_stage1_test_revised.csv",{0:int, 1:int}).replace("null",np.nan)
    df_res = df_test[[0,2,5]]
    
    #读预处理过的测试集。
    df_test = readAsChunks_hashead("test16.csv",{'0':int, '1':int, '4':float, '8':float, '9':float,'10':float,  '17':float,'18':float,'19':float, '20':float,'21':float,'22':float,'15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float})
    df_test.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
    df_test[4] = df_test[4].fillna(df_test[4].mean())
    print 'test read in ok'
    #选择特征列
    features_test = chooseFeatures(df_test) #.values
    #释放内存
    del df_test
    gc.collect()
    #XGBOOST得到新特征
    test_leaf = pd.DataFrame( gbm.predict(xgb.DMatrix(features_test), pred_leaf=True) )
    
    """    
    #标准化原始特征
    features_test = features_test.fillna(0)
    features_test = pd.DataFrame( scaler.transform(features_test) )
    #合并新特征和标准化后的原始特征
    test_leaf = pd.concat([test_leaf,features_test],axis=1)
    """
    #test_leaf = pd.DataFrame( pd.read_csv("test_leaf.csv",header=None)[99] )
    #test_leaf = pd.DataFrame( pd.read_csv("test_leaf.csv",header=None) )
    #哑编码    
    #test_leaf = enc.transform(test_leaf)
    
    #v4.8新哑编码
    mids = test_leaf.apply((lambda x:sparsifyMid(x)),axis=0)
    test_leaf = sparse.hstack(list(mids))
    #LR
    target_test = lr.predict_proba(test_leaf)[:,1]
    
    #保存结果
    df_res[4] = pd.DataFrame(target_test)
    #不想要科学计数法的结果
    df_res[4] = df_res[4].apply(lambda x: '%.15f' % x)

    df_res.to_csv("v4_9 xgb into lr_depth9.csv",header=None,index=False)
    return df_res

df_res = giveResultOnTestset_XGBLR()












"""
def clear():
    for key, value in globals().items():
        if callable(value) or value.__class__.__name__ == "module":
            continue
        del globals()[key]

def show_size_of_globals():
    l = []
    for key, value in globals().items():
        s = sys.getsizeof(key)
        print key, s
        l.append(s)
    return l
"""
#del df,X

print gc.collect()