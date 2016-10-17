# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 23:12:53 2016

@author: X93
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split  
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder

def readAsChunks(file_dir, types):
    chunks = []
    chunk_size = 1000000
    reader = pd.read_csv(file_dir,',', header=None, iterator=True, dtype=types)
    # '+'号表示匹配前面的子表达式一次或多次
    # dtype参数，指明表中各列的类型，避免python自己猜，可以提高速度、并减少内存占用
    while True:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print "Iteration is stopped."
            break
    df = pd.concat(chunks, ignore_index=True)
    #分块将.txt文件读入内存，放到一个 pandas 的 dataFrame 里。块大小（即每次读的行数）为chunk_size
    return df

def splitDiscountRateCol(df):
    #把优惠方式那一列切开
    df[8] = 0.0
    df[9] = 0.0
    #df[df[3].str.contains(':').fillna(False)][[8,9]] = df[3].str.split(':',expand=True).astype(float)
    df[[8,9]] = df[3].str.split(':',expand=True).astype(float)
    df[8][df[8]<1] = 0
    #df[df[3].str.contains(':').fillna(True)]
    df[10] = 1 - df[9]/df[8]
    
    df[[8,9]] = df[[8,9]].fillna(0)
    df[10] = df[10].fillna(1)
    
    #优惠方式这一列，直接是折扣率的，保留
    #df[3] = df[3].replace("\d*:\d*","0")#.astype(float).fillna(0)  #".*"为正则表达式，'.'表示任意字符，'*'表示出现任意次
        #但 replace 方法第一个参数并不是正则表达式，而是字符串。所以不用这种方法来替换字符串了。
    #print df[3].value_counts()
    #print df[df[3].str.contains(':').fillna(True)]
    df[3] = df[3].fillna(1)
    df[3][df[3].str.contains(':').fillna(True)] = '1'
    df[3] = df[3].astype(float)
    
    #填充距离字段的空值
    #df[4] = Imputer().fit_transform(df[4])
    df[4] = df[4].astype(float)
    df[4] = df[4].fillna(df[4].mean())
    return df
    
def markTarget(df):
    #正例反例标记
    df[11] = 0
    #print df[df[6].notnull() & df[2].notnull()]
    #df.loc( (df[6].notnull() & df[2].notnull()) ,11)= 1
    df[11][ df[6].notnull() & df[2].notnull() ] = 1
    df = df[df[2].notnull()]
    return df

df = readAsChunks("ccf_offline_stage1_train.csv", {0:int, 1:int,  3:str}).replace("null",np.nan)


#领券日期 和 消费日期 及 消费日期距离领券日期的差
df[5] = pd.to_datetime(df[5])
df[6] = pd.to_datetime(df[6])
df[7] = df[6]-df[5]

df = splitDiscountRateCol(df)
df = markTarget(df)
print df

#usrid, merchantid, discountrate, man, jian, approxi_discountrate
features = df[[0,1,3,4,8,9,10]].fillna(0).values
target_train = df[11]

#usrid, merchantid, coupon_id, discountrate, man, jian, approxi_discountrate
#features = df[[0,1,2,3,8,9,10]].fillna(0).values

#One-hot encode
enc = OneHotEncoder(categorical_features = np.array([0,1]) )
enc.fit(features)
features = enc.transform(features)


#训练模型

#lr = LogisticRegression(n_jobs=-1)
#lr.fit(features, target_train)
print 'LR score on train set:'
#print lr.score(features, target_train)

X = features
y = target_train
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=1) 
#rf = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1) 
#rf.fit(X_train,y_train)
#print 'RF score on train set:'
#print rf.score(X_train,y_train)
#print 'RF test score:'
#print rf.score(X_test,y_test)

#rf_whole = RandomForestClassifier(class_weight = 'auto', max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1) 
#rf_whole.fit(X,y)
#ab_whole = AdaBoostClassifier(n_estimators = 7)
#ab_whole.fit(X,y)
lr = LogisticRegression(n_jobs=-1)
lr.fit(X,y)
#ar = AdaBoostRegressor()
#ar.fit(X,y)


#读测试数据
df_test = readAsChunks("ccf_offline_stage1_test_revised.csv",{0:int, 1:int}).replace("null",np.nan)

df_test = splitDiscountRateCol(df_test)

#features_test = df_test[[0,1,3,8,9,10]].fillna(0).values
df_test = df_test[df_test[2].notnull()]
features_test = df_test[[0,1,3,4,8,9,10]].fillna(0).values
features_test = enc.transform(features_test)

#target_test = ab_whole.predict_proba(features_test)[:,1]
#target_test = rf_whole.predict_proba(features_test)[:,1]
target_test = lr.predict_proba(features_test)[:,1]
#target_test = ar.predict(features_test)

df_res = df_test[[0,2,5]]
df_res[4] = pd.DataFrame(target_test)
df_res.to_csv("v0_9_lr_with1 3_unbalanced_dummied 0 1.csv",header=None,index=False)
print df_res[4].value_counts()

#算auc
#roc_auc_score(y_true, y_scores)