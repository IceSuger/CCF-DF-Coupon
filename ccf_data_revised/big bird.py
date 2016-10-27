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
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler

#pd.set_option('display.float_format', lambda x: '%.13f' % x) #为了直观的显示数字，且为了输出提交文件的格式不出问题，不采用科学计数法

def readAsChunks_nohead(file_dir, types):
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

def readAsChunks_hashead(file_dir, types):
    chunks = []
    chunk_size = 1000000
    reader = pd.read_csv(file_dir,',', header=0, iterator=True, dtype=types)
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

    
def markTarget(df):
    #正例反例标记
    df[11] = 0
    #df[11][ (df[6].notnull() & df[2].notnull() )& (df[7].astype('timedelta64[D]').fillna(200).astype('int')<16)] = 1
    df[11][ df[6].notnull() & df[2].notnull()] =1
    return df
    

df = readAsChunks_hashead("offline7.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float, '20':float, '15':int, '16':int, '23':int, '24':float, '25':float}).replace("null",np.nan)
df.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df[5] = pd.to_datetime(df[5])
df[6] = pd.to_datetime(df[6])


#打上分类标记
df = markTarget(df)
#print df.columns



#统一为一个函数，在这里面统一选择作为特征的列
def chooseFeatures(df):
    #先把列顺序排好
    cols = list(df)
    cols.sort()
    df = df.ix[:,cols]
    #然后选出这些列作为特征，具体含义见FeatureExplaination.txt
    return df[[0,1]],df[[3,4,8,9,10,12,13,14,15,16,17,20,23,24,25]].fillna(0).values
    
#usrid, merchantid, discountrate, man, jian, approxi_discountrate
#features = df[[0,1,3,4,8,9,10,12,13,14]].fillna(0).values
features01, features = chooseFeatures(df)
target_train = df[11]

#下面把六月的拆出来，作为线下算平均auc的测试集
#print df.head()
#train_nojun = df.drop(df[df[5].apply(lambda x:x.month)==6] , axis=1)
train_nojun = df[df[5].apply(lambda x:x.month)!=6]
X_nojun01, X_nojun = chooseFeatures(train_nojun)
y_nojun = train_nojun[11]

test_jun = df[df[5].apply(lambda x:x.month)==6]
y_jun = test_jun[11]
X_jun01, X_jun = chooseFeatures(test_jun)

"""
#把usrid，merchantid合并进features
features = np.column_stack((features01,features))
X_nojun = np.column_stack((X_nojun01,X_nojun))
X_jun = np.column_stack((X_jun01,X_jun))
"""


"""
#多项式数据变换
pf = PolynomialFeatures()
pf.fit(features)
features = pf.transform(features)
X_nojun = pf.transform(X_nojun)
X_jun = pf.transform(X_jun)
print 'trans to polynomial ok'
"""

print features.shape
"""
#特征选择 在哑编码之前
rf = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1)
rf.fit(features, target_train)
sfm = SelectFromModel(rf, threshold=0.01, prefit=True)
#sfm.fit(features, target_train)
features = sfm.transform(features)
X_nojun = sfm.transform(X_nojun)
X_jun = sfm.transform(X_jun)
"""

""" 本来在这，移到前面试试看。结论是不能移到前面，不然的话，经过poly，那他妈的维数"""
#把usrid，merchantid合并进features
features = np.column_stack((features01,features))
X_nojun = np.column_stack((X_nojun01,X_nojun))
X_jun = np.column_stack((X_jun01,X_jun))
"""
Xrf = features.copy()
rf_whole = RandomForestClassifier( max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1) 
rf_whole.fit(Xrf,target_train)
"""
Xrf = features.copy()
#哑编码 One-hot encode
enc = OneHotEncoder(categorical_features = np.array([0,1]) ,handle_unknown ='ignore' )
enc.fit(features)
features = enc.transform(features)
X_nojun = enc.transform(X_nojun)
X_jun = enc.transform(X_jun)

print 'onehot ok', features.shape


#归一化/标准化
scaler = MaxAbsScaler()
#scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)
X_nojun = scaler.transform(X_nojun)
X_jun = scaler.transform(X_jun)
print 'scale ok'


"""
#特征选择
sfm = SelectFromModel(LR(threshold=0.5, C=0.1))
sfm.fit(features, target_train)
features = sfm.transform(features)
X_nojun = sfm.transform(X_nojun)
X_jun = sfm.transform(X_jun)
"""
print 'preprocess ok'


#训练模型
X = features
y = target_train

#rf = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1) 
#rf.fit(X_train,y_train)


#rf_whole = RandomForestClassifier( max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1) 
#rf_whole.fit(X,y)
#ab_whole = AdaBoostClassifier(n_estimators = 7)
#ab_whole.fit(X,y)
#lr = LogisticRegression(class_weight = 'auto', n_jobs=-1)
lr = LogisticRegression( n_jobs=4)
lr.fit(X,y)
#ar = AdaBoostRegressor()
#ar.fit(X,y)
#gbdt = GradientBoostingClassifier(n_estimators=100)
#rfr = RandomForestRegressor(n_estimators=10,n_jobs=-1,verbose=2)
model = lr
print 'training ok'

def giveResultOnTestset():
    #读测试数据。这里是题目给的原始数据，读它是为了保证提交结果的前三列格式不出问题
    df_test = readAsChunks_nohead("ccf_offline_stage1_test_revised.csv",{0:int, 1:int}).replace("null",np.nan)
    df_res = df_test[[0,2,5]]
    #读预处理过的测试集。
    df_test = readAsChunks_hashead("test3.csv",{'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float, '20':float, '15':int, '16':int, '23':int, '24':float, '25':float})
    df_test.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
    df_test[4] = df_test[4].fillna(df_test[4].mean())
    print 'test read in ok'
    #选择特征列
    features_test01, features_test = chooseFeatures(df_test)
    
    #features_test = pf.transform(features_test)
    #features_test = sfm.transform(features_test)
    features_test = np.column_stack((features_test01,features_test))
    features_test = enc.transform(features_test)
    features_test = scaler.transform(features_test)
    
    target_test = model.predict_proba(features_test)[:,1]
    
    
    df_res[4] = pd.DataFrame(target_test)
    #不想要科学计数法的结果
    #对目标series可以这样。注意，程序最开头那个设置pd的option的方法，只能影响控制台输出的格式，不能影响to_csv输出到文件的格式
    #Series(np.random.randn(3)).apply(lambda x: '%.3f' % x)
    df_res[4] = df_res[4].apply(lambda x: '%.15f' % x)

    df_res.to_csv("v0_26 wrong target train.csv",header=None,index=False)
    #print df_res[4].value_counts()
    return df_res
    

df_res = giveResultOnTestset()
print 'A result generated.'
#算auc
#roc_auc_score(y_true, y_scores)
#print cross_val_score(model,X,y,cv=5,scoring='roc_auc',n_jobs=2).mean()
#print cross_val_score(rf,X,y,cv=5,scoring='roc_auc',n_jobs=1).mean()

#算6月平均auc
def calcAucJun():
    #model_nojun = LogisticRegression(n_jobs=-1)
    model_nojun = model
    model_nojun.fit(X_nojun, y_nojun)
    print 'train ok'
    y_predict = model_nojun.predict_proba(X_jun)[:,1]
    print 'predict ok'
    y_predict = pd.DataFrame(y_predict)
    y_predict.columns=[100]
    test_jun_new = pd.concat([test_jun, y_predict], axis=1) #把预测出来的6月结果，合并到6月的完整表的最后一列，方便下面的groupby和计算
    #print test_jun.head()
    #return test_jun, y_predict
    auc_weight_list = []
    auc_list = []
    total = 0
    for name, group in test_jun_new.groupby(test_jun_new[2]):
        if group[11].unique().shape[0] != 1:
            aucvalue = roc_auc_score(group[11].values,group[100].values)
            auc_weight_list.append(  aucvalue*group.shape[0] )
            auc_list.append(aucvalue)
            total = total + group.shape[0]
    return test_jun_new, auc_weight_list, auc_list, total

aucs = []
aucs_w = []
#test_jun_new, aucs_w, aucs, total = calcAucJun()
s_w = pd.Series(aucs_w)
s = pd.Series(aucs)
print 'Weighted Mean auc is : ',s_w.sum()/total
print 'Direct Mean auc is: ',s.mean()