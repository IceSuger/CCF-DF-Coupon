# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 23:12:53 2016

@author: X93
"""
from __future__ import division #为了让整数相除得到小数，否则只能得到整数
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split  
from sklearn import cross_validation, metrics   #Additional     scklearn functions
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
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV   
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 5


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
    return df[[0,1,3,4,8,9,10,12,14,15,16,17,20,23,24,25]]

"""
def giveResultOnTestset():
    #读测试数据。这里是题目给的原始数据，读它是为了保证提交结果的前三列格式不出问题
    df_test = readAsChunks_nohead("ccf_offline_stage1_test_revised.csv",{0:int, 1:int}).replace("null",np.nan)
    df_res = df_test[[0,2,5]]
    #读预处理过的测试集。
    df_test = readAsChunks_hashead("test16.csv",{'0':int, '1':int, '4':float, '8':float, '9':float,'10':float,  '17':float,'18':float,'19':float, '20':float,'21':float,'22':float,'15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float})
    df_test.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
    df_test[4] = df_test[4].fillna(df_test[4].mean())
    print 'test read in ok'
    #选择特征列
    #features_test01, features_test = chooseFeatures(df_test)
    features_test = chooseFeatures(df_test) #.values
    
    #features_test = pf.transform(features_test)
    #features_test = sfm.transform(features_test)
    #features_test = np.column_stack((features_test01,features_test))
    #features_test = enc.transform(features_test)
    #features_test = scaler.transform(features_test)
    
    #target_test = model.predict_proba(features_test)[:,1]
    
    #XGBOOST
    target_test = gbm.predict(xgb.DMatrix(features_test), ntree_limit=gbm.best_iteration)

    
    df_res[4] = pd.DataFrame(target_test)
    #不想要科学计数法的结果
    #对目标series可以这样。注意，程序最开头那个设置pd的option的方法，只能影响控制台输出的格式，不能影响to_csv输出到文件的格式
    #Series(np.random.randn(3)).apply(lambda x: '%.3f' % x)
    df_res[4] = df_res[4].apply(lambda x: '%.15f' % x)

    df_res.to_csv("v3_14 using offline16 & test16 _with18 19 21 22 26 27 28 29_n 100_depth6.csv",header=None,index=False)
    #print df_res[4].value_counts()
    return df_res

#df_res = giveResultOnTestset()
print 'A result generated.'
"""
#画图，看feature importances
#pd.Series(gbm.get_fscore()).sort_values().plot(kind='barh',title='Feature importance')




#####
#v3.17
#####

def calcAuc(cid, y_pred, y_true): #cid: couponids
    conc = pd.concat([cid, y_pred, y_true], axis=1) #把预测出来的结果，合并到完整表的最后一列，方便下面的groupby和计算
    conc.columns = [2,100,31]
    auc_list = []
    for name, group in conc.groupby(conc[2]):
        if group[31].unique().shape[0] != 1:
            aucvalue = roc_auc_score(y_true = group[31].values, y_score = group[100].values) #列31为真实在15天内消费的结果
            auc_list.append(aucvalue)
    s = pd.Series(auc_list)
    print 'Direct Mean auc is: ',s.mean()
    return s.mean()

#1. shuffle stratified K fold
#得到一组分割
#2. for train_index, test_index in sss.split(X, y):
#循环内部，每循环一次，就计算一次平均auc：
#   需要2.1 用Xtrain训练
#       2.2 对Xtest预测，得到y_pred
#       2.3 把couponids,y_pred,y_true 传进calcAuc函数，函数返回一个平均auc
#       2.4 gc
#3. 循环内部，每得到一个auc，就加到list里
#4. 对list求mean。得到CV后的平均平均auc。


#读训练集
df = readAsChunks_hashead("offline16.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float,'18':float,'19':float, '20':float,'21':float,'22':float, '15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float}).replace("null",np.nan)
df.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df[5] = pd.to_datetime(df[5])
df[6] = pd.to_datetime(df[6])

#打上分类标记
df = markTarget(df)
#准备好这几个东西
X = chooseFeatures(df).copy()
y = df[11].copy()
couponids = df[2].copy()
y_true = df[31].copy()

#把df干掉，释放内存
import gc
del df
gc.collect()

#XGB初始化
params = {
        "objective": "binary:logistic",
        #"scale_pos_weight": weight,#设为正例反例比例
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "silent": 0,
        "nthread":4,
        "seed": 27,
    }
num_boost_round = 96

#1. 做shuffle stratified split (如果效果不好，还可以考虑换成 stratified K fold)
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(y, n_iter=10, test_size=0.1, random_state=0)
#2. 开始循环
times=0
mean_auc = []
for train_index, test_index in sss:
    times += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    cid = couponids[test_index]
    y_true_test = y_true[test_index]
    #训练XGB
    dtrain = xgb.DMatrix(X_train, label = y_train.values)
    gbm = xgb.train(params, dtrain, num_boost_round, verbose_eval=True)
    print times, 'training ok'
    #预测
    y_predict = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration)
    print times, 'predict ok'
    y_predict = pd.DataFrame(y_predict)
    #算一个平均AUC
    mean_auc.append( calcAuc(cid, y_predict, y_true_test) )
    #垃圾回收，释放内存
    gc.collect()
#算整个的平均平均AUC值，即伪CV的平均结果
print 'Mean mean auc is ', mean_auc.mean()






