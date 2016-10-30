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

#运行之前，先把下面这两行拷到console里运行一下
#import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


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


def modelfit(alg, X,y, useTrainCV=True, cv_folds=5):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X.values, label=y.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['auc'], #网上有文章说这个地方一定要放进一个list里，不然会有问题
            early_stopping_rounds=50, verbose_eval =1)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X, y,eval_metric='auc')
    
    #Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(y.values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob)
    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='barh', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


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
    

df = readAsChunks_hashead("offline14.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float,'18':float,'19':float, '20':float,'21':float,'22':float, '15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float}).replace("null",np.nan)
df.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df[5] = pd.to_datetime(df[5])
df[6] = pd.to_datetime(df[6])
#为了提高表现，有必要把字段14的空值填上了，毕竟他妈的在训练集里缺了将近一半
df[14] = df[14].fillna(35)

#打上分类标记
df = markTarget(df)

#统一为一个函数，在这里面统一选择作为特征的列
def chooseFeatures(df):
    #先把列顺序排好
    cols = list(df)
    cols.sort()
    df = df.ix[:,cols]
    #然后选出这些列作为特征，具体含义见FeatureExplaination.txt
    #return df[[0,1]],df[[3,4,8,9,10,12,14,17,20]] #.fillna(0).values
    
    #return df[[1,3,8,9,10,12,14,15,16,17,20,23,24,25]] #根据fscore，从下面这行里选出的比较重要的特征
    return df[[0,1,3,4,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
    
#features01, features = chooseFeatures(df)
features = chooseFeatures(df)
target_train = df[11]

#下面把六月的拆出来，作为线下算平均auc的测试集
#print df.head()
#train_nojun = df.drop(df[df[5].apply(lambda x:x.month)==6] , axis=1)
train_nojun = df[df[5].apply(lambda x:x.month)!=6]
#X_nojun01, X_nojun = chooseFeatures(train_nojun)
X_nojun = chooseFeatures(train_nojun)
y_nojun = train_nojun[11]

test_jun = df[df[5].apply(lambda x:x.month)==6]
y_jun = test_jun[31]
#X_jun01, X_jun = chooseFeatures(test_jun)
X_jun = chooseFeatures(test_jun)


print features.shape

""" 本来在这，移到前面试试看。结论是不能移到前面，不然的话，经过poly，那他妈的维数
#把usrid，merchantid合并进features
features = np.column_stack((features01,features))
X_nojun = np.column_stack((X_nojun01,X_nojun))
X_jun = np.column_stack((X_jun01,X_jun))
"""
"""
Xrf = features.copy()
rf_whole = RandomForestClassifier( max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1) 
rf_whole.fit(Xrf,target_train)
"""
Xrf = features.copy()
"""
#哑编码 One-hot encode
enc = OneHotEncoder(categorical_features = np.array([0,1,2,4,5,8]) ,handle_unknown ='ignore' )
enc.fit(features)
features = enc.transform(features)
X_nojun = enc.transform(X_nojun)
X_jun = enc.transform(X_jun)
"""
print 'onehot ok', features.shape

"""
#归一化/标准化
scaler = MaxAbsScaler()
#scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)
X_nojun = scaler.transform(X_nojun)
X_jun = scaler.transform(X_jun)
print 'scale ok'
"""

"""
#尝试哑编码一波？我猜会死机。
#pf = PolynomialFeatures()
pf.fit(features)
features = pf.transform(features)
X_nojun = pf.transform(X_nojun)
X_jun = pf.transform(X_jun)
print 'poly ok'
"""

print 'preprocess ok'


#训练模型
X = features
y = target_train

#model = lr
print 'training ok'



#XGB
weight = df[(df[11]==1)].shape[0]/df[(df[11]==0)].shape[0]
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
num_boost_round = 130
#features = features.values
target_train = target_train.values
dtrain = xgb.DMatrix(features,label = target_train)
gbm = xgb.train(params, dtrain, num_boost_round, verbose_eval=True)

"""
#XGB TUNE
#Choose all predictors except target & IDcols
#predictors = [x for x in train.columns if x not in [11]]
xgb1 = XGBClassifier(
     learning_rate =0.05,
     n_estimators=1000, #好像在eta 0.1时测出来是最好为417
     max_depth=7,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)
modelfit(xgb1, X,y)
"""



print 'xgb train ok'

def giveResultOnTestset():
    #读测试数据。这里是题目给的原始数据，读它是为了保证提交结果的前三列格式不出问题
    df_test = readAsChunks_nohead("ccf_offline_stage1_test_revised.csv",{0:int, 1:int}).replace("null",np.nan)
    df_res = df_test[[0,2,5]]
    #读预处理过的测试集。
    df_test = readAsChunks_hashead("test14.csv",{'0':int, '1':int, '4':float, '8':float, '9':float,'10':float,  '17':float,'18':float,'19':float, '20':float,'21':float,'22':float,'15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float})
    df_test.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
    #df_test[4] = df_test[4].fillna(df_test[4].mean())
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

    df_res.to_csv("v3_6 all useful features_n130.csv",header=None,index=False)
    #print df_res[4].value_counts()
    return df_res

df_res = giveResultOnTestset()
print 'A result generated.'
#算auc
#roc_auc_score(y_true, y_scores)
#print cross_val_score(model,X,y,cv=5,scoring='roc_auc',n_jobs=2).mean()
#print cross_val_score(rf,X,y,cv=5,scoring='roc_auc',n_jobs=1).mean()

#算6月平均auc
#v1.5尝试 先把X_nojun的userid统计出来放在uid_nojun这个list中
#uid_nojun = X_nojun[0].unique()
mid_nojun = X_nojun[1].unique()
#然后下面算auc的时候，只统计uid属于这个列表的
def calcAucJun():
    #model_nojun = model
    #model_nojun.fit(X_nojun, y_nojun)
    
    #xgb
    dtrain = xgb.DMatrix(X_nojun,label = y_nojun)
    gbm = xgb.train(params, dtrain, num_boost_round, verbose_eval=True)
    
    print 'train ok'
    #y_predict = model_nojun.predict_proba(X_jun)[:,1]
    y_predict = gbm.predict(xgb.DMatrix(X_jun), ntree_limit=gbm.best_iteration)

    print 'predict ok'
    y_predict = pd.DataFrame(y_predict)
    y_predict.columns=[100] #列100为针对六月预测出的结果
    test_jun_new = pd.concat([test_jun, y_predict], axis=1) #把预测出来的6月结果，合并到6月的完整表的最后一列，方便下面的groupby和计算
    
    #v1.5的尝试，在这里增加了一句，只取uid属于上面uid_nojun的来看
    #test_jun_new = test_jun_new[test_jun_new[0].isin(uid_nojun)]
    #test_jun_new = test_jun_new[test_jun_new[1].isin(uid_nojun)]
    print 'start calc mean auc'
    auc_weight_list = []
    auc_list = []
    total = 0
    for name, group in test_jun_new.groupby(test_jun_new[2]):
        if group[31].unique().shape[0] != 1:
            aucvalue = roc_auc_score(group[31].values,group[100].values) #列31为真实在15天内消费的结果
            auc_weight_list.append(  aucvalue*group.shape[0] )
            auc_list.append(aucvalue)
            total = total + group.shape[0]
    return test_jun_new, auc_weight_list, auc_list, total

aucs = []
aucs_w = []
total = 1
test_jun_new, aucs_w, aucs, total = calcAucJun()
s_w = pd.Series(aucs_w)
s = pd.Series(aucs)
print 'Weighted Mean auc is : ',s_w.sum()/total
print 'Direct Mean auc is: ',s.mean()
#画图，看feature importances
#pd.Series(gbm.get_fscore()).sort_values().plot(kind='barh',title='Feature importance')