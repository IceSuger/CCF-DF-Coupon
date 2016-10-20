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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel

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
    #df[11][ (df[6].notnull() & df[2].notnull() )& (df[7].astype('timedelta64[D]').fillna(200).astype('int')<16)] = 1
    df[11][ df[6].notnull() & df[2].notnull()] =1
    return df
    
def addFreqOfMerchant(df,sup_for_feature12, sup_for_feature13):
    df = df[df[2].notnull()]
    #当前商户在训练集中出现的频率：当前商户次数/训练集长度
    """
    chunks = []
    for name, group in df.groupby(df[1]):
        #merchant_freq = group.shape[0]/df.shape[0]
        group[12] = group.shape[0]/df.shape[0]
        chunks.append(group)
    df = pd.concat(chunks)
    """
    #df_sup = pd.DataFrame( [pd.Series(df[1].value_counts().index),  df[1].value_counts() / df.shape[0] ] )
    #print df_sup
    #df = df.merge(df_sup, left_on=1, right_on=1)
    merged = pd.merge( df, sup_for_feature12, left_on=1, right_index=True, how='left' )   
    print 'merge1 '
    print merged.head()
    print merged.columns
    merged = pd.merge( merged, sup_for_feature13, left_on=u'1_x', right_index=True, how='left' )
    print 'merge2 '
    print merged.head()
    print merged.columns
    #print merged
    
    return merged

def processDate(df):
    #领券日期 是当月的几号
    df[14] = df[5].apply(lambda x:x.day)
    return df

df = readAsChunks("ccf_offline_stage1_train.csv", {0:int, 1:int,  3:str}).replace("null",np.nan)
df[5] = pd.to_datetime(df[5])
df[6] = pd.to_datetime(df[6])
#为了计算当前商户在训练集中出现的频率：当前商户次数/训练集长度，在这里计算一下这个频率，存入sup(辅助)
#放在函数外面，是为了训练集、测试集都可以用它来merge
#sup = pd.DataFrame(df[1].value_counts()) #次数
#sup = pd.DataFrame(df[1].value_counts()/df.shape[0]) #比例
#sup = pd.DataFrame(df[1].value_counts()/df.shape[0] *10) #千分比例
sup_for_feature12 = pd.DataFrame(df[1].value_counts()/df.shape[0] *100) #百分比例
df = df[df[2].notnull()]
sup_for_feature13 = pd.DataFrame(df[1].value_counts()/df.shape[0] *100) #百分比例
#sup = pd.DataFrame(df[1].value_counts()/df.shape[0] *1000) #十分比例

df = addFreqOfMerchant(df,sup_for_feature12, sup_for_feature13)
df.columns=[0,1,2,3,4,5,6,12,13]
df = splitDiscountRateCol(df)
df = processDate(df)
#领券日期 和 消费日期 及 消费日期距离领券日期的差
df[7] = df[6]-df[5]
#打上分类标记
df = markTarget(df)
print df.columns



#统一为一个函数，在这里面统一选择作为特征的列
def chooseFeatures(df):
    #先把列顺序排好
    cols = list(df)
    cols.sort()
    df = df.ix[:,cols]
    #然后选出这些列作为特征，具体含义见FeatureExplaination.txt
    return df[[0,1]],df[[3,4,8,9,10,12,13,14]].fillna(0).values
    
#usrid, merchantid, discountrate, man, jian, approxi_discountrate
#features = df[[0,1,3,4,8,9,10,12,13,14]].fillna(0).values
features01, features = chooseFeatures(df)
target_train = df[11]

#下面把六月的拆出来，作为线下算平均auc的测试集
print df.head()
#train_nojun = df.drop(df[df[5].apply(lambda x:x.month)==6] , axis=1)
train_nojun = df[df[5].apply(lambda x:x.month)!=6]
#X_nojun = train_nojun[[0,1,3,4,8,9,10,12,13,14]].fillna(0).values
X_nojun01, X_nojun = chooseFeatures(train_nojun)
y_nojun = train_nojun[11]

test_jun = df[df[5].apply(lambda x:x.month)==6]
y_jun = test_jun[11]
#X_jun = test_jun[[0,1,3,4,8,9,10,12,13,14]].fillna(0).values
X_jun01, X_jun = chooseFeatures(test_jun)


#归一化/标准化
scaler = MaxAbsScaler()
scaler.fit(features)
features = scaler.transform(features)
X_nojun = scaler.transform(X_nojun)
X_jun = scaler.transform(X_jun)
print 'scale ok'
#多项式数据变换
pf = PolynomialFeatures()
pf.fit(features)
features = pf.transform(features)
X_nojun = pf.transform(X_nojun)
X_jun = pf.transform(X_jun)
print 'trans to polynomial ok'
print features.shape
#特征选择 在哑编码之前
rf = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1)
rf.fit(features, target_train)
sfm = SelectFromModel(rf, threshold=0.01, prefit=True)
#sfm.fit(features, target_train)
features = sfm.transform(features)
X_nojun = sfm.transform(X_nojun)
X_jun = sfm.transform(X_jun)

#把usrid，merchantid合并进features
features = np.column_stack((features01,features))
X_nojun = np.column_stack((X_nojun01,X_nojun))
X_jun = np.column_stack((X_jun01,X_jun))

#哑编码 One-hot encode
enc = OneHotEncoder(categorical_features = np.array([0,1]) )
enc.fit(features)
features = enc.transform(features)
X_nojun = enc.transform(X_nojun)
X_jun = enc.transform(X_jun)


"""

"""

#特征选择辅助函数
class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        #权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        #使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        #训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        #训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self
        """
#特征选择
sfm = SelectFromModel(LR(threshold=0.5, C=0.1))
sfm.fit(features, target_train)
features = sfm.transform(features)
X_nojun = sfm.transform(X_nojun)
X_jun = sfm.transform(X_jun)
"""
print 'selection ok'


#训练模型
X = features
y = target_train

#rf = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1) 
#rf.fit(X_train,y_train)


#rf_whole = RandomForestClassifier(class_weight = 'auto', max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1, n_jobs=-1) 
#rf_whole.fit(X,y)
#ab_whole = AdaBoostClassifier(n_estimators = 7)
#ab_whole.fit(X,y)
lr = LogisticRegression(n_jobs=-1)
lr.fit(X,y)
#ar = AdaBoostRegressor()
#ar.fit(X,y)
#gbdt = GradientBoostingClassifier(n_estimators=100)
#rfr = RandomForestRegressor(n_estimators=10,n_jobs=-1,verbose=2)
model = lr


def giveResultOnTestset():
    #读测试数据
    df_test = readAsChunks("ccf_offline_stage1_test_revised.csv",{0:int, 1:int}).replace("null",np.nan)
    df_res = df_test[[0,2,5]]    
    df_test[5] = pd.to_datetime(df_test[5])
    df_test = addFreqOfMerchant(df_test, sup_for_feature12, sup_for_feature13)
    df_test.columns=[0,1,2,3,4,5,12,13]
    df_test = splitDiscountRateCol(df_test)
    df_test = processDate(df_test)
    
    #features_test = df_test[[0,1,3,8,9,10]].fillna(0).values
    df_test = df_test[df_test[2].notnull()]
    
    #features_test = df_test[[0,1,3,4,8,9,10,12,13,14]].fillna(0).values
    features_test01, features_test = chooseFeatures(df_test)
    features_test = scaler.transform(features_test)
    features_test = pf.transform(features_test)
    features_test = sfm.transform(features_test)
    features_test = np.column_stack((features_test01,features_test))
    features_test = enc.transform(features_test)
    

    #target_test = ar.predict(features_test)
    target_test = model.predict_proba(features_test)[:,1]
    
    
    df_res[4] = pd.DataFrame(target_test)
    df_res.to_csv("v0_19_lr_poly_selected.csv",header=None,index=False)
    print df_res[4].value_counts()
    return df_res

#df_res = giveResultOnTestset()
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
    y_predict.columns=[15]
    test_jun_new = pd.concat([test_jun, y_predict], axis=1) #把预测出来的6月结果，合并到6月的完整表的最后一列，方便下面的groupby和计算
    #print test_jun.head()
    #return test_jun, y_predict
    auc_list = []
    for name, group in test_jun_new.groupby(test_jun_new[2]):
        if group[11].unique().shape[0] != 1:
            auc_list.append( roc_auc_score(group[11].values,group[15].values) )
    return test_jun_new, auc_list

aucs = []
test_jun_new, aucs = calcAucJun()
s = pd.Series(aucs)
print 'poly& select. Mean auc is : ',s.mean()