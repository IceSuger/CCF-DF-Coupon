# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 21:55:55 2016

@author: X93
"""

import pandas as pd
import numpy as np
import util
import ManipulatingFeatures as mf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 25, 8

def genDayOfWeek(df):
    df[34] = df[5].apply(lambda x:x.dayofweek)
    df[35] = df[6].apply(lambda x:x.dayofweek)
    return df

df = util.readAsChunks_hashead("..\\ccf_data_revised\\offline16.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float,'18':float,'19':float, '20':float,'21':float,'22':float, '15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float}).replace("null",np.nan)
df.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df[5] = pd.to_datetime(df[5])
df[6] = pd.to_datetime(df[6])
df = genDayOfWeek(df)

#读预处理过的测试集。
df_test = util.readAsChunks_hashead("..\\ccf_data_revised\\test16.csv",{'0':int, '1':int, '4':float, '8':float, '9':float,'10':float,  '17':float,'18':float,'19':float, '20':float,'21':float,'22':float,'15':int, '16':int, '23':int, '24':float, '25':float, '26':float, '27':float, '28':float, '29':float})
df_test.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df_test[5] = pd.to_datetime(df_test[5])



df = mf.markTarget(df)
#df_test = mf.markTarget(df_test)

pos = df[df[31]==1]
neg = df[df[31]==0]

i=35
comp = pd.concat([pos[i].value_counts()/pos.shape[0],neg[i].value_counts()/neg.shape[0]],axis=1)
    #plt.show(comp.plot(kind = 'bar'))
plt.show(comp.plot())

"""
for i in list(df.columns)[2:]:
    comp = pd.concat([pos[i].value_counts()/pos.shape[0],neg[i].value_counts()/neg.shape[0]],axis=1)
    #plt.show(comp.plot(kind = 'bar'))
    plt.show(comp.plot())
"""
"""
    plt.plot(list(pos.index), list(pos[i].value_counts()), 'r.-' ,label='pos'+str(i))
    plt.plot(list(neg.index), list(neg[i].value_counts()), 'g.-' ,label='neg'+str(i))
    plt.show()
"""
"""
    fig= plt.figure()
    ax1 = fig.add_subplot()
    ax1.scatter(comp.index, pos[i].value_counts())
    ax2 = ax1.twinx()
    comp.plot(kind='')
"""


























"""    
train_off = readAsChunks("ccf_offline_stage1_train.csv", {0:int, 1:int}).replace("null",np.nan)
train_on = readAsChunks("ccf_online_stage1_train.csv", {0:int, 1:int, 2:int}).replace("null",np.nan)
test = readAsChunks("ccf_offline_stage1_test_revised.csv", {0:int, 1:int}).replace("null",np.nan)
#print train_on.head()
#看看有多少缺失值
#nan_rate = (train_on.isnull().sum() / train_on.shape[0]) * 100

#只考虑完全正的正例，即领券并消费了券的，正例只占offline训练集的全部的4.29%


def fillinFeature4(df):
    gs=[]
    df[4] = df[4].astype(float)
    print df[4].isnull().value_counts()
    for name, group in df.groupby(df[0]):
        #print name
        group.loc[:,4] = group[4].fillna(group[4].mean())
        gs.append(group)
    print df[4].isnull().value_counts()
    df_g = pd.concat(gs)
    return df_g
    
#df_g = fillinFeature4(train_off)
#df_g.to_csv("ccf_offline_stage1_train_filled4.csv",header=None,index=False)

#下面是观察线上线下训练集里，用户和商户的重复情况
on_usrid = train_on[0].unique()
off_usrid = train_off[0].unique()
on_merid = train_on[1].unique()
off_merid = train_off[1].unique() #返回的是np。ndarray

intersec_usr = np.intersect1d( on_usrid, off_usrid)  #注意这个求交集的方法
intersec_mer = np.intersect1d( on_merid, off_merid)
#结论是，两个训练集中，用户有267448个重复的，商户没有重复的
t_merid = test[1].unique()
t_usrid = test[0].unique()
intersec_on_t_mer = np.intersect1d( on_merid, t_merid)
intersec_off_t_mer = np.intersect1d( off_merid, t_merid)
intersec_off_t_usr = np.intersect1d( off_usrid, t_usrid)
intersec_on_t_usr = np.intersect1d( on_usrid, t_usrid)

#list(set(b).difference(set(a))) #求差集
#上面这几行是拿训练集的做了对比



#sup_17 = pd.DataFrame( df[0].apply(lambda x:df[(df[0]==x & df[11]==1)].shape[0])/df[0].apply(lambda x:df[df[0]==x].shape[0]) )

#下面的部分，是在v3.5左右，发现了结果和历史最高分差别明显，怀疑是特征出了问题。于是对比两个版本的预处理结果，发现特征17和23出了大问题。
df14 = readAsChunks_hashead("offline14.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float}) #.replace("null",np.nan)
df14.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df7 = readAsChunks_hashead("offline7.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float}) #.replace("null",np.nan)
df7.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int

df7 = df7[[0,1,2,3,4,17,8,9,10,14,12,13,11,20,15,16,23,24,25]]
df14 = df14[[0,1,2,3,4,17,8,9,10,14,12,13,11,20,15,16,23,24,25]]
delta = df14 - df7
"""
"""
delta.describe()#发现17和23两个字段明显不同，13版本的值普遍比7版本的要大
df7[17]
df13[17]
df13[23]
df7[23]
delta[23].tail()
delta[23].value_counts()
df13[24]
df13[24].value_counts()
"""
