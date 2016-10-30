# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 21:55:55 2016

@author: X93
"""

import pandas as pd
import numpy as np
import seaborn as sns

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
"""
#下面的部分，是在v3.5左右，发现了结果和历史最高分差别明显，怀疑是特征出了问题。于是对比两个版本的预处理结果，发现特征17和23出了大问题。
df13 = readAsChunks_hashead("offline13.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float}) #.replace("null",np.nan)
df13.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df7 = readAsChunks_hashead("offline7.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float}) #.replace("null",np.nan)
df7.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int

df7 = df7[[0,1,2,3,4,17,8,9,10,14,12,13,11,20,15,16,23,24,25]]
df13 = df13[[0,1,2,3,4,17,8,9,10,14,12,13,11,20,15,16,23,24,25]]
delta = df13 - df7
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