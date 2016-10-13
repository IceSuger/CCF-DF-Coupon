# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 23:12:53 2016

@author: X93
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 

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

df = readAsChunks("ccf_offline_stage1_train.csv", {0:int, 1:int,  3:str}).replace("null",np.nan)
#领券日期 和 消费日期 及 消费日期距离领券日期的差
df[5] = pd.to_datetime(df[5])
df[6] = pd.to_datetime(df[6])
df[7] = df[6]-df[5]

#把优惠方式那一列切开
df[[8,9]] = df[3].str.split(':',expand=True).astype(float)
df[10] = 1 - df[9]/df[8]
#df[[2,3,8,9]] = df.fillna(0)

#优惠方式这一列，直接是折扣率的，保留
#df[3] = df[3].replace("\d*:\d*","0")#.astype(float).fillna(0)  #".*"为正则表达式，'.'表示任意字符，'*'表示出现任意次
    #但 replace 方法第一个参数并不是正则表达式，而是字符串。所以不用这种方法来替换字符串了。
#print df[3].value_counts()
#print df[df[3].str.contains(':').fillna(True)]
df[3][df[3].str.contains(':').fillna(True)] = '0'
df[3].astype(float)
print df[3].value_counts()

#print df.notnull()

#正例反例标记
df[11] = 0
#print df[df[6].notnull() & df[2].notnull()]
#df.loc( (df[6].notnull() & df[2].notnull()) ,11)= 1
df[11][ df[6].notnull() & df[2].notnull() ] = 1

#usrid, merchantid, discountrate, man, jian, approxi_discountrate
features = df[[0,1,3,8,9,10]].fillna(0).values
target_train = df[11]

#训练模型
lr = LogisticRegression()
lr.fit(features, target_train)

print lr.score(features, target_train)

#读测试数据
df_test = readAsChunks("ccf_offline_stage1_test_revised.csv",{0:int, 1:int}).replace("null",np.nan)
#把优惠方式那一列切开
df_test[[8,9]] = df_test[3].str.split(':',expand=True).astype(float)
df_test[10] = 1 - df_test[9]/df_test[8]
#优惠方式这一列，直接是折扣率的，保留;其他形式的都变成0
df_test[3][df_test[3].str.contains(':').fillna(True)] = '0'
df_test[3].astype(float)

features_test = df_test[[0,1,3,8,9,10]].fillna(0).values
target_test = lr.predict_proba(features_test)[:,1]

df_res = df_test[[0,2,5]]
df_res[4] = pd.DataFrame(target_test)
df_res.to_csv("result.csv",header=None,index=False)
print df_res[4].value_counts()
