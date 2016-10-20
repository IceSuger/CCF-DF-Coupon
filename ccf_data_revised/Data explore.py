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
print train_on.head()
#看看有多少缺失值
#nan_rate = (train_on.isnull().sum() / train_on.shape[0]) * 100

#只考虑完全正的正例，即领券并消费了券的，正例只占offline训练集的全部的4.29%


def fillinFeature4(df):
    df[4] = df[4].astype(float)
    print df[4].isnull().value_counts()
    for name, group in df.groupby(df[0]):
        group[4] = group[4].fillna(group[4].mean())
    print df[4].isnull().value_counts()
    return df
    
df = fillinFeature4(train_off)