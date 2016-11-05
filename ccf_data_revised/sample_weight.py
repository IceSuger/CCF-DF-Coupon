# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 12:37:28 2016

@author: X93
"""
import pandas as pd
import numpy as np

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
    
def genWeightOfTargetMark(df):
    #反例权重设置
    df[101] = 1
    #消费了(，没用券的)，赋值0.6
    df[101][ df[6].notnull()] = 0.8
    #消费了，用券了，但超出15天了，赋值0.85
    df[101][ ( (df[6].notnull()) & (df[2]>0) )] = 0.9
    #处理一下日期，便于找出严格正例
    df[5] = pd.to_datetime(df[5])
    df[6] = pd.to_datetime(df[6])
    df[7] = df[6]-df[5]
    #严格的正例，赋值1
    df[101][( ( (df[6].notnull()) & (df[2]>0) )& (df[7].astype('timedelta64[D]').fillna(200).astype('int')<16))] = 1
    
    #领券了，没消费的，赋值0.3
    df[101][ ((df[6].isnull() )&(df[2]>0))] = 0.65

    #df[11][ (df[6].notnull() & df[2].notnull() )& (df[7].astype('timedelta64[D]').fillna(200).astype('int')<16)] = 1
    #df[11][ df[6].notnull() & df[2].notnull()] =1
    #df[11][ ( (df[6].notnull()) & (df[2]>0) )] =1
    return pd.DataFrame(df[101])


df = readAsChunks_nohead("ccf_offline_stage1_train.csv", {0:int, 1:int}).replace("null",np.nan)
sample_weight = genWeightOfTargetMark(df)
sample_weight.to_csv("sample_weight.csv",header=None,index=False)