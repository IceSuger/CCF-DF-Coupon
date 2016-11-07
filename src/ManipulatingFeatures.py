# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 15:16:56 2016

@author: X93
"""
#正例反例标记
def markTarget(df):
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
    #return df[[0,1,3,4,8,9,10,12,14,15,16,17,20,23,24,25]]
    #v3.18的特征：    
    #return df[[3,4,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
    return df[[0,1,3,4,8,9,10,12,14,15,16,17,20,21,22,23,24,25,28,29]]
    