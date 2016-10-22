# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:46:13 2016

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

#train_off.rename(columns={7: 11}, inplace=True)
#train_on = readAsChunks("ccf_online_stage1_train.csv", {0:int, 1:int, 2:int}).replace("null",np.nan)
#test = readAsChunks("ccf_offline_stage1_test_revised.csv", {0:int, 1:int}).replace("null",np.nan)


def markTarget(df):
    #正例反例标记
    df[11] = 0
    df[11][ df[6].notnull() & df[2].notnull()] =1
    return df

def fill4(df_origin):
    #填充距离字段的空值
    df = df_origin[[0,4]]
    df[4]=df[4].astype(float)
    grouped = df.groupby(df[0])
    transformed = grouped.transform(lambda x: x.fillna(x.mean()))
    df_origin[4] = transformed.fillna(transformed[4].mean())
    return df_origin

def feature17(df_origin):
    #填入特征17
    m = pd.read_csv("m.csv",dtype={1:float},index_col=0,header=None )
    m.rename(columns={1:17}, inplace=True)
    merged = pd.merge( df_origin, m, left_on=0, right_index=True, how='left' )
    #merged.rename(columns={'0.0':17}, inplace=True)
    return merged
    
def generateSup17(df_origin):
    df = df_origin[[0,11]]
    grouped = df.groupby(df[0])
    m = grouped.mean()
    m.to_csv("m.csv",header=None,index=True)
    
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
    
    return df    

def processDate(df):
    #领券日期 是当月的几号
    df[14] = df[5].apply(lambda x:x.day)
    return df
    
def process5and6(df):
    df[5] = pd.to_datetime(df[5],format='%Y%m%d')
    df[6] = pd.to_datetime(df[6],format='%Y%m%d')
    #df[5] = pd.to_datetime(df[5])
    #df[6] = pd.to_datetime(df[6])
    return df

def addFreqOfMerchant(df,sup_for_feature12, sup_for_feature13):
    #df = df[df[2].notnull()]
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
    #print merged.head()
    #print merged.columns
    merged = pd.merge( merged, sup_for_feature13, left_on=u'1_x', right_index=True, how='left' )
    print 'merge2 '
    #print merged.head()
    #print merged.columns
    #print merged
    merged.rename(columns={1:13, u'1_x':1, u'1_y':12}, inplace=True)
    return merged
    
def feature7(df):
    #领券日期 和 消费日期 及 消费日期距离领券日期的差
    df[7] = df[6]-df[5]
    return df

def testSetPreprocess(df):
    df = df
    
#train_off = readAsChunks("ccf_offline_stage1_train.csv", {0:int, 1:int}) #.replace("null",np.nan)
train_off = readAsChunks_hashead("offline4.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float}) #.replace("null",np.nan)
train_off.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int

#train_off = process5and6(train_off) #feature5,6
#train_off = feature7(train_off)    #feature7
#train_off = processDate(train_off) #feaeture14

#为了计算当前商户在训练集中出现的频率：当前商户次数/训练集长度，在这里计算一下这个频率，存入sup(辅助)
#放在函数外面，是为了训练集、测试集都可以用它来merge
sup_for_feature12 = pd.DataFrame(train_off[1].value_counts()/train_off.shape[0] *100) #百分比例
size = train_off[(train_off[2]>0)].shape[0]
sup_for_feature13 = pd.DataFrame(train_off[1].value_counts()/size *100) #百分比例

#train_off = addFreqOfMerchant(train_off,sup_for_feature12, sup_for_feature13) #feature12,13
#df = markTarget(train_off) #feature/target 11
#df = fill4(df)     #feature4

print 'filled'
#save = feature17(df)   #feature17
print 'added 17'
#df = splitDiscountRateCol(train_off)   #feature3,8,9,10


#save.to_csv("offline3.csv",index=False)

#下面处理测试集
df_test = readAsChunks_nohead("ccf_offline_stage1_test_revised.csv",{0:int, 1:int}).replace("null",np.nan)
#df_test.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int

df_test = feature17(df_test)
df_test[5] = pd.to_datetime(df_test[5],format='%Y%m%d')
df_test = addFreqOfMerchant(df_test, sup_for_feature12, sup_for_feature13)
df_test = splitDiscountRateCol(df_test)
df_test = processDate(df_test)
df_test.to_csv("test.csv",index=False)
