# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:46:13 2016

@author: X93
"""
import pandas as pd
import numpy as np
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")

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
    df[11][ ((df[6].notnull()) & (df[2]>0)) ] =1
    #df[11][ (df[6].notnull()) ] =1
    return df

def fill4(df_origin):
    #填充距离字段的空值
    df = df_origin[[0,4]]
    df[4]=df[4].astype(float)
    grouped = df.groupby(df[0])
    transformed = grouped.transform(lambda x: x.fillna(x.mean()))
    df_origin[4] = transformed.fillna(transformed[4].mean())
    sup4 = transformed
    return df_origin #, sup4

def feature17(df_origin):
    #填入特征17（经过版本v3.1~v3.5的探索，发现这个特征，需要用正确标记——即用券消费才为正例的——来处理，效果才好）
    m = pd.read_csv("sup17.csv",dtype={1:float},index_col=0,header=None ) #sup17曾经叫m
    m.rename(columns={1:17}, inplace=True)
    merged = pd.merge( df_origin, m, left_on=0, right_index=True, how='left' )
    #merged.rename(columns={'0.0':17}, inplace=True)
    return merged
    
def generateSup17(df_origin):
    df = df_origin[[0,11]]
    grouped = df.groupby(df[0])
    m = grouped.mean()
    m.to_csv("sup17.csv",header=None,index=True)

def feature18(df_origin):
    #填入特征18
    m = pd.read_csv("sup18.csv",dtype={1:float},index_col=0,header=None ) 
    m.rename(columns={1:18}, inplace=True)
    merged = pd.merge( df_origin, m, left_on=0, right_index=True, how='left' )
    return merged
    
def generateSup18(df_origin):
    df = df_origin[[0,11]]
    grouped = df.groupby(df[0])
    m = grouped.mean()
    m.to_csv("sup18.csv",header=None,index=True)
    
def feature19(df_origin):
    #填入特征19
    m = pd.read_csv("sup19.csv",dtype={1:float},index_col=0,header=None ) 
    m.rename(columns={1:19}, inplace=True)
    merged = pd.merge( df_origin, m, left_on=0, right_index=True, how='left' )
    return merged
    
def generateSup19(df_origin):
    df = df_origin[[0,11]]
    grouped = df.groupby(df[0])
    m = grouped.mean()
    m.to_csv("sup19.csv",header=None,index=True)
    
def feature20(df_origin):
    #填入特征20
    m = pd.read_csv("sup20.csv",dtype={1:float},index_col=0,header=None )
    m.rename(columns={1:20}, inplace=True)
    merged = pd.merge( df_origin, m, left_on=0, right_index=True, how='left' )
    merged[20] = merged[20].fillna(0)
    return merged
    
def generateSup20(df_origin):
    df = df_origin[[0,11]][(df_origin[2]>0)]
    grouped = df.groupby(df[0])
    m = grouped.mean()
    m.to_csv("sup20.csv",header=None,index=True)
    
def feature21(df_origin):
    #填入特征21
    m = pd.read_csv("sup21.csv",dtype={1:float},index_col=0,header=None )
    m.rename(columns={1:21}, inplace=True)
    merged = pd.merge( df_origin, m, left_on=0, right_index=True, how='left' )
    merged[21] = merged[21].fillna(0)
    return merged
    
def generateSup21(df_origin):
    df = df_origin[[0,11]][(df_origin[2]>0)]
    grouped = df.groupby(df[0])
    m = grouped.mean()
    m.to_csv("sup21.csv",header=None,index=True)
    
def feature22(df_origin):
    #填入特征22
    m = pd.read_csv("sup22.csv",dtype={1:float},index_col=0,header=None )
    m.rename(columns={1:22}, inplace=True)
    merged = pd.merge( df_origin, m, left_on=0, right_index=True, how='left' )
    merged[22] = merged[22].fillna(0)
    return merged
    
def generateSup22(df_origin):
    df = df_origin[[0,11]][(df_origin[2]>0)]
    grouped = df.groupby(df[0])
    m = grouped.mean()
    m.to_csv("sup22.csv",header=None,index=True)
    
def splitDiscountRateCol(df):
    #先处理fixed
    df[3][df[3].str.contains('f').fillna(False)] = '1'
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

def feature30(df):
    df[30] = 0
    df[30][df[3].str.contains('f').fillna(False)] = 1
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

def generateSup15(df_origin):
    #用户在该店消费次数
    df = df_origin.copy()
    df[150] = 0
    df[150][ df[6].notnull() ] =1
    df = df[[0,1,150]]
    grouped = df.groupby([0,1], as_index=False)
    m = grouped.sum()
    m.to_csv("sup15.csv",header=None,index=False)
    
def generateSup16(df_origin):
    #用户在该店领券次数
    df = df_origin.copy()
    df[150] = 0
    df[150][ (df[2]>0) ] =1
    df = df[[0,1,150]]
    grouped = df.groupby([0,1], as_index=False)
    m = grouped.sum()
    m.to_csv("sup16.csv",header=None,index=False)
    
def generateSup23(df_origin):
    #用户在该店用券消费次数
    df = df_origin.copy()
    df[150] = 0
    df[150][ ((df[6].notnull()) & (df[2]>0) )] =1
    df = df[[0,1,150]]
    grouped = df.groupby([0,1], as_index=False)
    m = grouped.sum()
    m.to_csv("sup23.csv",header=None,index=False)

def feature15(df_origin):
    #填入特征15
    m = pd.read_csv("sup15.csv",dtype={2:int},header=None )
    m.rename(columns={2:15}, inplace=True)
    merged = pd.merge( df_origin, m, on=[0,1], how='left' )
    merged[15] = merged[15].fillna(0)
    return merged

def feature16(df_origin):
    #填入特征16
    m = pd.read_csv("sup16.csv",dtype={2:int},header=None )
    m.rename(columns={2:16}, inplace=True)
    merged = pd.merge( df_origin, m, on=[0,1], how='left' )
    merged[16] = merged[16].fillna(0)
    return merged
    
def feature23(df_origin):
    #填入特征23
    m = pd.read_csv("sup23.csv",dtype={2:int},header=None )
    m.rename(columns={2:23}, inplace=True)
    merged = pd.merge( df_origin, m, on=[0,1], how='left' )
    merged[23] = merged[23].fillna(0)
    return merged

def feature24and25(df):
    df[24] = (df[23]/df[15]).fillna(0)
    df[25] = (df[23]/df[16]).fillna(0)
    return df
    
def feature26(df_origin):
    #填入特征26(能用字段11来算，是基于错误标记——把消费了的都标记为正例了！)
    m = pd.read_csv("sup26.csv",dtype={1:float},index_col=0,header=None ) 
    m.rename(columns={1:26}, inplace=True)
    merged = pd.merge( df_origin, m, left_on=1, right_index=True, how='left' )
    return merged
    
def generateSup26(df_origin):
    df = df_origin[[1,11]][ ( (df_origin[11]==1) & (df_origin[2]>0) ) ]
    grouped = df.groupby(df[1])
    m = grouped.sum()
    m.to_csv("sup26.csv",header=None,index=True)
    
def feature27(df_origin):
    #填入特征27(能用字段11来算，是基于错误标记——把消费了的都标记为正例了！)
    m = pd.read_csv("sup27.csv",dtype={1:float},index_col=0,header=None ) 
    m.rename(columns={1:27}, inplace=True)
    merged = pd.merge( df_origin, m, left_on=1, right_index=True, how='left' )
    return merged
    
def generateSup27(df_origin):
    df = df_origin[[1,11]]
    grouped = df.groupby(df[1])
    m = grouped.sum()
    m.to_csv("sup27.csv",header=None,index=True)

def feature28(df):
    df[28] = (df[26]/df[27]).fillna(0)
    return df
    
def feature29(df_origin):
    #填入特征29(能用字段11来算，是基于错误标记——把消费了的都标记为正例了！)
    m = pd.read_csv("sup29.csv",dtype={1:float},index_col=0,header=None ) 
    m.rename(columns={1:29}, inplace=True)
    merged = pd.merge( df_origin, m, left_on=1, right_index=True, how='left' )
    return merged
    
def generateSup29(df_origin):
    df = df_origin[[0,1,11]][ (df_origin[11]==1) ]
    grouped = df.groupby(df[1])
    m = grouped.agg({0: lambda x:x.unique().shape[0]})
    m.to_csv("sup29.csv",header=None,index=True)


#train_off = readAsChunks_nohead("ccf_offline_stage1_train.csv", {0:int, 1:int}).replace("null",np.nan)
#train_off = readAsChunks_hashead("offline13.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float}) #.replace("null",np.nan)
#train_off.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int

"""
off13 = readAsChunks_hashead("offline13.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float}) #.replace("null",np.nan)
off13.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int

off13 = off13[[0,1,2,3,4,5,6,8,9,10,14,12,13,20,15,16,18,19,21,22]]
"""
"""
df = markTarget(train_off) #feature/target 11
#off13 = markTarget(off13)

generateSup17(df)
generateSup23(df)
"""
"""
off13 = feature17(off13)
off13 = feature23(off13) 
save = feature24and25(off13)
save.to_csv("offline14.csv",index=False)
"""



"""
train_off = process5and6(train_off) #feature5,6
#train_off = feature7(train_off)    #feature7
train_off = processDate(train_off) #feaeture14

#为了计算当前商户在训练集中出现的频率：当前商户次数/训练集长度，在这里计算一下这个频率，存入sup(辅助)
#放在函数外面，是为了训练集、测试集都可以用它来merge
sup_for_feature12 = pd.DataFrame(train_off[1].value_counts()/train_off.shape[0] *100) #百分比例
size = train_off[(train_off[2]>0)].shape[0]
sup_for_feature13 = pd.DataFrame(train_off[1].value_counts()/size *100) #百分比例

train_off = addFreqOfMerchant(train_off,sup_for_feature12, sup_for_feature13) #feature12,13
df = markTarget(train_off) #feature/target 11
#df = fill4(df)     #feature4

print 'targeted'
#save = feature17(df)   #feature17
df = splitDiscountRateCol(train_off)   #feature3,8,9,10

generateSup15(df)    #先生成特征??的辅助文件
generateSup16(df)
generateSup17(df)
generateSup20(df)
generateSup23(df)
print 'sups generated'
df = feature15(df)   #feature??
df = feature16(df) 
df = feature17(df)
df = feature20(df)
df = feature23(df) 
save = feature24and25(df)

#处理一下col7.有必要吗？其实没有。反正后面也不用它。算了，不处理了。
"""
"""
df = markTarget(train_off) #feature/target 11

generateSup26(df)
generateSup27(df)
generateSup29(df)
df = feature26(df)
df = feature27(df)
df = feature28(df)
df = feature29(df) 
save = df
save.to_csv("offline13.csv",index=False)
print 'saved'
"""


"""
#下面处理测试集
df_test = readAsChunks_nohead("ccf_offline_stage1_test_revised.csv",{0:int, 1:int}).replace("null",np.nan)
#df_test.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int

df_test = feature17(df_test)
print '17 ok'
df_test = feature20(df_test)
print '20 ok'
df_test[5] = pd.to_datetime(df_test[5],format='%Y%m%d')
df_test = addFreqOfMerchant(df_test, sup_for_feature12, sup_for_feature13)
df_test = splitDiscountRateCol(df_test)
df_test = processDate(df_test)
#df_test = df_test.fillna(df_test.mean()) #下一次尝试[17:13]，对于4的缺失值单独处理
"""

"""
df_test = readAsChunks_hashead("test13.csv",{'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float, '20':float})
df_test.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df_test = df_test[[0,1,2,3,4,5,8,9,10,14,12,13,20,15,16,18,19,21,22]]

df_test = feature17(df_test)
df_test = feature23(df_test) 
df_test = feature24and25(df_test)
df_test.to_csv("test14.csv",index=False)
"""

"""
df_test = feature15(df_test)   #feature??
df_test = feature16(df_test) 
df_test = feature23(df_test) 
df_test = feature24and25(df_test)
"""
"""
df_test = feature26(df_test) 
df_test = feature27(df_test) 
df_test = feature28(df_test) 
df_test = feature29(df_test)
df_test.to_csv("test13.csv",index=False)
"""


"""
#下面处理线上训练集
train_on = readAsChunks_nohead("ccf_online_stage1_train.csv", {0:int, 1:int}).replace("null",np.nan)
train_on.columns = [0,1,4,2,3,5,6]
df = feature30(train_on)
df = splitDiscountRateCol(df)
df = markTarget(df) #feature/target 11

generateSup18(df)
generateSup21(df)

df = feature18(df)
df = feature21(df)

train_off = readAsChunks_hashead("offline10.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float}) #.replace("null",np.nan)
train_off.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
train_off = markTarget(train_off) #feature/target 11

df_off = train_off
df_off = feature18(df_off)
df_off = feature21(df_off)

#下面把线上线下的表连起来，为了算字段19和22
df_onoff = pd.concat([df,train_off],axis=0)

df_onoff = df_onoff[[0,1,2,11]]
generateSup19(df_onoff)
generateSup22(df_onoff)
df_off = feature19(df_off)
df_off = feature22(df_off)
df_off.to_csv("offline12.csv",index=False)
"""
def sortcols(df):
    cols = list(df)
    cols.sort()
    return df.ix[:,cols]
    

#下面的部分，是在v3.5左右，发现了结果和历史最高分差别明显，怀疑是特征出了问题。于是对比两个版本的预处理结果，发现特征17和23出了大问题。
df14 = readAsChunks_hashead("offline14.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float}) #.replace("null",np.nan)
df14.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df7 = readAsChunks_hashead("offline7.csv", {'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float}) #.replace("null",np.nan)
df7.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
df14 = sortcols(df14)
df7 = sortcols(df7)

"""
df7 = df7[[0,1,2,3,4,17,8,9,10,14,12,13,11,20,15,16,23,24,25]]
df14 = df14[[0,1,2,3,4,17,8,9,10,14,12,13,11,20,15,16,23,24,25]]

delta = df14 - df7
"""

#老子是真没办法了，直接把好使的offline7里的某些字段拷过来放到新特征矩阵里吧，不重新处理了，他妈的没法再现了。
#df15 = pd.concat([df14.drop([14,17,23,24,25],axis=1),df7[[14,17,23,24,25]]],axis=1)
#df15.to_csv("offline15.csv",index=False)

#下面把测试集也处理出来一个test15版本
test14 = readAsChunks_hashead("test14.csv",{'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float, '20':float})
test14.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
test7 = readAsChunks_hashead("test3.csv",{'0':int, '1':int, '4':float, '8':float, '9':float,'10':float, '17':float, '20':float})
test7.rename(columns=lambda x:int(x), inplace=True) #因为读文件时直接读入了列名，但是是str类型，这里统一转换成int
test14 = sortcols(test14)
test7 = sortcols(test7)

#test15 = pd.concat([test14.drop([14,17,23,24,25],axis=1),test7[[14,17,23,24,25]]],axis=1)
#test15.to_csv("test15.csv",index=False)

#真是逼急眼了没辙了。
df16 = pd.concat([df7, df14[[18,19,21,22,26,27,28,29]]],axis=1)
df16.to_csv("offline16.csv",index=False)
test16 = pd.concat([test7, test14[[18,19,21,22,26,27,28,29]]],axis=1)
test16.to_csv("test16.csv",index=False)