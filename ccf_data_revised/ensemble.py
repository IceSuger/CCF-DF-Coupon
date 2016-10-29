# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 08:56:42 2016

@author: X93
"""

from __future__ import division #为了让整数相除得到小数，否则只能得到整数
import pandas as pd
import numpy as np

res1 = pd.read_csv("v1_11 no13_n is 96.csv",header=None)
res2 = pd.read_csv("v2_4 no18 19 21 22_balance.csv",header=None)
res3 = pd.read_csv("v1_9 n90.csv",header=None)

res = res1[[0,1,2]]
res[3] = (res1[3]*7+res2[3]*1 + res3[3]*2)/10

res.to_csv("ensembled1.csv",header=None,index=False)