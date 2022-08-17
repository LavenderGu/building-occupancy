#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:10:14 2021

@author: gu
"""
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from chinese_calendar import get_holiday_detail, is_workday
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import datetime as dt
import seaborn as sns
from fitter import Fitter


def check_zero_day_index(df):
    a = df.groupby(df.index.date).apply(lambda x : (x ==0).astype(int).sum(axis=0))
    all0 = {}
    part0 = {}
    for i in range(len(a)):
        allname = []
        pname = []
        for j in range(len(a.columns)):
            if a.iloc[i,j] == 24:
                allname.append(a.columns[j])
            if a.iloc[i,j] > 8 and a.iloc[i,j] < 24:
                pname.append(a.columns[j])
        if len(allname)>0: all0.update({a.index[i]:allname})
        if len(pname)>0: part0.update({a.index[i]:pname})
    return all0,part0
        

def insert_dtype(odf):
    df = odf.copy()
    df.insert(0,'date',pd.to_datetime(df.index).date)
    df.insert(0,'dtype',pd.to_datetime(df.index).date)
    for i in range(len(df)):
        dt = df['dtype'][i]
        if (get_holiday_detail(dt)[0] == True) and (get_holiday_detail(dt)[1] != None):
            df['dtype'][i] = 'Holiday'
        elif dt.weekday()+1 == 1:
            df['dtype'][i] = 'Mon'
        elif dt.weekday()+1 == 2:
            df['dtype'][i] = 'Tue'
        elif dt.weekday()+1 == 3:
            df['dtype'][i] = 'Wed'
        elif dt.weekday()+1 == 4:
            df['dtype'][i] = 'Thu'
        elif dt.weekday()+1 == 5:
            df['dtype'][i] = 'Fri'
        elif dt.weekday()+1 == 6:
            if is_workday(dt) == True:
                df['dtype'][i] = 'Shift'
            else:
                df['dtype'][i] = 'Sat'
        elif dt.weekday()+1 == 7:
            if is_workday(dt) == True:
                df['dtype'][i] = 'Shift'
            else:
                df['dtype'][i] = 'Sun'
    df = df.reset_index(drop = True)
    return df

def avg_data(rdf):
    return rdf.groupby([rdf.dtype,rdf.index.hour]).mean()


if __name__ == '__main__':
    sns.set_theme(style="white")
    
    resultpath = '...'
    opath = '...'
    df1 = pd.read_csv("...").dropna(how='all')
    df2 = pd.read_csv('...').dropna(how='all')
    nightlst = [...]
    #check 0 value and remove
    df1 = df1.fillna(0)
    df2 = df2.fillna(0)
    
    df1 = df1[168:] #2017年1月5日后数据有效
    df1['tm']=pd.to_datetime(df1['tm'])
    df1 = df1.set_index('tm')
    df2['tm']=pd.to_datetime(df2['tm'])
    df2 = df2.set_index('tm')
    
    all0,part0 = check_zero_day_index(df1)
    all0.update({dt.date(2019,1,31):list(df1.columns)})#1.31数据不可信
    part0.pop(dt.date(2019,1,31))
    df11 = insert_dtype(df1)
    
    #1月所有数据为2018.12和2019.2月平均
    df11 = df11.set_index(pd.to_datetime(df1.index))
    tdf1 = df11.loc['2018/12']
    tdf2 = df11.loc['2019/02']
    tdf = ((avg_data(tdf1) + avg_data(tdf2))/2).astype('int')
    for key,value in all0.items():
        d = df11.loc[str(key),'dtype'][0]
        for v in value:
            for i in df11.loc[str(key),v].index:
                k = (d,i.hour)
                df11.loc[i,v] = tdf.loc[k,v]
    #对于其他日期类型则选择前后一周做均值,假期和调休日则选择本月内假期调休日均值
    for key,value in part0.items():
        d = df11.loc[str(key),'dtype'][0]
        if d == 'Holiday' or d == 'Shift':
            tdf = avg_data(df11.loc[df11.index.month==key.month])
            for v in value:
                for i in df11.loc[str(key),v].index:
                    k = (d,i.hour)
                    df11.loc[i,v] = tdf.loc[k,v]
        else:
            for v in value:
                for i in df11.loc[str(key),v].index:
                    df11.loc[i,v] = (df11.loc[i-dt.timedelta(days=7),v]+
                            df11.loc[i+dt.timedelta(days=7),v])/2
    
    # 处理需要修正夜间建筑夜间值21-06点峰值为夜间值
    valuedf = pd.DataFrame()
    for i in np.unique(df11.index.date):
        value1 = df11.loc[(df11.index.date == i) & (df11.index.hour>=21),nightlst]
        value2 = df11.loc[(df11.index.date == i+dt.timedelta(days=1))&(df11.index.hour<=6),nightlst]
        if len(value2) ==0:value = value1.append(value2).max()
        else: value = value1.max()
        valuedf[i]=value
    valuedf = valuedf.T
    for bn in nightlst:
        for i in df11.index:
            if i.hour >= 21:
                df11.loc[i,bn] = valuedf.loc[i.date(),bn]
            elif i.hour<= 6:
                try:
                    df11.loc[i,bn] = valuedf.loc[i.date()-dt.timedelta(days=1),bn]
                except:
                    df11.loc[i,bn] = valuedf.loc[i.date(),bn] #仅限于第一天
            elif i.hour > 6 and i.hour <12:
                try:
                    if df11.loc[i,bn] < valuedf.loc[i.date()-dt.timedelta(days=1),bn]:
                        df11.loc[i,bn] = valuedf.loc[i.date()-dt.timedelta(days=1),bn]
                except:
                    if df11.loc[i,bn] < valuedf.loc[i.date(),bn]:
                        df11.loc[i,bn] = valuedf.loc[i.date(),bn]
    df11.to_csv(resultpath + '/'+'ocombine11.csv')
    




    
    

    
    
    
    