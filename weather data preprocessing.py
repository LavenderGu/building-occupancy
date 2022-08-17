#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 21:37:49 2022

@author: gu
"""

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.forecasting.stl import STLForecast
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection, metrics   #交叉验证和效果评估，其他模型也很常用。
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def finding_missing_datetime(start,end,freq,compare):
    totaltime = pd.date_range(start = start,end = end,freq = freq)
    totaltime = pd.DataFrame(totaltime,columns = ['tm'])
    totaltime.index = totaltime['tm']
    newdf = totaltime.merge(compare, how='left', left_index=True, right_index=True)
    missingDate = newdf[newdf.isnull().values==True].drop_duplicates().index
    return missingDate,newdf


    
def imputating_mv_bistlarima(data,freq = 24,method = 'stl',bi = True):
    missingDate = data[data.isnull().values==True].index
    mdlst = []
    impute_fw = data.copy()
    impute_bw = data.copy().sort_index(ascending = False)
    #判断连续缺失值状况以及位置,按照缺失划分数据
    a = [missingDate[0]]
    f = 24/freq
    for d in missingDate[1:]:
        if (d - a[-1]).total_seconds()/3600 == f:
            a.append(d)
        else:
            mdlst.append(a)
            a = [d]
        if d == missingDate[-1]:
            mdlst.append(a)
    #进行stl-arima正向预测，如果缺失值在开头则只能单向
    for i in range(len(mdlst)):
        print(mdlst[i])
        v = impute_fw[impute_fw.index < mdlst[i][0]]
        if method =='stl' :
            stlf = STLForecast(v, ARIMA, period = freq, model_kwargs=dict(order=(1, 1, 0), trend="t"))
            stlf_res = stlf.fit()
            forecast = stlf_res.forecast(len(mdlst[i])) # 预测
        if method =='sarima' :
            # to do
            pass
        impute_fw[mdlst[i]] = forecast #用填充后的数据进行后续缺失值填充过程
    #反向预测
    print('start backward imputating')
    if bi == True:
        for i in range(len(mdlst)-1,-1,-1):
            print(mdlst[i])
            v = impute_bw[impute_bw.index > mdlst[i][-1]]
            if method =='stl' :
                stlf = STLForecast(v, ARIMA, period = freq, model_kwargs=dict(order=(1, 1, 0), trend="t"))
                stlf_res = stlf.fit()
                forecast = stlf_res.forecast(len(mdlst[i])) # 预测
                forecast.index = mdlst[i][::-1]
            if method =='sarima' :
                pass
            impute_bw[mdlst[i][::-1]] = forecast #index要倒序一下
        impute_bw = impute_bw.sort_index(ascending = True)
    else:
        impute_bw = impute_fw
    return (impute_bw + impute_fw)/2

def dealing_with_hourly_weather_condition(data,freq = 24):
    missingDate = data[data.isnull().values==True].index
    orderlst = [np.nan,'Thunder','Fair','Fair / Windy','Thunder / Windy',
                'Mist','Partial Fog','Fog','Shallow Fog','Patches of Fog','Fog / Windy','Haze','Haze / Windy',
                'Partly Cloudy','Partly Cloudy / Windy','Cloudy','Cloudy / Windy',
                'Mostly Cloudy','Mostly Cloudy / Windy',
                'Light Rain','Light Rain with Thunder','Light Rain / Windy',
                'Rain','Rain / Windy','Light Rain Shower','Light Rain Shower / Windy',
                'Rain Shower','Rain Shower / Windy','Heavy Rain','Heavy Rain Shower','Heavy Rain Shower / Windy',
                'T-Storm', 'T-Storm / Windy','Heavy T-Storm','Heavy T-Storm / Windy',
                'Wintry Mix','Wintry Mix / Windy','Light Sleet','Light Snow','Light Snow / Windy','Snow']
    mdlst = []
    #判断连续缺失值状况以及位置,按照缺失划分数据
    a = [missingDate[0]]
    f = 24/freq
    for d in missingDate[1:]:
        if (d - a[-1]).total_seconds()/3600 == f:
            a.append(d)
        else:
            mdlst.append(a)
            a = [d]
        if d == missingDate[-1]:
            mdlst.append(a)
    for i in mdlst:
        if len(i)==1:
            data[i[0]] = data[i[0] - datetime.timedelta(hours=f)]
        if len(i)==2: 
            data[i[0]] = data[i[0] - datetime.timedelta(hours=f)]
            data[i[1]] = data[i[1] + datetime.timedelta(hours=f)]
    g = data.resample('H')
    new = []
    for key,value in g:
        if value[0] == value[1]:
            new.append(value[0])
        else:
            if orderlst.index(value[0]) < orderlst.index(value[1]):
                new.append(value[1])
            else:
                new.append(value[0]) 
    return new

def objective(trial,data = n,target = yy):
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=42)
    param = {
        'reg_alpha':trial.suggest_int('reg_alpha', 0, 5),
        'reg_lambda':trial.suggest_int('reg_lambda', 0, 5),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate',
                                                   [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        'gamma':trial.suggest_discrete_uniform('gamma', 0.1, 1,5),
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 10]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }
    model = xgb.XGBClassifier(**param)
    model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)
    return cross_val_score(model,test_x,test_y).mean()


def impute_missing_weather_condition(data,y, method = 'xgboost'):
    newdata = data.iloc[:,0:-2]
    le = preprocessing.LabelEncoder().fit(y)
    y2 = le.transform(y)
    y2 = pd.Series(y2)
    y2.index = y.index
    newdata['year'] = newdata.index.year
    newdata['month'] = newdata.index.month
    newdata['day'] = newdata.index.day
    newdata['hour'] = newdata.index.hour
    newdata['lag1'] = y2.shift(1)
    newdata['lag2'] = y2.shift(2)
    newdata['lag3'] = y2.shift(3)
    newdata['index'] = y.reset_index(drop = True).index
    n = np.array(newdata)
    #xgboost调参
    study = optuna.create_study(direction='maximize')
    study.optimize(objective,n_trials= 50)
    
    best_params = study.best_params
    best_model = xgb.XGBClassifier(best_params)
    best_model.fit(n,np.array(y2))
    #循环填入lag中nan
    for i in range(len(newdata)):
        try:
            if np.isnan(newdata.iloc[i,-1]) == True:
                x = np.array(newdata.iloc[i,1:-2]).reshape(1,-1)
                p = best_model.predict(x)
                newdata.iloc[i,-1] = le.inverse_transform([int(p[0])])[0]
                try:
                    newdata.iloc[i+1,-5] = p
                except:
                    pass
                try:
                    newdata.iloc[i+2,-4] = p
                except:
                    pass    
                try:
                    newdata.iloc[i+3,-3] = p
                except:
                    pass 
        except:
            pass
    
    return newdata['Condition']


def select_hourly_weatherDF(df):
    df = df[['temp', 'rh', 'pressure', 'vis','wspd','uv_index','wx_phrase']]
    df.index = pd.to_datetime(df.index)
    # check missing value
    missingDate,newdf = finding_missing_datetime(df.index[0],df.index[-1],'30T',df)
    if len(missingDate)==0:
        pass
    else:
        newdf['temp'] = imputating_mv_bistlarima(newdf['temp'],freq = 48,method = 'stl',bi = True)
        newdf['rh'] = imputating_mv_bistlarima(newdf['rh'],freq = 48,method = 'stl',bi = True)
        newdf['pressure'] = imputating_mv_bistlarima(newdf['pressure'],freq = 48,method = 'stl',bi = True)
        newdf['vis'] = imputating_mv_bistlarima(newdf['vis'],freq = 48,method = 'stl',bi = True)
        newdf['wspd'] = imputating_mv_bistlarima(newdf['wspd'],freq = 48,method = 'stl',bi = True)
        # categorical variable
    # getting hourly data
    hourDF = newdf.resample('H').mean()
    hourDF['wx_phrase'] = dealing_with_hourly_weather_condition(newdf['wx_phrase'],freq = 24)
    #imputing missing weather condition with xgboost
    
    hourDF.columns = ['T', 'RH', 'P', 'Vis','WS','UV','Condition']
    # check alldate
    return hourDF

def get_condition_dummies(df):
    d = df['Condition']
    # 合并相同含义情况
    for i in range(len(d)):
        if 'Smoke' in d[i]:
            d[i] = d[i].replace('Smoke','Haze')
        elif 'Widespread Dust' in d[i]:
            d[i] = d[i].replace('Widespread Dust','Sandstorm')
        elif 'Heavy T-Storm' in d[i]:
            d[i] = d[i].replace('Heavy T-Storm','T-Storm')
        elif 'Snow' in d[i]:
            temp = d[i].split('/')
            for t in range(len(temp)):
                if 'Snow' in temp[t]:
                    temp[t] = 'Snow '
            d[i] = "/".join(temp).strip()
        elif 'Fog' in d[i]:
            d[i] = 'Fog'
        elif 'Blowing Sand'in d[i]:
            d[i] = d[i].replace('Blowing Sand','Blowing Dust')
        elif ('Sleet' in d[i]) or ('Wintry Mix'in d[i]):
            d[i] = 'Sleet'
        elif ('with' in d[i]) :
            d[i] = d[i].replace('with','/')
    dd = pd.get_dummies(d)
    #复合变量拆分
    i = 0
    while i <(len(dd.columns.to_list())):
        c = dd.columns[i]
        if '/' in c:
            temp = c.split(' / ')
            for t in temp:
                try:
                    dd[t] = dd[t] + dd[c]
                except:
                    dd[t] = dd[c]
            dd = dd.drop(c,1)
            i = i-1 #因为删除了一列
        i += 1
    # 处理风速
    df['WS'] = df['WS']/3.6 #kmh to m/s
    #离散化
    for i in range(len(df['WS'])):
        if df['WS'][i]<5.5:
            df['WS'][i] = 0
        elif df['WS'][i]>=5.5 and df['WS'][i]<8:
            df['WS'][i] = 4
        elif df['WS'][i]>=8 and df['WS'][i]<10.8:
            df['WS'][i] = 5
        elif df['WS'][i]>=10.8 and df['WS'][i]<13.9:
            df['WS'][i] = 6
        elif df['WS'][i]>=13.9 and df['WS'][i]<17.2:
            df['WS'][i] = 7         
        elif df['WS'][i]>=17.2 and df['WS'][i]<20.8:
            df['WS'][i] = 8   
        elif df['WS'][i]>=20.8 and df['WS'][i]<24.5:
            df['WS'][i] = 9   
        elif df['WS'][i]>=24.5:
            df['WS'][i] = 10  
    df = pd.get_dummies(df,columns=['WS'])
    df = df.merge(dd,left_index=True, right_index=True)
    df = df.drop('Condition', 1)  
    df = df.drop('Windy', 1)  
    return df

if __name__ == '__main__':
    weatherDF = pd.read_csv('/Users/gu/python/occupancy/weather/ZSSS:9:CN2021-01-012020-12-31.csv', index_col=0).dropna(how='all')
    weatherDF2 = pd.read_csv('/Users/gu/python/occupancy/weather/ZBAA:9:CN2018-01-012017-12-31.csv', index_col=0).dropna(how='all')
    dataop = '/Users/gu/python/occupancy/prepared data/'
    #处理天气数据
    
    hourWDF = select_hourly_weatherDF(weatherDF)
    hourWDF2 = select_hourly_weatherDF(weatherDF2)
    dummyHWDF = get_condition_dummies(hourWDF)
    dummyHWDF2 = get_condition_dummies(hourWDF2)
    dummyHWDF.to_csv(dataop+'weatherSH.csv')
    dummyHWDF2.to_csv(dataop+'weatherBJ.csv')



    # beiyong
    wDF1 = pd.read_csv(inputPath + 'weatherSH_old.csv').dropna(how = 'all')
    wDF2 = pd.read_csv(inputPath + 'weatherBJ_old.csv').dropna(how = 'all')
    wDF1['Time'] = pd.to_datetime(wDF1['Time'])
    wDF2['Time'] = pd.to_datetime(wDF2['Time'])
    wDF1['Windy'] = 4*wDF1['WS_4.0']+5*wDF1['WS_5.0']+6*wDF1['WS_6.0']+7*wDF1['WS_7.0']
    wDF2['Windy'] = 4*wDF2['WS_4.0']+5*wDF2['WS_5.0']+6*wDF2['WS_6.0']+7*wDF2['WS_7.0']+8*wDF2['WS_8.0']
    wDF1 = wDF1.drop(['WS_4.0','WS_5.0','WS_0.0','WS_6.0', 'WS_7.0'], 1)
    wDF2 = wDF2.drop(['WS_4.0','WS_5.0','WS_0.0','WS_6.0', 'WS_7.0','WS_8.0'], 1)
    
    wDF1['Rainy'] = wDF1['Light Rain'] +wDF1['Light Rain Shower']+ 2*(wDF1['Rain']+wDF1['Rain Shower'])+3*wDF1['Sleet']+4*(wDF1['Heavy Rain']+wDF1['Heavy Rain Shower'])+5*wDF1['Snow']+6*wDF1['T-Storm']
    wDF2['Rainy'] = wDF2['Light Rain'] +wDF2['Light Rain Shower']+ 2*(wDF2['Rain']+wDF2['Rain Shower'])+3*wDF2['Sleet']+4*(wDF2['Heavy Rain']+wDF2['Heavy Rain Shower'])+5*wDF2['Snow']+6*wDF2['T-Storm']
    wDF1 = wDF1.drop(['Heavy Rain', 'Heavy Rain Shower', 'Light Rain','Light Rain Shower','Rain','Rain Shower', 'Sleet', 'Snow', 'T-Storm'],1)
    wDF2 = wDF2.drop(['Heavy Rain', 'Heavy Rain Shower', 'Light Rain','Light Rain Shower','Rain','Rain Shower', 'Sleet', 'Snow', 'T-Storm'],1)
         
    wDF1['Cloud'] = wDF1['Partly Cloudy'] + 2*wDF1['Cloudy']+ 3*wDF1['Mostly Cloudy']
    wDF2['Cloud'] = wDF2['Partly Cloudy'] + 2*wDF2['Cloudy']+ 3*wDF2['Mostly Cloudy']
    wDF1 = wDF1.drop(['Partly Cloudy', 'Cloudy','Mostly Cloudy'], 1)
    wDF2 = wDF2.drop(['Partly Cloudy', 'Cloudy','Mostly Cloudy'], 1)
    
    wDF1['Foggy'] = wDF1['Mist'] + 2*wDF1['Fog']+ 3*wDF1['Haze']
    wDF2['Foggy'] = wDF2['Mist'] + 2*wDF2['Fog']+ 3*wDF2['Haze']
    wDF1 = wDF1.drop(['Mist', 'Fog','Haze'], 1)
    wDF2 = wDF2.drop(['Mist', 'Fog','Haze'], 1)
    
    wDF2['Sandy'] = wDF2['Blowing Dust'] + 2*wDF2['Sandstorm']
    wDF2 = wDF2.drop(['Blowing Dust', 'Sandstorm'], 1)
    
    wDF1 = wDF1.drop(['Fair'], 1)
    wDF2 = wDF2.drop(['Fair'], 1)
    
    wDF1 = wDF1.drop(['Fair'], 1)
    wDF2 = wDF2.drop(['Fair'], 1)
       
    
    
    