# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:06:41 2021

@author: notfu
"""
import re
import pandas as pd
import pickle
import numpy as np
import datetime as dt
from datetime import datetime
#%%
filepath = 'C:/Users/notfu/Desktop/News/'
#%%
df_forbes = pd.read_csv(filepath+'forbes_us.csv', names = ['title', 'date', 'topic', 'link', 'summary', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'])
df_forbes = df_forbes.drop(['Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], axis = 1)
print(df_forbes.iloc[1,:])
#%%
df_CNBC = pd.read_excel(filepath+'CNBC_us.xlsx')
df_CNBC = df_CNBC.drop(['section', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8',
                        'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12',
                        'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16' ], axis = 1)
print(df_CNBC.iloc[1,:])
#%%
df = pd.concat([df_forbes, df_CNBC])
#%%
df = df.drop_duplicates()
#%%
#%%

df_replace = pd.DataFrame(columns = df.columns)
for i in df['topic']:
    if i != 'United States':
        df_replace = df_replace.append(df.loc[df['topic'] == i, :])
        df = df.drop(df.loc[df['topic'] == i, :].index)
#%%
for i in range(len(df_replace)):
    if type(df_replace.iloc[i, 4]) == float:
        for c in [4,3,2,1]:
            temp = df_replace.iloc[i, c-1]
            df_replace.iloc[i, c] = temp
        print(df_replace.iloc[i, :])
        date_match = re.search('(\d{4}/\d{1,2}/\d{1,2})', df_replace.iloc[i, 3])
        df_replace.iloc[i, 1] = date_match.group(0)
#%%
df_replace = df_replace.reindex(['index'])
#%%
#df_replace.index[df_replace['topic'] != 'United States'].tolist())
df_replace = df_replace.drop(df_replace.index[df_replace['topic'] != 'United States'].tolist())  
#%%
df = pd.concat([df, df_replace])
#%%
df = df.drop(df.index[df['topic'] != 'United States'].tolist())  
#%%
print(df['date'].unique())
#%%
wrong = datetime.strptime('2021-12-27', '%Y-%m-%d')
correct =  datetime.strptime('2020-12-27', '%Y-%m-%d')
#for i in range(5):
#df_sorted = df_sorted.replace('2021-12-31', '2020-12-31')
date = df['date'].unique()
#%%%
datelist = []
for date in df['date']:
        if type(date) == str:
            date_ = datetime.strptime(date,'%Y/%m/%d')
            date_ = datetime.strftime(date_,'%Y-%m-%d')
            
        else:
            date_ = datetime.strftime(date,'%Y-%m-%d')
        datelist.append(date_)
#%%
df['date'] = datelist
#%%Transform dateframe to dict
news_dict = {}
for topic in df['topic'].unique():
    news_dict[topic] = {}
    for date in df['date'].unique():
        print(date)
        news_dict[topic][date] = {}
        for index in ['title', 'link', 'summary']:
            news_dict[topic][date][index] = df[(df['topic'] == topic) & (df['date'] == date)][index]
    #print(topic)
#%%
with open('news_sorted_by_topic.pickle', 'wb') as f:
    pickle.dump(news_dict_all, f)
#%%Transform dateframe to dict
news_dict = {}
for date in df_sorted['date'].unique():
    news_dict[date] = {}
    for index in ['topic', 'title', 'link', 'summary']:
        news_dict[date][index] = df_sorted[(df_sorted['date'] == date)][index]
    print(date)
#%%
with open(filepath+'news_sorted_by_topic.pickle', 'rb') as f:
    news_dict_all = pickle.load(f)
#%%
news_dict_all.pop('United States')
news_dict_all['United States'] = news_dict['United States']
