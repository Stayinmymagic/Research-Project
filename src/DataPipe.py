#!/usr/local/bin/python

import numpy as np
import pandas as pd
import h5py
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ConfigLoader import path_parser,  topics


class DataPipe:

    def __init__(self):
        # load path
        self.targetName = path_parser.target
        self.fileName = path_parser.fileName
        self.topic_list = topics['topic_list']

   
 
    def get_newsdata_merge(self, time_delta = 7):
        print(self.targetName)
        target = pd.read_csv(self.targetName)
        data_array = np.zeros([len(target), time_delta, 24], dtype = object)
        with h5py.File(self.fileName, "r") as f:

            print(self.topic_list)
            # List all groups
            print("Keys: %s" % f.keys())
            day = 0
            for date in target['DATE']:
                
                date = datetime.strptime(date, "%Y-%m-%d")#+ timedelta(days=-3*time_delta)
                back_date = date + timedelta(days=-time_delta) 
    
                for delta in range(1, time_delta+1): 
                    
                    for topic in range(len(self.topic_list)):            
                        try:
                            data = f[self.topic_list[topic]][back_date.strftime('%Y-%m-%d')][:]
    
                            if data.shape[0] != 0:
                                data_array[day, delta-1, topic] = np.array(data)
                            
                        except KeyError:
                            # data_array[day, delta-1, topic] = 0
                            pass
                    back_date = back_date + timedelta(days=1) 
                
                day += 1
                
                    
        #刪除沒有蒐集到新聞資料的日期
        zero_list = []
        for i in range(data_array.shape[0]):
            count = 0
            for j in range(7):
                try:
                    if sum(data_array[i][j]) == 0:
                        count +=1
                except ValueError:
                    pass
            if count == 7:
                zero_list.append(i)    

        target = target.drop(zero_list, axis = 0)
        data_array = np.delete(data_array , zero_list, 0)
        
        return data_array, target
    
    def split_data(self, data_array, target):
        X_train, X_test, Y_train, Y_test = train_test_split(data_array, target, test_size=0.2, shuffle = False)
        train_mm = StandardScaler()
        cols = Y_train.columns[1:9]
        for col in cols:
            if col != 'DATE':
                Y_train.loc[:, col] = train_mm.fit_transform(np.array(Y_train.loc[:, col]).reshape(len(Y_train.loc[:, col]), 1))
        
        for col in cols:
            if col != 'DATE':
                Y_test.loc[:, col] = train_mm.transform(np.array(Y_test.loc[:, col]).reshape(len(Y_test.loc[:, col]), 1))
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, shuffle = False)
        
        Y_train = Y_train.reset_index(inplace = False).drop('index', axis=1)
        Y_valid = Y_valid.reset_index(inplace = False).drop('index', axis=1)
        Y_test = Y_test.reset_index(inplace = False).drop('index', axis=1)
        
        return  X_train, X_valid, X_test, Y_train, Y_valid, Y_test, train_mm
    
