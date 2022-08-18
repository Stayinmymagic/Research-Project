#!/usr/local/bin/python
from Model import Model
from Executor import Executor
from DataPipe import DataPipe

if __name__ == '__main__':
    model = Model()
    datapipe = DataPipe()
    
    data_array, target = datapipe.get_newsdata_merge()
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test, train_mm, test_mm = datapipe.split_data(data_array, target)
    
    exe = Executor(model)
    
    exe.train_dev(X_train, Y_train, train_mm, X_valid, Y_valid, X_test, Y_test, test_mm)
    
   
