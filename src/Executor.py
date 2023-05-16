#!/usr/local/bin/python
import torch
import numpy as np
from DataPipe import DataPipe
from datetime import datetime, timedelta
from ConfigLoader import config_model
import torch.optim as optim
import torch.nn as nn
from ConfigLoader import logger
from sklearn.metrics import matthews_corrcoef,precision_score, f1_score
from sklearn.metrics import recall_score
class Executor:

    def __init__(self, model, silence_step=200, skip_step=20):
        self.model = model
        self.silence_step = silence_step
        self.skip_step = skip_step
        self.pipe = DataPipe()
        self.pred_labels = 4
        self.delta = timedelta(40)
        self.device = config_model['device']
        self.lr = config_model['lr']
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train(self, x, y_):
        #train loss record 記錄x_train每一次backward的loss
        #train loss record 的length 會和k-fold後的x_train大小一樣
        train_loss_record = []
        y_pred_record = []
        for i in range(x.shape[0]): 
            
            
            y = torch.FloatTensor(np.array(y_.iloc[:, 1:])).to(self.device)
            previous_labels = y[i][:self.pred_labels].unsqueeze(0)
            for label in range(1, self.pred_labels):
                previous_labels = torch.cat((previous_labels,  y[i][:self.pred_labels].unsqueeze(0)), axis = 0).to(self.device)
            # print(previous_labels)
            self.optimizer.zero_grad()
            
            output, Success = self.model(x[i], previous_labels, 24)
            
            if Success:
                loss = 0
                loss_list = []
                for o in range(len(output)):
                    
                    loss += self.criterion(output[o], y[i][self.pred_labels+o].unsqueeze(0))
    
                    loss_list.append(self.criterion(output[o], y[i][self.pred_labels+o]).cpu().detach().numpy())
                loss.backward()
                # loss = loss/len(output)
                self.optimizer.step()
                
                # train_loss_record.append(loss.cpu().detach().numpy())
                train_loss_record.append(loss_list)
                # y_pred = [np.power(2, i.cpu().detach().numpy()[0]) for i in output]
                y_pred = [i.cpu().detach().numpy() for i in output]
                
                # y_pred = output.cpu().detach().numpy()
                y_pred_record.append(y_pred)
                # print(y_pred_record)
                
            else:
                print(i,'error')
    
        return y_pred_record, train_loss_record
    
    #test and valid可共用
    def valid(self, x, y_):
        valid_loss_record = []
        y_pred_record = []
        for i in range(x.shape[0]): 
            
            
            y = torch.FloatTensor(np.array(y_.iloc[:, 1:])).to(self.device)
            previous_labels = y[i][:self.pred_labels].unsqueeze(0)
            for label in range(1, self.pred_labels):
                previous_labels = torch.cat((previous_labels, y[i][:self.pred_labels].unsqueeze(0)), axis = 0).to(self.device)
            self.optimizer.zero_grad()
            output, Success = self.model(x[i], previous_labels, 24)
            if Success:
                loss = 0
                loss_list = []
                for o in range(len(output)):
                   
                    loss += self.criterion(output[o], y[i][self.pred_labels+o].unsqueeze(0))
                    # loss += criterion(output[o], y[i][pred_labels+o])
                    loss_list.append(self.criterion(output[o], y[i][self.pred_labels+o]).cpu().detach().numpy())
                # valid_loss_record.append(loss.cpu().detach().numpy())
                valid_loss_record.append(loss_list)
                # y_pred = [np.power(2, i.cpu().detach().numpy()[0]) for i in output]
                y_pred = [i.cpu().detach().numpy() for i in output]
                # y_pred = output.cpu().detach().numpy()
                y_pred_record.append(y_pred)
                # print(y_pred_record)
            else:
                print(i)
            # torch.cuda.empty_cache()
        return y_pred_record, valid_loss_record
                
    def calculate_loss(self, loss_record):
        loss_record = np.array(loss_record)
        return np.mean(loss_record, axis = 0)
        # return sum(loss_record)/len(loss_record)
    #分開計算各指數的acc
    def calculate_acc(self, y_pred_record, target, train_mm ):
        
        # y_pred_record = np.concatenate(np.array(y_pred_record), axis = 1)
        y_pred_record = np.array(y_pred_record, dtype = float)
        acc = []
        mcc = []
        precision = []
        recall = []
        f1 = []
        for i in range(self.pred_labels):
            # print(y_pred_record[:, i])
            temp = train_mm.inverse_transform(y_pred_record[:, i].reshape(y_pred_record.shape[0], 1))
            # temp = np.expm1(y_pred_record[:, i])
            
            temp = y_pred_record[:, i]
            # print(temp)
            
            if i == 0 or i == 3:
                temp =  (temp > 0.2).astype(int)
                
            else:
                temp =  (temp > 0).astype(int)
            
            # print('temp', temp)
            # print('target', target[:, 12+i])
            correct_num = np.sum(temp.reshape(1, y_pred_record.shape[0]) == target[:, 8+i])
            # print(correct_num)
            
            mcc.append(np.round(matthews_corrcoef(target[:, 8+i], temp), 4))
            acc.append(np.round(correct_num/len(temp), self.pred_labels))
            precision.append(np.round(precision_score(target[:, 8+i], temp), 4))
            recall.append(np.round(recall_score(target[:, 8+i], temp), 4))
            f1.append(np.round(f1_score(target[:, 8+i], temp), 4))
        return acc, mcc, precision, recall,  f1
    def train_dev(self, X_train, Y_train, train_mm, X_valid, Y_valid, X_test, Y_test, train_mm):
        self.epochs = config_model['epochs']

        torch.backends.cudnn.enable =True
        torch.backends.cudnn.benchmark = True
        the_last_loss = 100
        patience = 2
        trigger_times = 0
        for epoch in range(1, self.epochs+1):
            if trigger_times  >= patience:
                break

            #set training mode
            self.model.train()
            #feed a batch(a segment sliced by k-fold)
            # print(x_train.shape)
            # print(y_train.shape)
            y_pred_record, train_loss_record = self.train(X_train, Y_train)
            y_train_forloss = torch.FloatTensor(np.array(Y_train.iloc[:, 1:])).to(self.device)
            train_loss = self.calculate_loss(train_loss_record)
            train_acc, train_mcc, train_precision, train_recall, train_f1 = self.calculate_acc(
                y_pred_record, y_train_forloss.cpu().detach().numpy(), train_mm)
            self.model.eval()
            with torch.no_grad():
                
                y_pred_record, valid_loss_record = self.valid(X_valid, Y_valid)
                valid_loss = self.calculate_loss(valid_loss_record)
                # y_valid_forloss = torch.FloatTensor(np.array(Y_valid.iloc[:, 1:])).to(self.device)
                # valid_acc, valid_mcc, valid_precision, valid_recall, valid_f1 = calculate_acc(
                    # y_pred_record, y_valid_forloss.cpu().detach().numpy(), train_mm)
                # Early stopping
                # print('The current loss:', valid_loss)    
                if np.sum(valid_loss) > the_last_loss:
                    trigger_times += 1
                    print('trigger times:', trigger_times)
                   
        
                else:
                    print('trigger times: 0')
                    trigger_times = 0
                    the_last_loss = np.sum(valid_loss)
                if trigger_times >= patience:
                    y_pred_record, test_loss_record = self.valid(X_test, Y_test)
                    y_test_forloss = torch.FloatTensor(np.array(Y_test.iloc[:, 1:])).to(self.device)
                    test_loss = self.calculate_loss(test_loss_record)
                    test_acc, test_mcc, test_precision, test_recall, test_f1 = self.calculate_acc(
                    y_pred_record, y_test_forloss.cpu().detach().numpy(), train_mm)
                    print('epoch : {}, train loss = {}, valid loss = {} , test loss = {}'
                  .format(epoch, train_loss, valid_loss, test_loss))
                    print('train acc = {},train mcc = {}, train precision = {}, train recall = {},train f1 = {},\
                      test acc = {}, test mcc = {}, test precision = {},test recall= {}, test f1 = {}'.format(
                    train_acc, train_mcc, train_precision, train_recall, train_f1,\
                    test_acc, test_mcc, test_precision, test_recall, test_f1))   
                    print('Early stopping!.')
                    break
        
                y_pred_record, test_loss_record = self.valid(X_test, Y_test)
                y_test_forloss = torch.FloatTensor(np.array(Y_test.iloc[:, 1:])).to(self.device)
                test_loss = self.calculate_loss(test_loss_record)
                test_acc, test_mcc, test_precision, test_recall, test_f1 = self.calculate_acc(
                    y_pred_record, y_test_forloss.cpu().detach().numpy(), train_mm)
                
                

            iter_str = 'epoch : {}, train loss = {}, valid loss = {} , test loss = {}'.format(epoch, train_loss, valid_loss, test_loss)
  
            acc_str = 'train acc = {},train mcc = {}, train precision = {}, train recall = {},train f1 = {},\
                      test acc = {}, test mcc = {}, test precision = {},test recall= {}, test f1 = {}'.format(
                train_acc, train_mcc, train_precision, train_recall, train_f1,\
                test_acc, test_mcc, test_precision, test_recall, test_f1)
            
            logger.info(', '.join((iter_str, acc_str)))
