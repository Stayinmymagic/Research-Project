# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:45:00 2021

@author: notfu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import h5py
import calendar
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import matthews_corrcoef,precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
#%%
filePath = "C:/Users/notfu/Desktop/News/filtered_embeddings/"
fileName = "finbert_embeddings_filtered_sentiment.h5"
targetName = "target_0524.csv"
target = pd.read_csv(filePath+targetName)
device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
with h5py.File(filePath+fileName, "r") as f:
    topics = f.keys()
    print(topics)
    # removed_topics = ['asiadefense', 'media','sustainable', 'middle east', 'telecommunications', 
    # removed_topics = ['autos', 'sustainable','environment','energy','finance','currencies','technology','fed',
    #                   'media',  'telecommunications',  'uk','europe','china', 'asia','commodity', 'consumer','transportation','retail',
    #                     'United States','telecom',]
    
    # removed_topics = ['china', 'asia', 'autos', 'commodity', 'consumer',,'middle east','aerospace','middle east','aerospace','defense','politics',
    #                   'europe', 'defense', 'media','middle east', 'telecommunications', 'politics','finance','currencies','technology','fed',
    #                   'uk', 'retail', 'retails', 'United States','aerospace','telecom']
    # topic_list = [x for x in list(f.keys()) if x not in removed_topics]
    topic_list = list(topics)
    # List all groups
    print("Keys: %s" % f.keys())
    df_count = pd.DataFrame(columns = topic_list)

#%%
def get_newsdata_merge(time_delta = 21):
    data_array = np.zeros([len(target), time_delta, 24], dtype = object)
    with h5py.File(filePath+fileName, "r") as f:
        topics = f.keys()
        topic_list = list(topics)
        print(topic_list)
        # List all groups
        print("Keys: %s" % f.keys())
        day = 0
        for date in target['DATE']:
            
            date = datetime.strptime(date, "%Y-%m-%d")#+ timedelta(days=-3*time_delta)
            back_date = date + timedelta(days=-time_delta) 

            for delta in range(1, time_delta+1): 
                
                for topic in range(len(topic_list)):            
                    try:
                        data = f[topic_list[topic]][back_date.strftime('%Y-%m-%d')][:]

                        if data.shape[0] != 0:
                            data_array[day, delta-1, topic] = np.array(data)
                        
                    except KeyError:
                        # data_array[day, delta-1, topic] = 0
                        pass
                back_date = back_date + timedelta(days=1) 
            
            day += 1
            
                
    return data_array             
data_array = get_newsdata_merge()

#%%
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
#%%
target = target.drop(zero_list, axis = 0)
data_array = np.delete(data_array , zero_list, 0)
#%%
X_train, X_test, Y_train, Y_test = train_test_split(data_array, target, test_size=0.2, shuffle = False)#random_state= 66
train_mm = StandardScaler()
test_mm = StandardScaler()
cols = Y_train.columns[1:9]
for col in cols:
    if col != 'DATE':
        Y_train.loc[:, col] = train_mm.fit_transform(np.array(Y_train.loc[:, col]).reshape(len(Y_train.loc[:, col]), 1))

for col in cols:
    if col != 'DATE':
        Y_test.loc[:, col] = test_mm.fit_transform(np.array(Y_test.loc[:, col]).reshape(len(Y_test.loc[:, col]), 1))
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, shuffle = False)
#%%
Y_train = Y_train.reset_index(inplace = False).drop('index', axis=1)
Y_valid = Y_valid.reset_index(inplace = False).drop('index', axis=1)
Y_test = Y_test.reset_index(inplace = False).drop('index', axis=1)
#%%
embedding_dim = 768 # embedding size
n_hidden = 128  # number of hidden units in one cell
attn_matrix = 64
attn_news = 128
num_classes = 5  # 0 or 1
linear_matrix = 64
PCEDG_mat = 20

class BiLSTM_Attention_NoIntra(nn.Module):
    def __init__(self, use_attention=True):
        super(BiLSTM_Attention_NoIntra, self).__init__()
        self.use_attention = use_attention
        self.lstm_text = nn.LSTM(embedding_dim, n_hidden, num_layers = 2, 
                                  bidirectional=False, batch_first = True, dropout = 0.2)#
        self.lstm_topic = nn.LSTM(n_hidden, n_hidden, num_layers = 2, 
                                  bidirectional=False, batch_first = True, dropout = 0.2)#
        self.dropout = nn.Dropout(p=0.5)
        self.lstm_day = nn.LSTM(n_hidden, 64, num_layers = 2, bidirectional=False, batch_first = True, dropout = 0.2)#
        self.linear = nn.Linear(linear_matrix, 48)
        self.linear_2 = nn.Linear(48, 16)
        # self.out_Payems = nn.Linear(PCEDG_mat, 1)
        self.out_CPI = nn.Linear(PCEDG_mat, 1)
        self.out_Retails = nn.Linear(PCEDG_mat, 1)
        self.out_PCEDG = nn.Linear(PCEDG_mat , 1)
        self.out_CCI = nn.Linear(PCEDG_mat, 1) 
        self.linear_3 = nn.Linear(PCEDG_mat, 4) 
        if self.use_attention:
            self.W1 = torch.nn.Linear(attn_matrix, attn_matrix)
            self.W2 = torch.nn.Linear(attn_matrix, attn_matrix)
            self.V = torch.nn.Linear(attn_matrix, 1)
            
            self.w1 = torch.nn.Linear(n_hidden, n_hidden)
            self.w2 = torch.nn.Linear(attn_matrix, attn_matrix)
            
        
        
    def attention_net_topic(self, lstm_output, final_state, dim = 0):
        # score = self.V(torch.tanh(self.W1(final_state) + self.W2(lstm_output)))
        # score = torch.mm(lstm_output, final_state.permute(1, 0))

        score = torch.mm(self.w1(lstm_output), final_state.permute(1, 0))
        attention_weights = F.softmax(score, dim=dim)
        temp = attention_weights.squeeze(1)
        sort, indices = torch.sort(temp, descending=True)
        score = 0
        for i in range(len(sort)):
            score += sort[i]
            if score > 0.8:
                break

        context_vector = sort[:i+1].unsqueeze(1) *  torch.index_select(lstm_output, 0, indices[:i+1])
        # context_vector = attention_weights * lstm_output
        context_vector = torch.sum(context_vector, dim=dim)
        return context_vector, attention_weights
    
    def attention_net_day(self, lstm_output, final_state, dim = 0):
        # score = self.V(torch.tanh(self.W1(final_state) + self.W2(lstm_output)))
        # score = torch.mm(lstm_output, final_state.permute(1, 0))

        score = torch.mm(self.w2(lstm_output), final_state.permute(1, 0))
        attention_weights = F.softmax(score, dim=dim)
        context_vector = attention_weights * lstm_output
        context_vector = torch.sum(context_vector, dim=dim)
        return context_vector, attention_weights
        
    def forward(self, X, previous_labels, topics, date):
        
        #每一次送進lstm_inter是 N Days 所有News，1*len_seq *768(不同主題需分開), Days視為batchsize
        day_hidden_state = []
        for day in range(X.shape[0]):
            topic_hidden_state = []
            
            for topic in range(topics):
                    
                try:
                    x = torch.FloatTensor(X[day][topic]).to(device)
                    # print(x.size())
                    output, (final_hidden_state, final_cell_state) = self.lstm_text(x.unsqueeze(0))#每一個topic hidden state in each day
                    #因為有2 layer所以output第一維度是4，取
                    # topic_hidden_state.append(self.tanh((final_hidden_state[1,:,:]+final_hidden_state[3,:,:]).unsqueeze(0)))
                    topic_hidden_state.append(final_hidden_state[0,:,:].unsqueeze(0))
                    # topic_hidden_state.append(attn_output.unsqueeze(0))
                    # topic_hidden_state.append(final_hidden_state)
            
            
                except RuntimeError:
                    pass   
            if len(topic_hidden_state) != 0:
                topic_hidden_state = torch.cat(topic_hidden_state, dim = 0)
                topic_hidden_state = topic_hidden_state.permute(1,0,2)
                # print('topic',topic_hidden_state.shape)
                output, (final_hidden_state, final_cell_state) = self.lstm_topic(topic_hidden_state)
                attn_output, attention = self.attention_net_topic(output[0], final_hidden_state[0])
                
                # for i in attention:
                #     if i[0] > 0.1 and len(attention) == 24:
                #         print(date)
                #         print(attention)
                #         break
                # day_hidden_state.append(final_hidden_state)#attn_output.unsqueeze(0)
                day_hidden_state.append(attn_output.unsqueeze(0))#
        day_hidden_state = torch.cat(day_hidden_state, dim = 0)

        day_hidden_state = day_hidden_state.unsqueeze(0)
        # day_hidden_state = day_hidden_state.permute(1,0,2)
        output, (final_hidden_state, final_cell_state) = self.lstm_day(day_hidden_state)
        # final_hidden_state = self.dropout(final_hidden_state)
        # output = self.tanh(output)
        # final_hidden_state = self.tanh(final_hidden_state)
     
        attn_output, attention = self.attention_net_day(output[0], final_hidden_state[0])
        # for i in attention:
        #     if i[0] > 0.2 and len(attention) == 7:
        #         print(date)
        #         print(attention)
        #         break
        # _output = self.linear(torch.cat((final_hidden_state[1].squeeze(0), sp_data)))

        # _output = self.linear(final_hidden_state[1].squeeze(0))
        _output = self.linear(attn_output)#attn_output
        # print(_output.shape)
        _output = self.linear_2(_output)
        # _output = self.linear_2(torch.cat((_output, sp_data)))
        output_PCEDG = self.out_PCEDG(torch.cat((_output, previous_labels[0])))
        output_CCI = self.out_CCI(torch.cat((_output, previous_labels[1])))   
        output_Retails = self.out_Retails(torch.cat((_output, previous_labels[2])))     
        output_CPI = self.out_CPI(torch.cat((_output, previous_labels[3])))
        output = [output_PCEDG, output_CCI, output_Retails , output_CPI]

        return output, True
    

#%%%training
pred_labels = 4
delta = timedelta(40)
def train(x, y_):
    #train loss record 記錄x_train每一次backward的loss
    #train loss record 的length 會和k-fold後的x_train大小一樣
    train_loss_record = []
    y_pred_record = []
    for i in range(x.shape[0]): 
        
        date = datetime.strptime(y_['DATE'][i], "%Y-%m-%d")
        backdate = date - delta

        
        
        y = torch.FloatTensor(np.array(y_.iloc[:, 1:])).to(device)
        previous_labels = y[i][:pred_labels].unsqueeze(0)
        for label in range(1, pred_labels):
            previous_labels = torch.cat((previous_labels,  y[i][:pred_labels].unsqueeze(0)), axis = 0).to(device)
        # print(previous_labels)
        optimizer.zero_grad()
        
        output, Success = model(x[i], previous_labels, 24, date)
        
        if Success:
            loss = 0
            loss_list = []
            for o in range(len(output)):
                
                loss += criterion(output[o], y[i][pred_labels+o].unsqueeze(0))

                loss_list.append(criterion(output[o], y[i][pred_labels+o]).cpu().detach().numpy())
            loss.backward()
            # loss = loss/len(output)
            optimizer.step()
            
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
def valid(x, y_):
    valid_loss_record = []
    y_pred_record = []
    for i in range(x.shape[0]): 
        
        date = datetime.strptime(y_['DATE'][i], "%Y-%m-%d")
        # print(date)
        backdate = date - delta
        # sp_data = torch.FloatTensor(Sp500.loc[backdate :date,'Norm']).to(device)
        #print(previous_labels)
        # previous_labels = np.zeros((4, 3))
        # for label in range(pred_labels):
        #     temp = list(y_.iloc[i, 1:5])
        #     temp.pop(label)
        #     previous_labels[label,:] = temp
        # previous_labels = torch.Tensor(previous_labels).to(device)
        
        y = torch.FloatTensor(np.array(y_.iloc[:, 1:])).to(device)
        previous_labels = y[i][:pred_labels].unsqueeze(0)
        for label in range(1, pred_labels):
            previous_labels = torch.cat((previous_labels, y[i][:pred_labels].unsqueeze(0)), axis = 0).to(device)
        optimizer.zero_grad()
        output, Success = model(x[i], previous_labels, 24, date)
        if Success:
            loss = 0
            loss_list = []
            for o in range(len(output)):
               
                loss += criterion(output[o], y[i][pred_labels+o].unsqueeze(0))
                # loss += criterion(output[o], y[i][pred_labels+o])
                loss_list.append(criterion(output[o], y[i][pred_labels+o]).cpu().detach().numpy())
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
            
def calculate_loss(loss_record):
    loss_record = np.array(loss_record)
    return np.mean(loss_record, axis = 0)
    # return sum(loss_record)/len(loss_record)
#分開計算各指數的acc
def calculate_acc(y_pred_record, target, train_mm ):
    
    # y_pred_record = np.concatenate(np.array(y_pred_record), axis = 1)
    y_pred_record = np.array(y_pred_record, dtype = float)
    acc = []
    mcc = []
    precision = []
    recall = []
    f1 = []
    for i in range(pred_labels):
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
        acc.append(np.round(correct_num/len(temp), pred_labels))
        precision.append(np.round(precision_score(target[:, 8+i], temp), 4))
        recall.append(np.round(recall_score(target[:, 8+i], temp), 4))
        f1.append(np.round(f1_score(target[:, 8+i], temp), 4))
    return acc, mcc, precision, recall,  f1

#%%
epochs = 50
model = BiLSTM_Attention_NoIntra().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.002)
# criterion = nn.BCELoss()
criterion = nn.MSELoss()
train_loss_list = []
valid_loss_list = []
test_loss_list = []
train_acc_list = []
valid_acc_list = []
test_acc_list = []
train_mcc_list = []
valid_mcc_list = []
test_mcc_list = []

# y_test_all = torch.FloatTensor(np.array(Y_test.iloc[:, 1:])).to(device)
# y_train_all = torch.FloatTensor(np.array(Y_train.iloc[:, 1:])).to(device)
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
the_last_loss = 100
patience = 2
trigger_times = 0
for epoch in range(1, epochs+1):
    if trigger_times  >= patience:
        break
    # train_loss_in_a_epoch = []
    # valid_loss_in_a_epoch = []
    # test_loss_in_a_epoch = []
    # train_acc_in_a_epoch = []
    # valid_acc_in_a_epoch = []
    # test_acc_in_a_epoch = []
    # train_mcc_in_a_epoch = []
    # valid_mcc_in_a_epoch = []
    # test_mcc_in_a_epoch = []
    

        
        # y_train = torch.FloatTensor(y_train).to(device)
        # y_valid = torch.FloatTensor(y_valid).to(device)
        
        #set training mode
    model.train()
    #feed a batch(a segment sliced by k-fold)
    # print(x_train.shape)
    # print(y_train.shape)
    y_pred_record, train_loss_record = train(X_train, Y_train)
    y_train_forloss = torch.FloatTensor(np.array(Y_train.iloc[:, 1:])).to(device)
    train_loss = calculate_loss(train_loss_record)
    train_acc, train_mcc, train_precision, train_recall, train_f1 = calculate_acc(
        y_pred_record, y_train_forloss.cpu().detach().numpy(), train_mm)
    model.eval()
    with torch.no_grad():
        
        y_pred_record, valid_loss_record = valid(X_valid, Y_valid)
        valid_loss = calculate_loss(valid_loss_record)
        y_valid_forloss = torch.FloatTensor(np.array(Y_valid.iloc[:, 1:])).to(device)
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
            y_pred_record, test_loss_record = valid(X_test, Y_test)
            y_test_forloss = torch.FloatTensor(np.array(Y_test.iloc[:, 1:])).to(device)
            test_loss = calculate_loss(test_loss_record)
            test_acc, test_mcc, test_precision, test_recall, test_f1 = calculate_acc(
            y_pred_record, y_test_forloss.cpu().detach().numpy(), test_mm)
            print('epoch : {}, train loss = {}, valid loss = {} , test loss = {}'
          .format(epoch, train_loss, valid_loss, test_loss))
            print('train acc = {},train mcc = {}, train precision = {}, train recall = {},train f1 = {},\
              test acc = {}, test mcc = {}, test precision = {},test recall= {}, test f1 = {}'.format(
            train_acc, train_mcc, train_precision, train_recall, train_f1,\
            test_acc, test_mcc, test_precision, test_recall, test_f1))   
            print('Early stopping!.')
            break

        y_pred_record, test_loss_record = valid(X_test, Y_test)
        y_test_forloss = torch.FloatTensor(np.array(Y_test.iloc[:, 1:])).to(device)
        test_loss = calculate_loss(test_loss_record)
        test_acc, test_mcc, test_precision, test_recall, test_f1 = calculate_acc(
            y_pred_record, y_test_forloss.cpu().detach().numpy(), test_mm)
        
        
    # train_loss_in_a_epoch.append(train_loss)
    # valid_loss_in_a_epoch.append(valid_loss)
    # test_loss_in_a_epoch.append(test_loss)
    # train_acc_in_a_epoch.append(train_acc)
    # valid_acc_in_a_epoch.append(valid_acc)
    # test_acc_in_a_epoch.append(test_acc)
    # train_mcc_in_a_epoch.append(train_mcc)
    # valid_mcc_in_a_epoch.append(valid_mcc)
    # test_mcc_in_a_epoch.append(test_mcc)
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)

    test_acc_list.append(test_acc)
    train_mcc_list.append(train_mcc)
   
    test_mcc_list.append(test_mcc)
 
    print('epoch : {}, train loss = {}, valid loss = {} , test loss = {}'
          .format(epoch, train_loss, valid_loss, test_loss))
    
    print('train acc = {},train mcc = {}, train precision = {}, train recall = {},train f1 = {},\
              test acc = {}, test mcc = {}, test precision = {},test recall= {}, test f1 = {}'.format(
        train_acc, train_mcc, train_precision, train_recall, train_f1,\
        test_acc, test_mcc, test_precision, test_recall, test_f1))   
    
    # print('train acc = {}, valid acc = {}, test acc = {}'.format(
    #     train_acc,  valid_acc, test_acc))     

    
