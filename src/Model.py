# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:45:00 2021

@author: notfu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ConfigLoader import config_model

class Model(nn.Module):
    def __init__(self, use_attention=True):
        super(Model, self).__init__()
        
         # model config
        self.embedding_dim = config_model['embedding_dim']
        self.n_hidden = config_model['n_hidden']
        self.attn_matrix = config_model['attn_matrix']
        self.linear_matrix = config_model['linear_matrix']
        self.final_mat = config_model['final_mat']
        self.device = config_model['device']
        
        self.use_attention = use_attention
        self.lstm_text = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers = 2, 
                                  bidirectional=False, batch_first = True, dropout = 0.2)#
        self.lstm_topic = nn.LSTM(self.n_hidden, self.n_hidden, num_layers = 2, 
                                  bidirectional=False, batch_first = True, dropout = 0.2)#
        self.dropout = nn.Dropout(p=0.5)
        self.lstm_day = nn.LSTM(self.n_hidden, 64, num_layers = 2, bidirectional=False, batch_first = True, dropout = 0.2)#
        self.linear = nn.Linear(self.linear_matrix, 48)
        self.linear_2 = nn.Linear(48, 16)
        # self.out_Payems = nn.Linear(PCEDG_mat, 1)
        self.out_CPI = nn.Linear(self.final_mat, 1)
        self.out_Retails = nn.Linear(self.final_mat, 1)
        self.out_PCEDG = nn.Linear(self.final_mat , 1)
        self.out_CCI = nn.Linear(self.final_mat, 1) 
        self.linear_3 = nn.Linear(self.final_mat, 4) 
        if self.use_attention:
            self.W1 = torch.nn.Linear(self.attn_matrix, self.attn_matrix)
            self.W2 = torch.nn.Linear(self.attn_matrix, self.attn_matrix)
            self.V = torch.nn.Linear(self.attn_matrix, 1)
            
            self.w1 = torch.nn.Linear(self.n_hidden, self.n_hidden)
            self.w2 = torch.nn.Linear(self.attn_matrix, self.attn_matrix)
            
        
        
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
        
    def forward(self, X, previous_labels, topics):
        
        #每一次送進lstm_inter是 N Days 所有News，1*len_seq *768(不同主題需分開), Days視為batchsize
        day_hidden_state = []
        for day in range(X.shape[0]):
            topic_hidden_state = []
            
            for topic in range(topics):
                    
                try:
                    x = torch.FloatTensor(X[day][topic]).to(self.device)
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
                day_hidden_state.append(attn_output.unsqueeze(0))#
        day_hidden_state = torch.cat(day_hidden_state, dim = 0)

        day_hidden_state = day_hidden_state.unsqueeze(0)
        # day_hidden_state = day_hidden_state.permute(1,0,2)
        output, (final_hidden_state, final_cell_state) = self.lstm_day(day_hidden_state)

     
        attn_output, attention = self.attention_net_day(output[0], final_hidden_state[0])
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
    
