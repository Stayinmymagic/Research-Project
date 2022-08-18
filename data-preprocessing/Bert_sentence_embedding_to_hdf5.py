# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:29:52 2021

@author: notfu
"""

import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import pandas as pd
import h5py
import numpy as np
from sentence_transformers import SentenceTransformer
#%%
file_path = 'C:/Users/notfu/Desktop/News/'
#%%
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

#%%
import pickle
with open(file_path+'news_sorted_by_topic.pickle', 'rb') as f:
    news_dict = pickle.load(f)

#%%function for text to vector representation
device = torch.device('cuda')
def embeddings(news_dict, embedding_method = 'bert', embeddings_size = 768):
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer_sentiment = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    ##create hdf5 file
    print(embedding_method)
    filename = embedding_method+'_embeddings_filtered_sentiment.h5'
    file = h5py.File(filename, 'w')
    ##create groups
    #for group in news_dict.keys():
    #    group_ = file.create_group(group)        
    ##choose model
    if embedding_method == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = False, # Whether the model returns all hidden-states.
                                  )
    elif embedding_method == 'finbert':
        vocab = "finance-uncased"
        vocab_path = 'C:/Users/notfu/Desktop/News/analyst_tone/vocab'
        tokenizer = BertTokenizer(vocab_file = vocab_path, do_lower_case = True, do_basic_tokenize = True)
        model = BertModel.from_pretrained('C:/Users/notfu/Desktop/News/Finbert_uncased_weights/',
                                  output_hidden_states = False, # Whether the model returns all hidden-states.
                                  )
    else:
        model = SentenceTransformer('C:/Users/notfu/Desktop/News/paraphrase-MiniLM-L6-v2')
    model.to(device)
    #total_groups = file.create_group('topics')
    ##read text and embed to vectors
    for topic in news_dict.keys():      
        print(topic)
        
        group = file.create_group(topic)
        for date in news_dict[topic].keys():    
            CLS_token_matrix = np.empty(shape = (0,0))
            i=1
            for title in news_dict[topic][date]['title']:
                try:
                    inputs = tokenizer_sentiment(title, return_tensors="pt", padding=True)
                except ValueError:
                    print(inputs, date)
                outputs = finbert(**inputs)[0]
                
                label = np.argmax(outputs.detach().numpy())

                if label != 0:
                #preprocessing
                    title = title.lower() #lower case
                    title = remove_punctuation(title)# Removal of Punctuations
                    #如果embedding method 是Sbert, 修改text preprocessing
                    if embedding_method == 'Sbert':
                        embeddings = model.encode(title)
                        CLS_token_matrix = np.append(CLS_token_matrix, embeddings.reshape(1,-1))
                    else:
                        marked_text = "[CLS] " + title + " [SEP]" #add special tokens for bert model
                        tokenized_text = tokenizer.tokenize(marked_text)
                        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                        # Mark each of tokens as belonging to sentence "1".
                        segments_ids = [1] * len(tokenized_text)
                        # Convert inputs to PyTorch tensors
                        tokens_tensor = torch.tensor([indexed_tokens])
                        segments_tensors = torch.tensor([segments_ids])
                        with torch.no_grad():
                            tokens_tensor = tokens_tensor.to(device)
                            segments_tensors = segments_tensors.to(device)
                            outputs = model(tokens_tensor, segments_tensors)#output is tuple, first is last_hidden_state, second is pooler_output
                            #print(type(outputs[0][:,0,:]))#CLS token vector
                            #print(outputs[0][:,0,:].size())
                            output = outputs[0][:,0,:]
                            CLS_token_matrix = np.append(CLS_token_matrix, ((output.cpu()).numpy()).reshape(1,-1))
                    
                    CLS_token_matrix = CLS_token_matrix.reshape(i, 384)
                    i=i+1
                    
                #break
            
            #print(len(news_dict[topic][date]['title']))
            dataset = group.create_dataset(date, 
                              dtype = np.float32, data = CLS_token_matrix)#shape and dtype取決output tensor, outputs on a date的總長度
            
            
            #print(dataset.shape)
            #break
        #break
    #return CLS_token_matrix
    #break
    
    file.close()
#%%
file = h5py.File('Sbert_embeddings_filtered_sentiment.h5', 'r+')
model = SentenceTransformer('C:/Users/notfu/Desktop/News/paraphrase-MiniLM-L6-v2')
model.to(device)
topic = 'United States'
del file[topic] 
group = file.create_group(topic)
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer_sentiment = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
embedding_method = 'Sbert'
for date in news_dict[topic].keys():    
    CLS_token_matrix = np.empty(shape = (0,0))
    i=1
    for title in news_dict[topic][date]['title']:
        try:
            inputs = tokenizer_sentiment(title, return_tensors="pt", padding=True)
            
            #inputs = tokenizer_sentiment(title, return_tensors="pt", padding=True)
            outputs = finbert(**inputs)[0]
            
            label = np.argmax(outputs.detach().numpy())
        
            if label != 0:
            #preprocessing
                title = title.lower() #lower case
                title = remove_punctuation(title)# Removal of Punctuations
                #如果embedding method 是Sbert, 修改text preprocessing
                if embedding_method == 'Sbert':
                    embeddings_ = model.encode(title)
                    CLS_token_matrix = np.append(CLS_token_matrix, embeddings_.reshape(1,-1))
                
                
                CLS_token_matrix = CLS_token_matrix.reshape(i, 384)
                i=i+1
        except ValueError:
            pass
        #break

    #print(len(news_dict[topic][date]['title']))
    dataset = group.create_dataset(date, 
                      dtype = np.float32, data = CLS_token_matrix)

#%%Sbert embedding = 384
embeddings(news_dict, 'Sbert', 384
                          )  
#%%
f = h5py.File('Sbert_embeddings_filtered_sentiment.h5', 'r')
#%%
file.close()
#%%Bert_CLS_token test
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = False, # Whether the model returns all hidden-states.
                                  )
model.to(device)
# Put the model in "evaluation" mode, meaning feed-forward operation.
#model.eval()

#%% Bert_CLS_token
for date in news_dict['us'].keys():
    CLS_token_list = []
    for title in news_dict['us'][date]['title']:
        title = title.lower() #lower case
        title = remove_punctuation(title)# Removal of Punctuations
        marked_text = "[CLS] " + title + " [SEP]" #add special tokens for bert model
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # Mark each of tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            tokens_tensor = tokens_tensor.to(device)
            segments_tensors = segments_tensors.to(device)
            outputs = model(tokens_tensor, segments_tensors)#output is tuple, first is last_hidden_state, second is pooler_output
            print(type(outputs[0][:,0,:]))#CLS token vector
            print(outputs[0][:,0,:].size())
            CLS_token_list = CLS_token_list.append(outputs[0][:,0,:])
        break
    break
    news_dict[date]['CLS_token'] = CLS_token_list        

#%% Finbert test
model = BertModel.from_pretrained('C:/Users/notfu/Desktop/News/Finbert_uncased_weights/',
                                  output_hidden_states = False, # Whether the model returns all hidden-states.
                                  )
#%%
vocab = "finance-uncased"
vocab_path = 'C:/Users/notfu/Desktop/News/analyst_tone/vocab'
pretrained_weights_path = "C:/Users/notfu/Desktop/News/analyst_tone/pretrained_weights"

#%%
tokenizer = BertTokenizer(vocab_file = vocab_path, do_lower_case = True, do_basic_tokenize = True)
tokenized_sent = tokenizer.tokenize(news_dict['2021-07-01']['title'][4344])
ids_review  = tokenizer.convert_tokens_to_ids(tokenized_sent)
#%%Finbert_CLS_token

tokenizer = BertTokenizer(vocab_file = vocab_path, do_lower_case = True, do_basic_tokenize = True)
title = news_dict['2021-07-01']['title'][4344]
marked_text = "[CLS] " + title + " [SEP]" #add special tokens for bert model
tokenized_text = tokenizer.tokenize(marked_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [1] * len(tokenized_text)
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)#output is tuple, first is last_hidden_state, second is pooler_output
    print(type(outputs[0][:,0,:]))#CLS token vector
    print(outputs[0][:,0,:].shape)
#%%
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('C:/Users/notfu/Desktop/News/paraphrase-MiniLM-L6-v2')
embeddings = model.encode(news_dict['us']['2020-03-30']['title'][321566])
print(embeddings.shape)

#%%
max_len = 0
for topic in news_dict.keys():
    for date in news_dict[topic].keys():
        if len(news_dict[topic][date]['title']) > 50:
            print(topic, date, len(news_dict[topic][date]['title']))
            max_len = len(news_dict[topic][date]['title'])

#%%
#LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
device = torch.device('cuda')
df = pd.DataFrame(columns = ["date", "topic","title", "label"])
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3).to(device)
tokenizer_sentiment = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
i = 0
for topic in news_dict.keys():      
    print(topic)
    for date in news_dict[topic].keys():    
        for title in news_dict[topic][date]['title']:
            try:
                inputs = tokenizer_sentiment(title, return_tensors="pt", padding=True).to(device)
                outputs = finbert(**inputs)[0]
                
                label = np.argmax(outputs.detach().cpu().numpy())
                if label != 0:
                    df.loc[i] = [date ,topic, title, label]
                    i += 1
            except ValueError:
                pass
#%%
df.to_csv("news_filtered.csv", encoding = "utf_8_sig")
#%%
label_list = []
for title in df["title"]:
    inputs = tokenizer_sentiment(title, return_tensors="pt", padding=True).to(device)
    outputs = finbert(**inputs)[0]
    
    label = np.argmax(outputs.detach().cpu().numpy())
    label_list.append(label)
df["label"] = label_list