# Research-Project
Forecasting US economic indicators.
### Project intro

本研究以預測美國重要經濟領先指標為目標，利用現有的公開財經新聞資料，經語言預訓練模型進行資料預處理，並運用階層式時間序列模型對新聞資料提取有價值的資訊，搭配注意力機制計算不同新聞主題內容對預測經濟指標之重要性，將新聞主題之重要程度加入至特徵向量計算，最後以多任務學習架構預測多個經濟領先指標。經過實證研究，本研究對美國經濟領先指標確實有預測的能力，並且能在經濟指標公布前兩周至一個月前，就能預測下一期指標的走勢。

### Methods

* BERT
* Multi-task learning
* LSTM
* Attention

### Instructions
#### <資料預處理>
1. data-preprocessing/topic_pickle.py --->資料前處理，將新聞以主題排序，資料格式以pickle檔儲存，輸出檔案名稱：news_sorted_by_topic.pickle
2. data-preprocessing/Bert_sentence_embedding_to_hdf5.py --->資料前處理，新聞文字轉向量，資料格式：hdf5，輸出檔案名稱：bert_embedding_filtered_sentiment.h5。
**由於檔案過大，因此資料預處理輸出的檔案沒有放在github上，需要檔案者可寄信至信箱提供下載連結
#### <模型與訓練>
1. src/config.yml 修改模型參數
2. src/Datapipe.py 將新聞資料依據參數設定的天數與主題，整合新聞資料並分成train, valid, test資料集
3. src/Executor.py train, test, calculate accuracy等function
4. src/Model.py 主要模型
