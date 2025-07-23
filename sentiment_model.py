import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer,AutoModelForSequenceClassification,pipeline
import os

def get_file(i):
    path = './model_directory/shin_file'
    file_list = os.listdir(path)
    file_list_py = [file for file in file_list if file.endswith('{}.csv'.format(i))]
    file_list_0=file_list_py[0]
    return file_list_0

def get_csv(flist):
    file_lt=flist
    df=pd.read_csv('./model_directory/shin_file/{}'.format(file_lt),index_col=0,header=0,sep=',')    
    return df

def text_token():
    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert",return_tensors='pt')
    return tokenizer

def Finbert_Model():
    model=AutoModelForSequenceClassification.from_pretrained('./model_directory/')
    return model

def text_proprecessing(company):
    pt=re.compile(r'[^\w\s\d]+')
    cm_t=company['NewsTitle']
    
    for i in range(0,company.shape[0]):
        company['NewsTitle'][i] = str(cm_t[i]).replace('\n','').replace('\t','')
        pat_match=pt.sub('',company['NewsTitle'][i])
        company['NewsTitle'][i]=pat_match
    return company

def model_pipeline(token,models):
    classifier=pipeline('sentiment-analysis',tokenizer=token,model=models)
    return classifier

def sentiment_predict(company,pipeline):
    company=company
    classifier=pipeline
    label_=[]
    
    txt=classifier(company['NewsTitle'][0])
    for r in txt:
        label_.append(r['label'])
    # txt_label=txt['label'].map({'LABEL_0':1,'LABEL_1':-1})
    company['sentiment']=label_
    company['sentiment']=company['sentiment'].map({'LABEL_0':1,'LABEL_1':-1})
    
    return company

def time_over_sentiment_predict(company,pipeline):
    
    positive_count=0
    negative_count=0
    
    company=company
    classifier=pipeline
    
    
    label_,label_result,scores=[],[],[]
    
    cnt=company['NewsTitle']
    
    for title in cnt:
        txt=classifier(title)
        label_.append(txt)    
    
    for i,result in enumerate(label_):
        for r in result:
            label_result.append(r['label'])
            scores.append(r['score'])
        
    company['sentiment']=label_result
    company['sentiment']=company['sentiment'].map({'LABEL_0':1,'LABEL_1':-1})

    senti = company['sentiment']
    
    sum_scores=sum(scores)/len(scores)
    
    for senti in senti:
        if senti == 1:
            positive_count+=1
        elif senti == -1:
            negative_count+=1
    
    if positive_count > negative_count:
        return 1
    if positive_count < negative_count:
        return -1
    if positive_count == negative_count:
        if sum_scores > 0.5:
            return 1
        else:
            return -1
    