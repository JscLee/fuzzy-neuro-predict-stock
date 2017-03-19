from __future__ import division
import numpy as np
import pickle

def read_data():
    input=open('filter_cor.pkl','rb')
    result=pickle.load(input)
    data=pickle.load(input)
    index=pickle.load(input)
    return index,data,result

def normalize(index,data):
    for i in xrange(data.shape[0]):
        avg_=data[i].mean()
        min_=data[i].min()
        max_=data[i].max()
        data[i]=(data[i]-avg_)/(max_-min_)
    return data

def integrate_byday(data,result,num):
    #Data:matrix, rows:features, columns: dates
    #Integrate by 
    train_exp=[]
    for i in xrange(data.shape[1]-num+1):
        a={}
        a['x']=data[:,i:i+num]
        a['y']=result[i+num]
        train_exp.append(a)
    return train_exp

index,data,result=read_data()
data=normalize(index,data)
data_5=integrate_byday(data,result,1)
output=open('data_1day','wb')
pickle.dump(output,data_5)