# -*- coding:utf-8 -*-
import tushare as ts
import pandas as pd
import numpy as np
import pickle
from copy import copy
from pandas import DataFrame,Series
index=['open','close','high','low','volume','v_ma20','alpha#6','alpha#23','alpha#28','alpha#54','alpha#101']
lag=20

def cal_cor():
    cor=list()
    df=DataFrame()
    input= open('raw_data.pkl', 'rb')
    df=pickle.load(input)   
    data=np.matrix(df)
    data=(data.T)[1:,lag:-1]
    rate=df.as_matrix(['ReturnRate'])[lag+1:]
    #filter cor
    a=[]
    for i in xrange(data.shape[0]):
        cor.append(np.corrcoef(rate.T,data[i])[0][1])
        if abs(cor[i])<0.03:
            a.append(i)
            print index[i]+':'+str(cor[i])
        else:
            print index[i]+':'+str(cor[i])
    data=np.delete(data,a,axis=0)
    #delete filtered features in data
    a.reverse()
    index_cor=copy(index)
    for i in a:
        index_cor.pop(i)

    output=open('filter_cor.pkl','wb')
    pack={}
    pack['result']=rate
    pack['data']=data
    pack['index']=index_cor
    pickle.dump(pack,output)
    output.close()

def get_data():
    df=DataFrame()
    df=(ts.get_hist_data('hs300',start='2013-01-01',end='2017-01-01',ktype='D'))[index[:6]+['price_change']]
    df=df.sort_index()
    #ReturnRate=ln(s(t）/s(t-1))  lag=1
    df.insert(0,'ReturnRate',df['close'])    
    temp=1
    for i in df.index:
        df['ReturnRate'][i]=np.log(df['close'][i]/temp)
        temp=df['close'][i]

    #alpha#6 lag=10
    df.insert(7,'alpha#6',df['open'])
    for i in xrange(10,len(df.index)):
        df['alpha#6'][i]=np.corrcoef(df['alpha#6'][i-10:i],df['volume'][i-10:i])[0][1]

    #alpha#23 lag=20
    df.insert(8,'alpha#23',df['high'])   
        #过去20天最高价
    df['alpha#23'][20]=df['high'][:20].sum()/20.0   
    get_larger=lambda x,y:x if x>y else y
    for i in xrange(21,len(df.index)):
        df['alpha#23'][i]=(20*df['alpha#23'][i-1]-df['alpha#23'][i-21]+df['alpha#23'][i-1])/20.0
    for i in xrange(20,len(df.index)):
        if df['high'][i]>df['alpha#23'][i]:
            df['alpha#23'][i]=-1*(df['alpha#23'][i-2]-df['high'][i])
        else:
            df['alpha#23'][i]=0

    #alpha#28 lag=5
    df.insert(9,'alpha#28',df['high'])   
    temp=0
    for i in xrange(5,len(df.index)):
        df['alpha#28'][i]=np.corrcoef(df['v_ma20'][i-5:i],df['low'][i-5:i])[0][1]+(df['high'][i]+df['low'][i])/2.0-df['close'][i]
        temp=temp+abs(df['alpha#28'][i])
    for i in xrange(5,len(df.index)):
        df['alpha#28'][i]=df['alpha#28'][i]/temp
    
    #alpha#54 lag=0
    df.insert(10,'alpha#54',df['high'])  
    for i in xrange(len(df.index)):
        df['alpha#54'][i]=(-1*(df['low'][i]-df['close'][i])*pow(df['open'][i],5))/(-1*(df['low'][i]-df['high'][i])*pow(df['close'][i],5))
        
    #alpha#101 lag=0
    df.insert(11,'alpha#101',df['high'])  
    for i in xrange(len(df.index)):
        df['alpha#101'][i]=(df['close'][i]-df['open'][i])/(df['high'][i]-df['low'][i]+0.001) 
    
    df=df.drop(['price_change'],axis=1)       
        
    output = open('raw_data.pkl', 'wb')
    pickle.dump(df,output)
    pickle.dump(index,output)
    output.close()


get_data()
cal_cor()
