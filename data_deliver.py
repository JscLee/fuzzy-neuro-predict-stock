# -*- coding:utf-8 -*-
import tushare as ts
import pandas as pd
import numpy as np
import pickle
from copy import copy
from pandas import DataFrame,Series
from matplotlib.pyplot import *
index=['open','close','high','low','volume','v_ma20','alpha#6','alpha#23','alpha#28','alpha#54','alpha#101']
lag=25
dir='C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170330\\1'
def cal_cor():
    cor=list()
    df=DataFrame()
    input= open('raw_data.pkl', 'rb')
    df=pickle.load(input)   
    data=np.matrix(df,dtype=np.float64)
    data=(data.T)[1:,:-1]
    rate=np.matrix(df['ReturnRate'][1:],dtype=np.float64)
    #filter cor
    a=[]
    for i in xrange(data.shape[0]):
        cor.append(np.corrcoef(rate,data[i])[0][1])
        if abs(cor[i])<0.034:
            a.append(i)
            print 'del:'+index[i]+':'+str(cor[i])
        else:
            print 'keep:----'+index[i]+':'+str(cor[i])+'----'
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

    #绘图
    if True:
        fig=figure()
        #六个特征同时
        for i in xrange(len(pack['index'])):
            plot(pack['data'].tolist()[i],label=pack['index'][i])
        plot(pack['result'].T.tolist()[0],linewidth=2.5,label='result')
        legend(loc='upper left')
        savefig(dir+'\\raw_data.png',dpi=200)
        #分别描绘六个特征
        for i in xrange(len(pack['index'])):
            figure()
            plot(pack['data'].tolist()[i],label=pack['index'][i])
            legend(loc='upper left')
            savefig(dir+'\\'+pack['index'][i]+'_raw.png',dpi=200)
       

def get_data():
    df=DataFrame()
    df=(ts.get_k_data('000300',index=True,start='2009-01-01',end='2017-01-01',ktype='D'))[['date']+index[:5]]
    df=df.sort_index()    
    date=df['date']
    df=df[index[:5]]
    df['low'][1510]=3932.870 #接口包中数据错误

    #ReturnRate=ln(s(t）/s(t-1))  lag=1
    df.insert(0,'ReturnRate',df['close'])   
    for i in xrange(1,len(df.index)):
        df['ReturnRate'][i]=np.log(df['close'][i]/df['close'][i-1])

    #alpha#6 lag=10
    df.insert(6,'alpha#6',df['open'])
    for i in xrange(10,len(df.index)):
        df['alpha#6'][i]=np.corrcoef(df['open'][i-10:i],df['volume'][i-10:i])[0][1]

    
    #alpha#23 lag=20
    df.insert(7,'alpha#23',df['high'])   
    #过去20天最高价的均值
    df['alpha#23'][20]=df['high'][:20].sum()/20.0  
    for i in xrange(21,len(df.index)):
        df['alpha#23'][i]=(20*df['alpha#23'][i-1]-df['high'][i-21]+df['high'][i-1])/20.0
    '''
    plot(df['alpha#23'],df['date'],label='20_high_avg')
    plot(df['high'],df['date'],label='high')
    '''
    for i in xrange(20,len(df.index)):
        if df['high'][i]>df['alpha#23'][i]: #今日高于过去20天平均，呈上涨趋势
            df['alpha#23'][i]=-1*(df['high'][i-2]-df['high'][i])
        else:
            df['alpha#23'][i]=0
    '''
    plot(df['alpha#23'],df['date'],label='alpha#23')
    legend(loc='upper left')
    show()
    '''
    #alpha#28 lag=5
    df.insert(6,'v_ma20',df['volume'])   
    df['v_ma20'][20]=df['volume'][:20].sum()/20.0  
    for i in xrange(21,len(df.index)):
        df['v_ma20'][i]=(20*df['v_ma20'][i-1]-df['volume'][i-21]+df['volume'][i-1])/20.0

    df.insert(9,'alpha#28',df['high'])   
    temp=0
    for i in xrange(25,len(df.index)):
        df['alpha#28'][i]=np.corrcoef(df['v_ma20'][i-5:i],df['low'][i-5:i])[0][1]+(df['high'][i]+df['low'][i])/2.0-df['close'][i]

    temp=abs(df['alpha#28'][25:]).sum()
    for i in xrange(25,len(df.index)):
        df['alpha#28'][i]=df['alpha#28'][i]/temp



    #alpha#54 lag=0
    df.insert(10,'alpha#54',df['high'])  
    for i in xrange(len(df.index)):
        df['alpha#54'][i]=(-1*(df['low'][i]-df['close'][i])*pow(df['open'][i],5))/((df['low'][i]-df['high'][i])*pow(df['close'][i],5))

    #alpha#101 lag=0
    df.insert(11,'alpha#101',df['high']) 
    for i in xrange(len(df.index)):
        df['alpha#101'][i]=(df['close'][i]-df['open'][i])/(df['high'][i]-df['low'][i]+0.001) 
        
    
    df=df[lag:]
    '''
    #绘图
    if True:
        plot(df['alpha#6'],label='alpha#6')
        plot(df['alpha#23'],label='alpha#23')
        plot(df['alpha#28'],label='alpha#28')
        plot(df['alpha#54'],label='alpha#54')
        plot(df['alpha#101'],label='alpha#101')
        legend(loc='upper left')
        show()
    '''  
    output = open('raw_data.pkl', 'wb')
    pickle.dump(df,output)
    pickle.dump(index,output)
    output.close()




get_data()
cal_cor()
