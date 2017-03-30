# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pickle
from matplotlib.pyplot import *
dir='C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170330\\1'
def read_data():
    input=open('filter_cor.pkl','rb')
    pack=pickle.load(input)
    result=pack['result']
    data=pack['data']
    index=pack['index']
    return pack

def normalize(index,data):
    for i in xrange(data.shape[0]):
        avg_=data[i].mean()
        data[i]=data[i]-avg_
        min_=data[i].min()
        max_=data[i].max()
        data[i]=(data[i]-min_)/(max_-min_)
    return data
pack=read_data()
figure(1)
plot(pack['result'].T.tolist()[0])
data=np.matrix([])
data=np.mean(data, axis = 0)
data=normalize(pack['index'],pack['data'])
result=normalize(pack['index'],pack['result'])
#Data:matrix: index by features rows:features, columns: dates
#pack['data']:index by examples
pack['data']=data.T.tolist()
pack['result']=result.T.tolist()

#绘图
if True:
    
    for i in xrange(len(pack['index'])):
        plot(data.tolist()[i],label=pack['index'][i])
    #plot(pack['result'],label='result')
    #legend(loc='upper left')
    savefig(dir+'\\feature.png',dpi=200)

    for i in xrange(len(pack['index'])):
        figure()
        plot(data.tolist()[i],label=pack['index'][i])
        #plot(pack['result'],label='result')
        #legend(loc='upper left')
        savefig(dir+'\\'+pack['index'][i]+'.png',dpi=200)
    
    figure()
    plot(pack['result'],label='result')
    savefig(dir+'\\result.png',dpi=200)
    

output=open('1day_matrix.pkl','wb')
pickle.dump(pack,output)