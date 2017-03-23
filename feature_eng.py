# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pickle
from matplotlib.pyplot import *

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
        min_=data[i].min()
        max_=data[i].max()
        data[i]=(data[i]-min_)/(max_-min_)
    return data
pack=read_data()
figure(1)
plot(pack['result'].T.tolist()[0])
data=np.matrix([])
data=normalize(pack['index'],pack['data'])
result=normalize(pack['index'],pack['result'].T)
#Data:matrix: index by features rows:features, columns: dates
#pack['data']:index by examples
pack['data']=data.T.tolist()
pack['result']=result.tolist()[0]

#绘图
if True:
    for i in xrange(len(pack['index'])):
        plot(data.tolist()[i],label=pack['index'][i])
    plot(pack['result'],label='result')
    legend(loc='upper left')
    savefig('C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170320\\2\\feature.png',dpi=200)

    for i in xrange(len(pack['index'])):
        figure()
        plot(data.tolist()[i],label=pack['index'][i])
        plot(pack['result'],label='result')
        legend(loc='upper left')
        savefig('C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170320\\2\\'+pack['index'][i]+'.png',dpi=200)

output=open('1day_matrix.pkl','wb')
pickle.dump(pack,output)