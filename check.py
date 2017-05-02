import pickle
import numpy as np
from constant import *

f=open(dir+'\\prediction.pkl','rb')
pack=pickle.load(f)
x=pack['pred']
y=pack['expec']
MSE=np.sum(pow((np.array(x)-np.array(y)),2))/len(x)
    
f=open('test.pkl','rb')
pack=pickle.load(f)
data=pack['data']

close=np.array([d[1] for d in data])

count=0
for i in xrange(len(x)):
    if (x[i]-close[i])*(y[i]-close[i])>0:
        count=count+1
ratio=count*1.0/len(x)

print MSE,ratio