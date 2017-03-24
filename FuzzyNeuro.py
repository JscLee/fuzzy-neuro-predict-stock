# -*- coding:utf-8 -*-
import theano
import theano.tensor as T
from theano import function
import numpy as np
import pickle
from copy import copy
from random import random
from matplotlib.pyplot import *
theano.config.compute_test_value = 'warn'
from theano.printing import Print

def layout_1(inputs,outputs):
    #2初始化网络(2层隐含层)
    x = T.matrix('x')
    w1=theano.shared(np.array(np.random.rand(6,6), dtype=theano.config.floatX))
    w2=theano.shared(np.array(np.random.rand(6,6), dtype=theano.config.floatX))
    w3=theano.shared(np.array(np.random.rand(6,1), dtype=theano.config.floatX))

    b1 = theano.shared(1.)
    b2 = theano.shared(1.)
    b3 = theano.shared(1.)
    learning_rate = 0.01
    

    #构造四层全连接网络
    print 'Init network'
    a1=1/(1+T.exp(-T.dot(x,w1)-b1))
    a2=1/(1+T.exp(-T.dot(a1,w2)-b2))
    
    #tmp=copy(a2)
    #tmp.extend(a1)
    #x2 = T.stack(tmp,axis=1)
    a3 = 1/(1+T.exp(-T.dot(a2,w3)-b3))    

    a_hat = T.vector('a_hat') #Actual output

    cost = T.sum((a_hat - a3)**2)
    

    dw1,dw2,dw3,db1,db2,db3=T.grad(cost,[w1,w2,w3,b1,b2,b3])

    #定义学习规则
    u_list=[]
    u_list.append([w1,w1-learning_rate*dw1])
    u_list.append([w2,w2-learning_rate*dw2])
    u_list.append([w3,w3-learning_rate*dw3])
    u_list.append([b1,b1-learning_rate*db1])
    u_list.append([b2,b2-learning_rate*db2])
    u_list.append([b3,b3-learning_rate*db3])
    #训练函数
    train = function(
        inputs = [x,a_hat],
        outputs = [a2[0],a3,cost],
        updates = u_list
        #profile=True
    )

    print 'Start Training'
    # 遍历输入并计算输出:
    cost = []
    for iteration in range(300000):
        _pre,pred, cost_iter = train(inputs, outputs)
        print '###Iter: '+str(iteration)+'  cost: '+str(cost_iter)+' ###'
        cost.append(cost_iter)

    # 打印输出    
    print 'The outputs of the NN are:'
    for i in range(len(inputs)):
        print 'Real: %.5f | Predc: %.5f' % (outputs[i],pred[i])


    #存储
    output=open('C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170323\\1\\1.pkl','wb')
    temp={}
    temp['pred']=pred
    temp['expec']=outputs
    temp['cost']=cost
    temp['iter']=iteration
    temp['l_rate']=learning_rate
    temp['w']=[]
    temp['w'].append(w1)
    temp['w'].append(w2)
    temp['w'].append(w3)
    temp['w'].append(w4)
    temp['w'].append(w5)
    temp['w'].append(w6)

    pickle.dump(temp,output)

    # 绘制损失图:
    print '\nThe flow of cost during model run is as following:'
    #matplotlib inline
    figure(1)
    title('Variation of Cost')
    plot(cost)
    legend(loc='upper left')
    savefig('C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170323\\1\\cost.png')

    figure(2)
    title('Prediction and Expectation')
    plot(pred,label='p')
    plot(result,label='e')
    legend(loc='upper left')
    savefig('C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170323\\1\\prediction.png')
    show()

        

input=open('1day_matrix.pkl','rb')
pack=pickle.load(input)
result=pack['result']
data=pack['data']
layout_1(data,result)
