# -*- coding:utf-8 -*-
import theano
import theano.tensor as T
from theano import function
import numpy as np
import pickle
from random import random

def layout_1(inputs,outputs):
    #2层网络(隐含层)
    x = T.matrix('x')
    w1 = theano.shared(np.array([random(),random(),random(),random(),random(),random()]))
    w2 = theano.shared(np.array([random(),random(),random(),random(),random(),random()]))
    w3 = theano.shared(np.array([random(),random(),random(),random(),random(),random()]))
    w4 = theano.shared(np.array([random(),random(),random(),random(),random(),random()]))
    w5 = theano.shared(np.array([random(),random(),random(),random(),random(),random()]))
    w6 = theano.shared(np.array([random(),random(),random(),random(),random(),random()]))
    b1 = theano.shared(1.)
    b2 = theano.shared(1.)
    learning_rate = 0.000005
    

    print 'Init network'
    a1 = 1/(1+T.exp(-T.dot(x,w1)-b1))
    a2 = 1/(1+T.exp(-T.dot(x
,w2)-b1))
    a3 = 1/(1+T.exp(-T.dot(x,w3)-b1))
    a4 = 1/(1+T.exp(-T.dot(x,w4)-b1))
    a5 = 1/(1+T.exp(-T.dot(x,w5)-b1))
    a6 = 1/(1+T.exp(-T.dot(x,w6)-b1))
    x2 = T.stack([a1,a2,a3,a4,a5,a6],axis=1)
    a7 = 1/(1+T.exp(-T.dot(x2,w3)-b2))    

    a_hat = T.vector('a_hat') #Actual output
    cost = -(a_hat*T.log(a7) + (1-a_hat)*T.log(1-a7)).sum()
    dw1,dw2,dw3,dw4,dw5,dw6,db1,db2 = T.grad(cost,[w1,w2,w3,w4,w5,w6,b1,b2])

    train = function(
        inputs = [x,a_hat],
        outputs = [a3,cost],
        updates = [
            [w1, w1-learning_rate*dw1],
            [w2, w2-learning_rate*dw2],
            [w3, w3-learning_rate*dw3],
            [w4, w1-learning_rate*dw4],
            [w5, w2-learning_rate*dw5],
            [w6, w3-learning_rate*dw6],
            [b1, b1-learning_rate*db1],
            [b2, b2-learning_rate*db2]
        ]
    )

    print 'Start Training'
    # 遍历输入并计算输出:
    cost = []
    for iteration in range(30000):
        pred, cost_iter = train(inputs, outputs)
        print '###Iter: '+str(iteration)+'  cost: '+str(cost_iter)+' ###'
        cost.append(cost_iter)
    # 打印输出    
    print 'The outputs of the NN are:'
    for i in range(len(inputs)):
        print 'Real: %.5f | Predc: %.5f' % (outputs[i],pred[i])


    #存储
    output=open('C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170320\\1\\1.pkl','wb')
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
    import matplotlib.pyplot as plt
    #matplotlib inline
    plt.figure(1)
    plt.title('Variation of Cost')
    plt.plot(cost)

    plt.figure(2)
    plt.title('Prediction and Expectation')
    plt.plot(pred)
    plt.plot(result)
    plt.show()
        
        

input=open('1day_matrix.pkl','rb')
pack=pickle.load(input)
result=pack['result']
data=pack['data']
layout_1(data,result)
