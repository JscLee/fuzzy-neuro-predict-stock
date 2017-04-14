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
#from theano.printing import Print
import random
from os.path import exists
from os import mkdir

dir='C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170414\\1'
pred=[]
outputs=[]

if not exists(dir):
    mkdir(dir)
def plot_pre():
    plot(pred)
    plot(outputs)
    show()
def shuffle_exp(inputs,outputs,i):
    random.seed(i)
    random.shuffle(inputs,random.random)
    random.seed(i)
    random.shuffle(outputs,random.random)

def layout_1(inputs,outputs):
    #2初始化网络(2层隐含层)
    x = T.matrix('x')
    #x.tag.test_value=inputs[:3]
    #x.tag.test_value=np.matrix(np.ones([3,6]),dtype=theano.config.floatX)

    w1=theano.shared(np.array(np.random.rand(6,10), dtype=theano.config.floatX))
    w2=theano.shared(np.array(np.random.rand(10,30), dtype=theano.config.floatX))
    w3=theano.shared(np.array(np.random.rand(30,1), dtype=theano.config.floatX))


    b1 = theano.shared(1.)
    b2 = theano.shared(1.)
    b3 = theano.shared(1.)
    
    learning_rate = T.scalar('learning_rate')

    #构造四层全连接网络
    print 'Init network'
    #print tmp.tag.test_value
    a1=T.nnet.relu((T.dot(x,w1)+b1),0.1)
    #print a1.tag.test_value

    a2=T.nnet.relu((T.dot(a1,w2)+b2),0.1)
    #print a2.tag.test_value
    
    #tmp=copy(a2)
    #tmp.extend(a1)
    #x2 = T.stack(tmp,axis=1)
    a3 = T.nnet.relu((T.dot(a2,w3)+b3),0.1)
    #print a3.tag.test_value

    '''
    tmp=a_hat-a3
    print 'tmp test value 1:'
    print tmp.test_value
    tmp=tmp**2
    print 'tmp test value 2:'
    print tmp.test_value
    '''    
    a_hat = T.matrix('a_hat') #Actual output

    cost = T.sum((a_hat - a3)**2)
    #print 'cost test value:'
    #print cost.tag.test_value
    

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
        inputs = [x,a_hat,learning_rate],
        outputs = [a1,a2,a3,w1,w2,w3,dw1,dw2,dw3,a3,cost],
        updates = u_list
        #profile=True
    )

    print 'Start Training'
    # 迭代
    cost = []
    w_=[]
    dw_=[]
    b_size=30
    lr= 0.00001
    itr=10000
    end_train=False #结束标志

    for iteration in xrange(itr):
        batch=b_size
        cost_iter=0
        pred=[]

        while batch<len(inputs):
            a1,a2,a3,w_1,w_2,w_3,dw_1,dw_2,dw_3,p_, c_ = train(inputs[batch-b_size:batch], outputs[batch-b_size:batch],lr)
            cost_iter=cost_iter+c_
            batch=batch+b_size
            pred.extend(p_)
        a1,a2,a3,w1,w2,w3,dw1,dw2,dw3,p_, c_ = train(inputs[batch-b_size:len(inputs)], outputs[batch-b_size:len(inputs)],lr) #last batch
        cost_iter=cost_iter+c_
        pred.extend(p_)
        #w_.append([w1,w2,w3])    
        #dw_.append([dw1,dw2,dw3])
        cost.append(cost_iter)  #######Stop here#####

        print '###Iter: '+str(iteration)+'  cost: '+str(cost_iter)+' ###'   

        shuffle_exp(inputs,outputs,iteration)   #打乱训练集，结束一轮迭代
        if end_train:   #结束训练
            break
    '''
    f1=open(dir+'\\w.txt','wb')
    f2=open(dir+'\\wb.txt','wb')
    print >>f1,w_[200:500]
    print >>f2,dw_[200:500]
    f1.close()
    f2.close()
    '''
    '''
    # 打印输出    
    print 'The outputs of the NN are:'
    for i in range(len(inputs)):
        print 'Real: %.5f | Predc: %.5f' % (outputs[i][0],pred[i][0])
    print 'Final cost: %.5f'%(cost[-1])
    '''

    #存储
    output=open(dir+'\\data.pkl','wb')
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

    pickle.dump(temp,output)

    # 绘制损失图:
    print '\nThe flow of cost during model run is as following:'
    #matplotlib inline
    figure(1)
    title('Variation of Cost')
    plot(cost,label='cost')
    legend(loc='upper left')
    savefig(dir+'\\cost.png')

    figure(2)
    title('Prediction and Expectation')
    plot(pred,label='p')
    plot(result,label='e')
    legend(loc='upper left')
    savefig(dir+'\\prediction.png')
    show()

        

input=open('1day_matrix.pkl','rb')
pack=pickle.load(input)
result=pack['result']
data=pack['data']
layout_1(data,result)
