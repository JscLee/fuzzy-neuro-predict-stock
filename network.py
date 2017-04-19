# -*- coding:utf-8 -*-
import theano
import theano.tensor as T
from theano import function
import numpy as np
import pickle
from copy import copy
from random import random
from matplotlib.pyplot import *
from constant import *
theano.config.compute_test_value = 'warn'
#from theano.printing import Print
import random
from os.path import exists
from os import mkdir

if not exists(dir):
    mkdir(dir)

class network:
    train_=0
    predict_=0 
    inputs=[]
    outputs=[]
    
    def __init__(self):
        #初始化网络(3层隐含层)
        x = T.matrix('x')
        #x.tag.test_value=inputs[:3]
        #x.tag.test_value=np.matrix(np.ones([3,6]),dtype=theano.config.floatX)

        w1=theano.shared(np.array(np.random.rand(11,15), dtype=theano.config.floatX))
        w2=theano.shared(np.array(np.random.rand(15,15), dtype=theano.config.floatX))
        w3=theano.shared(np.array(np.random.rand(15,5), dtype=theano.config.floatX))
        w4=theano.shared(np.array(np.random.rand(5,1), dtype=theano.config.floatX))


        b1 = theano.shared(1.)
        b2 = theano.shared(1.)
        b3 = theano.shared(1.)
        b4 = theano.shared(1.)
    
        learning_rate = T.scalar('learning_rate')

        #构造四层全连接网络
        print 'Init network'
        #print tmp.tag.test_value
        a1=T.nnet.relu((T.dot(x,w1)+b1),0.1)
        #print a1.tag.test_value

        a2=T.nnet.relu((T.dot(a1,w2)+b2),0.1)
        #print a2.tag.test_value
        a3=T.nnet.relu((T.dot(a2,w3)+b3),0.1)
    
        #tmp=copy(a2)
        #tmp.extend(a1)
        #x2 = T.stack(tmp,axis=1)
        a4 = T.nnet.relu((T.dot(a3,w4)+b4),0.1)
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

        cost = T.sum((a_hat - a4)**2)
        #print 'cost test value:'
        #print cost.tag.test_value
    

        dw1,dw2,dw3,dw4,db1,db2,db3,db4=T.grad(cost,[w1,w2,w3,w4,b1,b2,b3,b4])

        #定义学习规则
        u_list=[]
        u_list.append([w1,w1-learning_rate*dw1])
        u_list.append([w2,w2-learning_rate*dw2])
        u_list.append([w3,w3-learning_rate*dw3])
        u_list.append([w4,w4-learning_rate*dw4])
        u_list.append([b1,b1-learning_rate*db1])
        u_list.append([b2,b2-learning_rate*db2])
        u_list.append([b3,b3-learning_rate*db3])
        u_list.append([b4,b3-learning_rate*db4])

        #训练函数
        self.train_ = function(
            inputs = [x,a_hat,learning_rate],
            outputs = [a1,a2,a3,w1,w2,w3,w4,dw1,dw2,dw3,dw4,a4,cost],
            updates = u_list
            #profile=True
        )

        #预测函数
        self.predict_ = function(
            inputs = [x],
            outputs = [a4]
            #profile=True
        )

    def load_data(self,path):
        input=open(path,'rb')
        pack=pickle.load(input)
        self.outputs=pack['result']
        self.inputs=pack['data']
    def shuffle_exp(self,i):
        random.seed(i)
        random.shuffle(self.inputs,random.random)
        random.seed(i)
        random.shuffle(self.outputs,random.random)


    def train(self):
        print 'Start Training'
        # 迭代
        cost = []
        w_=[]
        dw_=[]
        b_size=60
        lr= 0.00000001
        itr=100000
        end_train=False #结束标志
        iteration=0

        while iteration<itr:
            batch=b_size
            cost_iter=0
            pred=[]

            while batch<len(self.inputs):
                a1,a2,a3,w_1,w_2,w_3,w_4,dw_1,dw_2,dw_3,dw_4,p_, c_ = self.train_(self.inputs[batch-b_size:batch], self.outputs[batch-b_size:batch],lr)
                cost_iter=cost_iter+c_
                batch=batch+b_size
                pred.extend(p_)
            a1,a2,a3,w_1,w_2,w_3,w_4,dw_1,dw_2,dw_3,dw_4,p_, c_ = self.train_(self.inputs[batch-b_size:len(self.inputs)], self.outputs[batch-b_size:len(self.inputs)],lr) #last batch
            cost_iter=cost_iter+c_
            pred.extend(p_)
            #w_.append([w1,w2,w3])    
            #dw_.append([dw1,dw2,dw3])
            cost.append(cost_iter)  ## #####Stop here#####

            print '###Iter: '+str(iteration)+'  cost: '+str(cost_iter)+' ###'   

            self.shuffle_exp(iteration)   #打乱训练集，结束一轮迭代
            iteration=iteration+1

            if end_train:   #结束训练
                break
            '''
            if iteration%200==0 and iteration>200:   
                figure(0)
                plot(np.array(outputs)-np.array(pred),label=iteration)
                legend(loc='upper left')
            '''     


        #存储
        output=open(dir+'\\data.pkl','wb')
        temp={}
        temp['pred']=pred
        temp['expec']=self.outputs
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
        plot(result,label='e')
        plot(pred,label='p')
        legend(loc='upper left')
        savefig(dir+'\\prediction.png')
        show()

    def predict(self):
        self.outputs=self.predict_(inputs)      
    
n=network()
n.load_data('train.pkl')
n.train()
n.load_data('test.pkl')
n.predict()
n.check()         