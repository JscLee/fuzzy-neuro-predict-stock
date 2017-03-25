#import theano
#import theano.tensor as T
import pickle
from matplotlib.pyplot import *

input=open('C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170323\\2\\1.pkl','rb')
temp=pickle.load(input)
pred=temp['pred']
outputs=temp['expec']
cost=temp['cost']
iteration=temp['iter']
learning_rate=temp['l_rate']
w=temp['w']

#matplotlib inline
figure(1)
title('Variation of Cost')
plot(cost[10:],label='cost')
legend(loc='upper left')
savefig('C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170323\\2\\cost.png')

figure(2)
title('Prediction and Expectation')
plot(pred,label='p')
plot(result,label='e')
legend(loc='upper left')
savefig('C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170323\\2\\prediction.png')
show()
