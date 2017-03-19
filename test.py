import theano
import theano.tensor as T

x=T.fvector('x')
s=10-x
f=theano.function([x],s)

f([1,2,3])