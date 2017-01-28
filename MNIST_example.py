from Totem import *
import theano.tensor as T
import theano
import numpy as np
import gzip
import cPickle as pickle

def get_data(data_xy):
    datax, datay=data_xy
    sharedx=theano.shared(np.asarray(datax, dtype=theano.config.floatX), borrow=True)
    sharedy=theano.shared(np.asarray(datay, dtype=theano.config.floatX), borrow=True)
    return sharedx, sharedy

r=RNG(12234)
f=gzip.open("/home/tanmaya/mnist.pkl.gz", "rb")
Train,Valid,Test=pickle.load(f)
Trainx, Trainy=get_data(Train)
Vx, Vy=get_data(Valid)
n_batches=100
batch_size=500
x=Model((1,28,28))
n_iters=100
x.add_layer(ConvLayer(10,(4,4),r, activation=None, init_method="he", strides=(2,2)))
x.add_layer(BNLayer())
x.add_layer(ActLayer(activation="relu"))
x.add_layer(PoolLayer((2,2), mode="max"))
x.add_layer(ConvLayer(5,(3,3),r, activation=None))
x.add_layer(BNLayer())
x.add_layer(ActLayer(activation="relu"))
x.add_layer(PoolLayer((2,2)))
x.add_layer(FlattenLayer())
x.add_layer(FCLayer(100, r, "relu"))
x.add_layer(FCLayer(10,r,"softmax"))

opt=SGD_momentum(0.3,0.05,"cce", False, Trainx.reshape((50000,1,28,28)), Trainy)
run=x.get_runner(Vx.reshape((10000, 1, 28, 28)), Vy)
x.build_optimizer(opt)

g=np.arange(50000)
h=np.arange(10000)

for i in xrange(n_iters):
    cost=[]
    r.Shuffle(g)
    x.change_is_training(True)
    for j in xrange(n_batches):
        cost.append(opt.train_step(g[j*batch_size: (j+1)*batch_size]))
    print "Average cost for the iteration: {}".format(np.mean(cost))
    x.change_is_training(False)
    print "The validation error: {}".format(run.error())

f=gzip.open("MNIST_trained.pkl.gz", "wb")
x.save(f)