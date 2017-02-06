from Totem import *
import numpy as np
import gzip
import cPickle as pickle
import time

r = RNG(12234)
f = gzip.open("/home/tanmaya/mnist.pkl.gz", "rb")
Train, Valid, Test = pickle.load(f)
Trainx, Trainy = Train
Vx, Vy = Valid
n_batches = 100
batch_size = 500
x = Model((1, 28, 28))
n_iters = 5
x.add_layer(ConvLayer("Conv1", 10, (3, 3), r, activation="relu", init_method="glorot"))
x.add_layer(PoolLayer("Pool1", (2, 2), mode="max"))
x.add_layer(ConvLayer("Conv2", 5, (3, 3), r, activation="relu", init_method="glorot"))
x.add_layer(PoolLayer("Pool2", (2, 2)))
x.add_layer(PoolLayer("Sub1", (2, 2), mode="avg"), source="inputs")
x.add_layer(ConvLayer("Conv3", 10, (3, 3), r, activation="relu", init_method="glorot"))
x.add_layer(PoolLayer("Pool3", (2, 2)))
x.add_layer(ConvLayer("Conv4", 5, (2, 2), r, activation="relu", init_method="glorot"))
x.add_layer(PoolLayer("Sub2", (2, 2)), source="Sub1")
x.add_layer(ConvLayer("Conv5", 10, (2, 2), r))
x.add_layer(ConvLayer("Conv6", 10, (2, 2), r))
x.add_layer(JoinLayer("Join1", axis=1), source=("Pool2", "Conv4", "Conv6"))
x.add_layer(ConvLayer("Conv7", 10, (2, 2), r))
x.add_layer(FlattenLayer("Flat1"))
x.add_layer(BNLayer("BN1"))
x.add_layer(FCLayer("FC1", 400, r, "relu"))
x.add_layer(FCLayer("FC2", 10, r, "softmax"))

opt = ADAM("cce", False, Trainx.reshape((50000, 1, 28, 28)), Trainy, decay=1e-3)
run = x.get_runner(Vx.reshape((10000, 1, 28, 28)), Vy)
x.build_optimizer(opt)

g = np.arange(50000)
h = np.arange(10000)
start = time.time()
for i in xrange(n_iters):
    cost = []
    r.Shuffle(g)
    x.change_is_training(True)
    for j in xrange(n_batches):
        cost.append(opt.train_step(g[j * batch_size: (j + 1) * batch_size]))
    print "Average cost for the iteration: {}".format(np.mean(cost))
    x.change_is_training(False)
    print "The validation error: {}".format(run.error())
end = time.time()
print end - start
sss = Test[0].shape[0]
run.set_value(Test[0].reshape(sss, 1, 28, 28), Test[1])
print "Test error is {}".format(run.error())
f = gzip.open("MNIST_trained.pkl.gz", "wb")
x.save(f)
