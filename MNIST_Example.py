from __future__ import print_function
from Totem import *
import numpy as np
import gzip
import cPickle as pickle
import time

r = rng.RNG(12234)
f = gzip.open("/home/tanmaya/mnist.pkl.gz", "rb")
Train, Valid, Test = pickle.load(f)
Trainx, Trainy = Train
Vx, Vy = Valid
n_batches = 100
batch_size = 500
x = model.Model((1, 28, 28))
n_iters = 20
x.add_layer(layers.ConvLayer("Conv1", 10, (3, 3), r, activation="leaky_relu", init_method="glorot"))
x.add_layer(layers.PoolLayer("Pool1", (2, 2), mode="max"))
x.add_layer(layers.ConvLayer("Conv2", 5, (3, 3), r, activation=("leaky_relu", {"alpha": 0.02}), init_method="glorot"))
x.add_layer(layers.PoolLayer("Pool2", (2, 2)))
x.add_layer(layers.PoolLayer("Sub1", (2, 2), mode="avg"), source="inputs")
x.add_layer(layers.ConvLayer("Conv3", 10, (3, 3), r, activation="leaky_relu", init_method="glorot"))
x.add_layer(layers.PoolLayer("Pool3", (2, 2)))
x.add_layer(layers.ConvLayer("Conv4", 5, (2, 2), r, activation="leaky_relu", init_method="glorot"))
x.add_layer(layers.DropOutLayer("Drop1", r))
x.add_layer(layers.PoolLayer("Sub2", (2, 2)), source="Sub1")
x.add_layer(layers.ConvLayer("Conv5", 10, (2, 2), r, activation="leaky_relu"))
x.add_layer(layers.ConvLayer("Conv6", 10, (2, 2), r, activation="leaky_relu"))
x.add_layer(layers.DropOutLayer("Dropout1", r))
x.add_layer(layers.JoinLayer("Join1", axis=1), source=("Pool2", "Conv4", "Conv6"))
x.add_layer(layers.ConvLayer("Conv7", 10, (2, 2), r, activation="leaky_relu"))
x.add_layer(layers.FlattenLayer("Flat1"))
x.add_layer(layers.BNLayer("BN1"))
x.add_layer(layers.FCLayer("FC1", 400, r, "elu"))
x.add_layer(layers.FCLayer("FC2", 10, r, "softmax"))

opt = optimizers.ADAM("cce", False, Trainx.reshape((50000, 1, 28, 28)), Trainy)
run = x.get_runner(Vx.reshape((10000, 1, 28, 28)), Vy)
# x.build_optimizer(opt, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, "Conv7", "Conv6"])  # this will train only particular layers
x.build_optimizer(opt)  # this will train all layers
g = np.arange(50000)
h = np.arange(10000)
start = time.time()
for i in xrange(n_iters):
    cost = []
    r.shuffle(g)
    x.change_is_training(True)
    for j in xrange(n_batches):
        cost.append(opt.train_step(g[j * batch_size: (j + 1) * batch_size]))
    print ("Average cost for the iteration: {}".format(np.mean(cost)))
    x.change_is_training(False)
    print ("The validation error: {}".format(run.error()))
end = time.time()
print ("Time taken is {}".format(end - start))
run.set_value(Test[0].reshape(10000, 1, 28, 28), Test[1])
print ("Test error is {}".format(run.error()))
f = gzip.open("MNIST_trained.pkl.gz", "wb")
x.save(f)

