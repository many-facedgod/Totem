import theano.tensor as T
from theano import shared, function
import theano
import numpy as np
from numpy import ones, zeros, asarray
from theano.ifelse import ifelse
from sklearn.metrics import roc_auc_score
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

_floatX = theano.config.floatX


def floatX(x):
    """
    Converts the given scalar/ndarray to theano.config.floatX
    :param x: the scalar/ndarray
    :return: the typecasted value
    """
    return np.cast[_floatX](x)


tensors = [T.scalar, T.vector, T.matrix, T.tensor3, T.tensor4]


def exp_avg(X_curr, X_new, momentum):
    """
    Returns an update tuple for an exponential average update
    :param X_curr: The current value of the average
    :param X_new: The new value of the average
    :param momentum: The momentum to be used
    :return: The update tuple
    """
    return X_curr, X_curr * momentum + X_new * (1 - momentum)


def batch_normalize(X, mean, variance, gamma, beta, epsilon=1e-03, mode="low_mem"):
    """
    Returns a batch normalization graph for a batch
    :param X: Input batch
    :param mean: The mean for normalization
    :param variance: The variance for normalization
    :param gamma: The scaling parameter
    :param beta: The shifting parameter
    :param epsilon: The smoothing factor
    :param mode: The mode for theano's batch normalization method: "low_mem" or "high_mem". Recommended is "low_mem"
    :return: Tensor representing the normalized input
    """

    stddev = T.sqrt(variance + epsilon)
    return T.nnet.bn.batch_normalization(X, gamma, beta, mean, stddev, mode)


def conv2d(image, filter, image_shape, filter_shape, strides=(1, 1), mode="valid"):
    """
    Returns a 2D convolution of the image with the filter
    :param image: A 4D tensor of the shape (batch_size, channels, height, width)
    :param filter: A 4D tensor representing the filters
    :param image_shape: Tuple having the last 3 dimensions of the image (not including batch size)
    :param filter_shape: Tuple having the filter size
    :param strides: The sub-sampling strides
    :param mode: The padding mode: "valid",  "full"
    :return: (The output of the convolution, The shape of the output (if possible) without the batch_size)
    """
    modes = {"valid": (0, 0), "half": (filter_shape[2] // 2, filter_shape[3] // 2),
             "full": (filter_shape[2] - 1, filter_shape[3] - 1), "same": (filter_shape[2] // 2, filter_shape[3] // 2)}

    if type(mode) == str:
        padding = modes[mode]
    else:
        padding = mode

    if type(strides) == int:
        strides = (strides, strides)

    result = T.nnet.conv2d(image, filter, input_shape=(None,) + image_shape, filter_shape=filter_shape,
                           border_mode=padding, subsample=strides)
    op_shape = (filter_shape[0], ((image_shape[1] + 2 * padding[0] - filter_shape[2]) // strides[0]) + 1,
                ((image_shape[2] + 2 * padding[1] - filter_shape[3]) // strides[1]) + 1)
    return result, op_shape


def pool(image, image_shape, ds, mode="max"):
    """
    Returns a downscaled image using the given mode.
    :param image: Image tensor to be pooled. Only the last 2 dimensions are pooled
    :param image_shape: The shape of the input image (not including the batch_size)
    :param ds: The downscale factor tuple for each dimension.
    :param mode: "max", "avg", "sum"
    :return: The downscaled image and the shape of it without the batch_size
    """
    if mode == "avg":
        mode = "average_inc_pad"
    result = T.signal.pool.pool_2d(image, ds, ignore_border=True, mode=mode)
    op_shape = (image_shape[0], image_shape[1] // ds[0], image_shape[2] // ds[1])
    return result, op_shape
