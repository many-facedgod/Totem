import theano.tensor as T
import theano.tensor.signal
import theano.tensor.signal.pool
import theano
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
import theano.ifelse

_floatX=theano.config.floatX
activations={"sigmoid":T.nnet.sigmoid, "relu": T.nnet.relu, "tanh": T.tanh, "softmax": T.nnet.softmax, "None": lambda x: x, None: lambda x: x}
tensors=[T.scalar, T.vector, T.matrix, T.tensor3, T.tensor4]


def floatX(x):
    '''
    Converts the given scalar/ndarray to theano.config.floatX
    :param x: the scalar/ndarray
    :return: the typecasted value
    '''
    return np.cast[_floatX](x)

def increment(X, inc):
    """
    Returns an update tuple to increment a particular shared tensor
    :param X: The shared tensor
    :param inc: The increment value
    :return: The update tuple
    """
    return (X, X+inc)

def decrement(X, dec):
    """
    Returns an update tuple to decrement a particular shared tensor
    :param X: The shared tensor
    :param dec: The decrement value
    :return: The update tuple
    """
    return (X, X-dec)

def exp_avg(X_curr, X_new, momentum):
    """
    Returns an update tuple for an exponential average update
    :param X_curr: The current value of the average
    :param X_new: The new value of the average
    :param momentum: The momentum to be used
    :return: The update tuple
    """
    return (X_curr, X_curr*momentum+X_new*(1-momentum))


def BNormalize(X, mean, variance, gamma, beta, epsilon=1e-03, mode="low_mem"):
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

    stddev=T.sqrt(variance + epsilon)
    return T.nnet.bn.batch_normalization(X, gamma, beta, mean, stddev, mode)


def conv2d(image, filter, image_shape, filter_shape, strides = (1,1), mode="valid"):
    """
    Returns a 2D convolution of the image with the filter
    :param image: A 4D tensor of the shape (batch_size, channels, height, width)
    :param filter: A 4D tensor representing the filters
    :param image_shape: Tuple having the last 3 dimensions of the image (not including batch size)
    :param filter_shape: Tuple having the filter size
    :param strides: The subsampling strides
    :param mode: The padding mode: "valid",  "full"
    :return: (The output of the convolution, The shape of the output (if possible) without the batch_size)
    """
    modes={"valid": (0, 0), "half": (filter_shape[2] // 2, filter_shape[3] // 2), "full": (filter_shape[2] - 1, filter_shape[3] -1), "same": (filter_shape[2] // 2, filter_shape[3] // 2)}

    if type(mode)==str:
        padding=modes[mode]
    else:
        padding=mode

    if type(strides)==int:
        strides=(strides, strides)

    result=T.nnet.conv2d(image, filter, input_shape=(None, ) + image_shape, filter_shape=filter_shape, border_mode=padding, subsample=strides)
    op_shape=(filter_shape[0], ((image_shape[1] + 2 * padding[0] - filter_shape[2]) // strides[0]) + 1, ((image_shape[2] + 2 * padding[1] - filter_shape[3]) // strides[1]) + 1)
    return (result, op_shape)




def pool(image, image_shape, ds, mode="max"):
    """
    Returns a downscaled image using the given mode.
    :param image: Image tensor to be pooled. Only the last 2 dimensions are pooled
    :param image_shape: The shape of the input image (not including the batch_size)
    :param ds: The downscale factor tuple for each dimension.
    :param mode: "max", "avg", "sum"
    :return: The downscaled image and the shape of it without the batch_size
    """
    if mode=="avg":
        mode="average_inc_pad"
    result=T.signal.pool.pool_2d(image, ds, ignore_border=True, mode=mode)
    op_shape=(image_shape[0], image_shape[1]//ds[0], image_shape[2]//ds[1])
    return (result, op_shape)


def categorical_crossentropy(Y_true, Y_pred, one_hot=False, epsilon=1e-15):
    """
    Cross-entropy loss useful for multi-class classifications, i.e., where the output vector represents sums to one.
    If one_hot is True, then the Y_true for each example is expected to be a vector with probability for each class.
    If one_hot is False, then the Y_true for for each example is expected to be a single integer value representing the class number

    Note: Y_true and Y_pred are expected to be a collection of such vectors (representing a batch)

    :param Y_true: The ground truth
    :param Y_pred: The output predicted by the model
    :param one_hot: Flag that determines how Y_true is encoded
    :return: Tensor representing the loss.
    """
    if not one_hot:
        return -T.mean(T.log(T.clip(Y_pred[T.arange(Y_pred.shape[0]), Y_true], floatX(epsilon), floatX(np.inf))))
    else:
        return T.mean(T.nnet.categorical_crossentropy(Y_pred, Y_true))

def mean_squared_error(Y_true, Y_pred, one_hot=True):
    """
    Returns the mean squared error between the true outputs and the predicted outputs

    :param Y_true: The ground truth (must be one-hot if classification)
    :param Y_pred:  The output predicted by the model
    :return: Tensor representing the loss
    """

    if not one_hot:
        raise ValueError ("Not implemented")

    return T.mean(T.square(Y_true-Y_pred))

def binary_crossentropy(Y_true, Y_pred, one_hot=True):
    """
    Cross-entropy loss useful for multi-label classification, i.e., when more than one label can be assigned to one instance
    :param Y_true: The ground truth (in one-hot form)
    :param Y_pred: The output predicted by the model
    :return: Tensor representing the loss
    """
    if not one_hot:
        raise ValueError ("Not implemented")

    return T.mean(T.nnet.binary_crossentropy(Y_pred, Y_true))

def mean_absolute_error(Y_true, Y_pred, one_hot=True):
    """
    Returns the mean absolute error between the true outputs and the predicted outputs

    :param Y_true: The ground truth (must be one-hot if classification)
    :param Y_pred: The output predicted by the model
    :return: Tensor representing the loss
    """
    if not one_hot:
        raise ValueError ("Not implemented")

    return T.mean(T.abs_(Y_true-Y_pred))



objectives = {"cce":categorical_crossentropy, "mse": mean_squared_error, "bce": binary_crossentropy, "mae": mean_absolute_error}

class RNG:
    """
    Wrapper around numpy's random number generated
    """
    def __init__(self, seed):
        self.np_rng=np.random.RandomState(seed)
        self.th_rng=T.shared_randomstreams.RandomStreams(seed)


    def random(self, dtype=_floatX):
        """
        Returns a single random scalar
        :return: a random number between 0 and 1
        """
        return np.cast[dtype](self.np_rng.uniform())

    def Gaussian(self, mean, stddev, shape, dtype):
        return self.np_rng.normal(loc=mean, scale=stddev, size=shape).astype(dtype)

    def Uniform(self, mean, stddev, shape, dtype):
        """
        Returns an array of samples drawn from a uniform distribution with the given params
        :param mean: The mean of the distribution
        :param stddev: The standard deviaion of the dis
        :param shape:
        :param dtype:
        :return:
        """
        low = mean - stddev*np.sqrt(3)
        high = mean + stddev*np.sqrt(3)
        return self.np_rng.uniform(low = low, high = high, size = shape).astype(dtype)

    def get_weights(self, shape, distribution = "normal", method = "glorot", mode="FC", scale="relu", dtype=_floatX):
        """
        Returns a numpy ndarray of weights with Glorot style initialization
        :param shape: Shape of the required weight matrix (a tuple)
        :param mode: "FC" or "CONV", for fully connected or convolutional layers
        :param scale: Scales for different activation functions: "relu", "sigmoid", "tanh"
        :param dtype: The data type of the output array
        :return: The generated weights
        """
        distributions = {"normal": self.Gaussian, "gaussian": self.Gaussian, "uniform": self.Uniform}

        if scale=="sigmoid":
            z=4
        else:
            z=1

        if mode=="FC":
            fan_in=shape[0]
            fan_out=shape[1]
        elif mode=="CONV":
            fan_in=shape[1]*shape[2]*shape[3]
            fan_out=shape[0]*shape[2]*shape[3]
        else:
            raise ValueError ("Unrecognized mode")

        if method=="glorot":
            stddev=np.sqrt(2.0 / ( fan_in + fan_out))
        elif method=="he":
            stddev=np.sqrt(2.0 / (fan_in))
        else:
            raise ValueError ("Not supported")

        return z*(distributions[distribution](mean=0.0, stddev=stddev, shape=shape, dtype=dtype))


    def Orthogonal(self, shape, mode="FC", scale="relu", dtype=_floatX):
        """
        Returns a numpy ndarray of weights with orthogonal initialization
        :param shape: Shape of the required weight matrix (a tuple)
        :param mode: "FC" or "CONV", for fully connected or convolutional layers
        :param scale: Scales for different activation functions: "relu", "sigmoid", "tanh"
        :param dtype: The data type of the output array
        :return: The generated weights
        """
        if scale=="relu":
            z=np.sqrt(2)
        elif scale=="sigmoid":
            z=4
        else:
            z=1

        if mode=="FC":
            sample=self.np_rng.normal(size=shape)
            U,_,_=np.linalg.svd(sample)
            weights=(z*U).astype(dtype)
        elif mode=="CONV":
            sample=self.np_rng.normal(size=(shape[0], shape[1]*shape[2]*shape[3]))
            _,_,V=np.linalg.svd(sample)
            weights=(z*V).astype(type).reshape(shape)
        else:
            raise TypeError
        return weights

    def DropoutMask(self, shape, keep_prob=0.7, dtype=_floatX):
        """
        Returns a tensor that acts as a dropout mask for the dropout layer. The returned tensor has the shape (1, ) + shape
        :param shape: The shape of the required mask. Note: Do not include the batch size in this tuple
        :param keep_prob: The probability of keeping the element
        :return: A tensor that serves as a mask for a dropout layer.
        """
        mask=self.th_rng.binomial(shape, p=keep_prob, dtype=dtype)/np.cast[dtype](keep_prob)
        return mask

    def SymbolicShuffle(self, X, size=None):
        """
        Returns a view of the tensor with symbolically shuffled leftmost dimension. Useful for shuffling training data.
        :param X: The input tensor
        :param size: The size of the leftmost dimension, if known
        :return: The shuffled view of the tensor
        """
        if size==None:
            return X[self.th_rng.permutation(n=X.shape[0])]
        else:
            return X[self.th_rng.permutation(n=size)]

    def Shuffle(self, x):
        """
        Shuffles a numpy array
        :param x: The numpy array
        """
        self.np_rng.shuffle(x)




class Layer:
    """
    Abstract class for the layer interface
    """
    def __init__(self):
        self.params=None
        self.inputs=None
        self.outputs=None
        self.input_shape=None
        self.output_shape=None
        self.updates=None
        self.L1=None
        self.L2=None
        self.is_training=None

    def build(self, inputs, input_shape, is_training):
        return None

class FCLayer(Layer):

    """
    A layer implementing f(Wx+b)
    """
    def __init__(self, n_units, rng, activation="relu", init_method="glorot"):
        """
        Constructor
        :param n_units: Number of fully connected units in the layer
        :param rng: An RNG instance
        :param activation: The name of the activation function to be used
        """
        Layer.__init__(self)
        self.n_units=n_units
        self.rng=rng
        self.activation=activation
        self.init_method=init_method

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to this layer
        :param input_shape: The shape of the inputs as a tuple (not including batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        self.input_shape=input_shape
        self.is_training=is_training
        self.output_shape=(self.n_units, )
        self.weights=theano.shared(self.rng.get_weights((self.input_shape[0], self.n_units), distribution="normal", method=self.init_method, mode="FC", scale=self.activation, dtype=_floatX) , borrow=True)
        self.bias=theano.shared(np.zeros(self.n_units, dtype=_floatX), borrow=True)
        self.inputs=inputs
        self.outputs=activations[self.activation](T.dot(self.inputs, self.weights)+self.bias)
        self.params=[self.weights, self.bias]
        self.L1=T.sum(T.abs_(self.weights))+T.sum(T.abs_(self.bias))
        self.L2=T.sum(T.square(self.weights))+T.sum(T.square(self.bias))
        self.updates=[]


class ConvLayer(Layer):
    """
    A layer implementing a convolution on images
    """

    def __init__(self, filter_num, filter_size, rng, activation="relu", mode="valid", strides=(1,1), init_method="glorot"):
        """
        Constructor for a convolution layer
        :param filter_num: Number of filters
        :param filter_size: Size of each filter (2D)
        :param rng: RNG instance for weight initialization
        :param activation: activation function
        :param mode: Convolution padding mode
        """
        Layer.__init__(self)
        self.filter_num=filter_num
        self.filter_size=filter_size
        self.rng=rng
        self.activation=activation
        self.mode=mode
        self.init_method=init_method
        self.strides=strides

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to this layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        self.input_shape=input_shape
        self.inputs=inputs
        self.is_training=is_training
        self.filter_shape=(self.filter_num, input_shape[0])+ self.filter_size
        self.filter=theano.shared(self.rng.get_weights(self.filter_shape, distribution="normal", method=self.init_method, mode="CONV", scale=self.activation, dtype=_floatX), borrow=True)
        self.bias=theano.shared(np.zeros(self.filter_num, dtype=_floatX), borrow=True)
        conv, shape=conv2d(inputs, self.filter, input_shape, self.filter_shape, mode=self.mode, strides=self.strides)
        self.outputs=activations[self.activation](conv+ self.bias.dimshuffle('x',0,'x','x'))
        self.params=[self.filter, self.bias]
        self.L1=T.sum(T.abs_(self.filter))+T.sum(T.abs_(self.bias))
        self.L2=T.sum(T.square(self.filter))+T.sum(T.square(self.bias))
        self.output_shape=shape
        self.updates=[]


class PoolLayer(Layer):
    """
    A layer implementing 2D pooling
    """
    def __init__(self, down_sample_size, mode="max"):
        """
        Constructor for the layer
        :param down_sample_size: Tuple of length two having the down sample size in both dimesions
        :param mode: Mode for pooling: "max", "avg"
        """
        Layer.__init__(self)
        self.down_sample_size=down_sample_size
        self.mode=mode

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        self.input_shape=input_shape
        self.inputs=inputs
        self.is_training=is_training
        self.outputs, self.output_shape=pool(inputs, input_shape, self.down_sample_size, self.mode)
        self.params=[]
        self.L1=T.constant(0.)
        self.L2=T.constant(0.)
        self.updates=[]


class ActLayer(Layer):
    """
    A layer representing just an activation function
    """
    def __init__(self, activation):
        """
        Constructor for the layer
        :param activation: The activation function
        """
        Layer.__init__(self)
        self.activation=activation

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        self.inputs=inputs
        self.input_shape=input_shape
        self.is_training=is_training
        self.outputs=activations[self.activation](self.inputs)
        self.output_shape=input_shape
        self.params=[]
        self.L1 = T.constant(0.)
        self.L2 = T.constant(0.)
        self.updates=[]



class ReshapeLayer(Layer):
    """
    A layer representing the reshape option
    """

    def __init__(self, new_shape):
        Layer.__init__(self)
        self.output_shape=new_shape

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer

        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        self.inputs=inputs
        self.input_shape=input_shape
        self.outputs=T.reshape(inputs, (inputs.shape[0], )+self.output_shape, ndim=len(self.output_shape)+1)
        self.params=[]
        self.L1=T.constant(0.)
        self.L2=T.constant(0.)
        self.updates=[]


class DropOutLayer(Layer):
    """
    A layer representing noise
    """

    def __init__(self, rng, keep_prob=0.7):
        """
        Constructor for the layer
        :param rng: An RNG instance for generating the noise
        :param keep_prob: The probability of a signal to remain uncorrupted
        """
        Layer.__init__(self)
        self.rng=rng
        self.keep_prob=keep_prob

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        self.input_shape=input_shape
        self.output_shape=input_shape
        self.is_training=is_training
        self.inputs=inputs
        self.outputs=theano.ifelse.ifelse(is_training, inputs*self.rng.DropoutMask(shape=input_shape, keep_prob=self.keep_prob), inputs)
        self.params=[]
        self.L1=T.constant(0.)
        self.L2=T.constant(0.)
        self.updates=[]

class FlattenLayer(Layer):
    """
    A layer that flattens all dimensions but one (leaves the batch dimension untouched)
    """

    def __init__(self):
        """
        Constructor for the layer
        """
        Layer.__init__(self)

    def build(self, inputs, input_shape,is_training):
        """
        Building the actual layer
        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        self.inputs=inputs
        self.input_shape=input_shape
        self.is_training=is_training
        self.outputs=T.flatten(inputs, 2)
        self.output_shape= (np.prod(input_shape),)
        self.L1 = T.constant(0.)
        self.L2 = T.constant(0.)
        self.updates = []
        self.params=[]


class BNLayer(Layer):
    """
    A layer that implements batch normalization between layers
    """

    def __init__(self, epsilon=1e-03, momentum=0.99, mode="low_mem"):
        """
        Constructor for the layer
        :param epsilon: The epsilon parameter in batch normalization
        :param momentum: The momentum for the running average of mean and variance
        :param mode: The mode for theano's batch normalization function: "low_mem" or "high_mem" Recommended is low_mem.
        """
        Layer.__init__(self)
        self.epsilon=epsilon
        self.momentum=momentum
        self.mode=mode

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        self.inputs=inputs
        self.input_shape=input_shape
        self.is_training=is_training
        self.mean=theano.shared(np.zeros(input_shape, dtype=_floatX), borrow=True)
        self.var=theano.shared(np.zeros(input_shape, dtype=_floatX), borrow=True)
        self.gamma=theano.shared(np.ones(input_shape, dtype=_floatX), borrow=True)
        self.beta=theano.shared(np.zeros(input_shape, dtype=_floatX), borrow=True)
        cur_mean=T.mean(inputs, axis=0)
        cur_var=T.var(inputs, axis=0)
        self.outputs=theano.ifelse.ifelse(is_training, BNormalize(self.inputs, cur_mean, cur_var, self.gamma, self.beta, self.epsilon, self.mode), BNormalize(self.inputs, self.mean, self.var, self.gamma, self.beta, self.epsilon, self.mode))
        self.output_shape=self.input_shape
        self.params=[self.gamma, self.beta]
        self.updates=[exp_avg(self.mean, cur_mean, self.momentum), exp_avg(self.var, cur_var, self.momentum)]
        self.L1=T.sum(T.abs_(self.gamma))+ T.sum(T.abs_(self.beta))
        self.L2=T.sum(T.square(self.gamma))+T.sum(T.square(self.beta))

class Optimizer:
    """
    The optimizer super class
    """

    def __init__(self, cost, one_hot, data_input, data_output, L1=None, L2=None):

        self.model_params=None
        self.model_grads=None
        self.layer_updates=None
        self.train_step=None
        self.model_inputs=None
        self.model_outputs=None
        self.cost_function=cost
        self.truth_placeholder=None
        self.one_hot=one_hot
        self.cost=None
        self.updates=None
        self.data_input=data_input
        self.data_output=data_output
        self.L1=L1
        self.L2=L2

    def build(self, model_params, layer_updates, model_inputs, model_outputs, L1_val, L2_val):

        return None

    def set_value(self, data_input, data_output):
        """
        Change the training sets
        :param data_input: numpy array of the new input set
        :param data_output: numpy array of the new output set
        """
        self.data_input.set_value(data_input)
        self.data_output.set_value(data_output)

class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer with mini-batches
    """

    def __init__(self, learning_rate, cost, one_hot, data_input, data_output, L1=None, L2=None):
        """
        The constructor for the optimizer

        :param learning_rate: The learning rate parameter
        :param cost: The cost function name
        :param one_hot: Whether the training labels are one_hot or not
        :param data_input: The input data tensor. Expected shape (training_samples, ) + model.input_shape
        :param data_output: The outputs data tensor. Expected shape (training_samples, ) if one_hot else (training_samples, number_of_classes)
        """
        Optimizer.__init__(self, cost, one_hot, data_input, data_output, L1, L2)
        self.learning_rate=learning_rate

    def build(self, model_params, layer_updates, model_inputs, model_outputs, L1_val, L2_val):
        """
        Building the actual optimizer. To be called by the model's build_optimizer method

        :param model_params: The parameters of the model
        :param layer_updates: The updates required for proper functioning of the layers
        :param model_inputs: The input_placeholder of the model
        :param model_outputs: The outputs of the model
        """
        self.model_params=model_params
        self.layer_updates=layer_updates
        self.model_inputs=model_inputs
        self.model_outputs=model_outputs
        self.truth_placeholder=T.vector() if not self.one_hot else T.matrix()
        self.cost=objectives[self.cost_function](T.cast(self.truth_placeholder, "int32"), model_outputs, self.one_hot)
        cost_regular=self.cost
        if self.L1!=None:
            cost_regular=cost_regular+self.L1*L1_val
        if self.L2!=None:
            cost_regular=cost_regular+self.L2*L2_val
        self.model_grads=T.grad(cost_regular, wrt=self.model_params)
        self.updates=[(param_i, param_i-self.learning_rate*grad_i) for (param_i, grad_i) in zip(self.model_params, self.model_grads)]+self.layer_updates
        indices=T.lvector()
        self.train_step=theano.function([indices], self.cost, updates=self.updates, givens={self.model_inputs: self.data_input[indices], self.truth_placeholder: self.data_output[indices]})

class SGD_momentum(Optimizer):
    """
    Stochastic gradient descent over mini-batches with added momentum
    """

    def __init__(self, learning_rate, momentum, cost, one_hot, data_input, data_output, L1=None, L2=None):
        """
        The constructor for the optimizer

        :param learning_rate: The learning rate parameter
        :param momentum: The momentum parameter
        :param cost: The cost function name
        :param one_hot: Whether the training labels are one-hot or not
        :param data_input: The input data tensor. Expected shape (training_samples, ) + model.input_shape
        :param data_output: The outputs data tensor. Expected shape (training_samples, ) if one_hot else (training_samples, number of classes)
        """
        Optimizer.__init__(self, cost, one_hot, data_input, data_output, L1, L2)
        self.learning_rate=learning_rate
        self.momentum=momentum

    def build(self, model_params, layer_updates, model_inputs, model_outputs, L1_val, L2_val):
        """
        Building the actual optimizer. To be called by the model's build_optimizer method

        :param model_params: The parameters of the model
        :param layer_updates: The updates required for proper functioning of the layers
        :param model_inputs: The input_placeholder of the model
        :param model_outputs: The outputs of the model
        """
        self.model_params=model_params
        self.layer_updates=layer_updates
        self.model_inputs=model_inputs
        self.model_outputs=model_outputs
        self.truth_placeholder=T.vector() if not self.one_hot else T.matrix()
        self.cost=objectives[self.cost_function](T.cast(self.truth_placeholder, "int32"), model_outputs, self.one_hot)
        cost_regular=self.cost
        if self.L1!=None:
            cost_regular=cost_regular+self.L1*L1_val
        if self.L2!=None:
            cost_regular=cost_regular+self.L2*L2_val
        self.model_grads=T.grad(cost_regular, wrt=self.model_params)
        old_deltas=[theano.shared(np.zeros(shape=x.shape.eval(), dtype=_floatX), borrow=True) for x in self.model_params]
        new_deltas=[-self.learning_rate*grad_i+self.momentum*old_i for (grad_i, old_i) in zip(self.model_grads, old_deltas)]
        self.updates=[(param_i, param_i + new_i) for (param_i, new_i) in zip(self.model_params, new_deltas)]+[(old_i, new_i) for (old_i, new_i) in zip(old_deltas, new_deltas)]+self.layer_updates
        indices=T.lvector()
        self.train_step=theano.function([indices], self.cost, updates=self.updates, givens={self.model_inputs: self.data_input[indices], self.truth_placeholder: self.data_output[indices]})



class Runner:
    """
    Use the methods of this class to run the model
    """
    def __init__(self, input_placeholder, input_shape, model_outputs, is_training, test_input, test_output):
        """
        Constructor. To be called by the model "get_runner" method"
        :param input_placeholder: The input_placeholder for the model
        :param input_shape: The shape of the input to the model
        :param model_outputs: The model output tensor
        :param is_training: The is_training tensor of the model
        :param test_input: The tensor containing the test input data
        :param test_output: The tensor containing the test output data (required to calculate errors)
        """
        self.input_placeholder=input_placeholder
        self.model_outputs=model_outputs
        self.is_training=is_training
        self.input_shape=input_shape
        self.test_input=test_input
        self.test_output=test_output
        index=T.lvector()
        self.run=theano.function([index], self.model_outputs, givens={self.input_placeholder: test_input[index]})

    def set_value(self, test_input, test_output):
        """
        Change the value of the test_input and test_output
        :param test_input: numpy array of the new input values
        :param test_output: numpy array of the new output values
        """
        self.test_input.set_value(test_input)
        self.test_output.set_value(test_output)

    def error(self, one_hot=False, thresh=0.5, at_a_time=20):
        """
        Returns the average zero-one error for the test data
        :param one_hot: Whether the outputs of the test data are one-hot or not
        :param thresh: Threshold for decision
        :param at_a_time: How many to be run at a time. Make sure the total is divisible by this.
        :return The error score
        """

        iters = int(np.ceil(self.test_input.shape[0].eval() / floatX(at_a_time)))
        indices=np.arange(self.test_input.shape[0].eval())
        op=np.asarray(self.run(indices[:at_a_time]))
        for i in range(1,iters):
            op=np.append(op, self.run(indices[i*at_a_time: (i+1)*at_a_time]), axis=0)
        actual=np.asarray(self.test_output.get_value(), dtype=np.int32)
        if one_hot:
            threshed=np.clip(np.ceil(op-thresh), 0.0, 1.0).astype(np.int32)
            return np.mean(np.not_equal(threshed, actual))
        else:
            return np.mean(np.not_equal(np.argmax(op, axis=1), actual))


    def auc_score(self, mode="macro", at_a_time=20):
        """
        Returns the AUC score using sklearn's method
        :param mode: The averaging mode for AUC calculation
        :param at_a_time: How many to be run at a time. Make sure the total is divisible by this.
        :return: The AUC score
        """
        iters = int(np.ceil(self.test_input.shape[0].eval() / floatX(at_a_time)))
        indices = np.arange(self.test_input.shape[0].eval())
        op = np.asarray(self.run(indices[:at_a_time]))
        for i in range(1, iters):
            op = np.append(op, self.run(indices[i * at_a_time: (i + 1) * at_a_time]), axis=0)

        actual = np.asarray(self.test_output.eval())
        return roc_auc_score(actual, op, average=mode)




class Model:

    def __init__(self, input_shape):
        """
        Constructor for a model
        :param input_shape: The shape of the expected input. (Not including batch_size)
        """
        self.params=[]
        self.layers=[]
        self.updates=[]
        self.input_shape=input_shape
        self.input_placeholder=tensors[len(input_shape)+1]("input_placeholder")
        self.is_training=theano.shared(1)
        self.L1=None
        self.L2=None

    def change_is_training(self, mode):
        """
        Change the mode from training to running
        :param mode: True for training, False for running
        """
        if mode==True:
            mode=1
        elif mode==False:
            mode=0
        self.is_training.set_value(mode)

    def add_layer(self, layer):
        """
        Add a layer to the model
        :param layer: The layer to be added
        """
        self.layers.append(layer)
        if len(self.layers)==1:
            layer.build(self.input_placeholder, self.input_shape, self.is_training)
            self.L1=layer.L1
            self.L2=layer.L2
        else:
            layer.build(self.layers[len(self.layers)-2].outputs, self.layers[len(self.layers)-2].output_shape, self.is_training)
            self.L1=self.L1+layer.L1
            self.L2=self.L2+layer.L2

        self.params=self.params+layer.params
        self.updates=self.updates+layer.updates

    def build_optimizer(self, optimizer):
        """
        Build an optimizer for this model
        :param optimizer: The optimizer to be built
        """
        optimizer.build(self.params, self.updates, self.input_placeholder, self.layers[len(self.layers)-1].outputs, self.L1, self.L2)

    def get_runner(self, test_input, test_output):
        """
        Get a runner instance to run on this model
        :param test_input: The test input data for the runner
        :param test_output: The test output data for the runner
        :return:
        """
        return Runner(self.input_placeholder,self.input_shape, self.layers[len(self.layers)-1].outputs, self.is_training, test_input, test_output)

    def save(self, file):
        """
        Save the model to a file
        :param file: The file object to which it is to be written
        """
        pickle.dump(self, file, protocol=-1)