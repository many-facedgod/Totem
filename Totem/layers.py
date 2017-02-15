from . import utils as U
from .activations import activate


class Layer:
    """
    Abstract class for the layer interface
    """

    def __init__(self, name):
        self.params = []
        self.inputs = None
        self.outputs = None
        self.input_shape = None
        self.output_shape = None
        self.updates = []
        self.L1 = U.T.constant(0.0)
        self.L2 = U.T.constant(0.0)
        self.is_training = None
        self.name = name

    def build(self, inputs, input_shape, is_training):
        self.inputs = inputs
        self.input_shape = input_shape
        self.is_training = is_training

    def get_output_shape(self):
        return self.output_shape


class JoinLayer(Layer):
    """
    This layer concatenates the input of two or more layers.
    """

    def __init__(self, name, axis):
        """
        Constructor
        :param name: The name of this layer. Not optional.
        :param axis: Axis along which concatenation is to happen. Note that the batch axis is 0.
        """
        Layer.__init__(self, name)
        self.axis = axis

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: A list of inputs of previous layers
        :param input_shape: A list of input_shape tuples of previous layers
        :param is_training: Whether the model is training or not
        :return:
        """
        Layer.build(self, inputs, input_shape, is_training)
        self.outputs = U.T.concatenate(inputs, axis=self.axis)
        if self.axis == 0:
            self.output_shape = self.input_shape[0]
        else:
            total = U.np.sum(self.input_shape, axis=0)[self.axis - 1]
            self.output_shape = tuple([x if ind != (self.axis - 1) else total for ind, x in enumerate(input_shape[0])])


class FCLayer(Layer):
    """
    A layer implementing f(Wx+b)
    """

    def __init__(self, name, n_units, rng, activation="relu", init_method="glorot"):
        """
        Constructor
        :param name: The name of this layer. Not optional.
        :param n_units: Number of fully connected units in the layer
        :param rng: An RNG instance
        :param activation: The name of the activation function. If it takes parameters, give a tuple of
                           (name, dict_of_params)
        """
        Layer.__init__(self, name)
        self.n_units = n_units
        self.rng = rng
        self.activation = activation
        self.init_method = init_method
        self.weights = None
        self.bias = None

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to this layer
        :param input_shape: The shape of the inputs as a tuple (not including batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        Layer.build(self, inputs, input_shape, is_training)
        if isinstance(inputs, list) or isinstance(input_shape[0], tuple):
            raise ValueError("Layer takes input from only one source")
        self.output_shape = (self.n_units,)
        self.weights = U.shared(
            self.rng.get_weights((self.input_shape[0], self.n_units), distribution="normal", method=self.init_method,
                                 mode="FC", scale=self.activation, dtype=U._floatX), borrow=True)
        self.bias = U.shared(U.zeros(self.n_units, dtype=U._floatX), borrow=True)
        self.outputs, param_list = activate(U.T.dot(self.inputs, self.weights) + self.bias, self.output_shape,
                                            self.activation)
        self.params = [self.weights, self.bias] + param_list
        self.L1 = U.T.sum(U.T.abs_(self.weights)) + U.T.sum(U.T.abs_(self.bias))
        self.L2 = U.T.sum(U.T.square(self.weights)) + U.T.sum(U.T.square(self.bias))


class ConvLayer(Layer):
    """
    A layer implementing a convolution on images
    """

    def __init__(self, name, filter_num, filter_size, rng, activation="relu", mode="valid", strides=(1, 1),
                 init_method="glorot"):
        """
        Constructor for a convolution layer
        :param name: The name of this layer. Not optional.
        :param filter_num: Number of filters
        :param filter_size: Size of each filter (2D)
        :param rng: RNG instance for weight initialization
        :param activation: The name of the activation function. If it takes parameters, give a tuple of
                           (name, dict_of_params)
        :param mode: Convolution padding mode
        """
        Layer.__init__(self, name)
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.rng = rng
        self.activation = activation
        self.mode = mode
        self.init_method = init_method
        self.strides = strides
        self.filter = None
        self.filter_shape = None
        self.bias = None

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to this layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        Layer.build(self, inputs, input_shape, is_training)
        if isinstance(inputs, list) or isinstance(input_shape[0], tuple):
            raise ValueError("Layer takes input from only one source")
        self.filter_shape = (self.filter_num, input_shape[0]) + self.filter_size
        self.filter = U.shared(
            self.rng.get_weights(self.filter_shape, distribution="normal", method=self.init_method, mode="CONV",
                                 scale=self.activation, dtype=U._floatX), borrow=True)
        self.bias = U.shared(U.zeros(self.filter_num, dtype=U._floatX), borrow=True)
        conv, shape = U.conv2d(inputs, self.filter, input_shape, self.filter_shape, mode=self.mode,
                               strides=self.strides)
        self.output_shape = shape
        self.outputs, param_list = activate(conv + self.bias.dimshuffle('x', 0, 'x', 'x'), self.output_shape,
                                            self.activation)
        self.params = [self.filter, self.bias] + param_list
        self.L1 = U.T.sum(U.T.abs_(self.filter)) + U.T.sum(U.T.abs_(self.bias))
        self.L2 = U.T.sum(U.T.square(self.filter)) + U.T.sum(U.T.square(self.bias))


class PoolLayer(Layer):
    """
    A layer implementing 2D pooling
    """

    def __init__(self, name, down_sample_size, mode="max"):
        """
        Constructor for the layer
        :param name: The name of this layer. Not optional.
        :param down_sample_size: Tuple of length two having the down sample size in both dimesions
        :param mode: Mode for pooling: "max", "avg"
        """
        Layer.__init__(self, name)
        self.down_sample_size = down_sample_size
        self.mode = mode

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        Layer.build(self, inputs, input_shape, is_training)
        if isinstance(inputs, list) or isinstance(input_shape[0], tuple):
            raise ValueError("Layer takes input from only one source")
        self.outputs, self.output_shape = U.pool(inputs, input_shape, self.down_sample_size, self.mode)


class ActLayer(Layer):
    """
    A layer representing just an activation function
    """

    def __init__(self, name, activation):
        """
        Constructor for the layer
        :param name: The name of this layer. Not optional.
        :param activation: The name of the activation function. If it takes parameters, give a tuple of
                           (name, dict_of_params)
        """
        Layer.__init__(self, name)
        self.activation = activation

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        Layer.build(self, inputs, input_shape, is_training)
        if isinstance(inputs, list) or isinstance(input_shape[0], tuple):
            raise ValueError("Layer takes input from only one source")
        self.output_shape = input_shape
        self.outputs, param_list = activate(self.inputs, self.output_shape, self.activation)
        self.params += param_list


class ReshapeLayer(Layer):
    """
    A layer representing the reshape option
    """

    def __init__(self, name, new_shape):
        """
        Constructor for the layer
        :param name: The name of this layer. Not optional.
        :param new_shape: The new shape. Must be compatible.
        """
        Layer.__init__(self, name)
        self.output_shape = new_shape

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer

        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        Layer.build(self, inputs, input_shape, is_training)
        if isinstance(inputs, list) or isinstance(input_shape[0], tuple):
            raise ValueError("Layer takes input from only one source")
        self.outputs = U.T.reshape(inputs, (inputs.shape[0],) + self.output_shape, ndim=len(self.output_shape) + 1)


class DropOutLayer(Layer):
    """
    A layer representing noise
    """

    def __init__(self, name, rng, keep_prob=0.7):
        """
        Constructor for the layer
        :param name: The name of this layer. Not optional.
        :param rng: An RNG instance for generating the noise
        :param keep_prob: The probability of a signal to remain uncorrupted
        """
        Layer.__init__(self, name)
        self.rng = rng
        self.keep_prob = keep_prob

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        Layer.build(self, inputs, input_shape, is_training)
        if isinstance(inputs, list) or isinstance(input_shape[0], tuple):
            raise ValueError("Layer takes input from only one source")
        self.outputs = U.ifelse(is_training,
                                inputs * self.rng.get_dropout_mask(shape=input_shape,
                                                                   keep_prob=self.keep_prob),
                                inputs)
        self.output_shape = self.input_shape


class FlattenLayer(Layer):
    """
    A layer that flattens all dimensions but one (leaves the batch dimension untouched)
    """

    def __init__(self, name):
        """
        Constructor for the layer
        :param name: The name of this layer. Not optional.
        """
        Layer.__init__(self, name)

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        Layer.build(self, inputs, input_shape, is_training)
        self.outputs = U.T.flatten(inputs, 2)
        self.output_shape = (U.np.prod(input_shape),)


class BNLayer(Layer):
    """
    A layer that implements batch normalization between layers
    """

    def __init__(self, name, epsilon=1e-03, momentum=0.99, mode="low_mem"):
        """
        Constructor for the layer
        :param name: The name of this layer. Not optional.
        :param epsilon: The epsilon parameter in batch normalization
        :param momentum: The momentum for the running average of mean and variance
        :param mode: The mode for theano's batch normalization function: "low_mem" or "high_mem" Recommended is low_mem.
        """
        Layer.__init__(self, name)
        self.epsilon = epsilon
        self.momentum = momentum
        self.mode = mode
        self.mean = None
        self.var = None
        self.gamma = None
        self.beta = None

    def build(self, inputs, input_shape, is_training):
        """
        Building the actual layer
        :param inputs: The inputs to the layer
        :param input_shape: The shape of the inputs (excluding the batch size)
        :param is_training: Decides whether the model is currently training or not
        """
        Layer.build(self, inputs, input_shape, is_training)
        self.mean = U.shared(U.zeros(input_shape, dtype=U._floatX), borrow=True)
        self.var = U.shared(U.ones(input_shape, dtype=U._floatX), borrow=True)
        self.gamma = U.shared(U.ones(input_shape, dtype=U._floatX), borrow=True)
        self.beta = U.shared(U.zeros(input_shape, dtype=U._floatX), borrow=True)
        cur_mean = U.T.mean(inputs, axis=0)
        cur_var = U.T.var(inputs, axis=0)
        self.outputs = U.ifelse(is_training,
                                U.batch_normalize(self.inputs, cur_mean, cur_var, self.gamma, self.beta,
                                                  self.epsilon, self.mode),
                                U.batch_normalize(self.inputs, self.mean, self.var, self.gamma, self.beta,
                                                  self.epsilon, self.mode))
        self.output_shape = self.input_shape
        self.params = [self.gamma, self.beta]
        self.updates = [U.exp_avg(self.mean, cur_mean, self.momentum), U.exp_avg(self.var, cur_var, self.momentum)]
        self.L1 = U.T.sum(U.T.abs_(self.gamma)) + U.T.sum(U.T.abs_(self.beta))
        self.L2 = U.T.sum(U.T.square(self.gamma)) + U.T.sum(U.T.square(self.beta))
