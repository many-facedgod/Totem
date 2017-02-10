from . import utils as U


def relu(inputs, shape):
    """
    Activates the inputs by applying rectified linear units.
    :param inputs: The input tensor
    :param shape: The shape of the input tensor (without the batch size).
    :return: The activated tensor, list of learnable parameters (None in this case).
    """

    return U.T.nnet.relu(inputs), []


def sigmoid(inputs, shape):
    """
    Activates the inputs by applying the sigmoid function
    :param inputs: The input tensor
    :param shape: The shape of the input tensor (without the batch size).
    :return: The activated tensor, list of learnable parameters (None in this case).
    """

    return U.T.nnet.sigmoid(inputs), []


def tanh(inputs, shape):
    """
    Activates the inputs by applying the tanh function
    :param inputs: The input tensor
    :param shape: The shape of the input tensor (without the batch size).
    :return: The activated tensor, list of learnable parameters (None in this case).
    """

    return U.T.tanh(inputs), []


def softmax(inputs, shape):
    """
    Activates the inputs by applying the softmax function
    :param inputs: The input tensor
    :param shape: The shape of the input tensor (without the batch size).
    :return: The activated tensor, list of learnable parameters (None in this case).
    """

    return U.T.nnet.softmax(inputs), []


def softplus(inputs, shape):
    """
    Activates the inputs by applying the softplus function
    :param inputs: The input tensor
    :param shape: The shape of the input tensor (without the batch size).
    :return: The activated tensor, list of learnable parameters (None in this case).
    """

    return U.T.log(1 + U.T.exp(inputs)), []


def identity(inputs, shape):
    """
    Returns the input tensor as it is
    :param inputs: The input tensor
    :param shape: The shape of the input tensor (without the batch size)
    :return: The activated tensor, list of learnable parameters (None in this case).
    """

    return inputs, []


def leaky_relu(inputs, shape, alpha=0.1):
    """
    Activates the inputs by applying the "leaky" version of relu
    :param inputs: The input tensor
    :param shape: The shape of the input tensor (without the batch size)
    :param alpha: The slope of the function when input is negative
    :return: The activated tensor, list of learnable parameters (None in this case).
    """

    return U.T.nnet.relu(inputs, alpha), []


def elu(inputs, shape, alpha=1.0):
    """
    Activates the inputs by applying the exponential linear unit
    :param inputs: The input tensor
    :param shape: The shape of the input tensor (without the batch size)
    :param alpha: The alpha parameter for the function.
    :return: The activated tensor, list of learnable parameters (None in this case).
    """

    return U.T.nnet.elu(inputs, alpha), []


def prelu(inputs, shape, axes=(1,), init=0.25):
    """
    Activates the inputs by applying the parametrized (learnable) relu
    :param inputs: The input tensor
    :param shape: The shape of the input tensor (without the batch size)
    :param axes: The axes over which the parameters should differ. Must be a tuple. 0 is the batch axis so the tuple
                 cannot have a 0. The dimensions must be in the increasing order.
    :param init: The initial value of all the alphas.
    :return: The activated tensor, list of learnable parameters.
    """

    alpha_shape = [shape[i - 1] for i in axes]
    alphas = U.shared(U.ones(alpha_shape, dtype=U._floatX) * U.floatX(init), borrow=True)
    dimshuf = ['x']
    k = 0
    for i in range(1, len(shape) + 1):
        if i in axes:
            dimshuf.append(k)
            k += 1
        else:
            dimshuf.append('x')
    return U.T.nnet.relu(inputs, alphas.dimshuffle(dimshuf)), [alphas]


activations = {"sigmoid": sigmoid, "relu": relu, "tanh": tanh, "softmax": softmax,
               "softplus": softplus, "elu": elu,
               "prelu": prelu, "leaky_relu": leaky_relu, "None": identity, None: identity}


def activate(inputs, input_shape, activation):
    """
    Common interface for all activations
    :param inputs: The input tensor
    :param input_shape: The shape of the input tensor (without the batch size)
    :param activation: The activation name or a tuple (act_name, dict_of_params)
    :return: The activated tensor, list of learnable parameters.
    """

    if type(activation) is tuple or type(activation) is list:
        act_name = activation[0]
        kwargs = activation[1]
    else:
        act_name = activation
        kwargs = {}
    return activations[act_name](inputs, input_shape, **kwargs)
