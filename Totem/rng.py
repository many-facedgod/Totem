from . import utils as U


class RNG:
    """
    Wrapper around numpy's and theano's random number generated
    """

    def __init__(self, seed):
        self.np_rng = U.np.random.RandomState(seed)
        self.th_rng = U.RandomStreams(seed)

    def random(self, dtype=U._floatX):
        """
        Returns a single random scalar
        :return: a random number between 0 and 1
        """
        return U.np.cast[dtype](self.np_rng.uniform())

    def gaussian(self, mean, stddev, shape, dtype):
        """
        Returns an array of samples drawn from a gaussian distribution with the given params
        :param mean: The mean of the distribution
        :param stddev: The standard deviaion of the dis
        :param shape: The shape of the required output
        :param dtype: The data type required
        :return: The generated random sample
        """
        return self.np_rng.normal(loc=mean, scale=stddev, size=shape).astype(dtype)

    def uniform(self, mean, stddev, shape, dtype):
        """
        Returns an array of samples drawn from a uniform distribution with the given params
        :param mean: The mean of the distribution
        :param stddev: The standard deviaion of the dis
        :param shape: The shape of the required output
        :param dtype: The data type required
        :return: The generated random sample
        """
        low = mean - stddev * U.np.sqrt(3)
        high = mean + stddev * U.np.sqrt(3)
        return self.np_rng.uniform(low=low, high=high, size=shape).astype(dtype)

    def get_weights(self, shape, distribution="normal", method="glorot", mode="FC", scale="relu", dtype=U._floatX):
        """
        Returns a numpy ndarray of weights with Glorot style initialization
        :param shape: Shape of the required weight matrix (a tuple)
        :param distribution: The distribution to be used for sampling
        :param method: Which method to use for weight initialization. "glorot" or "he"
        :param mode: "FC" or "CONV", for fully connected or convolutional layers
        :param scale: Scales for different activation functions: "relu", "sigmoid", "tanh"
        :param dtype: The data type of the output array
        :return: The generated weights
        """

        distributions = {"normal": self.gaussian, "gaussian": self.gaussian, "uniform": self.uniform}
        if type(scale) is tuple:
            scale = scale[0]
        if scale == "sigmoid":
            z = 4
        else:
            z = 1

        if mode == "FC":
            fan_in = shape[0]
            fan_out = shape[1]
        elif mode == "CONV":
            fan_in = shape[1] * shape[2] * shape[3]
            fan_out = shape[0] * shape[2] * shape[3]
        else:
            raise ValueError("Unrecognized mode")

        if method == "glorot":
            stddev = U.np.sqrt(2.0 / (fan_in + fan_out))
        elif method == "he":
            stddev = U.np.sqrt(2.0 / fan_in)
        else:
            raise ValueError("Not supported")

        return z * (distributions[distribution](mean=0.0, stddev=stddev, shape=shape, dtype=dtype))

    def get_dropout_mask(self, shape, keep_prob=0.7, dtype=U._floatX):
        """
        Returns a tensor that acts as a dropout mask for the dropout layer.
        :param shape: The shape of the required mask. Note: Do not include the batch size in this tuple
        :param keep_prob: The probability of keeping the element
        :param dtype: The data type of the tensor required
        :return: A tensor that serves as a mask for a dropout layer.
        """
        mask = self.th_rng.binomial(shape, p=keep_prob, dtype=dtype) / U.np.cast[dtype](keep_prob)
        return mask

    def shuffle(self, x):
        """
        Shuffles a numpy array
        :param x: The numpy array
        """
        self.np_rng.shuffle(x)
