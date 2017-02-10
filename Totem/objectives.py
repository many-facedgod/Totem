from . import utils as U


def categorical_crossentropy(Y_true, Y_pred, one_hot=False, epsilon=1e-15):
    """
    Cross-entropy loss useful for multi-class classifications, i.e., where the output vector represents sums to one.
    If one_hot is True, then the Y_true for each example is expected to be a vector with probability for each class.
    If one_hot is False, then the Y_true for for each example is expected to be a single integer value representing the
    class number

    Note: Y_true and Y_pred are expected to be a collection of such vectors (representing a batch)

    :param Y_true: The ground truth
    :param Y_pred: The output predicted by the model
    :param one_hot: Flag that determines how Y_true is encoded
    :param epsilon: The clipping factor for the log function
    :return: Tensor representing the loss.
    """
    if not one_hot:
        return -U.T.mean(U.T.log(U.T.clip(Y_pred[U.T.arange(Y_pred.shape[0]), Y_true], epsilon, U.np.inf)))
    else:
        return U.T.mean(U.T.nnet.categorical_crossentropy(Y_pred, Y_true))


def mean_squared_error(Y_true, Y_pred, one_hot=True):
    """
    Returns the mean squared error between the true outputs and the predicted outputs

    :param Y_true: The ground truth (must be one-hot if classification)
    :param Y_pred:  The output predicted by the model
    :param one_hot: Whether Y_true is one-hot or not. Non one-hot not supported.
    :return: Tensor representing the loss
    """

    if not one_hot:
        raise NotImplementedError("Not implemented")

    return U.T.mean(U.T.square(Y_true - Y_pred))


def binary_crossentropy(Y_true, Y_pred, one_hot=True):
    """
    Cross-entropy loss useful for multi-label classification, i.e., when more than one label can
    be assigned to one instance
    :param Y_true: The ground truth (in one-hot form)
    :param Y_pred: The output predicted by the model
    :return: Tensor representing the loss
    """
    if not one_hot:
        raise NotImplementedError("Not implemented")

    return U.T.mean(U.T.nnet.binary_crossentropy(Y_pred, Y_true))


def mean_absolute_error(Y_true, Y_pred, one_hot=True):
    """
    Returns the mean absolute error between the true outputs and the predicted outputs

    :param Y_true: The ground truth (must be one-hot if classification)
    :param Y_pred: The output predicted by the model
    :return: Tensor representing the loss
    """
    if not one_hot:
        raise NotImplementedError("Not implemented")

    return U.T.mean(U.T.abs_(Y_true - Y_pred))


objectives = {"cce": categorical_crossentropy, "mse": mean_squared_error, "bce": binary_crossentropy,
              "mae": mean_absolute_error}
