from . import utils as U


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
        self.input_placeholder = input_placeholder
        self.model_outputs = model_outputs
        self.is_training = is_training
        self.input_shape = input_shape
        self.test_input = U.shared(U.asarray(test_input, dtype=U._floatX), borrow=True)
        self.test_output = U.shared(U.asarray(test_output, dtype=U._floatX), borrow=True)
        index = U.T.lvector()
        self.run = U.function([index], self.model_outputs, givens={self.input_placeholder: self.test_input[index]})

    def set_value(self, test_input, test_output):
        """
        Change the value of the test_input and test_output
        :param test_input: numpy array of the new input values
        :param test_output: numpy array of the new output values
        """
        self.test_input.set_value(U.asarray(test_input, dtype=U._floatX))
        self.test_output.set_value(U.asarray(test_output, dtype=U._floatX))

    def run_all(self, at_a_time=20):
        """
        Runs the model on the entire test set.
        :param at_a_time: The number of inputs to be run in one batch
        :return: numpy array of outputs
        """
        iters = int(U.np.ceil(self.test_input.shape[0].eval() / U.floatX(at_a_time)))
        indices = U.np.arange(self.test_input.shape[0].eval())
        op = U.asarray(self.run(indices[:at_a_time]))
        for i in range(1, iters):
            op = U.np.append(op, self.run(indices[i * at_a_time: (i + 1) * at_a_time]), axis=0)
        return op

    def error(self, one_hot=False, thresh=0.5, at_a_time=20):
        """
        Returns the average zero-one error for the test data
        :param one_hot: Whether the outputs of the test data are one-hot or not
        :param thresh: Threshold for decision
        :param at_a_time: How many to be run at a time. Make sure the total is divisible by this.
        :return The error score
        """

        op = self.run_all(at_a_time)
        actual = U.asarray(self.test_output.get_value(), dtype=U.np.int32)
        if one_hot:
            threshed = U.np.clip(U.np.ceil(op - thresh), 0.0, 1.0).astype(U.np.int32)
            return U.np.mean(U.np.not_equal(threshed, actual))
        else:
            return U.np.mean(U.np.not_equal(U.np.argmax(op, axis=1), actual))

    def auc_score(self, mode="macro", at_a_time=20):
        """
        Returns the AUC score using sklearn's method
        :param mode: The averaging mode for AUC calculation
        :param at_a_time: How many to be run at a time. Make sure the total is divisible by this.
        :return: The AUC score
        """
        op = self.run_all(at_a_time)
        actual = U.asarray(self.test_output.eval())
        return U.roc_auc_score(actual, op, average=mode)
