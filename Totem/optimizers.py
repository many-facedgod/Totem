from . import utils as U
from .objectives import objectives


class Optimizer:
    """
    The optimizer super class
    """

    def __init__(self, cost, one_hot, data_input, data_output, decay=None, L1=None, L2=None):
        self.model_params = None
        self.model_grads = None
        self.layer_updates = None
        self.train_step = None
        self.model_inputs = None
        self.model_outputs = None
        self.cost_function = cost
        self.truth_placeholder = None
        self.one_hot = one_hot
        self.cost = None
        self.t = U.shared(U.floatX(0), borrow=True)
        self.decay = decay
        self.updates = [(self.t, self.t + 1)]
        self.data_input = U.shared(U.asarray(data_input, dtype=U._floatX), borrow=True, name="Train_Input")
        self.data_output = U.shared(U.asarray(data_output, dtype=U._floatX), borrow=True, name="Train_Output")
        self.L1 = L1
        self.L2 = L2

    def calc_updates(self):
        return None

    def build(self, model_params, layer_updates, model_inputs, model_outputs, l1_val, l2_val):
        """
        Does the operations common to all optimizers
        :param model_params: The parameters of the model
        :param layer_updates: The updates that are required by layers apart from training
        :param model_inputs: The model input tensor
        :param model_outputs: The model outputs tensor
        :param l1_val: The weight for L1 norm
        :param l2_val: The weight for L2 norm
        """
        self.model_params = model_params
        self.layer_updates = layer_updates
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs
        self.truth_placeholder = U.T.vector() if not self.one_hot else U.T.matrix()
        self.cost = objectives[self.cost_function](U.T.cast(self.truth_placeholder, "int32"), model_outputs,
                                                   self.one_hot)
        cost_regular = self.cost
        if self.L1 is not None:
            cost_regular = cost_regular + self.L1 * l1_val
        if self.L2 is not None:
            cost_regular = cost_regular + self.L2 * l2_val
        self.model_grads = U.T.grad(cost_regular, wrt=self.model_params)
        if self.decay is not None:
            assert hasattr(self, "learning_rate"), "The optimizer has no learning rate to decay"
            self.learning_rate /= (1 + self.decay * self.t)
        self.calc_updates()
        self.get_train_step()

    def set_value(self, data_input, data_output):
        """
        Change the training sets
        :param data_input: numpy array of the new input set
        :param data_output: numpy array of the new output set
        """
        self.data_input.set_value(U.asarray(data_input, dtype=U._floatX))
        self.data_output.set_value(U.asarray(data_output, dtype=U._floatX))

    def get_train_step(self):
        """
        Generates the train_step function for the optimizer
        """
        indices = U.T.lvector()
        self.train_step = U.function([indices], self.cost, updates=self.updates + self.layer_updates,
                                     givens={self.model_inputs: self.data_input[indices],
                                             self.truth_placeholder: self.data_output[indices]})


class RMSProp(Optimizer):
    """
    RMSProp proposed by Hinton et. al.
    """

    def __init__(self, cost, one_hot, data_input, data_output, learning_rate=0.001, rho=0.9, epsilon=1e-8, decay=None,
                 L1=None, L2=None):
        """
        The constructor for the optimizer
        :param cost: The cost function name
        :param one_hot: Whether the training labels are one-hot or not
        :param data_input: The input data tensor. Expected shape (training_samples, ) + model.input_shape
        :param data_output: The outputs data tensor. Expected shape (training_samples, ) if one_hot
                            else (training_samples, number_of_classes)
        :param learning_rate: The learning rate parameter
        :param rho: The forgetting factor
        :param epsilon: The smoothing factor
        :param decay: The decay rate
        :param L1: The weight for L1 norm
        :param L2: The weight for L2 norm
        """

        Optimizer.__init__(self, cost, one_hot, data_input, data_output, decay, L1, L2)
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def calc_updates(self):
        """
        Calculates the updates for RMSProp optimizer
        """
        v_t = [U.shared(U.zeros(x.shape.eval()).astype(U._floatX), borrow=True) for x in self.model_params]
        new_v = [self.rho * v + (1 - self.rho) * (g ** 2) for v, g in zip(v_t, self.model_grads)]
        self.updates += [(param, param - self.learning_rate * g / (U.T.sqrt(v) + self.epsilon)) for param, g, v in
                         zip(self.model_params, self.model_grads, new_v)] + [(v, new) for v, new in zip(v_t, new_v)]


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer proposed by Duchi et. al.
    """

    def __init__(self, cost, one_hot, data_input, data_output, learning_rate=0.01, epsilon=1e-8, decay=None, L1=None,
                 L2=None):
        """
        The constructor for the Optimizer
        :param cost: The cost function name
        :param one_hot: Whether the training labels are one-hot or not
        :param data_input: The input data tensor. Expected shape (training_samples, ) + model.input_shape
        :param data_output: The outputs data tensor. Expected shape (training_samples, ) if one_hot
                            else (training_samples, number_of_classes)
        :param learning_rate: The learning rate parameter
        :param epsilon: The smoothing factor
        :param decay: The decay rate
        :param L1: The weight for L1 norm
        :param L2: The weight for L2 norm
        """

        Optimizer.__init__(self, cost, one_hot, data_input, data_output, decay, L1, L2)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def calc_updates(self):
        """
        Calculates the updates for the AdaGrad optimizer
        """
        accumulator = [U.shared(U.zeros(x.shape.eval(), dtype=U._floatX), borrow=True) for x in self.model_params]
        new_acc = [acc + U.T.square(g) for acc, g in zip(accumulator, self.model_grads)]
        self.updates += [(param, param - self.learning_rate * g / (U.T.sqrt(acc) + self.epsilon)) for param, g, acc in
                         zip(self.model_params, self.model_grads, new_acc)] + [(acc, new) for acc, new in
                                                                               zip(accumulator, new_acc)]


class ADAM(Optimizer):
    """
    ADAM optimizer proposed by Diederik et. al.
    """

    def __init__(self, cost, one_hot, data_input, data_output, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 decay=None, L1=None, L2=None):
        """
        The constructor for the Optimizer
        :param cost: The cost function name
        :param one_hot: Whether the training labels are one-hot or not
        :param data_input: The input data tensor. Expected shape (training_samples, ) + model.input_shape
        :param data_output: The outputs data tensor. Expected shape (training_samples, ) if one_hot
                            else (training_samples, number_of_classes)
        :param alpha: The alpha parameter in the paper. (Learning rate)
        :param beta_1: The beta_1 parameter. (Forget rate)
        :param beta_2: The beta_2 parameter. (Forget rate)
        :param decay: The decay rate
        :param L1: The weight for L1 norm
        :param L2: The weight for L2 norm
        :param epsilon: The smoothing factor
        """

        Optimizer.__init__(self, cost, one_hot, data_input, data_output, decay, L1, L2)
        self.learning_rate = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def calc_updates(self):
        """
        Calculates the update for the ADAM optimizer
        """
        m_t = [U.shared(U.zeros(x.shape.eval()).astype(U._floatX), borrow=True) for x in self.model_params]
        v_t = [U.shared(U.zeros(x.shape.eval()).astype(U._floatX), borrow=True) for x in self.model_params]
        new_m = [self.beta_1 * m + (1 - self.beta_1) * g for m, g in zip(m_t, self.model_grads)]
        new_v = [self.beta_2 * v + (1 - self.beta_2) * (g ** 2) for v, g in zip(v_t, self.model_grads)]
        m_cap = [m / (1 - U.T.power(self.beta_1, self.t + 1)) for m in new_m]
        v_cap = [v / (1 - U.T.power(self.beta_2, self.t + 1)) for v in new_v]
        self.updates += [(param_i, param_i - self.learning_rate * m / (U.T.sqrt(v) + self.epsilon)) for param_i, m, v in
                         zip(self.model_params, m_cap, v_cap)] + [(m, new) for m, new in
                                                                  zip(m_t, new_m)] + [(v, new) for v, new
                                                                                      in zip(v_t,
                                                                                             new_v)]


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer with mini-batches
    """

    def __init__(self, cost, one_hot, data_input, data_output, learning_rate, decay=None, L1=None, L2=None):
        """
        The constructor for the optimizer

        :param cost: The cost function name
        :param one_hot: Whether the training labels are one_hot or not
        :param data_input: The input data tensor. Expected shape (training_samples, ) + model.input_shape
        :param data_output: The outputs data tensor. Expected shape (training_samples, ) if one_hot
                            else (training_samples, number_of_classes)
        :param learning_rate: The learning rate parameter
        :param decay: The decay rate
        :param L1: The weight for L1 norm
        :param L2: The weight for L2 norm
        """
        Optimizer.__init__(self, cost, one_hot, data_input, data_output, decay, L1, L2)
        self.learning_rate = learning_rate

    def calc_updates(self):
        """
        Calculates the updates for the SGD optimizer.
        """
        self.updates += [(param_i, param_i - self.learning_rate * grad_i) for (param_i, grad_i) in
                         zip(self.model_params, self.model_grads)]


class SGDMomentum(Optimizer):
    """
    Stochastic gradient descent over mini-batches with added momentum
    """

    def __init__(self, cost, one_hot, data_input, data_output, learning_rate, momentum, decay=None, L1=None, L2=None):
        """
        The constructor for the optimizer
        :param cost: The cost function name
        :param one_hot: Whether the training labels are one-hot or not
        :param data_input: The input data tensor. Expected shape (training_samples, ) + model.input_shape
        :param data_output: The outputs data tensor. Expected shape (training_samples, ) if one_hot
                            else (training_samples, number of classes)
        :param learning_rate: The learning rate parameter
        :param momentum: The momentum parameter
        :param decay: The decay rate
        :param L1: The weight for L1 norm
        :param L2: The weight for L2 norm
        """
        Optimizer.__init__(self, cost, one_hot, data_input, data_output, decay, L1, L2)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def calc_updates(self):
        """
        Calculates the updates for the SGDMomentum optimizer.
        """
        old_deltas = [U.shared(U.zeros(shape=x.shape.eval(), dtype=U._floatX), borrow=True) for x in
                      self.model_params]
        new_deltas = [-self.learning_rate * grad_i + self.momentum * old_i for (grad_i, old_i) in
                      zip(self.model_grads, old_deltas)]
        self.updates += [(param_i, param_i + new_i) for (param_i, new_i) in zip(self.model_params, new_deltas)] + [
            (old_i, new_i) for (old_i, new_i) in zip(old_deltas, new_deltas)]
