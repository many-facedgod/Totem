from . import utils as U
from .layers import JoinLayer
from .runner import Runner


class Model:
    def __init__(self, input_shape):
        """
        Constructor for a model
        :param input_shape: The shape of the expected input. (Not including batch_size)
        """
        self.params = []
        self.layers = []
        self.layer_names = []
        self.updates = []
        self.input_shape = input_shape
        self.input_placeholder = U.tensors[len(input_shape) + 1]("input_placeholder")
        self.is_training = U.shared(1)
        self.L1 = U.T.constant(0.0)
        self.L2 = U.T.constant(0.0)

    def change_is_training(self, mode):
        """
        Change the mode from training to running
        :param mode: True for training, False for running
        """
        if mode is True:
            mode = 1
        elif mode is False:
            mode = 0
        self.is_training.set_value(mode)

    def get_is_training(self):
        return self.is_training.eval() == 1

    def add_layer(self, layer, source=-1):
        """
        Add a layer to the model
        :param layer: The layer to be added
        :param source: The source of the input to this layer. It can be the name of a layer, -1 for the last added layer
                        and "inputs" for the original. If the layer is a JoinLayer, use a tuple of names/-1/"inputs"
        """

        if len(self.layers) == 0:
            assert not isinstance(layer, JoinLayer), "Cannot have JoinLayer as the first layer"
            inp = self.input_placeholder
            inpsh = self.input_shape

        elif not isinstance(layer, JoinLayer):
            if source == -1:
                inp = self.layers[-1].outputs
                inpsh = self.layers[-1].output_shape
            elif source == "inputs":
                inp = self.input_placeholder
                inpsh = self.input_shape
            else:
                ind = self.layer_names.index(source)
                inp = self.layers[ind].outputs
                inpsh = self.layers[ind].output_shape
        else:
            assert isinstance(source, tuple) or isinstance(source,
                                                           list), "Need a tuple or a list of sources for JoinLayer"
            inp = []
            inpsh = []
            for elem in source:
                if elem == -1:
                    inp.append(self.layers[-1].outputs)
                    inpsh.append(self.layers[-1].output_shape)
                elif elem == "inputs":
                    inp.append(self.input_placeholder)
                    inpsh.append(self.input_shape)
                else:
                    ind = self.layer_names.index(elem)
                    inp.append(self.layers[ind].outputs)
                    inpsh.append(self.layers[ind].output_shape)

        layer.build(inp, inpsh, self.is_training)
        self.layers.append(layer)
        self.layer_names.append(layer.name)
        self.L1 = self.L1 + layer.L1
        self.L2 = self.L2 + layer.L2
        self.params = self.params + layer.params
        self.updates = self.updates + layer.updates

    def build_optimizer(self, optimizer):
        """
        Build an optimizer for this model
        :param optimizer: The optimizer to be built
        """
        optimizer.build(self.params, self.updates, self.input_placeholder, self.layers[-1].outputs, self.L1, self.L2)

    def get_runner(self, test_input, test_output):
        """
        Get a runner instance to run on this model
        :param test_input: The test input data for the runner
        :param test_output: The test output data for the runner
        :return:
        """
        return Runner(self.input_placeholder, self.input_shape, self.layers[len(self.layers) - 1].outputs,
                        self.is_training, test_input, test_output)

    def get_output_shape(self, layer_name):
        """
        Get the output shape of a particular layer
        :param layer_name: The name of the layer
        :return: A tuple repersenting the shape of the layer output
        """
        ind = self.layer_names.index(layer_name)
        return self.layers[ind].get_output_shape()

    def run_direct(self, inputs):
        """
        Run the model directly on the given inputs
        :param inputs: A numpy array of inputs
        :return: The output of the model for the given input
        """
        curr_mode = self.get_is_training()
        self.change_is_training(False)
        op = U.asarray(self.layers[-1].outputs.eval({self.input_placeholder: inputs}))
        self.change_is_training(curr_mode)
        return op

    def save(self, f):
        """
        Save the model to a file
        :param f: The file object to which it is to be written
        """
        U.pickle.dump(self, f, protocol=-1)
