from . import utils as U
from .runner import Runner


class Model:
    def __init__(self, input_shape):
        """
        Constructor for a model
        :param input_shape: The shape of the expected input. (Not including batch_size)
        """
        self.params = []
        self.layers = {}
        self.layer_names = []
        self.updates = []
        self.input_shape = input_shape
        self.input_placeholder = U.tensors[len(input_shape) + 1]("input_placeholder")
        self.is_training = U.shared(1)
        self.L1 = U.T.constant(0.0)
        self.L2 = U.T.constant(0.0)

    def get_layer(self, layer_id):
        """
        Get the layer with the layer_id name or the layer_id index
        :param layer_id: The name of the layer or the index of the layer.
        :return: The layer requested
        """
        if isinstance(layer_id, int):
            return self.layers[self.layer_names[layer_id]]
        elif isinstance(layer_id, str):
            return self.layers[layer_id]
        else:
            raise ValueError("Unsupported layer ID format")

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
            inp = self.input_placeholder
            inpsh = self.input_shape

        elif not isinstance(source, tuple) and not isinstance(source, list):
            if source == "inputs":
                inp = self.input_placeholder
                inpsh = self.input_shape
            else:
                l = self.get_layer(source)
                inp = l.outputs
                inpsh = l.output_shape
        else:
            inp = []
            inpsh = []
            for elem in source:
                if elem == "inputs":
                    inp.append(self.input_placeholder)
                    inpsh.append(self.input_shape)
                else:
                    l = self.get_layer(elem)
                    inp.append(l.outputs)
                    inpsh.append(l.output_shape)

        layer.build(inp, inpsh, self.is_training)
        self.layers[layer.name] = layer
        self.layer_names.append(layer.name)
        self.L1 = self.L1 + layer.L1
        self.L2 = self.L2 + layer.L2
        self.params = self.params + layer.params
        self.updates = self.updates + layer.updates

    def build_optimizer(self, optimizer, trainables=[]):
        """
        Build an optimizer for this model
        :param optimizer: The optimizer to be built
        :param trainables: The layers (index or name) to be trained by this optimizer. If empty, trains all.
        """
        if not trainables:
            optimizer.build(self.params, self.updates, self.input_placeholder, self.get_layer(-1).outputs, self.L1,
                            self.L2)
        else:
            params = []
            updates = []
            L1 = U.T.constant(0.)
            L2 = U.T.constant(0.)
            for id in trainables:
                layer = self.get_layer(id)
                params += layer.params
                updates += layer.updates
                L1 += layer.L1
                L2 += layer.L2
            optimizer.build(params, updates, self.input_placeholder, self.get_layer(-1).outputs, L1, L2)

    def get_runner(self, test_input, test_output):
        """
        Get a runner instance to run on this model
        :param test_input: The test input data for the runner
        :param test_output: The test output data for the runner
        :return:
        """
        return Runner(self.input_placeholder, self.input_shape, self.layers[self.layer_names[-1]].outputs,
                      self.is_training, test_input, test_output)

    def get_output_shape(self, layer_id):
        """
        Get the output shape of a particular layer
        :param layer_id: The name or the index of the layer
        :return: A tuple representing the shape of the layer output
        """
        layer = self.get_layer(layer_id)
        return layer.output_shape

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
        U.pickle.dump(self, f, protocol=2)
