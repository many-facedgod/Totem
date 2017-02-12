# Totem

A Keras like wrapper built specifically for theano. Currently, it supports only feed forward networks including multiple parallel branches and concatenation of outputs. Recurrent nets yet to be implemented.

The given MNIST example runs convolutions on multiple parallel branches, each running on a subsampled version of the input with different subsampling sizes. They are finally joined a la Google Inception's DepthConcat for further convolutions.

### Prerequisites

Currently, the list of dependencies include Theano, Numpy, scikit-learn and Pickle/cPickle.

## Authors

* **Tanmaya Shekhar Dabral** - [many-facedgod](https://github.com/many-facedgod)
