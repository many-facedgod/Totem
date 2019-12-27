# Totem

A wrapper over [Theano](http://deeplearning.net/software/theano/) for quick creation and training of feedforward neural networks. Written as a part of the implementation for [A Multi-scale Convolutional Neural Network Architecture for Music Auto-Tagging](https://link.springer.com/chapter/10.1007/978-981-13-1592-3_60), the code for which can be found [here](https://github.com/many-facedgod/Music-Tagger). This was written in Python 2.

## Requirements

- Python 2.7
- Theano >= 0.8
- Numpy
- Scikit-learn

## Usage
The class `model.Model` represents a feedforward network. After initializing the model, the layers can be added to it using the `model.add_layer` function. The layers are defined in `layers.py` and contain most of the standard layers. Once all the layers have been added, the `model.build` function can be used to build the entire graph. The optimizer can be build using the `model.build_optimizer` method by passing one of the optimizers defined in `optimizers.py`.

A detailed working example is shown in `MNIST_Example.py`.

## Installation
    git clone https://github.com/many-facedgod/Totem
    cd Totem
    pip install .
## Authors

* **Tanmaya Shekhar Dabral** - [many-facedgod](https://github.com/many-facedgod)
