from setuptools import setup

setup(
   name='totem',
   version='0.0.1',
   description='A Theano wrapper for neural networks',
   author='Tanmaya Dabral',
   author_email='tanmaya.dabral@gmail.com',
   packages=['totem'],
   install_requires=['theano', 'numpy', 'scikit-learn']
)
