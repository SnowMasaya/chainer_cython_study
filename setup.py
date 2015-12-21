#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("train_mnist_c.pyx"))
