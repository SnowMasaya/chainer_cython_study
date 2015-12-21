#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""


import pstats, cProfile
import pyximport
pyximport.install()
import train_mnist_c2

cProfile.runctx("train_mnist_c2.train()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
