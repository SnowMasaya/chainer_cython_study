#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import train_mnist

import pstats, cProfile

cProfile.runctx("train_mnist.train()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

