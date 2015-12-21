#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from train_mnist_class import Train_Mnist_C

graphviz = GraphvizOutput()
graphviz.output_file = 'basic.png'

with PyCallGraph(output=graphviz):
    train_mnist = Train_Mnist_C()
    train_mnist.train()


