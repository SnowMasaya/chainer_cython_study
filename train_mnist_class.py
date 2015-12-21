#!/usr/bin/env python
# cython: profile=True
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
from __future__ import print_function

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import data
import net


class Train_Mnist_C():

    def __init__(self):
        self.batchsize = 100
        self.n_epoch = 20
        self.n_units = 1000

        # Prepare dataset
        print('load MNIST dataset')
        self.mnist = data.load_mnist_data()
        self.mnist['data'] = self.mnist['data'].astype(np.float32)
        self.mnist['data'] /= 255
        self.mnist['target'] = self.mnist['target'].astype(np.int32)

        self.N = 60000
        self.x_train, self.x_test = np.split(self.mnist['data'],   [self.N])
        self.y_train, self.y_test = np.split(self.mnist['target'], [self.N])
        self.N_test = self.y_test.size

        # Prepare multi-layer perceptron model, defined in net.py
        self.model = L.Classifier(net.MnistMLP(784, self.n_units, 10))
        self.xp = np

        # Setup optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    # Learning loop
    def train(self):
        for epoch in six.moves.range(1, self.n_epoch + 1):
            print('epoch', epoch)

            # training
            perm = np.random.permutation(self.N)
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, self.N, self.batchsize):
                x = chainer.Variable(self.xp.asarray(self.x_train[perm[i:i + self.batchsize]]))
                t = chainer.Variable(self.xp.asarray(self.y_train[perm[i:i + self.batchsize]]))

                # Pass the loss function (Classifier defines it) and its arguments
                self.optimizer.update(self.model, x, t)

                if epoch == 1 and i == 0:
                    with open('graph.dot', 'w') as o:
                        g = computational_graph.build_computational_graph(
                            (self.model.loss, ), remove_split=True)
                        o.write(g.dump())
                    print('graph generated')

                sum_loss += float(self.model.loss.data) * len(t.data)
                sum_accuracy += float(self.model.accuracy.data) * len(t.data)

            print('train mean loss={}, accuracy={}'.format(
                sum_loss / self.N, sum_accuracy / self.N))

            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, self.N_test, self.batchsize):
                x = chainer.Variable(self.xp.asarray(self.x_test[i:i + self.batchsize]),
                                     volatile='on')
                t = chainer.Variable(self.xp.asarray(self.y_test[i:i + self.batchsize]),
                                     volatile='on')
                loss = self.model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(self.model.accuracy.data) * len(t.data)

            print('test  mean loss={}, accuracy={}'.format(
                sum_loss / self.N_test, sum_accuracy / self.N_test))

        # Save the model and the optimizer
        print('save the model')
        serializers.save_hdf5('mlp.model', self.model)
        print('save the optimizer')
        serializers.save_hdf5('mlp.state', self.optimizer)
