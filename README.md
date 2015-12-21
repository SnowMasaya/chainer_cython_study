Chainer with Cython 
====


## Description
# Multi-Layer Perceptron for MNIST Classification

This is a minimal example to write a feed-forward net.
The code consists of three parts: dataset preparation, network and optimizer definition and learning loop.
This is a common routine to write a learning process of networks with dataset that is small enough to fit into memory.

This example requires scikit-learn to load the MNIST dataset.
#
### Install

If you don't install pyenv and virtualenv you have to install bellow
####Prepare Install
linux
```
apt-get install pyenv 
apt-get install virtualenv 
```
Mac
```
brew install pyenv 
brew install virtualenv 
```

####Prepare Inastall2
```
pyenv install 3.4.1
pyenv rehash
pyenv local 3.4.1
virtualenv -p ~/.pyenv/versions/3.4.1/bin/python3.4 my_env
source my_env/bin/activate

```

```
pip install -r requirement.txt 
```
Installing a library bellow
##Requirements

    Python 3.4+
    h5py
    chainer
    pycallgraph

#
### Usage 
#
```
*You execute python normal 
python execute_mnist_p.py

*You execute python cython 
python execute_mnist_p_c.py

*You execute python cython + cdef 
python execute_mnist_p_c2.py

*You execute python cython + cdef + adam.py(cython)
python execute_mnist_p_c3.py

*You execute python cython + cdef + adam.py(cython)
python execute_mnist_p_c3.py

*You execute python graph visualize 
python execute_mnist.py
```
#
### Licence
#
```
The MIT License (MIT)

Copyright (c) 2015 Masaya Ogushi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
#
### Author
#
[SnowMasaya](https://github.com/SnowMasaya)
### References 
#
>[Chainer]http://chainer.org/<br>
