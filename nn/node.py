# -*- coding: utf-8 -*-

import numpy as np

from .function import sigmoid, sigmoid_prime, se_loss


class Node(object):
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self, dz):
        pass


class Input(Node):
    def __init__(self):
        self._x = None

    def set(self, x):
        self._x = x

    def forward(self):
        return self._x

    def backward(self, dz):
        # recursion ends here
        return


class Weight(Node):
    def __init__(self, w, rate):
        self._w = w
        self._rate = rate

    def forward(self):
        return self._w

    def backward(self, dz):
        self._w -= self._rate * dz
        # recursion ends here too
        return


class Mul(Node):
    def __init__(self, node1, node2, bias, rate):
        self._node1 = node1
        self._node2 = node2
        self._bias = bias
        self._rate = rate
        self._x, self._y = None, None
        self._dx, self._dy = None, None

    def forward(self):
        self._x = self._node1.forward()
        self._y = self._node2.forward()
        self._z = np.dot(self._x, self._y) + self._bias
        return self._z

    def backward(self, dz):
        self._bias -= self._rate * dz
        self._dx = np.dot(dz, self._y.T)
        self._dy = np.dot(self._x.T, dz)
        self._node1.backward(self._dx)
        self._node2.backward(self._dy)


class SELoss(Node):
    def __init__(self, prev):
        self._prev = prev
        self._t, self._p = None, None

    def forward(self, guess, correct):
        self._t = correct
        self._p, _loss = se_loss(guess, correct)
        return _loss

    def backward(self):
        dl = self._p - self._t
        self._prev.backward(dl)


class Active(Node):
    def __init__(self, prev, func='sigmoid'):
        self._prev = prev
        if func == 'sigmoid':
            self._f = sigmoid
            self._df = sigmoid_prime
        self._in, self._out = None, None

    def forward(self):
        self._in = self._prev.forward()
        self._out = self._f(self._in)
        return self._out

    def backward(self, dz):
        dout = self._df(self._in)
        dout *= dz
        self._prev.backward(dout)


# Add, SMul, SNum for test
class SMul(Node):
    def __init__(self, node1, node2):
        self._node1 = node1
        self._node2 = node2
        self._x, self._y = None, None
        self._dx, self._dy = None, None
        self._sum = None

    def forward(self):
        self._x = self._node1.forward()
        self._y = self._node2.forward()
        self._num = self._x * self._y
        return self._num

    def backward(self, dz):
        print(self.__class__)
        print(dz)
        self._dx = dz * self._y
        self._dy = dz * self._x
        self._node1.backward(self._dx)
        self._node2.backward(self._dy)


class SAdd(Node):
    def __init__(self, node1, node2):
        self._node1 = node1
        self._node2 = node2
        self._x, self._y = None, None
        self._dx, self._dy = None, None
        self._sum = None

    def forward(self):
        self._x = self._node1.forward()
        self._y = self._node2.forward()
        self._num = self._x + self._y
        return self._num

    def backward(self, dz):
        print(self.__class__)
        print(dz)
        self._dx = dz
        self._dy = dz
        self._node1.backward(self._dx)
        self._node2.backward(self._dy)


class Num(Node):
    def __init__(self, num):
        self._num = num

    def set(self, num):
        self._num = num

    def forward(self):
        return self._num

    def backward(self, dz):
        print(self.__class__)
        print(dz)
        return dz
