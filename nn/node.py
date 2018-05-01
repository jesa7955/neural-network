# -*- coding: utf-8 -*-

import numpy as np


class Node(object):
    def forward(self):
        pass

    def backward(self, dz):
        pass


class Input(Node):
    def __init__(self, x):
        self._x = x.copy()
        self._x  # add a 1 for bias

    def set(self, x):
        self._x = x.copy()

    def forward(self):
        return self._x

    def backward(self, dz):
        pass


class Weight(Node):
    def __init__(self, w):
        self._w = w.copy()
        self._w  # add a 0 for bias

    def set(self, w):
        self._w = w

    def forward(self):
        return self._w

    def backward(self, dz):
        # TODO
        pass


class Mul(Node):
    def __init__(self, node1, node2):
        self._node1 = node1
        self._node2 = node2

    def forward(self):
        self._z = np.dot(node1.forward(), node2.forward())
        return self._z.copy()

    def backward(self, dz):
        # TODO
        pass


class Loss(Node):
    def __init__(self, prev, func):
        self._prev = prev
        self._func = func

    def forward(self):
        return self._func(self._prev.forward())

    def backward(self, dz):
        # TODO
        pass


class Active(Node):
    def __init__(self, prev, func):
        self._prev = prev
        self._func = func

    def forward(self):
        return self._func(self._prev.forward())

    def backward(self, dz):
        # TODO
        pass


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
