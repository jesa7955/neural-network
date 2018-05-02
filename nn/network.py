# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .node import Mul, Active, SELoss, SMul, SAdd, Num, Input, Weight


class NeuralNetwork:
    def __init__(self, sizes, l_rate, read=False, active='sigmoid', loss='se'):
        self.sizes = sizes
        self.losses = []  # keep data to plot graph
        self.parameter = {}  # store parameters to dump
        self.parameter['l_rate'] = l_rate
        x = Input()
        self.input = x
        for index in range(len(sizes) - 1):
            w_key = 'w{0}'.format(index)
            b_key = 'b{0}'.format(index)
            if read:
                # TODO read model from file
                break
            else:
                self.parameter[w_key] = np.random.randn(sizes[index + 1],
                                                        sizes[index])
                self.parameter[b_key] = np.random.randn(sizes[index + 1], 1)
            w = Weight(self.parameter[w_key], l_rate)
            a = Mul(w, x, self.parameter[b_key], l_rate)
            if index == len(sizes) - 2:
                z = a
            else:
                z = Active(a, active)
                x = z
        if loss == 'se':
            self.loss = SELoss(z)
        self.end = z

    def predict(self, x):
        self.input.set(x)
        return self.end.forward()

    def fit(self, x, correct):
        guess = self.predict(x)
        loss = self.loss.forward(guess, correct)
        self.losses.append(loss)
        self.loss.backward()

    def plot(self):
        plt.plot(self.losses)
        plt.show()


# A simple computation graph
class TestNetwork:
    def __init__(self):
        """
        A simple computation graph like this
        x ---
             \
             * ---
             /    \
        2 ---      |
                   + ---> z
        y ---      |
             \    /
             * ---
             /
        3 ---
        """
        x = Num(0)
        cx = Num(2)
        xn = SMul(x, cx)
        y = Num(0)
        cy = Num(3)
        yn = SMul(y, cy)
        add = SAdd(xn, yn)
        self.end = add
        self.begin = [x, y]

    def fit(self, dz=1):
        self.end.backward(dz)

    def predict(self, i):
        self.begin[0].set(i[0])
        self.begin[1].set(i[1])
        print(self.end.forward())
