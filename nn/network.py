# -*- coding: utf-8 -*-

import numpy as np
from .node import Mul, Active, Loss, SMul, SAdd, Num

class NeuralNetwork:

    def __init__(self, sizes, read=False, active='sigmoid', loss='se'):
        self.losses = [] # keep data to plot graph
        self.parameter = {} # store parametres to dump
        self.sizes = sizes
        x = Input(np.zeros(sizes[0]))
        for index in range(len(sizes)-1):
            key = 'w{0}'.format(index)
            if read:
                # TODO
                # read model from file
                pass
            else:
                self.parametre[key] = np.random.randn(sizes[index], sizes[index+1])
            w = Weight(self.parametre[key])
            a = Mul(x, w)
            z = Active(a, active)
            x = z
        self.loss = Loss(z, loss)
        self.end = z

    def fit(self, x, correct):
        guess = self.predict(x)
        loss = self.loss.forward(correct, guess)
        self.losses.append(loss)
        self.end.backward(correct, guess)

    def predict(self, x):
        self.input.set(x)
        return self.end.forward()

    def plot(self):
        # Plot self.losses
        # TODO
        pass

# A simple computation graph
class TestNetwork:

    def __init__(self):
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
