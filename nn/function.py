# -*- coding: utf-8 -*-

import numpy as np


def softmax(x):
    """The softmax function"""
    exp_x = np.exp(x - np.max(x))
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def se_loss(y, t):
    """Combine softmax and cross entropy"""
    p = softmax(y)
    l = -np.sum(t * np.log(p))
    return (p, l)


def sigmoid(x):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
