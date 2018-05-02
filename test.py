import random

import numpy as np
from tqdm import tqdm
from nn.network import NeuralNetwork
import utils.mnist_loader as mnist_loader

def test(nn, data, train_data=False):
    if train_data:
        result = [(np.argmax(nn.predict(x)), np.argmax(y)) for (x, y) in data]
    else:
        result = [(np.argmax(nn.predict(x)), y) for (x, y) in data]
    correct = sum([int(x == y) for (x, y) in result])
    return "{0} / {1}".format(correct, len(result))

sizes = [784, 100, 10]
n = NeuralNetwork(sizes, 0.1)
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print('initial: ' + test(n, test_data))
bs = 100
for i in range(30):
    random.shuffle(training_data)
    batches = [training_data[k:k+bs] for k in range(0, len(training_data), bs)]
    for batch in batches:
        for x, y in batch:
            n.fit(x, y)
    print('test: ' + test(n, test_data))
n.plot()
