import random

import numpy as np
from tqdm import tqdm
from nn.network import NeuralNetwork
import utils.mnist_loader as mnist_loader

def test(nn, test_data):
    result = [(np.argmax(nn.predict(x)), y) for (x, y) in test_data]
    print("{0} / {1}".format(sum([int(x == y) for (x, y) in result]), len(result)))

sizes = [784, 50, 10]
n = NeuralNetwork(sizes, 0.01)
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
test(n, test_data)
bs = 100
for i in range(20):
    random.shuffle(training_data)
    batches = [training_data[k:k+bs] for k in range(0, len(training_data), bs)]
    for batch in batches:
        for x, y in batch:
            n.fit(x, y)
    test(n, test_data)
