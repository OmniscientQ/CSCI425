'''
Demonstrates Recurrent Neural Networks
'''

import sys
import numpy as np
import tensorflow as tf
import keras
from keras import layers


def main() -> int:
    model = keras.Sequential()

    model.add(layers.Embedding(input_dim=1024, output_dim=3))
    model.add(layers.GRU(128))
    model.add(layers.SimpleRNN(128))
    model.add(layers.Dense(10))

    model.summary()

    return 0


if __name__ == '__main__':
    sys.exit(main())
