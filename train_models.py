## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
import os

def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1,
          init=None) -> keras.Sequential:
    """
    Standard neural network training procedure.

    Create model
      params are size of: [conv0, conv1, conv2, conv3, dense, dense]
    load weights from init
    fit model to data for num_epochs, with batch_size
    save weights to file_name
    return model

    Model as defined in setup_mnist,
     but: with dropout
          image shape driven by parameter array
    """
    model = Sequential()

    print(data.train_data.shape)

    model.add(Conv2D(params[0], (3, 3),
                     input_shape=data.train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(10))

    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        """Cross entropy of how much correct labels match predicted."""
        return tf.nn.softmax_cross_entropy_with_logits(
          labels=correct,
          logits=predicted/train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)

    if file_name != None:
        model.save(file_name)

    return model

if not os.path.isdir('models'):
    os.makedirs('models')

train(MNIST(), "models/mnist", [32, 32, 64, 64, 200, 200], num_epochs=50)
