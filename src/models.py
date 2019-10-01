import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import Dense
from tf.keras.layers import Dropout
from tf.keras.layers import LSTM
from tf.keras.layers import Activation


def lstm_model(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        clipvalue=0.5,
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True, clipvalue=0.5))
    model.add(Dropout(0.3))
    model.add(LSTM(512, clipvalue=0.5))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model
