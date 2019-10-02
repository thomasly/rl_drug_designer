import pickle as pk

import tensorflow as tf
from tf.keras.callbacks import ModelCheckpoint

from models import lstm_model
from utils.paths import Path
from utils.smiles_reader import smiles_sampler
from utils.smiles_reader import smiles2sequence
from utils.smiles_reader import get_smiles_tokens


def prepare_sequences(tokens, n_vocab, seq_len=100):
    """ Prepare the sequences used by the Neural Network """

     # create a dictionary to map pitches to integers
    token_to_int = dict((token, number) for number, token in enumerate(tokens))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - seq_len, 1):
        sequence_in = notes[i:i + seq_len]
        sequence_out = notes[i + seq_len]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, seq_len, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=64,
              callbacks=callbacks_list)

