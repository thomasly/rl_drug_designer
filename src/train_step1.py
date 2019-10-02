import os
import pickle as pk

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback

from models import lstm_model
from utils.paths import Path
from utils.smiles_reader import smiles_sampler
from utils.smiles_reader import smiles2sequence
from utils.smiles_reader import get_smiles_tokens


def sequences_generator(batch_size=32,
                        seq_len=100, n_samples=1000):
    """ Generate sequences for training the Neural Network.
    vocab_len (int): length of the vocabulary
    batch_size (int): size of mini batches
    seq_len (int): the max length of the sequences
    n_samples (int): number of SMILES to sample from pubChem dataset
    """

    
    tokens = get_smiles_tokens()
    vocab = dict((name, index) for index, name in enumerate(tokens))
    vocab_len = len(tokens)
     # create a dictionary to map tokens to integers
    sampler = smiles_sampler(n_samples)

    while 1:
        X = np.zeros((batch_size, seq_len, vocab_len))
        Y = np.zeros((batch_size, seq_len, vocab_len))

        # create input sequences and the corresponding outputs
        for i in range(batch_size):
            ss = next(sampler)
            int_sequence = smiles2sequence(ss, vocab)
            for j in range(len(int_sequence)-1):
                X[i, j, int_sequence[j]] = 1
                Y[i, j, int_sequence[j+1]] = 1
                if int_sequence[j+1] == 0:
                    break
        yield X, Y


def make_name(model, seq_len=100):
    tokens = get_smiles_tokens()
    index_to_char = dict((index, name) for index, name in enumerate(tokens))
    vocab_len = len(tokens)
    name = []
    x = np.zeros((1, seq_len, vocab_len))
    end = False
    i = 0
    
    while end==False:
        probs = list(model.predict(x)[0,i])
        probs = probs / np.sum(probs)
        index = np.random.choice(range(vocab_len), p=probs)
        if i == seq_len-2:
            character = '\n'
            end = True
        else:
            character = index_to_char[index]
        name.append(character)
        x[0, i+1, index] = 1
        i += 1
        if character == '\n':
            end = True
    
    return ''.join(name)


def generate_name_loop(epoch, _):
    if epoch % 24 == 0:
        print('SMILES generated after epoch %d:' % epoch)
        for i in range(3):
            name = make_name(model)
            print(name)
        print()


def train(model, batch_size=32, epochs=100, n_samples=1000):
    """ train the neural network """
    saving_path = Path.checkpoints
    os.makedirs(saving_path, exist_ok=True)
    filepath = os.path.join(
        saving_path, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    name_generator = LambdaCallback(on_epoch_end = generate_name_loop)
    callbacks_list = [checkpoint, name_generator]
    
    data_generator = sequences_generator(
        batch_size=batch_size, n_samples=n_samples)

    steps = int(n_samples/batch_size)
    model.fit_generator(
        data_generator, steps_per_epoch=steps, epochs=epochs,
        callbacks=callbacks_list)


if __name__ == "__main__":
    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument("-s", "--n_samples", type=int, default=1000)
    paser.add_argument("-b", "--batch_size", type=int, default=32)
    paser.add_argument("-e", "--epochs", type=int, default=100)
    args = paser.parse_args()

    n_samples = args.n_samples
    batch_size = args.batch_size
    epochs = args.epochs

    tokens = get_smiles_tokens()
    vocab_len = len(tokens)
    seq_len=100
    input_shape = (seq_len, vocab_len)
    model = lstm_model(input_shape)
    train(model, batch_size, epochs, n_samples)
