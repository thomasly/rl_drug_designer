import numpy as np

from .smiles_reader import get_int2token


def make_name(model, seq_len=100):
    index_to_char = get_int2token()
    vocab_len = len(index_to_char)
    name = []
    x = np.zeros((1, seq_len, vocab_len))
    end = False
    i = 0

    while not end:
        probs = list(model.predict(x)[0, i])
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
