import os
import gzip
import pickle as pk
from random import shuffle

from rdkit import Chem
import numpy as np

from .paths import Path


def get_smiles_from_sdf(path, max_len=100):
    smiles = list()
    with gzip.open(path, "rb") as sdf:
        line = sdf.readline()
        while line:
            if b"ISO_SMILES" in line:
                line = sdf.readline()
                if len(line) > max_len:
                    line = sdf.readline()
                    continue
                smiles.append(line)
            line = sdf.readline()
    return smiles


def is_valid(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return True


def smiles_sampler(n_samples):
    """ Sample n_samples isomeric SMILES string from pubChem dataset
    """
    files = list(os.scandir(Path.sdf_full))
    shuffle(files)

    smiles = list()
    counter = 0
    for gzf in files:
        if not gzf.name.startswith("Compound"):
            continue
        smiles_in_gzf = get_smiles_from_sdf(gzf.path)
        for ss in smiles_in_gzf:
            if is_valid(ss):
                smiles.append(ss)
                counter += 1
                if counter == n_samples:
                    break
        if counter == n_samples:
            break

    if counter < n_samples:
        print("No enough samples in the dataset. Sampled {}.".format(counter))
    else:
        print(
            "Successfully sampled {} samples from the datset.".format(counter))
    while 1:
        shuffle(smiles)
        for ss in smiles:
            ss = ss.decode("utf-8")
            yield ss


def smiles2sequence(smiles, vocab, max_len=100):
    vocab_len = len(vocab)
    sequence = [0] * max_len
    idx_ss = 0
    idx_seq = 0
    while idx_ss < len(smiles):
        if idx_ss == len(smiles) - 1:
            sequence[idx_seq] = vocab[smiles[idx_ss]]
            return sequence
        if smiles[idx_ss:idx_ss+2] in vocab:
            sequence[idx_seq] = vocab[smiles[idx_ss:idx_ss+2]]
            idx_ss += 2
            idx_seq += 1
        else:
            sequence[idx_seq] = vocab[smiles[idx_ss]]
            idx_ss += 1
            idx_seq += 1
    return sequence


def get_smiles_tokens():
    """ Get all the notes and chords from the midi files in the ./midi_songs \
        directory 
    """
    with open(Path.smiles_tokens, "rb") as f:
        tokens = pk.load(f)
    return tokens


if __name__ == "__main__":
    gen = smiles_sampler(100)
    for i in range(5):
        ss = next(gen)
        print(ss)
        print("length:", len(ss))
        tokens = get_smiles_tokens()
        vocab = dict((token, number) for number, token in enumerate(tokens))
        seq = smiles2sequence(ss, vocab)
        print("sequence:", seq)
