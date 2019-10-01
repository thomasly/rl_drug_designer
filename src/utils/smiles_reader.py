import os
import gzip
from random import sample, shuffle

from .paths import Path


def get_smiles_from_sdf(path):
    smiles = list()
    with gzip.open(path, "rb") as sdf:
        line = sdf.readline()
        while line:
            if b"ISO_SMILES" in line:
                line = sdf.readline()
                if len(line) > 100:
                    line = sdf.readline()
                    continue
                smiles.append(line)
            line = sdf.readline()
    return smiles


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


if __name__ == "__main__":
    gen = smiles_sampler(100)
    for i in range(20):
        print(next(gen))
