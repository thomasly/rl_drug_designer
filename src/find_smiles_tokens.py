import os
import argparse
import pickle as pk
from random import shuffle

from tqdm import tqdm
from rdkit import Chem

from utils.paths import Path
from utils.smiles_reader import get_smiles_from_sdf


def get_nonelement_tokens(smiles):
    nonelements = set()
    for t in set(smiles):
        if t.isalpha():
            continue
        nonelements.add(t)
    return nonelements


def get_tokens(smiles):
    tokens = set()
    mol = Chem.MolFromSmiles(smiles)
    try:
        atoms = mol.GetAtoms()
        atoms = set(map(lambda x: x.GetSymbol(), atoms))
        tokens = tokens.union(atoms)
    except AttributeError:
        print("{} is not valid".format(smiles))
    other_tokens = get_nonelement_tokens(smiles)
    tokens = tokens.union(other_tokens)
    return tokens


def find_tockens(threshold):
    gzfiles = list(os.scandir(Path.sdf_full))
    shuffle(gzfiles)
    smiles_token = set()

    round = 0
    last_len = 0
    for gzfile in tqdm(gzfiles):
        if not gzfile.name.startswith("Compound"):
            continue
        smiles = get_smiles_from_sdf(gzfile.path)
        for ss in smiles:
            smiles_token = smiles_token.union(get_tokens(ss.decode("utf-8")))
        if len(smiles_token) == last_len:
            round += 1
            if round == threshold:
                break
        else:
            round = 0
            last_len = len(smiles_token)

    print(smiles_token)
    print("Total tokens: {}".format(len(smiles_token)))
    save_path = os.path.join(Path.data, "smiles_token")

    os.makedirs(Path.data, exist_ok=True)
    with open(save_path, "wb") as f:
        smiles_token = list(smiles_token)
        smiles_token.sort()
        pk.dump(smiles_token, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=int, help="Threshold number")
    args = parser.parse_args()
    find_tockens(threshold=args.threshold)
