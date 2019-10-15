import argparse

from utils.smiles_reader import is_valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True,
                        help="Path to the file containing SMILES.")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        contents = f.readlines()

    counter = 0
    n_smiles = 0
    for line in contents:
        if line == "\n":
            continue
        n_smiles += 1
        if is_valid(line):
            counter += 1
    print("Total: {}, valid: {}, valid rate: {:.2f}".format(
        n_smiles, counter, counter/n_smiles))
