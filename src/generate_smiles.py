import os
import argparse

import tensorflow as tf
from tqdm import tqdm

from models import lstm_model
from utils.smiles_reader import get_smiles_tokens
from utils.model_actions import make_name
from utils.paths import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Path to the saved model.")
    parser.add_argument("-n", "--n_samples", type=int, default=100, 
                        help="Number of samples to generate")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    log_file = os.path.join(Path.log, "SMILES_after_step1.txt")
    with open(log_file, "w") as f:
        for _ in tqdm(range(args.n_samples)):
            name = make_name(model)
            print(name, file=f)
