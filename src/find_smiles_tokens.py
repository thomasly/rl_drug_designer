import os
import gzip
import pickle as pk
from random import shuffle

from tqdm import tqdm
from utils.paths import Path
from utils.smiles_reader import get_smiles_from_sdf


threshold = 10
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
        smiles_token = smiles_token.union(set(ss.decode("utf-8")))
        if len(smiles_token) == last_len:
            round +=1
            if round == threshold:
                break
        else:
            round = 0
            last_len = len(smiles_token)
    
print(smiles_token)
print(f"Total tokens: {len(smiles_token)}")
save_path = os.path.join(Path.data, "smiles_token")
with open(save_path, "wb") as f:
    pk.dump(smiles_token, f)
