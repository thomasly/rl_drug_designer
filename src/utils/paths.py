from os import path as op


class Path:

    root = op.join(op.dirname(op.abspath(__file__)), op.pardir, op.pardir)
    shared = op.join(root, op.pardir)
    data = op.join(root, "data")
    datasets = op.join(shared, "datasets")
    src = op.join(root, "src")
    pubChem = op.join(datasets, "pubChem_compound")
    sdf_full = op.join(pubChem, "SDF_full")
    smiles_tokens = op.join(data, "smiles_token")
    checkpoints = op.join(root, "checkpoints")
    log = op.join(root, "logs")
