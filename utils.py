import pandas as pd
import numpy as np

def load_ds(path):
    df = pd.read_csv("{}/paths.csv".format(path))
    betas = np.load("{}/betas.npy".format(path))
    return df, betas

def load_seq_ds(path):
    df = pd.read_csv("{}/paths.csv".format(path))
    seq_features = np.load("{}/seq_features.npy".format(path))
    return df, seq_features

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)