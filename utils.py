import pandas as pd
import numpy as np

def load_ds(path):
    df = pd.read_csv("{}/paths.csv".format(path))
    betas = np.load("{}/betas.npy".format(path))
    return df, betas
    