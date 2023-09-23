import utils
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import h5py
from PIL import Image

class fNIRSDataset(Dataset):
    def __init__(self, path, preprocess):
        self.preprocess = preprocess
        self.df, seq_features = utils.load_seq_ds(path) # (num_examples, d, seq_len)
        self.seq_features = seq_features.transpose(0, 2, 1) # (num_examples, seq_len, d)
        self.images = h5py.File('nsd_stimuli.hdf5.1', 'r')
        self.img_ids = self.df["img_ids"]
        assert self.seq_features.shape[0] == len(self.img_ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # image = preprocess(Image.open(self.img_paths[idx]))
        img_id = self.img_ids[idx] - 1 # important!
        image = self.preprocess(Image.fromarray(self.images['imgBrick'][img_id]))
        features = self.seq_features[idx]
        return image, features, img_id