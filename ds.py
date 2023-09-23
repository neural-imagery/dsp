import utils
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import h5py
from PIL import Image

class fNIRSDataset(Dataset):
    def __init__(self, path, preprocess, scaler=None, trunc=None):
        self.preprocess = preprocess
        self.df, seq_features = utils.load_seq_ds(path)
        
        # seq_features (num_examples, d, seq_len)
        seq_features = seq_features.transpose(0, 2, 1)
        seq_len = seq_features.shape[1]
        # (num_examples, seq_len, d)
        if trunc:
            print("truncating to {}".format(trunc))
            print(seq_features.shape)
            seq_features = seq_features[:, :, :trunc]

        d = seq_features.shape[-1]
        # (N*5, 319)
        seq_features = seq_features.reshape(-1, d)
        if not scaler:
            scaler = StandardScaler()
            seq_features = scaler.fit_transform(seq_features)
        else:
            print("using existing scaler!")
            seq_features = scaler.transform(seq_features)
        self.scaler = scaler

        seq_features = seq_features.reshape(-1, seq_len, d)
        self.seq_features = seq_features

        self.images = h5py.File('nsd_stimuli.hdf5.1', 'r')
        self.img_ids = self.df["img_ids"]
        assert self.seq_features.shape[0] == len(self.img_ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # image = preprocess(Image.open(self.img_paths[idx]))
        img_id = self.img_ids[idx] - 1
        image = self.preprocess(Image.fromarray(self.images['imgBrick'][img_id]))
        # image = preprocess(Image.open(
        #     "images/shared0001_nsd02951.png"))  # dummy for now
        features = self.seq_features[idx]
        return image, features, img_id