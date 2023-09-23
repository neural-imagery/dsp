import glob
import pandas as pd
import numpy as np
import os
import h5py
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


ds_type = "val"
runs = glob.glob("{}_data/**/head_seg.npy".format(ds_type), recursive=True)
if ds_type == "train":
    scaler = StandardScaler()
    good_rows = []
else:
    print("load scaler and good_rows!")
    scaler = load("scaler.joblib")
    good_rows = np.load("good_rows.npy")
seq_len = 3
out = "{}_seq_dataset_{}".format(ds_type, seq_len)
if not os.path.exists(out):
    os.makedirs(out)


img_ids = []
seq_features = []
event_paths = []

images = h5py.File('nsd_stimuli.hdf5.1', 'r')

# create the scaler
if ds_type == "train":
    print("fitting scaler!")
    all_data = []
    for path in runs:
        folder, file = os.path.split(path)
        fnirs_folder = os.path.join(folder, "fnirs_data")
        fnirs = glob.glob("{}/*.npy".format(fnirs_folder))
        data = []
        for x in fnirs:
            d = np.load(x)
            # print("path: {}, shape: {}".format(x, d.shape))
            data.append(d)
        data = np.vstack(data)
        if not len(good_rows):
            for i in range(data.shape[0]):
                if data[i].sum() != 0:
                    good_rows.append(i)
            print("len good rows: {}".format(len(good_rows)))
            good_rows = np.array(good_rows)
            np.save("good_rows.npy", good_rows)
        data = data[good_rows]
        all_data.append(data)
    all_data = np.vstack(all_data)
    all_data = scaler.fit_transform(all_data)
    dump(scaler, "scaler.joblib")
    # breakpoint()

print(f"{len(good_rows) = }")

for path in runs:
    folder, file = os.path.split(path)
    events_path = os.path.join("events_" + folder, "events.tsv")
    df = pd.read_csv(events_path, sep="\t")
    fnirs_folder = os.path.join(folder, "fnirs_data")
    fnirs = glob.glob("{}/*.npy".format(fnirs_folder))
    data = []
    for x in fnirs:
        d = np.load(x)
        # print("path: {}, shape: {}".format(x, d.shape))
        data.append(d)
    data = np.vstack(data)
    data = data[good_rows]
    # normalize!
    data = scaler.transform(data)

    times = np.arange(data.shape[-1]) * 1.6
    num_trials = len(df)
    img_ids.extend(list(df["73k_id"].values))
    onsets = df["onset"].values

    for i, on in enumerate(onsets):
        time_idx = np.where(times > on)[0][0]
        feat = data[:, time_idx:time_idx+seq_len]
        if feat.shape[-1] < seq_len:
            print(i, feat.shape)
            breakpoint()
            feat = np.pad(feat, ((0, 0), (0, seq_len - feat.shape[-1])))
        seq_features.append(feat)

    print(f"{data.shape = }, {path = }")
    event_paths.extend([path] * num_trials)


assert len(seq_features) == len(event_paths)
assert len(event_paths) == len(img_ids)
# seq_features = np.array(seq_features)
max_channels = max([feat.shape[0] for feat in seq_features])
seq_features = [np.pad(feat, ((0, max_channels-feat.shape[0]), (0, 0)))
                for feat in seq_features]
seq_features = np.array(seq_features)
print(f"{seq_features.shape = }")
assert seq_features.shape[0] == len(event_paths)
ds = pd.DataFrame({
    "img_ids": img_ids,
    "event_paths": event_paths
})
ds.to_csv("{}/paths.csv".format(out))
np.save("{}/seq_features.npy".format(out), seq_features)
breakpoint()
