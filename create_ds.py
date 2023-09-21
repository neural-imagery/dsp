import glob
import pandas as pd
import numpy as np
import os
import h5py


runs = glob.glob("data/**/head_seg.npy", recursive=True)
out = "dataset"
if not os.path.exists(out):
    os.makedirs(out)


img_ids = []
betas = []
event_paths = []

images = h5py.File('nsd_stimuli.hdf5.1', 'r')
# breakpoint()

for path in runs:
    folder, file = os.path.split(path)
    betas_path = os.path.join("glm_" + folder, "glm_betas.npy")
    events_path = os.path.join("events_" + folder, "events.tsv")
    run_betas = np.load(betas_path)
    df = pd.read_csv(events_path, sep="\t")

    fnirs_folder = os.path.join(folder, "fnirs_data")
    fnirs = glob.glob("{}/*.npy".format(fnirs_folder))
    data = []
    for x in fnirs:
        d = np.load(x)
        # print("path: {}, shape: {}".format(x, d.shape))
        data.append(d)
    data = np.vstack(data)
    times = np.arange(data.shape[-1]) * 1.6
    # 10k_ids = df["10k_id"]
    # 73k_ids = df["73k_id"]
    num_trials = len(df)
    # breakpoint()
    img_ids.extend(list(df["73k_id"].values))
    # for id73k, id10k in zip(df["73k_id"], df["10k_id"]):
    #     img_path = "images/shared{}_nsd{}.png".format(str(id10k).zfill(4), str(id73k).zfill(5))
    #     # print(img_path)
    #     img_paths.append(img_path)

    # for on in df["onset"]:
    #     # print(on)
    #     if on == 100:
    #         breakpoint()
    #     trial_ind = np.where(times > on)[0][0]
    #     betas.append(run_betas[:, trial_ind])
    # breakpoint()
    betas.append(run_betas)
    event_paths.extend([path] * num_trials)
    assert num_trials == run_betas.shape[-1]

max_channels =  max([b.shape[0] for b in betas])
betas = [np.pad(b, ((0,max_channels-b.shape[0]), (0,0))) for b in betas]
betas = np.hstack(betas)
assert betas.shape[-1] == len(event_paths)
assert len(event_paths) == len(img_ids)
ds = pd.DataFrame({
    "img_ids": img_ids,
    "event_paths": event_paths
})
ds.to_csv("{}/paths.csv".format(out))
np.save("{}/betas.npy".format(out), betas)
breakpoint()