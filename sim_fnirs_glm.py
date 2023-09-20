import ray
import glob
import os
import numpy as np
from nilearn.plotting import plot_design_matrix
import nilearn.glm
from nilearn.glm.first_level import run_glm as nilearn_glm
import pandas as pd


@ray.remote
def run_glm(
    i, raw_channel, design_matrix, noise_model="ar1", bins=0, n_jobs=1, verbose=0
):
    # num_channels = raw.shape[0]
    # bins = num_channels
    # results = dict()
    # for i in range(num_channels):
    try:
        labels, glm_estimates = nilearn_glm(
            # raw[i : i + 1].T,
            raw_channel.T,
            design_matrix.values,
            noise_model=noise_model,
            bins=bins,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    except Exception as e:
        print("channel {}: {}, sum: {}".format(i, raw_channel, raw_channel.sum()))
        return None
    # results[i] = glm_estimates[labels[0]]
    estimate = glm_estimates[labels[0]]
    return estimate.theta.squeeze()


runs = glob.glob("data/**/head_seg.npy", recursive=True)


def make_first_level_design_matrix(
    frame_times,
    conditions,
    onsets,
    stim_dur=1.0,
    hrf_model="glover",
    drift_model="cosine",
    high_pass=0.01,
    drift_order=1,
    fir_delays=[0],
    add_regs=None,
    add_reg_names=None,
    min_onset=-24,
    oversampling=50,
):
    from nilearn.glm.first_level import make_first_level_design_matrix
    from pandas import DataFrame

    # frame_times = raw.times

    # Create events for nilearn
    # conditions = raw.annotations.description
    # onsets = raw.annotations.onset - raw.first_time
    duration = stim_dur * np.ones(len(conditions))
    events = DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": duration}
    )

    dm = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model=drift_model,
        drift_order=drift_order,
        hrf_model=hrf_model,
        min_onset=min_onset,
        high_pass=high_pass,
        add_regs=add_regs,
        oversampling=oversampling,
        add_reg_names=add_reg_names,
        fir_delays=fir_delays,
    )
    return dm


@ray.remote
def process(path):
    folder, file = os.path.split(path)
    glm_folder = "glm_" + folder
    events_folder = "events_" + folder
    events_path = os.path.join(events_folder, "events.tsv")
    if not os.path.exists(glm_folder):
        os.makedirs(glm_folder)
    # print(path, folder)
    fnirs_folder = os.path.join(folder, "fnirs_data")
    fnirs = glob.glob("{}/*.npy".format(fnirs_folder))
    data = []
    for x in fnirs:
        d = np.load(x)
        # print("path: {}, shape: {}".format(x, d.shape))
        data.append(d)
    data = np.vstack(data)
    # print(fnirs)
    # print(f"{data.shape = }")
    # filter the data
    data = np.array([row for row in data if row.sum() >= 1])
    # print("post filter data shape: {}".format(data.shape))

    times = np.arange(data.shape[-1]) * 1.6
    df = pd.read_csv(events_path, sep="\t")
    dm = make_first_level_design_matrix(times, df["trial_number"], df["onset"])
    # print(dm.shape)
    num_channels = data.shape[0]
    # print(f"{num_channels = }")
    thetas = ray.get(
        [
            run_glm.remote(i, data[i : i + 1], dm, bins=num_channels)
            for i in range(num_channels)
        ]
    )
    thetas = np.array(thetas)
    # print(thetas.shape)
    num_trials = len(df)
    # print(num_trials, len(df["onset"]), dm.columns[: len(df["onset"])])
    trial_thetas = thetas[:, :num_trials]
    glm_path = os.path.join(glm_folder, "glm_betas.npy")
    np.save(glm_path, trial_thetas)


if __name__ == "__main__":
    ray.init()
    job_ids = [process.remote(file) for file in runs[:]]
    results = ray.get(job_ids)
