import ray
import glob
import os
import requests

fnirs = glob.glob("data/**/head_seg.npy", recursive=True)


@ray.remote
def process(path):
    folder, file = os.path.split(path)
    print(folder)
    parts = folder.split("/")
    run_num = parts[-1].split("run")[-1]
    sess_num = parts[-2].split("sess")[-1]
    sub_num = parts[-4].split("sub")[-1]
    tsv_url = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata_rawdata/sub-{}/ses-nsd{}/func/sub-{}_ses-nsd{}_task-nsdcore_run-{}_events.tsv".format(
        sub_num, sess_num, sub_num, sess_num, run_num
    )
    # print(run_num, sess_num, tsv_path)
    events_folder = "events_" + folder
    if not os.path.exists(events_folder):
        os.makedirs(events_folder)
    events_path = os.path.join(events_folder, "events.tsv")
    # Ensure the directory exists before running the script
    # You need to put your own directory in 'your_directory'
    response = requests.get(tsv_url)
    with open(events_path, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    ray.init()
    job_ids = [process.remote(file) for file in fnirs]
    results = ray.get(job_ids)
