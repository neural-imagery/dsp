import clip
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
import h5py
from IPython import embed
import utils
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from sklearn.preprocessing import StandardScaler
import os
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
config = {
    # "batch_size": 2048,
    "batch_size": 2400,
    "epochs": 100,
    "lr": 5e-5,
    "weight_decay": 0.001,
    "name": "exp",
    "seq_len": 3
}
ds_name = "seq_dataset_{}".format(config["seq_len"])

ckpt_dir = "ckpts/{}".format(config["name"])
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# https://huggingface.co/docs/accelerate/concept_guides/performance
print(f"{accelerator.num_processes = }")
config["lr"] *= accelerator.num_processes
config["num_processes"] = accelerator.num_processes
clip_model, preprocess = clip.load(
    "ViT-B/32", device=device, jit=False, download_root="/lfs/ampere1/0/suppakit")


class fNIRSDataset(Dataset):
    def __init__(self, path):
        self.df, seq_features = utils.load_seq_ds(path)
        # seq_features (num_examples, d, seq_len)
        seq_features = seq_features.transpose(0, 2, 1)
        # (num_examples, seq_len, d)
        d = seq_features.shape[-1]
        # (N*5, 319)
        seq_features = seq_features.reshape(-1, d)
        scaler = StandardScaler()
        seq_features = scaler.fit_transform(seq_features)
        seq_features = seq_features.reshape(-1, config["seq_len"], d)  # (N, seq_len, 319)
        self.seq_features = seq_features

        self.images = h5py.File('nsd_stimuli.hdf5.1', 'r')
        self.img_ids = self.df["img_ids"]
        assert self.seq_features.shape[0] == len(self.img_ids)

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        # image = preprocess(Image.open(self.img_paths[idx]))
        img_id = self.img_ids[idx]
        image = preprocess(Image.fromarray(self.images['imgBrick'][img_id]))
        # image = preprocess(Image.open(
        #     "images/shared0001_nsd02951.png"))  # dummy for now
        features = self.seq_features[idx]
        return image, features


ds = fNIRSDataset(ds_name)
print(f"{ds_name = }, {len(ds) = }")
dl = DataLoader(ds, batch_size=config["batch_size"], shuffle=True)

accelerator.init_trackers("clip1", config=config)
loss_img = nn.CrossEntropyLoss()
loss_fnirs = nn.CrossEntropyLoss()

# embedding_size stolen from CLIP
num_params = utils.count_parameters(clip_model)
accelerator.log({"num_params": num_params})
print("num params: {}".format(num_params))
clip_model = clip_model.to(device)
clip_model.train()
optimizer = optim.Adam(clip_model.parameters(), lr=config["lr"],
                       betas=(0.9, 0.98), eps=1e-6, weight_decay=config["weight_decay"])

step = 1
clip_model, optimizer, training_dataloader = accelerator.prepare(
    clip_model, optimizer, dl
)
save_every = 10
for epoch in range(config["epochs"]):
    for batch in dl:
        optimizer.zero_grad()
        images, features = batch

        images = images.to(device)
        features = features.to(device).float()

        ground_truth = torch.arange(
            len(images), dtype=torch.long, device=device)
        logits_per_image, logits_per_fnirs = clip_model(images, features)
        loss = (loss_img(logits_per_image, ground_truth) +
                loss_fnirs(logits_per_fnirs, ground_truth))/2
        if not step % save_every and accelerator.is_main_process:
            embed()
            accelerator.save_model(clip_model, ckpt_dir)

        accelerator.backward(loss)

        if accelerator.is_main_process:
            print("epoch: {}, step: {}, loss: {}".format(
                epoch, step, loss.item()))
            accelerator.log({"training_loss": loss, "epoch": epoch}, step=step)
        optimizer.step()
        step += 1
accelerator.end_training()
