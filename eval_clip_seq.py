import clip
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
import IPython
import h5py
from IPython import embed
import utils
import numpy as np
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
import os
import matplotlib.pyplot as plt
import wandb
from ds import fNIRSDataset
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
config = {
    "batch_size": 2048,
    # "batch_size": 2400,
    "epochs": 100_000_000,
    "lr": 5e-5,
    "weight_decay": 0.001,
    "name": "6gpu-6",
    "seq_len": 3,
    "d_model": 256,
    "nhead": 16,
    "num_layers": 24,
    "dim_feedforward": 4096,
    "dropout": 0.1,
    # "resume_path": "/lfs/ampere1/0/suppakit/ml/ckpts/6gpu-5/800.pt"
    # "resume_path": "/lfs/ampere1/0/suppakit/ml/ckpts/6gpu-testckpt1/10"
    # "resume_path": "/lfs/ampere1/0/suppakit/ml/ckpts/6gpu-6/780"
    "resume_path": "/lfs/ampere1/0/suppakit/ml/ckpts/6gpu-10/50"
}

config["ds_name"] = "seq_dataset_{}".format(config["seq_len"])
config["val_ds_name"] = "val_seq_dataset_{}".format(config["seq_len"])

# https://huggingface.co/docs/accelerate/concept_guides/performance
if accelerator.is_main_process:
    print(f"{accelerator.num_processes = }")
config["lr"] *= accelerator.num_processes
config["num_processes"] = accelerator.num_processes
clip_model, preprocess = clip.load(
    "ViT-B/32", device=device, jit=False, download_root="/lfs/ampere1/0/suppakit")


train_ds = fNIRSDataset(config["ds_name"], preprocess)
print(f"{len(train_ds) = }")
train_dl = DataLoader(train_ds, batch_size=config["batch_size"])
val_ds = fNIRSDataset(config["val_ds_name"], preprocess, scaler=train_ds.scaler, trunc=train_ds.seq_features.shape[-1])
val_dl = DataLoader(val_ds, batch_size=config["batch_size"])
# breakpoint()

accelerator.init_trackers("clip1", config=config)
# loss_img = nn.CrossEntropyLoss()
# loss_fnirs = nn.CrossEntropyLoss()


step = 1


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout), num_layers)
        self.dense = nn.Linear(d_model, 512)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.dense.bias.data.zero_()
        self.dense.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # rearrange input to have sequence length dimension first, for the Positional Encoding
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # mean used to generate a single embedding
        output = self.dense(output.mean(dim=0))
        return output


# d_model = ds.seq_features.shape[-1]
print(f"{config['d_model'] = }")

model = TimeSeriesTransformer(
    config["d_model"], config["nhead"], config["num_layers"], config["dim_feedforward"], config["dropout"]).to(device)

# embedding_size stolen from CLIP
num_params = utils.count_parameters(model)
accelerator.log({"num_params": num_params})
print("num params: {}".format(num_params))


optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                       betas=(0.9, 0.98), eps=1e-6, weight_decay=config["weight_decay"])

model, optimizer, train_dl, val_dl = accelerator.prepare(
    model, optimizer, train_dl, val_dl
)

if "resume_path" in config:
    resume_dir  = config["resume_path"]
    accelerator.load_state(resume_dir)
    base, folder = os.path.split(resume_dir)
    step = int(folder)
    if accelerator.is_main_process:
        print("resuming from resume_dir: {}".format(resume_dir))


def eval_ds(dl, ds_type="train"):
    fnirs_features = []
    gt_img_ids = []
    for i, batch in enumerate(dl):
        print("[{}] batch: {}".format(ds_type, i))
        images, features, img_ids = batch
        features = features.to(device)
        features = features[:, :, :config["d_model"]]
        fnirs_features.append(model(features).cpu().detach().numpy())
        gt_img_ids.append(img_ids.cpu().detach().numpy())
        break
    # IPython.embed()
    gt_img_ids = np.vstack(gt_img_ids).squeeze()
    image_features = np.load("stimuli.py.npy")
    image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    fnirs_features = np.vstack(fnirs_features)
    fnirs_features = fnirs_features / np.linalg.norm(fnirs_features, axis=1, keepdims=True)
    logit_scale = np.exp(clip_model.logit_scale.cpu().detach().numpy())
    logits_per_fnirs = logit_scale * fnirs_features @ image_features.T
    top_img_inds = np.argmax(logits_per_fnirs, axis=1)
    acc = (gt_img_ids == top_img_inds).sum() / gt_img_ids.shape[0]
    print("[{}] acc: {}".format(ds_type, acc))

    accelerator.log({"{}_acc".format(ds_type): acc})

    # get the ranks
    all_ranks = []
    for i in range(logits_per_fnirs.shape[0]):
        logits = logits_per_fnirs[i]
        img_id = gt_img_ids[i]
        sort_indices = np.argsort(logits)[::-1]
        ranks = np.empty_like(sort_indices)
        ranks[sort_indices] = np.arange(len(logits))
        rank = ranks[img_id]
        all_ranks.append(rank)

    all_ranks = np.array(all_ranks)
    plt.hist(all_ranks, bins=50, alpha=0.5, color='g', edgecolor='black')
    plt.title('ranks distr')
    plt.xlabel('ranks')
    plt.ylabel('freq')
    plt.savefig('{}_ranks_hist.png'.format(ds_type))
    accelerator.log({"{}_ranks_hist".format(ds_type): wandb.Image("{}_ranks_hist.png".format(ds_type))})
    top5_acc = sum(all_ranks < 5) / len(all_ranks)
    print("[{}] top5 acc: {}".format(ds_type, top5_acc))
    accelerator.log({"{}_top5_acc".format(ds_type): top5_acc})

eval_ds(train_dl, ds_type="train")
eval_ds(val_dl, ds_type="val")




# image_features = image_features / \
#     image_features.norm(dim=1, keepdim=True)
# fnirs_features = fnirs_features / \
#     fnirs_features.norm(dim=1, keepdim=True)

# logit_scale = clip_model.logit_scale.exp()
# logits_per_image = logit_scale * image_features.float() @ fnirs_features.t()
# logits_per_fnirs = logits_per_image.t()