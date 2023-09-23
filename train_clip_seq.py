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
from ds import fNIRSDataset
import os
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
    "name": "6gpu-10",
    "seq_len": 3,
    "d_model": 256,
    "nhead": 16,
    "num_layers": 24,
    "dim_feedforward": 4096,
    "dropout": 0.1,
    # "resume_path": "/lfs/ampere1/0/suppakit/ml/ckpts/6gpu-testckpt1/10"
}

ds_name = "seq_dataset_{}".format(config["seq_len"])

ckpt_dir = "ckpts/{}".format(config["name"])
if accelerator.is_main_process:
    assert not os.path.exists(ckpt_dir)
    os.makedirs(ckpt_dir)


# https://huggingface.co/docs/accelerate/concept_guides/performance
if accelerator.is_main_process:
    print(f"{accelerator.num_processes = }")
config["lr"] *= accelerator.num_processes
config["num_processes"] = accelerator.num_processes
clip_model, preprocess = clip.load(
    "ViT-B/32", device=device, jit=False, download_root="/lfs/ampere1/0/suppakit")


# class fNIRSDataset(Dataset):
#     def __init__(self, path):
#         self.df, seq_features = utils.load_seq_ds(path)
#         # seq_features (num_examples, d, seq_len)
#         seq_features = seq_features.transpose(0, 2, 1)
#         # (num_examples, seq_len, d)
#         d = seq_features.shape[-1]
#         # (N*5, 319)
#         seq_features = seq_features.reshape(-1, d)
#         scaler = StandardScaler()
#         seq_features = scaler.fit_transform(seq_features)
#         # (N, seq_len, 319)
#         seq_features = seq_features.reshape(-1, config["seq_len"], d)
#         self.seq_features = seq_features

#         self.images = h5py.File('nsd_stimuli.hdf5.1', 'r')
#         self.img_ids = self.df["img_ids"]
#         assert self.seq_features.shape[0] == len(self.img_ids)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         # image = preprocess(Image.open(self.img_paths[idx]))
#         img_id = self.img_ids[idx]
#         image = preprocess(Image.fromarray(self.images['imgBrick'][img_id]))
#         # image = preprocess(Image.open(
#         #     "images/shared0001_nsd02951.png"))  # dummy for now
#         features = self.seq_features[idx]
#         return image, features


ds = fNIRSDataset(ds_name, preprocess)
print(f"{ds_name = }, {len(ds) = }")
dl = DataLoader(ds, batch_size=config["batch_size"], shuffle=True)

accelerator.init_trackers("clip1", config=config)
loss_img = nn.CrossEntropyLoss()
loss_fnirs = nn.CrossEntropyLoss()


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

model, optimizer, training_dataloader = accelerator.prepare(
    model, optimizer, dl
)


save_model_every = 10

if "resume_path" in config:
    resume_dir  = config["resume_path"]
    accelerator.load_state(resume_dir)
    base, folder = os.path.split(resume_dir)
    step = int(folder)
    if accelerator.is_main_process:
        print("resuming from resume_dir: {}".format(resume_dir))
    # model = accelerator.unwrap_model(model)
    # pkg = torch.load(config["resume_path"])
    # breakpoint()
    # model.load_state_dict(pkg['model'])
    # optim.load_state_dict(pkg['optim'])
    # folder, file = os.path.split(config["resume_path"])
    # step = int(file.split(".")[0])    
    # print("resuming from step: {}, loaded from path: {}", step, config["resume_path"])

for epoch in range(config["epochs"]):
    for batch in dl:
        optimizer.zero_grad()
        images, features, img_ids = batch

        images = images.to(device)
        features = features.to(device).float()
        features = features[:, :, :config["d_model"]]
        # print(f"{features.shape = }")

        image_features = clip_model.encode_image(images)
        fnirs_features = model(features)

        image_features = image_features / \
            image_features.norm(dim=1, keepdim=True)
        fnirs_features = fnirs_features / \
            fnirs_features.norm(dim=1, keepdim=True)

        logit_scale = clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features.float() @ fnirs_features.t()
        logits_per_fnirs = logits_per_image.t()  # (B, B)

        ground_truth = torch.arange(
            len(images), dtype=torch.long, device=device)
        loss = (loss_img(logits_per_image, ground_truth) +
                loss_fnirs(logits_per_fnirs, ground_truth))/2
        # total_loss.backward()
        accelerator.backward(loss)

        if accelerator.is_main_process and not (step % save_model_every):
            ckpt_path = "{}/{}".format(ckpt_dir, step)
            os.makedirs(ckpt_path, exist_ok=True)
            accelerator.save_state(output_dir=ckpt_path)
            # accelerator.save_model(model, model_path)
            # save(model_path, accelerator, optimizer, model)
            print(f'{step}: saving model to {ckpt_path}')

        if accelerator.is_main_process:
            print("epoch: {}, step: {}, loss: {}".format(
                epoch, step, loss.item()))
            accelerator.log({"training_loss": loss, "epoch": epoch}, step=step)
        optimizer.step()
        step += 1
accelerator.end_training()
