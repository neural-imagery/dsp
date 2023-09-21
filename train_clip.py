import clip
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
import h5py
import utils
from accelerate import Accelerator
accelerator = Accelerator(log_with="wandb")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
config = {
    "batch_size": 64,
    "epochs": 100,
    "lr": 5e-5
}

# https://huggingface.co/docs/accelerate/concept_guides/performance
print(f"{accelerator.num_processes = }")
config["lr"] *= accelerator.num_processes
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False, download_root="/lfs/ampere1/0/suppakit")



class fNIRSDataset(Dataset):
    def __init__(self, path):
        self.df, self.betas = utils.load_ds(path)
        self.betas = self.betas.T
        self.images = h5py.File('nsd_stimuli.hdf5.1', 'r')
        self.img_ids = self.df["img_ids"]
        assert len(self.betas) == len(self.img_ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # image = preprocess(Image.open(self.img_paths[idx]))
        img_id = self.img_ids[idx]
        image = preprocess(Image.fromarray(self.images['imgBrick'][img_id]))
        # image = preprocess(Image.open(
        #     "images/shared0001_nsd02951.png"))  # dummy for now
        betas = self.betas[idx]
        return image, betas


ds = fNIRSDataset("dataset")
breakpoint()
dl = DataLoader(ds, batch_size=config["batch_size"], shuffle=True)

accelerator.init_trackers("clip1", config=config)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()


# class fNIRSEmbedding(nn.Module):
#     def __init__(self, input_size, embedding_size, hidden_size=64):
#         super(fNIRSEmbedding, self).__init__()

#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, embedding_size)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

class fNIRSEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size=256):
        super(fNIRSEmbedding, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size*4)
        self.fc4 = nn.Linear(hidden_size*4, hidden_size*2)
        self.fc5 = nn.Linear(hidden_size*2, hidden_size)
        self.fc6 = nn.Linear(hidden_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        return x



# embedding_size stolen from CLIP
model = fNIRSEmbedding(ds.betas.shape[-1], 512)
model = model.to(device)
num_params =utils.count_parameters(model)
accelerator.log({"num_params": num_params})
print("num params: {}".format(num_params))
optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                       betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

step = 0
model, optimizer, training_dataloader = accelerator.prepare(
    model, optimizer, dl
)
for epoch in range(config["epochs"]):
    for batch in dl:
        optimizer.zero_grad()
        images, betas = batch

        images = images.to(device)
        betas = betas.to(device).float()

        image_features = clip_model.encode_image(images)
        beta_features = model(betas)

        image_features = image_features / \
            image_features.norm(dim=1, keepdim=True)
        beta_features = beta_features / beta_features.norm(dim=1, keepdim=True)

        logit_scale = clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features.float() @ beta_features.t()
        logits_per_beta = logits_per_image.t()  # (B, B)

        ground_truth = torch.arange(
            len(images), dtype=torch.long, device=device)
        loss = (loss_img(logits_per_image, ground_truth) +
                loss_txt(logits_per_beta, ground_truth))/2
        # total_loss.backward()
        accelerator.backward(loss)

        print("epoch: {}, step: {}, loss: {}".format(
            epoch, step, loss.item()))
        accelerator.log({"training_loss": loss, "epoch": epoch}, step=step)
        optimizer.step()
        step += 1
accelerator.end_training()
