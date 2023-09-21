import clip
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
import utils

device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 64
epochs = 100
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


class fNIRSDataset(Dataset):
    def __init__(self, path):
        self.df, self.betas = utils.load_ds(path)
        self.betas = self.betas.T
        self.img_paths = self.df["img_paths"]
        assert len(self.betas) == len(self.img_paths)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # image = preprocess(Image.open(self.img_paths[idx]))
        image = preprocess(Image.open(
            "images/shared0001_nsd02951.png"))  # dummy for now
        betas = self.betas[idx]
        return image, betas


ds = fNIRSDataset("dataset")
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()


class fNIRSEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size=64):
        super(fNIRSEmbedding, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# embedding_size stolen from CLIP
model = fNIRSEmbedding(ds.betas.shape[-1], 512)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5,
                       betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

it = 0
for epoch in range(epochs):
    for batch in dl:
        optimizer.zero_grad()
        images, betas = batch

        images = images.to(device)
        betas = betas.to(device).float()

        # breakpoint()
        image_features = clip_model.encode_image(images)
        beta_features = model(betas)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        beta_features = beta_features / beta_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = clip_model.logit_scale.exp()
        # breakpoint()
        logits_per_image = logit_scale * image_features.float() @ beta_features.t()
        logits_per_beta = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]

        ground_truth = torch.arange(
            len(images), dtype=torch.long, device=device)
        total_loss = (loss_img(logits_per_image, ground_truth) +
                      loss_txt(logits_per_beta, ground_truth))/2
        total_loss.backward()
        print("epoch: {}, it: {}, loss: {}".format(epoch, it, total_loss.item()))
        it += 1
        optimizer.step()
