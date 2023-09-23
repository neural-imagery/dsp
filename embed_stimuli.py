import clip
import h5py
from PIL import Image
import numpy as np
import torch
device = "cuda"
clip_model, preprocess = clip.load(
    "ViT-B/32", device=device, jit=False, download_root="/lfs/ampere1/0/suppakit")
images = h5py.File('nsd_stimuli.hdf5.1', 'r')
num_images = images["imgBrick"].shape[0]
batch_size = 2048
all_embeddings = []
for i in range(0, num_images, batch_size):
    print(f"{i = }")
    batch = images["imgBrick"][i:i+batch_size]
    print("got batch")
    processed = [preprocess(Image.fromarray(batch[i])) for i in range(batch.shape[0])]
    print("got preprocessed")
    processed = torch.tensor(np.array(processed)).to(device)    
    embeddings = clip_model.encode_image(processed)
    embeddings = embeddings.cpu().detach().numpy()
    all_embeddings.append(embeddings)

all_embeddings = np.vstack(all_embeddings)
print(f"{all_embeddings.shape = }")
np.save("stimuli.npy", all_embeddings)
breakpoint()
