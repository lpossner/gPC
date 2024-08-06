import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from model import model


# seed RNG
seed = 1
np.random.seed(seed)

# load coordinates
coords = np.load("data/coords.npy")

# run simulations with coords
label = 1
image = Image.open(f"data/{label}.png")
results = model(coords, image, label)[:, np.newaxis]

# save results
os.makedirs("data", exist_ok=True)
np.save("data/results.npy", results)
