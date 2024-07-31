import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from model import model


# seed RNG
seed = 1
np.random.seed(seed)

# load coordinates
coords = np.load("data/coords.npy")

# run simulations with coords
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
img, label = testset[0]
results = model(coords, img, label)[:, np.newaxis]

# save results
os.makedirs("data", exist_ok=True)
np.save("data/results.npy", results)
