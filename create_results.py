import os
import numpy as np
from tqdm import tqdm
from model import moons_model


# seed RNG
seed = 1
np.random.seed(seed)

# load coordinates
coords = np.load("data/coords.npy")

# run simulations with coords
accuracy_lst = []
precision_lst = []
recall_lst = []
for noise in tqdm(coords):
    accuracy, precision, recall, _, _, _ = moons_model(noise=noise.item(),
                                                       runs=30)
    accuracy_lst.append(accuracy)
    precision_lst.append(precision)
    recall_lst.append(recall)
results = np.stack([accuracy_lst, precision_lst, recall_lst])

# save results
os.makedirs("data", exist_ok=True)
np.save("data/results.npy", results)
