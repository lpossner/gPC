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
class MyDataset(Dataset):

    def __init__(self, path: str, transform=None, train: bool=True):
        self.transform = transform
        self.path = path
        self.mode = 'train' if train else 'test'
        
        with open (f"{path}/{self.mode}/{self.mode}.json", "r") as f:
            self.dataset_info = json.load(f)
        
        self.data = [(k, v) for k, v in self.dataset_info.items()]

    def __len__(self):
        return len(self.dataset_info)
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(f"{self.path}/{self.mode}/{image_path}")
        if self.transform:
            image = self.transform(image)
        return image, label
    
train_dataset = MyDataset(
    path="./data/al5083",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    train=True
)

img, label = train_dataset[0]

results = model(coords, img, label)[:, np.newaxis]

# save results
os.makedirs("data", exist_ok=True)
np.save("data/results.npy", results)
