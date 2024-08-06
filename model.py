import json
from tqdm import tqdm
import timm
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as Ft
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader


# Load the pre-trained ResNet50 model
clf_model = timm.create_model(
    "resnet18.a2_in1k", pretrained=True, in_chans=1, num_classes=6
)

# Load the weights
checkpoint = torch.load("data/resnet50_finetuned_epoch_1.pth", weights_only=True)
clf_model.load_state_dict(checkpoint["model_state_dict"])

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf_model.to(device)

# Move the model to another backend if available
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
device = torch.device(device)
clf_model.to(device)


class CustomDataset(Dataset):

    def __init__(self, path: str, transform=None, train: bool = True):
        self.transform = transform
        self.path = path
        self.train = train
        self.mode = "train" if train else "test"

        with open(f"{path}/{self.mode}/{self.mode}.json", "r") as f:
            self.dataset_info = json.load(f)

        self.data = [(k, v) for k, v in self.dataset_info.items()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(f"{self.path}/{self.mode}/{image_path}")
        if self.transform:
            image = self.transform(image)
        return image, label


transform = lambda images, angle, brightness: Ft.rotate(
    Ft.adjust_brightness(
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(0.5, 0.5),
            ]
        )(images).unsqueeze(dim=1),
        brightness,
    ),
    angle,
)

test_dataset = CustomDataset(
    path="/Users/lpossner/Projects/welding_defect/data/al5083",
    transform=transform,
    train=False,
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define model function
def model(coords, image, label):
    results_lst = []
    clf_model.eval()
    for angle, brightness in coords:
        with torch.no_grad():
            images = transform(image, angle, brightness)
            images = images.to(device)
            logits = clf_model(images)
            probas = F.softmax(logits, dim=1)
            results_lst.append((probas.squeeze()[label]).item())
            # results_lst.append((logits.squeeze()[label]).item())
    return np.array(results_lst)


if __name__ == "__main__":
    # coords = np.array([[0, 1]], dtype=float)
    # image_lst = [transform(Image.open(f"data/{label}.png"), 0, 1) for label in range(6)]
    # images = torch.concatenate(image_lst)
    # images = images.to(device)
    # logits = clf_model(images)
    # labels_pred = torch.argmax(logits, dim=1)
    # probas = F.softmax(logits, dim=1)
    # print(labels_pred)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    N = 100
    rotation = np.linspace(-90, 90, N)  # 100 points from -5 to 5
    brightness = np.linspace(0, 2, N)

    rotation, brightness = np.meshgrid(rotation, brightness)
    coords = np.stack([rotation.reshape(-1), brightness.reshape(-1)], axis=1)
    label = 0
    image = Image.open(f"data/{label}.png")
    
    results = model(coords, image, label)
    results = results.reshape(N, N)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(rotation, brightness, results, cmap='viridis', edgecolor='none')

    plt.show()
