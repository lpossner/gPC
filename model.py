import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the model
net = Net()
net.load_state_dict(torch.load("./data/model.pth", weights_only=True))
net.eval()

img_transform = lambda img, angle, brightness: transforms.functional.rotate(
    transforms.functional.adjust_brightness(img, brightness), angle
)
# scale_proba = lambda proba: proba * 20 - 10
scale_proba = lambda proba: proba


def model(coords, img, label):
    probas = []
    with torch.no_grad():
        for angle, brightness in coords:
            proba = scale_proba(
                F.softmax(net(img_transform(img, angle, brightness)), dim=1)
            )[0, label].item()
            probas.append(proba)
    return np.array(probas)


if __name__ == "__main__":

    N = 100
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    img, label = testset[0]

    angles = np.random.uniform(-90, 90, size=N)
    brightness = np.random.uniform(1, 3, size=N)
    coords = np.stack([angles, brightness], axis=1)

    results = model(coords, img, label)
    print(results)
