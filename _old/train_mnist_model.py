import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import accuracy_score

# Load and transform the dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Download the training set
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Split the training set into training and evaluation sets
train_size = int(0.8 * len(trainset))
eval_size = len(trainset) - train_size
trainset, evalset = random_split(trainset, [train_size, eval_size])

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
evalloader = DataLoader(evalset, batch_size=32, shuffle=True)

# Download the test set
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=32, shuffle=True)


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


net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Early stopping parameters
early_stop_patience = 3
best_loss = float("inf")
patience_counter = 0


def evaluate_model(loader, model):
    model.eval()
    all_labels = []
    all_predictions = []
    eval_loss = 0.0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    eval_loss /= len(loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    return eval_loss, accuracy


# Train the network with early stopping
for epoch in range(50):  # maximum number of epochs
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}")
            running_loss = 0.0

    # Evaluate the model on the evaluation set
    eval_loss, eval_accuracy = evaluate_model(evalloader, net)
    print(
        f"Epoch {epoch + 1}, Evaluation Loss: {eval_loss:.3f}, Accuracy: {eval_accuracy:.2f}"
    )

    # Early stopping check
    if eval_loss < best_loss:
        best_loss = eval_loss
        patience_counter = 0
        torch.save(net.state_dict(), "./data/model.pth")  # Save the entire model
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

print("Training Finished")

# Load the best model
net = Net()
net.load_state_dict(torch.load("./data/model.pth"))
net.eval()

# Test the network
test_loss, test_accuracy = evaluate_model(testloader, net)
print(f"Accuracy of the network on the 10000 test images: {test_accuracy:.2f}")
