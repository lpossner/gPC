import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score


def moons_model(noise, runs=1, verbose=False, random_state=None):
    # Lists for results
    accuracy_lst = []
    precision_lst = []
    recall_lst = []
    for _ in range(runs):
        # Generate moon dataset
        X, y = make_moons(n_samples=1000, noise=noise, random_state=random_state)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )

        # Standardize the dataset
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Define the neural network model
        class MoonClassifier(nn.Module):
            def __init__(self):
                super(MoonClassifier, self).__init__()
                self.fc1 = nn.Linear(2, 16)
                self.fc2 = nn.Linear(16, 16)
                self.fc3 = nn.Linear(16, 2)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = MoonClassifier()

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        num_epochs = 50

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if verbose:
                print(
                    f"""
                    Epoch {epoch+1}/{num_epochs}, 
                    Loss: {running_loss/len(train_loader):.4f}
                    """
                )

        # Evaluate the model
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Calculate metrics
        accuracy_lst.append(accuracy_score(y_true, y_pred))
        precision_lst.append(precision_score(y_true, y_pred))
        recall_lst.append(recall_score(y_true, y_pred))

    return (
        np.mean(accuracy_lst),
        np.mean(precision_lst),
        np.mean(recall_lst),
        model,
        X_test,
        y_test,
    )
