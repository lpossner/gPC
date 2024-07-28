import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import moons_model

noise = 0.1

accuracy, precision, recall, model, X_test, y_test = moons_model(noise=noise,
                                                                 runs=1)

# Print metrics
def print_metrics(accuracy, precision, recall):
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')


# Visualize the decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid)
        # _, Z = torch.max(Z, 1)
        Z = F.softmax(Z, dim=1)[:, 0]
        Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
    plt.title('Decision Boundary')
    plt.show()

print_metrics(accuracy, precision, recall)
plot_decision_boundary(model, X_test.numpy(), y_test.numpy())
