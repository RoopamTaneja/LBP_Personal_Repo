import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float().unsqueeze(1)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layer_stack(x)


model = NeuralNetwork()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    with torch.no_grad():
        Z = model(grid_tensor)
    Z = Z.reshape(xx.shape)
    Z = Z > 0.5  # Threshold predictions

    plt.figure(figsize=(10, 6))

    plt.contourf(xx, yy, Z, cmap="Spectral", alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap="Spectral", edgecolors="k")
    plt.title("Neural Network Decision Boundary (PyTorch)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_decision_boundary(model, X, y)
