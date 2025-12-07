import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

NUM_CLASSES = 4
X, y = make_blobs(n_samples=1000, centers=NUM_CLASSES, n_features=2, cluster_std=1.5, random_state=42)

X = torch.from_numpy(X).float()
# Note: For CrossEntropyLoss, y must be Long (int64) and 1D (no unsqueeze)
y = torch.from_numpy(y).long()


class MultiClassNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            # Output features = Number of classes (4)
            nn.Linear(in_features=10, out_features=NUM_CLASSES),
            # nn.CrossEntropyLoss handles the softmax internally.
        )

    def forward(self, x):
        return self.layer_stack(x)


model = MultiClassNN()

# CrossEntropyLoss requires raw logits as input and class indices as target
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    y_logits = model(X)
    loss = criterion(y_logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def plot_multiclass_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    with torch.no_grad():
        logits = model(grid_tensor)
        y_pred = torch.argmax(logits, dim=1)

    Z = y_pred.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap="Spectral", alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Spectral", edgecolors="k")
    plt.title("Multi-class Decision Boundary (PyTorch)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_multiclass_boundary(model, X, y)
