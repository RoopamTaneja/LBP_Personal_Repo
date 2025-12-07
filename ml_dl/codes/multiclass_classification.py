import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# DATA PREPARATION
X_np, y_np = make_blobs(n_samples=1000, centers=4, n_features=2, cluster_std=2.0, random_state=SEED, return_centers=False)
X_tensor = torch.from_numpy(X_np).float()
y_tensor = torch.from_numpy(y_np).long()
dataset = TensorDataset(X_tensor, y_tensor)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# MODEL DEFINITION
input_dim = 2
output_dim = 4
model = nn.Sequential(nn.Linear(input_dim, output_dim)).to(device)

# LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# TRAINING LOOP
epochs = 100
epoch_losses = []

model.train()

for epoch in range(epochs):
    batch_loss_accum = 0.0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss_accum += loss.item()

    avg_loss = batch_loss_accum / len(dataloader)
    epoch_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


# VISUALIZATION
def plot_decision_boundary(model, X, y):
    model.eval()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)

    # Predict
    with torch.no_grad():
        logits = model(grid_tensor)
        predicted = torch.argmax(logits, dim=1)
        predicted = predicted.cpu().reshape(xx.shape).numpy()

    plt.figure(figsize=(12, 5))

    # Subplot 1: Decision Boundary
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, predicted, alpha=0.3, cmap="viridis")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="viridis", s=20)
    plt.title("Softmax Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Subplot 2: Training Loss
    plt.subplot(1, 2, 2)
    plt.plot(epoch_losses)
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


plot_decision_boundary(model, X_np, y_np)
