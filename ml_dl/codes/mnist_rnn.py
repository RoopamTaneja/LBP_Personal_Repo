# RNN to read MNIST images as sequence of rows (see row of 28 pixels as one time step of 28 features)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters & Config
BATCH_SIZE = 64
INPUT_SIZE = 28  # Number of pixels in one row
SEQ_LENGTH = 28  # Number of rows (time steps)
HIDDEN_SIZE = 128  # Number of features in hidden state
NUM_LAYERS = 2  # Number of stacked LSTM layers
NUM_CLASSES = 10  # 0-9 digits
LEARNING_RATE = 0.001
EPOCHS = 5

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class MNIST_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MNIST_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        # batch_first=True expects input shape: (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully Connected Layer for the final prediction
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape from loader: (batch, 1, 28, 28)
        # We need to remove the channel dim (1) to get (batch, 28, 28)
        # Interpreted as: (batch, sequence_length, input_size)
        x = x.squeeze(1)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # out shape: (batch, seq_length, hidden_size) containing output features from the last layer of LSTM for each t
        # _ (hidden states) are ignored here as we just need the output
        out, _ = self.lstm(x, (h0, c0))

        # out has hidden state output for all time steps, but we only want the last time step

        # We take out[:, -1, :] -> All batches, Last time step, All hidden features
        out = out[:, -1, :]

        # Pass through Linear layer
        out = self.fc(out)
        return out


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = MNIST_RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = train_loss / num_batches
    accuracy = correct / size
    print(f"Training Loss: {avg_loss:>8f} | Accuracy: {(100*accuracy):>0.1f}%")


def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Loss:     {test_loss:>8f} | Accuracy: {(100*correct):>0.1f}% \n")


print(f"Training on {len(train_data)} samples")

for t in range(EPOCHS):
    print(f"Epoch {t+1}")
    train(train_loader, model, criterion, optimizer)

print(f"Testing the model on the {len(test_data)} test samples")
test(test_loader, model, criterion)
