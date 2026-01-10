import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Linear layer for x_t (Input -> Hidden)
        self.x2h = nn.Linear(input_size, hidden_size)

        # Linear layer for h_{t-1} (Hidden -> Hidden)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        h_t = torch.tanh(self.x2h(x_t) + self.h2h(h_prev))
        return h_t


class BiRNNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiRNNNetwork, self).__init__()
        self.hidden_size = hidden_size

        # Two separate cells: one for Forward, one for Backward
        self.cell_fwd = CustomRNNCell(input_size, hidden_size)
        self.cell_bwd = CustomRNNCell(input_size, hidden_size)

        # Final classifier: Takes (Hidden_Fwd + Hidden_Bwd) -> Output
        # We concatenate the hidden states, so input dim is 2 * hidden_size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # 1. Initialize hidden states (one for each direction)
        h_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
        h_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)

        fwd_states = []
        bwd_states = []

        # 2. Forward Pass Loop (Time 0 -> T)
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_fwd = self.cell_fwd(x_t, h_fwd)
            fwd_states.append(h_fwd)

        # 3. Backward Pass Loop (Time T -> 0)
        # We iterate backwards using range(seq_len - 1, -1, -1)
        for t in range(seq_len - 1, -1, -1):
            x_t = x[:, t, :]
            h_bwd = self.cell_bwd(x_t, h_bwd)
            # We append to a list, effectively storing [h_T, h_{T-1}, ..., h_0]
            bwd_states.append(h_bwd)

        # 4. Align the Backward states
        # The backward loop stored states in reverse order (T to 0).
        # We reverse the list to match the forward order (0 to T).
        bwd_states.reverse()

        # 5. Concatenate states at each time step
        # out_t = [h_fwd_t; h_bwd_t]
        # We stack the lists into tensors first
        fwd_tensor = torch.stack(fwd_states, dim=1)  # (batch, seq, hidden)
        bwd_tensor = torch.stack(bwd_states, dim=1)  # (batch, seq, hidden)

        # Concatenate along the feature dimension (dim=2)
        combined_features = torch.cat((fwd_tensor, bwd_tensor), dim=2)
        # Shape: (batch, seq, hidden * 2)

        # 6. Classification Strategy
        # For sequence classification, we usually pool the features.
        # Here we use Mean Pooling (average over time dimension).
        pooled_out = torch.mean(combined_features, dim=1)  # (batch, hidden * 2)

        # 7. Final Prediction
        logits = self.fc(pooled_out)
        return logits


def generate_wave_classification_data(seq_len, num_samples):
    X = []
    y = []

    for _ in range(num_samples):
        # Randomly choose class 0 (Sine) or 1 (Triangle)
        label = np.random.randint(0, 2)

        # Create time axis
        start = np.random.rand() * 2 * np.pi
        t = np.linspace(start, start + 4 * np.pi, seq_len)

        if label == 0:
            # Sine Wave
            wave = np.sin(t)
        else:
            # Triangle Wave (approx using arcsin of sin)
            # This creates a sharp, linear wave visually distinct from sine
            wave = (2 / np.pi) * np.arcsin(np.sin(t))

        # Add slight noise to make it harder
        wave += np.random.normal(0, 0.05, seq_len)

        X.append(wave)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.longlong)


# Parameters
SEQ_LEN = 50
HIDDEN_SIZE = 32
INPUT_SIZE = 1
OUTPUT_SIZE = 2  # Two classes: Sine (0) vs Triangle (1)
NUM_SAMPLES = 1000
EPOCHS = 100
LR = 0.005

X_np, y_np = generate_wave_classification_data(SEQ_LEN, NUM_SAMPLES)
X_tensor = torch.tensor(X_np).unsqueeze(-1)  # (Batch, Seq, Input)
y_tensor = torch.tensor(y_np)  # (Batch) - Class labels

model = BiRNNNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Starting BiRNN Training (Sine vs Triangle)...")

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_tensor).float().mean() * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%")

model.eval()
with torch.no_grad():
    # Generate 2 new samples (one of each class)
    t = np.linspace(0, 4 * np.pi, SEQ_LEN)

    # Sample 1: Sine
    sine_wave = np.sin(t)
    sine_input = torch.tensor(sine_wave, dtype=torch.float32).view(1, SEQ_LEN, 1)
    pred_sine = model(sine_input).argmax().item()

    # Sample 2: Triangle
    tri_wave = (2 / np.pi) * np.arcsin(np.sin(t))
    tri_input = torch.tensor(tri_wave, dtype=torch.float32).view(1, SEQ_LEN, 1)
    pred_tri = model(tri_input).argmax().item()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(sine_wave, label="Input (Sine)", color="blue")
    ax1.set_title(f"Truth: Sine (0) | Pred: {pred_sine}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(tri_wave, label="Input (Triangle)", color="orange")
    ax2.set_title(f"Truth: Triangle (1) | Pred: {pred_tri}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"BiRNN Classification Results")
    plt.show()
