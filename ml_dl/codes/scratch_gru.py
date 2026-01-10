import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # We define linear layers for x (input) and h (hidden) separately.
        # This makes the "Reset Gate" logic clearer than combining them into one matrix immediately.

        # x_t -> Gates (Update z, Reset r, Candidate n)
        self.x2h = nn.Linear(input_size, 3 * hidden_size)

        # h_{t-1} -> Gates (Update z, Reset r, Candidate n)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x_t, h_prev):
        """
        x_t: (batch_size, input_size)
        h_prev: (batch_size, hidden_size)
        """

        # 1. Linear projections
        # We compute the linear transformations for inputs and hidden states
        # chunk(3, dim=1) splits the tensor into 3 parts: for Update(z), Reset(r), and Candidate(n)
        x_z, x_r, x_n = self.x2h(x_t).chunk(3, dim=1)
        h_z, h_r, h_n = self.h2h(h_prev).chunk(3, dim=1)

        # 2. Compute Reset Gate (r_t) and Update Gate (z_t)
        # r_t = sigmoid(W_xr*x + W_hr*h)
        # z_t = sigmoid(W_xz*x + W_hz*h)
        r_t = torch.sigmoid(x_r + h_r)
        z_t = torch.sigmoid(x_z + h_z)

        # 3. Compute New Candidate Hidden State (n_t)
        # This is where GRU differs significantly. The reset gate acts on h_prev
        # BEFORE the linear transformation for the candidate.
        # n_t = tanh(W_xn*x + r_t * (W_hn*h))
        # Note: In standard texts, the reset gate is applied to h_prev.
        # Depending on implementation (PyTorch vs papers), biases might differ slightly.
        # Here we apply r_t to the hidden component 'h_n'.
        n_t = torch.tanh(x_n + (r_t * h_n))

        # 4. Update Hidden State (h_t)
        # h_t = (1 - z_t) * n_t + z_t * h_prev
        # Note: PyTorch's built-in GRU typically uses: (1 - z_t) * n_t + z_t * h_prev
        # Some papers swap the (1-z) and z terms. We stick to the standard PyTorch convention here.
        h_t = (1 - z_t) * n_t + z_t * h_prev

        return h_t


class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = CustomGRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # Manual loop over sequence steps
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.gru_cell(x_t, h_t)

        # Final prediction
        out = self.fc(h_t)
        return out


def generate_sine_data(seq_len, num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.rand()
        t = np.linspace(start, start + 2 * np.pi, seq_len + 1)
        sine_wave = np.sin(t)
        X.append(sine_wave[:-1])
        y.append(sine_wave[-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# Parameters
SEQ_LEN = 50
HIDDEN_SIZE = 32
INPUT_SIZE = 1
OUTPUT_SIZE = 1
NUM_SAMPLES = 1000
EPOCHS = 20
LR = 0.01

X, y = generate_sine_data(SEQ_LEN, NUM_SAMPLES)
X_tensor = torch.tensor(X).unsqueeze(-1)  # (1000, 50, 1)
y_tensor = torch.tensor(y).unsqueeze(-1)  # (1000, 1)

model = GRUNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Starting GRU Training...")
losses = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    output = model(X_tensor)
    loss = criterion(output, y_tensor)

    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    # Test on a single new sample
    test_X, test_y = generate_sine_data(SEQ_LEN, 1)
    test_tensor = torch.tensor(test_X).unsqueeze(-1)
    prediction = model(test_tensor).item()

    plt.figure(figsize=(10, 5))
    plt.plot(range(SEQ_LEN), test_X[0], label="Input Sequence", color="blue")
    plt.plot(SEQ_LEN, test_y[0], "go", label="True Future Value", markersize=10)
    plt.plot(SEQ_LEN, prediction, "rx", label="GRU Prediction", markersize=10)
    plt.legend()
    plt.title(f"Custom GRU Sine Wave Prediction\nFinal Loss: {losses[-1]:.5f}")
    plt.grid(True, alpha=0.3)
    plt.show()
