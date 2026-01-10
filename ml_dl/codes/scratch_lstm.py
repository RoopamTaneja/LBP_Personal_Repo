import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # We combine all weights for efficiency (W_ii|W_if|W_ig|W_io)
        # Input size is (input_size + hidden_size) because we concat x_t and h_{t-1}
        combined_dim = input_size + hidden_size

        # Weights for Forget, Input, Candidate(Gate), Output
        self.W_f = nn.Linear(combined_dim, hidden_size)
        self.W_i = nn.Linear(combined_dim, hidden_size)
        self.W_g = nn.Linear(combined_dim, hidden_size)  # candidate
        self.W_o = nn.Linear(combined_dim, hidden_size)

    def forward(self, x_t, hidden_states):
        """
        x_t: Input at current time step (batch_size, input_size)
        hidden_states: Tuple (h_{t-1}, c_{t-1})
        """
        h_prev, c_prev = hidden_states

        # 1. Concatenate input and previous hidden state
        combined = torch.cat((x_t, h_prev), dim=1)

        # 2. Compute Gates
        f_t = torch.sigmoid(self.W_f(combined))  # Forget Gate
        i_t = torch.sigmoid(self.W_i(combined))  # Input Gate
        g_t = torch.tanh(self.W_g(combined))  # Candidate Gate
        o_t = torch.sigmoid(self.W_o(combined))  # Output Gate

        # 3. Update Cell State
        # c_t = (forget_old) + (add_new)
        c_t = f_t * c_prev + i_t * g_t

        # 4. Update Hidden State
        h_t = o_t * torch.tanh(c_t)

        return h_t, (h_t, c_t)


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = CustomLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)  # Map hidden state to output

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # Initialize hidden and cell states with zeros
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # Loop through sequence manually (This is usually what nn.LSTM does internally)
        for t in range(seq_len):
            x_t = x[:, t, :]  # Select input at time t
            h_t, (h_t, c_t) = self.lstm_cell(x_t, (h_t, c_t))

        # We only care about the final prediction for this example
        out = self.fc(h_t)
        return out


def generate_sine_data(seq_len, num_samples):
    # Create sine waves with random phases
    X = []
    y = []
    for _ in range(num_samples):
        # Random start point
        start = np.random.rand()
        # Generate sequence
        t = np.linspace(start, start + 2 * np.pi, seq_len + 1)
        sine_wave = np.sin(t)

        # X = [sin(t1), sin(t2), ..., sin(tn)]
        # y = sin(tn+1) (The next value)
        X.append(sine_wave[:-1])
        y.append(sine_wave[-1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# Parameters
SEQ_LEN = 50
HIDDEN_SIZE = 32
INPUT_SIZE = 1  # We feed 1 number at a time
OUTPUT_SIZE = 1  # We predict 1 number
NUM_SAMPLES = 1000

X, y = generate_sine_data(SEQ_LEN, NUM_SAMPLES)
X_tensor = torch.tensor(X).unsqueeze(-1)  # Shape: (1000, 50, 1)
y_tensor = torch.tensor(y).unsqueeze(-1)  # Shape: (1000, 1)

model = LSTMNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Starting Training...")
epochs = 20
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    output = model(X_tensor)
    loss = criterion(output, y_tensor)

    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    # Generate a new sample to test
    test_X, test_y = generate_sine_data(SEQ_LEN, 1)
    test_tensor = torch.tensor(test_X).unsqueeze(-1)
    prediction = model(test_tensor).item()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(SEQ_LEN), test_X[0], label="Input Sequence")
    plt.plot(SEQ_LEN, test_y[0], "go", label="True Future Value")
    plt.plot(SEQ_LEN, prediction, "rx", label="Predicted Value")
    plt.legend()
    plt.title(f"LSTM Sine Wave Prediction\nLoss: {losses[-1]:.5f}")
    plt.grid(True)
    plt.show()
