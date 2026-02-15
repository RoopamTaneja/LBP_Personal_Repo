import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
plt.figure(figsize=(15, 5))

print("--- PART 1: STANDARD LINEAR REGRESSION ---")

# 1. Data Generation (y = x^T*theta + epsilon)
N = 100
true_theta_1 = 2.5  # Slope
true_theta_0 = 1.0  # Intercept
sigma = 0.5  # Noise

x = torch.randn(N, 1)
epsilon = torch.randn(N, 1) * sigma
y = true_theta_1 * x + true_theta_0 + epsilon

# 2. Analytical Solution via Normal Equations
# We create the Design Matrix X by adding a column of 1s
X_design = torch.cat([torch.ones(N, 1), x], dim=1)

# theta_ml = (X^T * X)^-1 * X^T * y
XT_X = X_design.T @ X_design
XT_y = X_design.T @ y
theta_ml = torch.linalg.inv(XT_X) @ XT_y

print(f"Analytical Theta: Intercept={theta_ml[0].item():.4f}, Slope={theta_ml[1].item():.4f}")

# 3. Numerical Solution via Gradient Descent
model = nn.Linear(1, 1)  # Holds weights (theta) and bias
criterion = nn.MSELoss()  # Equivalent to neg log-likelihood for Gaussian noise
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

losses = []
epochs = 50

for epoch in range(epochs):
    # Forward
    y_pred = model(x)
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

gd_slope = model.weight.item()
gd_intercept = model.bias.item()
print(f"Gradient Descent Theta: Intercept={gd_intercept:.4f}, Slope={gd_slope:.4f}")

# --- VISUALIZATION 1: Linear Fits ---
plt.subplot(1, 3, 1)
plt.scatter(x, y, color="gray", alpha=0.5, label="Noisy Data")
# Generate line for plotting
x_line = torch.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_analytical = theta_ml[1] * x_line + theta_ml[0]
y_gd = gd_slope * x_line + gd_intercept

plt.plot(x_line, y_analytical, "r--", label="Analytical (Normal Eq)")
plt.plot(x_line, y_gd, "b:", linewidth=3, label="Gradient Descent")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()

# --- VISUALIZATION 2: Convergence ---
plt.subplot(1, 3, 2)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Convergence")
plt.grid(True)

# ==========================================
# PART 2: POLYNOMIAL REGRESSION (Features)
# ==========================================

print("\n--- PART 2: POLYNOMIAL REGRESSION ---")

# 1. Generate Non-Linear Data
x_poly = torch.linspace(-5, 5, 30).reshape(-1, 1)
y_poly = -torch.sin(x_poly / 2) + 0.1 * torch.randn(x_poly.shape)


# 2. Feature Map function phi(x)
def poly_features(x_tensor, degree):
    # [x, x^2, x^3, ...]
    features = [x_tensor]
    for d in range(2, degree + 1):
        features.append(x_tensor**d)
    return torch.cat(features, dim=1)


# 3. Fit Model (Degree 4)
degree = 4
phi_x = poly_features(x_poly, degree)

# Note: Input dim is 'degree', output is 1. Bias handled by nn.Linear
poly_model = nn.Linear(degree, 1)
poly_opt = torch.optim.Adam(poly_model.parameters(), lr=0.05)

for epoch in range(1000):
    y_pred_poly = poly_model(phi_x)
    loss_poly = criterion(y_pred_poly, y_poly)

    poly_opt.zero_grad()
    loss_poly.backward()
    poly_opt.step()

print(f"Polynomial Fit Completed (Degree {degree})")

# --- VISUALIZATION 3: Polynomial Fit ---
plt.subplot(1, 3, 3)
plt.scatter(x_poly, y_poly, color="gray", label="Non-linear Data")

# Smooth curve for plotting
x_test = torch.linspace(-6, 6, 100).reshape(-1, 1)
phi_test = poly_features(x_test, degree)
with torch.no_grad():
    y_test = poly_model(phi_test)

plt.plot(x_test, y_test, color="green", label=f"Poly Fit (Deg {degree})")
plt.xlabel("x")
plt.title("Feature Spaces)")
plt.legend()

plt.tight_layout()
plt.show()
