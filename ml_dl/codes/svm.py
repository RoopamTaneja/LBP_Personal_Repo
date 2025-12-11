import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset generation
X, y = datasets.make_moons(n_samples=300, noise=0.20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
svm_models = {
    "Linear SVM": SVC(kernel="linear", C=1.0),
    "RBF SVM (Gaussian)": SVC(kernel="rbf", gamma=0.7, C=1.0),
    "Polynomial SVM (Degree 3)": SVC(kernel="poly", degree=3, coef0=1, C=5.0),
}


def plot_decision_boundary(clf, X, y, ax, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict class for each point in the mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot contours and data points
    ax.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30)

    ax.set_title(title)
    ax.set_xticks(())
    ax.set_yticks(())


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, svm_models.items()):
    # Train
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Visualize
    plot_title = f"{name}\nAccuracy: {acc:.2f}"
    plot_decision_boundary(model, X_train_scaled, y_train, ax, plot_title)
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=100, linewidth=1, facecolors="none", edgecolors="k", alpha=0.5, label="Support Vectors")

plt.tight_layout()
plt.show()
