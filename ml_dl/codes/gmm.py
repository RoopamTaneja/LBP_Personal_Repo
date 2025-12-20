import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

plt.style.use("seaborn-v0_8-whitegrid")
np.random.seed(42)


def generate_data(n_samples=500):
    # Parameters for three true components (Ground Truth)
    means = np.array([[0, 0], [5, 5], [0, 5]])
    covs = np.array(
        [
            [[1, 0.2], [0.2, 1]],
            [[1, -0.2], [-0.2, 1]],
            [[0.5, 0], [0, 0.5]],
        ]
    )
    weights = [0.4, 0.3, 0.3]

    # Generate data
    X = []
    labels = []
    for _ in range(n_samples):
        k = np.random.choice(len(weights), p=weights)
        x = np.random.multivariate_normal(means[k], covs[k])
        X.append(x)
        labels.append(k)

    return np.array(X), np.array(labels)


class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_ = np.array([])
        self.covariances_ = np.array([])
        self.weights_ = np.array([])

    def initialize(self, X):
        n_samples, n_features = X.shape
        self.weights_ = np.ones(self.K) / self.K
        indices = np.random.choice(n_samples, self.K, replace=False)
        self.means_ = X[indices]
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.K)])

    def e_step(self, X):
        N = X.shape[0]
        K = self.K
        r_nk = np.zeros((N, K))

        for k in range(K):
            p_k = multivariate_normal.pdf(X, mean=self.means_[k], cov=self.covariances_[k])
            r_nk[:, k] = self.weights_[k] * p_k

        row_sums = r_nk.sum(axis=1)[:, np.newaxis] + 1e-15
        r_nk = r_nk / row_sums
        return r_nk

    def m_step(self, X, r_nk):
        N, D = X.shape
        N_k = r_nk.sum(axis=0)

        # 1. Update Means
        self.means_ = (r_nk.T @ X) / N_k[:, np.newaxis]

        # 2. Update Covariances
        for k in range(self.K):
            diff = X - self.means_[k]
            weighted_diff = r_nk[:, k][:, np.newaxis] * diff
            sigma_k = (weighted_diff.T @ diff) / N_k[k]
            # Add small epsilon to diagonal for stability
            self.covariances_[k] = sigma_k + np.eye(D) * 1e-6

        # 3. Update Weights
        self.weights_ = N_k / N

    def compute_log_likelihood(self, X):
        N = X.shape[0]
        K = self.K
        likelihoods = np.zeros((N, K))

        for k in range(K):
            likelihoods[:, k] = self.weights_[k] * multivariate_normal.pdf(X, self.means_[k], self.covariances_[k])

        total_log_likelihood = np.sum(np.log(np.sum(likelihoods, axis=1) + 1e-15))
        return total_log_likelihood

    def fit(self, X):
        self.initialize(X)
        old_ll = None
        for i in range(self.max_iter):
            r_nk = self.e_step(X)
            self.m_step(X, r_nk)
            ll = self.compute_log_likelihood(X)

            # Check convergence
            if old_ll is not None and abs(ll - old_ll) < self.tol:
                print(f"Manual GMM converged at iteration {i}")
                break

            old_ll = ll

    def predict(self, X):
        # Assign clusters based on highest responsibility
        r_nk = self.e_step(X)
        return np.argmax(r_nk, axis=1)


def plot_compare_gmms(manual_model, sklearn_model, X, y_true, titles=("Manual GMM", "Scikit-Learn GMM")):
    # Create grid for contours
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    pos = np.dstack((xx, yy))

    # Manual model density
    means_m = manual_model.means_
    covs_m = manual_model.covariances_
    weights_m = manual_model.weights_
    z_m = np.zeros(xx.shape)
    for k in range(len(weights_m)):
        z_m += weights_m[k] * multivariate_normal.pdf(pos, means_m[k], covs_m[k])

    # Sklearn model density
    means_s = sklearn_model.means_
    covs_s = sklearn_model.covariances_
    weights_s = sklearn_model.weights_
    z_s = np.zeros(xx.shape)
    for k in range(len(weights_s)):
        z_s += weights_s[k] * multivariate_normal.pdf(pos, means_s[k], covs_s[k])

    # Plot side-by-side
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, z, means, title in zip(axes, [z_m, z_s], [means_m, means_s], titles):
        ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", s=10, alpha=0.5)
        ax.contour(xx, yy, z, levels=10, cmap="inferno", linewidths=1.2)
        ax.scatter(means[:, 0], means[:, 1], c="red", marker="x", s=100, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    plt.tight_layout()
    plt.show()


X, y_true = generate_data()

# Manual GMM
gmm = GMM(n_components=3, max_iter=50)
gmm.fit(X)

# Sklearn GMM for comparison
gmm_sklearn = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm_sklearn.fit(X)

plot_compare_gmms(gmm, gmm_sklearn, X, y_true, titles=("Manual GMM", "Scikit-Learn GMM"))
