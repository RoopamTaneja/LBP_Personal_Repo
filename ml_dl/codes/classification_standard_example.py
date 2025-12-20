import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

data = load_breast_cancer()
X = data.data
y = 1 - data.target
# INVERSION LOGIC: Sklearn default is 0=Malignant, 1=Benign.
# We want 1=Cancer (Positive) to measure Recall correctly.

models = {
    "Logistic Reg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42))]),
    "KNN (k=5)": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(max_depth=5, random_state=42))]),
    "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))]),
    "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))]),
    "Gradient Boosting": Pipeline([("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))]),
    "XGBoost": Pipeline([("clf", XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric="logloss", random_state=42))]),
}

scoring = {"accuracy": "accuracy", "recall": "recall"}
results_data = []
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, pipe in models.items():
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    results_data.append({"Model": name, "Avg Accuracy": scores["test_accuracy"].mean(), "Cancer Recall": scores["test_recall"].mean()})

df_results = pd.DataFrame(results_data).sort_values(by="Cancer Recall", ascending=False)
best_model_name = df_results.iloc[0]["Model"]
best_recall_score = df_results.iloc[0]["Cancer Recall"]
print(f"\nBest Model for Safety (Recall): {best_model_name} (Recall: {best_recall_score:.2%})")

plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
df_melted = df_results.melt(id_vars=["Model"], value_vars=["Avg Accuracy", "Cancer Recall"], var_name="Metric", value_name="Score")
sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette=["#bdc3c7", "#e74c3c"])
plt.title("Model Performance: Accuracy vs. Cancer Detection (Recall)")
plt.ylim(0.85, 1.01)
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)

# Confusion Matrix for Best Model
plt.subplot(1, 2, 2)
# Create a fresh split to visualize the confusion matrix
# The previous step used Cross-Validation (which produces scores, not a fitted model). To plot a specific Confusion Matrix, we need to manually fit the best model one last time on a standard Train/Test split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

best_pipe = models[best_model_name]
best_pipe.fit(X_train, y_train)

# Using from_estimator automatically handles predictions
disp = ConfusionMatrixDisplay.from_estimator(best_pipe, X_test, y_test, display_labels=["Benign (0)", "Cancer (1)"], cmap="Reds", colorbar=False, ax=plt.gca())
plt.title(f"Confusion Matrix: {best_model_name}\n(Test Set Evaluation)")
plt.grid(False)
plt.tight_layout()
plt.show()

print("\n--- FINAL EVALUATION REPORT ---")
print(df_results.to_string(index=False))
print("-" * 60)
print(f"Interpretation for {best_model_name}:")
print("Top-Left:  True Negatives (Benign correctly identified as Benign)")
print("Bottom-Right: True Positives (Cancer correctly identified as Cancer)")
print("Bottom-Left:  False Negatives (CRITICAL ERROR -> Cancer missed)")
