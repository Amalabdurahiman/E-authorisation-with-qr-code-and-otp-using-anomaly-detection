import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000
n_features = 10

# Generate normal data (e.g., Gaussian distribution)
normal_data = np.random.normal(0, 1, size=(n_samples, n_features))

# Generate anomalous data with overlapping distribution (e.g., Gaussian distribution close to normal)
n_anomalies = 100
anomalous_data = np.random.normal(2, 1.5, size=(n_anomalies, n_features))

# Introduce noise features (e.g., Gaussian noise)
noise_features = np.random.normal(0, 10, size=(n_samples + n_anomalies, 2))  # Adding 2 noise features

# Combine normal and anomalous data
X = np.vstack([normal_data, anomalous_data])

# Add noise features to the dataset
X = np.hstack([X, noise_features])

# Labels for the data
y = np.hstack([np.zeros(n_samples), np.ones(n_anomalies)])  # 0 = normal, 1 = anomaly

# Shuffle the dataset to mix normal and anomalous instances
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the Isolation Forest model
iso_forest = IsolationForest(contamination=n_anomalies / (n_samples + n_anomalies), random_state=42)
iso_forest.fit(X)

# Predict using Isolation Forest
y_pred_iso = iso_forest.predict(X)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]

# Train the One-Class SVM model
one_class_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
one_class_svm.fit(X)

# Predict using One-Class SVM
y_pred_svm = one_class_svm.predict(X)
y_pred_svm = [1 if x == -1 else 0 for x in y_pred_svm]

# Combine the predictions by averaging
y_pred_combined = np.mean([y_pred_iso, y_pred_svm], axis=0)
y_pred_combined = np.where(y_pred_combined >= 0.5, 1, 0)

# Function to safely calculate ROC-AUC
def safe_roc_auc_score(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return "Undefined (only one class present)"
    else:
        return f"{roc_auc_score(y_true, y_pred):.2f}"

# Calculate evaluation metrics for Isolation Forest
roc_auc_iso = safe_roc_auc_score(y, y_pred_iso)
precision_iso = precision_score(y, y_pred_iso)
recall_iso = recall_score(y, y_pred_iso)
f1_iso = f1_score(y, y_pred_iso)

# Calculate evaluation metrics for One-Class SVM
roc_auc_svm = safe_roc_auc_score(y, y_pred_svm)
precision_svm = precision_score(y, y_pred_svm)
recall_svm = recall_score(y, y_pred_svm)
f1_svm = f1_score(y, y_pred_svm)

# Calculate evaluation metrics for the combined model
roc_auc_combined = safe_roc_auc_score(y, y_pred_combined)
precision_combined = precision_score(y, y_pred_combined)
recall_combined = recall_score(y, y_pred_combined)
f1_combined = f1_score(y, y_pred_combined)

# Print the results
print("Isolation Forest Results:")
print(f"ROC-AUC: {roc_auc_iso}")
print(f"Precision: {precision_iso:.2f}")
print(f"Recall: {recall_iso:.2f}")
print(f"F1-Score: {f1_iso:.2f}")

print("\nOne-Class SVM Results:")
print(f"ROC-AUC: {roc_auc_svm}")
print(f"Precision: {precision_svm:.2f}")
print(f"Recall: {recall_svm:.2f}")
print(f"F1-Score: {f1_svm:.2f}")

print("\nCombined Model Results:")
print(f"ROC-AUC: {roc_auc_combined}")
print(f"Precision: {precision_combined:.2f}")
print(f"Recall: {recall_combined:.2f}")
print(f"F1-Score: {f1_combined:.2f}")