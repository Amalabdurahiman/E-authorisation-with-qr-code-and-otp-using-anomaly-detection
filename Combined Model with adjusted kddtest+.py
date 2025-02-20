import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Load the KDDTest+ dataset
df = pd.read_csv('nsl-kdd/KDDTest+.txt', header=None)

# Define the column names
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
           "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
           "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
           "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
           "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
           "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

df.columns = columns + ['target']

# Map the labels to binary values (normal = 0, anomaly = 1)
df['target'] = df['target'].apply(lambda x: 0 if x == 'normal' else 1)

# Check the distribution of the classes in the dataset
class_distribution = df['target'].value_counts()
print("Class distribution in the dataset:")
print(class_distribution)

# Separate normal and anomalous instances
normal_instances = df[df['target'] == 0]
anomalous_instances = df[df['target'] == 1]

# Check if both subsets have data
if len(normal_instances) == 0 or len(anomalous_instances) == 0:
    raise ValueError("One of the classes has no instances, cannot create a balanced dataset.")

# Determine the number of samples for balance (min of both counts)
sample_size = min(len(normal_instances), len(anomalous_instances))

# Sample from both subsets to get equal numbers
normal_sample = normal_instances.sample(sample_size, random_state=42)
anomaly_sample = anomalous_instances.sample(sample_size, random_state=42)

# Combine the samples to create a balanced dataset
balanced_df = pd.concat([normal_sample, anomaly_sample])

# Shuffle the dataset to mix normal and anomalous instances
balanced_df = shuffle(balanced_df, random_state=42)

# Reset the index
balanced_df.reset_index(drop=True, inplace=True)

# Prepare the features and labels from the balanced dataset
features = balanced_df.drop(columns=['target'])
labels = balanced_df['target']

# Convert categorical variables to dummy variables
features = pd.get_dummies(features)

# Ensure the dataset is not empty
if features.shape[0] == 0:
    raise ValueError("The balanced dataset is empty after processing.")

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train and test using the balanced dataset
X = features
y = labels

# Train the Isolation Forest model
iso_forest = IsolationForest(contamination=0.1, random_state=42)
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