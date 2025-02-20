import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Load the KDD Cup 1999 dataset
url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
           "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
           "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
           "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
           "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
           "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

df = pd.read_csv(url, names=columns)

# Map the labels to binary values (normal = 0, anomaly = 1)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

# Select a subset of features and label for simplicity
features = df.drop(columns=['label'])
labels = df['label']

# Convert categorical variables to dummy variables
features = pd.get_dummies(features)

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Define the Stratified K-Fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store metrics for each fold
roc_auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Cross-validation loop
for train_index, test_index in kf.split(features, labels):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Train the Isolation Forest model
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_train)

    # Predict on the test set using Isolation Forest
    y_pred_iso = iso_forest.predict(X_test)
    y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]

    # Train the One-Class SVM model
    one_class_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
    one_class_svm.fit(X_train)

    # Predict on the test set using One-Class SVM
    y_pred_svm = one_class_svm.predict(X_test)
    y_pred_svm = [1 if x == -1 else 0 for x in y_pred_svm]

    # Combine the predictions by averaging
    y_pred_combined = np.mean([y_pred_iso, y_pred_svm], axis=0)
    y_pred_combined = np.where(y_pred_combined >= 0.5, 1, 0)

    # Calculate and store evaluation metrics for the fold
    roc_auc_scores.append(roc_auc_score(y_test, y_pred_combined))
    precision_scores.append(precision_score(y_test, y_pred_combined))
    recall_scores.append(recall_score(y_test, y_pred_combined))
    f1_scores.append(f1_score(y_test, y_pred_combined))

# Print the average results over all folds
print("Combined Model Results with Cross-Validation (Isolation Forest + One-Class SVM):")
print(f"Average ROC-AUC: {np.mean(roc_auc_scores):.2f}")
print(f"Average Precision: {np.mean(precision_scores):.2f}")
print(f"Average Recall: {np.mean(recall_scores):.2f}")
print(f"Average F1-Score: {np.mean(f1_scores):.2f}")
