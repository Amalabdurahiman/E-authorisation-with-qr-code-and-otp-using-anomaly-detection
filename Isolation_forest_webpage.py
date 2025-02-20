import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load the dataset
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

# Select features and labels
features = df.drop(columns=['label'])
labels = df['label']

# Convert categorical variables to dummy variables
features = pd.get_dummies(features)

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA for dimensionality reduction (optional, but can improve model performance)
pca = PCA(n_components=10)  # Reduce to 10 principal components
features_pca = pca.fit_transform(features_scaled)

# Split the data into training and test sets ensuring both classes are represented
X_train, X_test, y_train, y_test = train_test_split(
    features_pca, labels, test_size=0.3, stratify=labels, random_state=42)

# Tune the contamination parameter for Isolation Forest
contamination_value = 0.05  # Adjust this based on the actual proportion of anomalies in your data

# Train the Isolation Forest model
iso_forest = IsolationForest(contamination=contamination_value, random_state=42)
iso_forest.fit(X_train)

# Predict on the test set
y_pred = iso_forest.predict(X_test)

# Convert the predictions to binary (anomalies = 1, normal = 0)
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Calculate evaluation metrics
try:
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"ROC-AUC: {roc_auc:.2f}")
except ValueError as e:
    print(f"ROC-AUC could not be calculated: {e}")

# Calculate precision, recall, and F1-Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")