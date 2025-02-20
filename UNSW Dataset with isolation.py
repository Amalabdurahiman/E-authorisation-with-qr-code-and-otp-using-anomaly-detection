import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load the UNSW-NB15 dataset (you may need to adjust the file path)
df_train = pd.read_csv('UNSW_NB15_training-set.csv')
df_test = pd.read_csv('UNSW_NB15_testing-set.csv')

# Display basic info about the dataset
print("Training set:")
print(df_train.info())
print("\nTesting set:")
print(df_test.info())

# Preprocess the data
# Convert categorical variables to dummy variables
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

# Ensure that both training and testing datasets have the same columns
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

# Separate features and labels
X_train = df_train.drop(columns=['label'])  # Assuming 'label' is the target column
y_train = df_train['label']

X_test = df_test.drop(columns=['label'])
y_test = df_test['label']

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Isolation Forest model
iso_forest = IsolationForest(contamination='auto', random_state=42)
iso_forest.fit(X_train)

# Predict using Isolation Forest on the test set
y_pred_test = iso_forest.predict(X_test)
y_pred_test = [1 if x == -1 else 0 for x in y_pred_test]

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

# Print the results
print("Isolation Forest Results on UNSW-NB15 Test Set:")
print(f"ROC-AUC: {roc_auc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")