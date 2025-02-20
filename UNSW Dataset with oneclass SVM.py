import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load the UNSW-NB15 dataset
df_train = pd.read_csv('UNSW_NB15_training-set.csv')
df_test = pd.read_csv('UNSW_NB15_testing-set.csv')

# Preprocess the data
# Convert categorical variables to dummy variables
df_train = pd.get_dummies(df_train, columns=['proto', 'service', 'state', 'attack_cat'])
df_test = pd.get_dummies(df_test, columns=['proto', 'service', 'state', 'attack_cat'])

# Ensure that both training and testing datasets have the same columns
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

# Separate features and labels
X_train = df_train.drop(columns=['label', 'id'])  # Dropping 'id' and 'label' columns
y_train = df_train['label']

X_test = df_test.drop(columns=['label', 'id'])
y_test = df_test['label']

# Map the labels to binary values (normal = 0, anomaly = 1)
y_train = y_train.apply(lambda x: 0 if x == 0 else 1)
y_test = y_test.apply(lambda x: 0 if x == 0 else 1)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the One-Class SVM model
one_class_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
one_class_svm.fit(X_train)

# Predict using One-Class SVM on the test set
y_pred_test = one_class_svm.predict(X_test)
y_pred_test = [1 if x == -1 else 0 for x in y_pred_test]

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

# Print the results
print("One-Class SVM Results on UNSW-NB15 Test Set:")
print(f"ROC-AUC: {roc_auc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")