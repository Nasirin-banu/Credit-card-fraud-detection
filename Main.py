import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
train_data = pd.read_csv(r"C:\Users\Welcome\Downloads\Book1.csv")

# Display information about the dataset
print("Dataset Columns:", train_data.columns)
train_data.info()

# Check if required columns exist
required_columns = ['trans_date_trans_time', 'dob', 'is_fraud', 'merchant', 'category', 'gender', 'job']
missing_columns = [col for col in required_columns if col not in train_data.columns]
if missing_columns:
    print(f"Missing Columns: {missing_columns}")
    exit()

# Convert date columns to datetime
train_data["trans_date_trans_time"] = pd.to_datetime(train_data["trans_date_trans_time"])
train_data["dob"] = pd.to_datetime(train_data["dob"])

# Drop unnecessary columns
columns_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'trans_date_trans_time']
train_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Drop rows with missing values
train_data.dropna(ignore_index=True, inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
for col in ['merchant', 'category', 'gender', 'job']:
    train_data[col] = encoder.fit_transform(train_data[col])

# Visualize class distribution
exit_counts = train_data["is_fraud"].value_counts()
plt.figure(figsize=(12, 6))
plt.pie(exit_counts, labels=["No", "Yes"], autopct="%0.0f%%")
plt.title("is_fraud Counts")
plt.tight_layout()
plt.show()

# Split data into features and target
X = train_data.drop(columns=["is_fraud"], inplace=False)
Y = train_data["is_fraud"]

# Standardize feature values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train SVC model
model = SVC()
model.fit(X_train, Y_train)

# Evaluate the model
train_accuracy = model.score(X_train, Y_train)
test_accuracy = model.score(X_test, Y_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Hyperparameter tuning (Optional)
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, Y_train)
print("Best Hyperparameters:", grid_search.best_params_)
