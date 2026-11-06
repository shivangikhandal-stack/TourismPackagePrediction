
import pandas as pd
import os

from sklearn.model_selection import train_test_split

# Define file paths
DATA_PATH = "tourism_project/data/tourism.csv"
OUTPUT_DIR = "tourism_project/data"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove unnecessary identifier columns
columns_to_drop = ["Unnamed: 0", "CustomerID"]

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)

# Define target column
TARGET_COLUMN = "ProdTaken"

# Validate target column
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column {TARGET_COLUMN} not found")

# Separate features and target
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Combine features and target
train_df = X_train.copy()
train_df[TARGET_COLUMN] = y_train

test_df = X_test.copy()
test_df[TARGET_COLUMN] = y_test

# Save training and testing datasets
train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)

print("Data preparation completed successfully!")
print("-----------------------------------------")
print("Original dataset shape:", df.shape)
print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)
print("Training and testing files saved successfully!")
