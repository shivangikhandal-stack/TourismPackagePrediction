
import pandas as pd
import os



# Dataset location
DATA_PATH = "tourism_project/data/tourism.csv"

# Check whether dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Expected columns
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact",
    "CityTier", "DurationOfPitch", "Occupation", "Gender",
    "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome"
]

# Validate columns
missing_columns = [
    col for col in EXPECTED_COLUMNS if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Print registration summary
print("Dataset registration successful!")
print("----------------------------------")
print("Dataset shape:", df.shape)
print("Number of records:", df.shape[0])
print("Number of columns:", df.shape[1])
print("Target variable: ProdTaken")

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["ProdTaken"].value_counts())

print("\nDataset preview:")
print(df.head())

print("\nDataset summary:")
print(df.describe(include="all").T)
