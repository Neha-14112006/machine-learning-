# fraud_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================
# STEP 1: LOAD DATASET
# =====================================
df = pd.read_csv("fraud.csv")

print("=" * 50)
print("DATASET LOADED")
print("=" * 50)

print("Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# =====================================
# STEP 2: CHECK MISSING VALUES
# =====================================
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values with median
df = df.fillna(df.median(numeric_only=True))

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# =====================================
# STEP 3: REMOVE DUPLICATES
# =====================================
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# =====================================
# STEP 4: SEPARATE FEATURES & TARGET
# =====================================
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# =====================================
# STEP 5: ENCODE CATEGORICAL DATA
# =====================================
X = pd.get_dummies(X, drop_first=True)

print("\nShape after Encoding:", X.shape)

# =====================================
# STEP 6: TRAIN-TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# =====================================
# STEP 7: FEATURE SCALING
# =====================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed")

print("X_train Shape:", X_train_scaled.shape)
print("X_test Shape:", X_test_scaled.shape)

# =====================================
# STEP 8: SAVE PREPROCESSED DATA
# =====================================
pd.DataFrame(X_train_scaled).to_csv("X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("X_test_scaled.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nPreprocessed files saved:")
print("1. X_train_scaled.csv")
print("2. X_test_scaled.csv")
print("3. y_train.csv")
print("4. y_test.csv")

print("\nPREPROCESSING COMPLETED SUCCESSFULLY")