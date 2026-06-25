import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("fraud.csv")

print("Shape:", df.shape)

# -----------------------------
# DATA PREPROCESSING
# -----------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values (numeric columns)
df = df.fillna(df.median(numeric_only=True))

# Remove duplicate rows
df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)

# Separate features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Convert categorical columns if any
X = pd.get_dummies(X, drop_first=True)

print("\nFeature Shape:", X.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nPreprocessing Completed Successfully!")