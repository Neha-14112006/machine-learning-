import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("fraud_knn_model.pkl")
scaler = joblib.load("fraud_scaler.pkl")

print("=" * 50)
print("FRAUD DETECTION SYSTEM")
print("=" * 50)

print("\nEnter transaction details:")
print("\nExample:")
print("150 2 0 3 35 5 30 1 75 0 0 5")
print("\nFormat:")
print("amount hour weekend items age prev_txn distance device network first_txn store velocity\n")

# User input
values = input("Enter values: ").split()

# Convert to float
values = [float(x) for x in values]

# Feature names (must match training data)
columns = [
    "transaction_amount",
    "hour_of_day",
    "is_weekend",
    "num_items",
    "customer_age",
    "prev_transactions",
    "distance_from_home",
    "device_type",
    "network_quality",
    "is_first_transaction",
    "store_type",
    "velocity_score"
]

# Create dataframe
new_data = pd.DataFrame([values], columns=columns)

# Scale input
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)[0]
probability = model.predict_proba(new_data_scaled)[0]

print("\n" + "=" * 50)

if prediction == 1:
    print("RESULT : FRAUD TRANSACTION")
else:
    print("RESULT : LEGITIMATE TRANSACTION")

print(f"\nLegitimate Probability : {probability[0]*100:.2f}%")
print(f"Fraud Probability      : {probability[1]*100:.2f}%")

print("=" * 50)