import pandas as pd
import joblib

# =========================
# LOAD MODEL AND SCALER
# =========================
model = joblib.load("fraud_knn_model.pkl")
scaler = joblib.load("fraud_scaler.pkl")

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

print("=" * 60)
print("      FRAUD DETECTION SYSTEM")
print("=" * 60)

print("\nExample Input:")
print("150 2 0 3 35 5 30 1 75 0 0 5")

values = input("\nEnter 12 values: ").split()

if len(values) != 12:
    print("ERROR: Enter exactly 12 values.")
    exit()

try:
    values = [float(x) for x in values]
except:
    print("ERROR: Numeric values only.")
    exit()

# Create dataframe
new_data = pd.DataFrame([values], columns=columns)

# Scale data
new_data_scaled = scaler.transform(new_data)

# Prediction
prediction = model.predict(new_data_scaled)[0]

# Probabilities
probability = model.predict_proba(new_data_scaled)[0]

legit_prob = probability[0] * 100
fraud_prob = probability[1] * 100

# Neighbor distances
distances, indices = model.kneighbors(new_data_scaled)

avg_distance = distances.mean()

# Distance confidence
distance_confidence = (1 / (1 + avg_distance)) * 100

# Combined confidence
confidence = (
    0.6 * max(legit_prob, fraud_prob)
    + 0.4 * distance_confidence
)

if confidence > 100:
    confidence = 100

print("\n" + "=" * 60)

if prediction == 1:
    print("RESULT           : FRAUD TRANSACTION")
else:
    print("RESULT           : LEGITIMATE TRANSACTION")

print("-" * 60)

print(f"Legitimate Chance: {legit_prob:.2f}%")
print(f"Fraud Chance     : {fraud_prob:.2f}%")
print(f"Confidence Score : {confidence:.2f}%")

if confidence >= 90:
    level = "VERY HIGH"
elif confidence >= 75:
    level = "HIGH"
elif confidence >= 60:
    level = "MEDIUM"
else:
    level = "LOW"

print(f"Confidence Level : {level}")

print("=" * 60)