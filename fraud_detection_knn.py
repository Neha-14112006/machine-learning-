"""
KNN Fraud Detection Classifier
================================
This script loads fraud detection data, preprocesses it, trains a KNN model,
evaluates performance, and saves the trained model for future predictions.

Author: ML Pipeline
Date: 2024
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = 'fraud.csv'  # Path to your fraud dataset
MODEL_PATH = 'fraud_knn_model.pkl'  # Where to save the trained model
SCALER_PATH = 'fraud_scaler.pkl'  # Where to save the scaler
K_NEIGHBORS = 5  # Number of neighbors for KNN
TEST_SIZE = 0.2  # 80-20 train-test split
RANDOM_STATE = 42  # For reproducibility

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data(filepath):
    """Load CSV data from file."""
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

# ============================================================================
# STEP 2: EXPLORE & UNDERSTAND DATA
# ============================================================================

def explore_data(df):
    """Explore and display data information."""
    print("\n" + "=" * 70)
    print("STEP 2: DATA EXPLORATION")
    print("=" * 70)
    
    # Display column names and types
    print("\n📊 Column Names & Data Types:")
    print("-" * 70)
    print(df.dtypes)
    
    # Display first few rows
    print("\n📋 First 5 Rows of Data:")
    print("-" * 70)
    print(df.head())
    
    # Display basic statistics
    print("\n📈 Basic Statistics:")
    print("-" * 70)
    print(df.describe())
    
    # Check for missing values
    print("\n⚠️  Missing Values:")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
        print(f"\nTotal missing values: {missing.sum()}")
    else:
        print("✅ No missing values found!")
    
    # Check class distribution
    print("\n🎯 Class Distribution (Fraud vs Legitimate):")
    print("-" * 70)
    fraud_dist = df['is_fraud'].value_counts()
    print(fraud_dist)
    print(f"\nFraud Rate: {(fraud_dist[1] / len(df) * 100):.2f}%")
    print(f"Legitimate Rate: {(fraud_dist[0] / len(df) * 100):.2f}%")
    
    # Display all column headers
    print("\n📝 All Column Headers:")
    print("-" * 70)
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return df

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Preprocess data: handle missing values, separate features and target."""
    print("\n" + "=" * 70)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 70)
    
    df = df.copy()
    
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    print(f"✅ Separated features (X) and target (y)")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Handle missing values - Strategy: Fill with median
    print(f"\n🔧 Handling Missing Values:")
    print(f"   Strategy: Fill with column median")
    
    missing_cols = X.columns[X.isnull().any()].tolist()
    if missing_cols:
        print(f"   Columns with missing values: {missing_cols}")
        for col in missing_cols:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"      - {col}: filled {X[col].isnull().sum()} missing values")
    else:
        print("   ✅ No missing values to handle!")
    
    # Verify no missing values remain
    if X.isnull().sum().sum() == 0:
        print(f"   ✅ All missing values handled successfully!")
    
    print(f"\n✅ Preprocessing Complete:")
    print(f"   Final feature shape: {X.shape}")
    print(f"   Final target shape: {y.shape}")
    
    return X, y

# ============================================================================
# STEP 4: SPLIT DATA
# ============================================================================

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split data into train and test sets."""
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Data split successfully (Stratified):")
    print(f"   Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"   Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    print(f"\n   Train fraud rate: {(y_train.sum() / len(y_train) * 100):.2f}%")
    print(f"   Test fraud rate: {(y_test.sum() / len(y_test) * 100):.2f}%")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# STEP 5: FEATURE SCALING
# ============================================================================

def scale_features(X_train, X_test):
    """Scale features using StandardScaler (important for KNN)."""
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE SCALING")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Features scaled using StandardScaler:")
    print(f"   Training set mean: {X_train_scaled.mean():.4f}")
    print(f"   Training set std: {X_train_scaled.std():.4f}")
    print(f"   Test set mean: {X_test_scaled.mean():.4f}")
    print(f"   Test set std: {X_test_scaled.std():.4f}")
    
    return X_train_scaled, X_test_scaled, scaler

# ============================================================================
# STEP 6: TRAIN KNN MODEL
# ============================================================================

def train_knn_model(X_train, y_train, k=K_NEIGHBORS):
    """Train KNN classifier."""
    print("\n" + "=" * 70)
    print("STEP 6: TRAINING KNN MODEL")
    print("=" * 70)
    
    print(f"🤖 Training KNN with k={k} neighbors...")
    
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='euclidean',
        n_jobs=-1  # Use all CPU cores
    )
    
    knn.fit(X_train, y_train)
    
    print(f"✅ Model trained successfully!")
    print(f"   Algorithm: KNN (K-Nearest Neighbors)")
    print(f"   K value: {k}")
    print(f"   Distance metric: Euclidean")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Features used: {X_train.shape[1]}")
    
    return knn

# ============================================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================================

def make_predictions(model, X_train, X_test, y_train, y_test):
    """Make predictions on train and test sets."""
    print("\n" + "=" * 70)
    print("STEP 7: MAKING PREDICTIONS")
    print("=" * 70)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get probability predictions for AUC
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"✅ Predictions made on both train and test sets")
    
    return y_train_pred, y_test_pred, y_test_pred_proba

# ============================================================================
# STEP 8: EVALUATE MODEL
# ============================================================================

def evaluate_model(y_train, y_train_pred, y_test, y_test_pred, y_test_pred_proba):
    """Evaluate model performance with multiple metrics."""
    print("\n" + "=" * 70)
    print("STEP 8: MODEL EVALUATION")
    print("=" * 70)
    
    # Training metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\n📊 TRAINING SET METRICS:")
    print(f"   Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # Test metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    print(f"\n📊 TEST SET METRICS:")
    print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"   Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"   F1-Score:  {test_f1:.4f}")
    print(f"   ROC-AUC:   {test_auc:.4f}")
    
    # Classification report
    print(f"\n📋 CLASSIFICATION REPORT:")
    print("-" * 70)
    print(classification_report(y_test, y_test_pred, 
                              target_names=['Legitimate', 'Fraud']))
    
    # Confusion matrix
    print(f"\n🎯 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_test_pred)
    print("-" * 70)
    print(f"   True Negatives:  {cm[0,0]}")
    print(f"   False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}")
    print(f"   True Positives:  {cm[1,1]}")
    
    # Additional insights
    print(f"\n💡 MODEL INSIGHTS:")
    print(f"   Fraud Detection Rate (Recall): {test_recall*100:.2f}%")
    print(f"   False Alarm Rate: {(1 - test_precision)*100:.2f}%")
    
    metrics = {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'auc': test_auc
    }
    
    return metrics, cm

# ============================================================================
# STEP 9: VISUALIZE RESULTS
# ============================================================================

def visualize_results(y_test, y_test_pred, y_test_pred_proba, cm):
    """Create visualizations of model performance."""
    print("\n" + "=" * 70)
    print("STEP 9: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('KNN Fraud Detection - Model Performance', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. Class Distribution
    class_counts = [len(y_test) - y_test.sum(), y_test.sum()]
    axes[0, 1].bar(['Legitimate', 'Fraud'], class_counts, color=['green', 'red'], alpha=0.7)
    axes[0, 1].set_title('Test Set Class Distribution')
    axes[0, 1].set_ylabel('Count')
    for i, v in enumerate(class_counts):
        axes[0, 1].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    auc = roc_auc_score(y_test, y_test_pred_proba)
    axes[1, 0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    axes[1, 0].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Prediction Probability Distribution
    axes[1, 1].hist(y_test_pred_proba[y_test == 0], bins=30, alpha=0.7, 
                    label='Legitimate', color='green', edgecolor='black')
    axes[1, 1].hist(y_test_pred_proba[y_test == 1], bins=30, alpha=0.7,
                    label='Fraud', color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Predicted Probability of Fraud')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved as 'fraud_detection_results.png'")
    plt.show()

# ============================================================================
# STEP 10: SAVE MODEL & SCALER
# ============================================================================

def save_model(model, scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Save trained model and scaler to disk."""
    print("\n" + "=" * 70)
    print("STEP 10: SAVING MODEL")
    print("=" * 70)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved to '{model_path}'")
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to '{scaler_path}'")
    
    print(f"\n💾 You can load these files later using:")
    print(f"   model = joblib.load('{model_path}')")
    print(f"   scaler = joblib.load('{scaler_path}')")

# ============================================================================
# STEP 11: PREDICT ON NEW DATA (EXAMPLE)
# ============================================================================

def predict_new_transaction(model, scaler, new_data_dict):
    """Example: Predict fraud on a new transaction."""
    print("\n" + "=" * 70)
    print("STEP 11: EXAMPLE PREDICTION ON NEW DATA")
    print("=" * 70)
    
    # Convert dict to DataFrame
    new_data = pd.DataFrame([new_data_dict])
    
    # Scale features
    new_data_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction = model.predict(new_data_scaled)[0]
    probability = model.predict_proba(new_data_scaled)[0]
    
    print(f"\n🔮 New Transaction Prediction:")
    print(f"   Data: {new_data_dict}")
    print(f"   Prediction: {'🚨 FRAUD' if prediction == 1 else '✅ LEGITIMATE'}")
    print(f"   Confidence (Legitimate): {probability[0]*100:.2f}%")
    print(f"   Confidence (Fraud): {probability[1]*100:.2f}%")
    
    return prediction, probability

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "KNN FRAUD DETECTION CLASSIFIER" + " " * 23 + "║")
    print("║" + " " * 18 + "End-to-End ML Pipeline" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # Step 1: Load data
        df = load_data(DATA_PATH)
        
        # Step 2: Explore data
        explore_data(df)
        
        # Step 3: Preprocess
        X, y = preprocess_data(df)
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Step 5: Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Step 6: Train model
        model = train_knn_model(X_train_scaled, y_train, k=K_NEIGHBORS)
        
        # Step 7: Make predictions
        y_train_pred, y_test_pred, y_test_pred_proba = make_predictions(
            model, X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Step 8: Evaluate
        metrics, cm = evaluate_model(y_train, y_train_pred, y_test, 
                                     y_test_pred, y_test_pred_proba)
        
        # Step 9: Visualize
        visualize_results(y_test, y_test_pred, y_test_pred_proba, cm)
        
        # Step 10: Save model
        save_model(model, scaler)
        
        # Step 11: Example prediction
        example_transaction = {
            'transaction_amount': 150.0,
            'hour_of_day': 2.0,
            'is_weekend': 0.0,
            'num_items': 3.0,
            'customer_age': 35.0,
            'prev_transactions': 5.0,
            'distance_from_home': 30.0,
            'device_type': 1.0,
            'network_quality': 75.0,
            'is_first_transaction': 0.0,
            'store_type': 0.0,
            'velocity_score': 5.0
        }
        predict_new_transaction(model, scaler, example_transaction)
        
        # Final summary
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📁 Output Files Generated:")
        print(f"   1. {MODEL_PATH} - Trained KNN model")
        print(f"   2. {SCALER_PATH} - Feature scaler")
        print(f"   3. fraud_detection_results.png - Performance visualizations")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Please check your data file and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
