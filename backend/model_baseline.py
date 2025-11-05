import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pickle

print("\n" + "="*70)
print("BASELINE MODEL - LOGISTIC REGRESSION")
print("="*70)

# Load prepared data
X_train = pd.read_csv("data/X_train_scaled.csv")
X_test = pd.read_csv("data/X_test_scaled.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

print(f"\nLoaded training data: {X_train.shape}")
print(f"Loaded testing data: {X_test.shape}")

# ============================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

print("\n" + "="*70)
print("TRAINING LOGISTIC REGRESSION")
print("="*70)

# Create model
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

# Train model
print("\nTraining model...")
model.fit(X_train, y_train)
print("✓ Model trained successfully!")

# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\n" + "="*70)
print("MAKING PREDICTIONS")
print("="*70)

# Predictions on training set
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)[:, 1]  # Probability of winning

# Predictions on testing set
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of winning

print("✓ Predictions made on training and testing data")

# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Training metrics
print("\nTRAINING SET PERFORMANCE:")
print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"  Precision: {precision_score(y_train, y_train_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_train, y_train_pred, zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_train, y_train_pred, zero_division=0):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_train, y_train_pred_proba):.4f}")

# Testing metrics (most important!)
print("\nTESTING SET PERFORMANCE:")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  ROC-AUC:   {test_auc:.4f}")

# Confusion Matrix
print("\nCONFUSION MATRIX (Testing Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(f"  True Negatives:  {cm[0, 0]}")
print(f"  False Positives: {cm[0, 1]}")
print(f"  False Negatives: {cm[1, 0]}")
print(f"  True Positives:  {cm[1, 1]}")

# Detailed classification report
print("\nDETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_test_pred, target_names=['Not Winner', 'Winner']))

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "="*70)
print("FEATURE IMPORTANCE (Coefficients)")
print("="*70)

feature_names = X_train.columns
coefficients = model.coef_[0]

# Sort by importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Importance': np.abs(coefficients)
}).sort_values('Abs_Importance', ascending=False)

print("\nHow much each feature affects winning prediction:")
for idx, row in importance_df.iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {row['Feature']:30s} {direction:10s} win probability (coef: {row['Coefficient']:+.4f})")

# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

with open("data/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✓ Model saved to: logistic_regression_model.pkl")

# Save metrics
metrics = {
    'accuracy': test_accuracy,
    'precision': test_precision,
    'recall': test_recall,
    'f1_score': test_f1,
    'roc_auc': test_auc,
    'feature_importance': importance_df.to_dict()
}

with open("data/baseline_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("✓ Metrics saved to: baseline_metrics.pkl")

print("\n" + "="*70)
print("BASELINE MODEL COMPLETE")
print("="*70)
