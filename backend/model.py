import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

print("\n" + "="*70)
print("ADVANCED MODEL - RANDOM FOREST ONLY")
print("="*70)

# Load prepared data
X_train = pd.read_csv("backend/data/X_train_scaled.csv")
X_test = pd.read_csv("backend/data/X_test_scaled.csv")
y_train = pd.read_csv("backend/data/y_train.csv").squeeze()
y_test = pd.read_csv("backend/data/y_test.csv").squeeze()

print(f"\nLoaded data: X_train {X_train.shape}, X_test {X_test.shape}")

print("\n" + "="*70)
print("TRAINING RANDOM FOREST")
print("="*70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✓ Random Forest trained!")

y_test_pred_rf = rf_model.predict(X_test)
y_test_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

rf_metrics = {
    'accuracy': accuracy_score(y_test, y_test_pred_rf),
    'precision': precision_score(y_test, y_test_pred_rf, zero_division=0),
    'recall': recall_score(y_test, y_test_pred_rf, zero_division=0),
    'f1_score': f1_score(y_test, y_test_pred_rf, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_test_pred_proba_rf)
}

print("\nTesting Set Performance:")
print(f"  Accuracy:  {rf_metrics['accuracy']:.4f}")
print(f"  Precision: {rf_metrics['precision']:.4f}")
print(f"  Recall:    {rf_metrics['recall']:.4f}")
print(f"  F1-Score:  {rf_metrics['f1_score']:.4f}")
print(f"  ROC-AUC:   {rf_metrics['roc_auc']:.4f}")

feature_importance_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 5 Most Important Features:")
for idx, row in feature_importance_rf.head(5).iterrows():
    print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

with open("backend/data/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("✓ Random Forest saved")

with open("backend/data/all_models_metrics.pkl", "wb") as f:
    pickle.dump({'Random Forest': rf_metrics}, f)
print("✓ Metrics saved")

print("\n" + "="*70)
print("ADVANCED MODEL COMPLETE")
print("="*70)
