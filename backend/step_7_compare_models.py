import pandas as pd
import matplotlib.pyplot as plt
import pickle

print("\n" + "="*70)
print("STEP 7: MODEL COMPARISON VISUALIZATION")
print("="*70)

# Load all metrics
with open("data/all_models_metrics.pkl", "rb") as f:
    all_metrics = pickle.load(f)

# Add baseline metrics
with open("data/baseline_metrics.pkl", "rb") as f:
    baseline_metrics = pickle.load(f)

all_metrics['Logistic Regression'] = baseline_metrics

# Create comparison dataframe
comparison_df = pd.DataFrame(all_metrics).T

print("\nModel Performance Comparison:")
print(comparison_df.to_string())

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('F1 Winner Prediction - Model Comparison', fontsize=16, fontweight='bold')

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

for (i, j), metric in zip(positions, metrics_to_plot):
    ax = axes[i, j]
    comparison_df[metric].plot(kind='bar', ax=ax, color=['#E10600', '#0082FA', '#FFA800'])
    ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# Remove the 6th subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig('src/model_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved model comparison chart: model_comparison.png")
plt.close()

# Create a summary report
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

best_model = comparison_df['roc_auc'].idxmax()
print(f"\nBest Overall Model: {best_model}")
print(f"  ROC-AUC Score: {comparison_df.loc[best_model, 'roc_auc']:.4f}")
print(f"  Accuracy: {comparison_df.loc[best_model, 'accuracy']:.4f}")
print(f"  Recall: {comparison_df.loc[best_model, 'recall']:.4f}")
