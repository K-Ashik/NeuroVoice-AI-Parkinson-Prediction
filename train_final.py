import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import os

# --- SETUP ---
# Create directory for final visuals
output_dir = "Final_Model_Results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Load Processed Data
print(">>> Loading Data from 'Processed_Data_New'...")
X_train = pd.read_csv('Processed_Data_New/X_train.csv')
y_train = pd.read_csv('Processed_Data_New/y_train.csv').values.ravel()
X_test = pd.read_csv('Processed_Data_New/X_test.csv')
y_test = pd.read_csv('Processed_Data_New/y_test.csv').values.ravel()

print(f"Train Shape: {X_train.shape}")
print(f"Test Shape:  {X_test.shape}")

# 2. Train XGBoost Model
# We tune hyperparameters for this specific small dataset to prevent overfitting.
print("\n>>> Training XGBoost Model...")
model = xgb.XGBClassifier(
    n_estimators=100,      # Reasonable number of trees
    learning_rate=0.1,     # Standard learning rate
    max_depth=4,           # Shallow trees to prevent memorizing noise
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model Training Complete.")

# 3. Evaluation
print("\n>>> Evaluating Performance...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate Metrics
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print(f"--------------------------------")
print(f"🚀 TEST ACCURACY:   {acc:.2%}")
print(f"📊 ROC-AUC SCORE:   {roc:.4f}")
print(f"--------------------------------")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 4. Visualization: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, annot_kws={"size": 16})
plt.title(f"Confusion Matrix (Acc: {acc:.1%})", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
plt.close()
print(f"Saved: confusion_matrix.png")

# 5. Explainable AI: SHAP Analysis
print("\n>>> Generating SHAP Explanations...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot 1: Summary (Global Importance)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("What features drive the diagnosis?", fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_dir}/shap_summary.png", dpi=300)
plt.close()
print(f"Saved: shap_summary.png")

# Plot 2: Waterfall (Local Explanation for Patient #1)
# We pick the first patient in the test set
patient_idx = 0
plt.figure(figsize=(8, 6))
# Using the object-oriented API for cleaner waterfall plots
explainer_obj = shap.Explainer(model, X_train)
shap_values_obj = explainer_obj(X_test)
shap.plots.waterfall(shap_values_obj[patient_idx], show=False)
plt.title(f"Explanation for Test Patient #{patient_idx}", fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_dir}/shap_waterfall.png", bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: shap_waterfall.png")

# 6. Save Final Model
joblib.dump(model, 'Processed_Data_New/model.pkl')
print(f"\n✅ FINAL MODEL SAVED: 'Processed_Data_New/model.pkl'")