import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# --- SETUP ---
output_dir = "Processed_Data_New"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Load Data
print(">>> Loading Data...")
with zipfile.ZipFile('parkinsons.zip', 'r') as z:
    with z.open('parkinsons.data') as f:
        df = pd.read_csv(f)

print(f"Original Shape: {df.shape}")

# 2. Feature Selection (Removing Redundancy)
# We drop the ID column ('name') and the highly correlated "clones"
cols_to_drop = [
    'name',              # Not a feature
    'MDVP:Jitter(Abs)',  # Redundant with Jitter(%)
    'MDVP:RAP',          # Redundant with Jitter(%)
    'MDVP:PPQ',          # Redundant with Jitter(%)
    'Jitter:DDP',        # Redundant with Jitter(%)
    'MDVP:Shimmer(dB)',  # Redundant with Shimmer
    'Shimmer:APQ3',      # Redundant with Shimmer
    'Shimmer:APQ5',      # Redundant with Shimmer
    'MDVP:APQ',          # Redundant with Shimmer
    'Shimmer:DDA'        # Redundant with Shimmer
]

df_clean = df.drop(cols_to_drop, axis=1)
print(f"\n--- FEATURE SELECTION ---")
print(f"Dropped {len(cols_to_drop)} columns.")
print(f"New Shape: {df_clean.shape}")
print(f"Remaining Features: {list(df_clean.columns)}")

# 3. Split Data
X = df_clean.drop('status', axis=1)
y = df_clean['status']

# Stratify is CRITICAL here because of the class imbalance (147 vs 48)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n--- DATA SPLIT ---")
print(f"Training Set: {X_train.shape[0]} patients (Stratified)")
print(f"Test Set:     {X_test.shape[0]} patients (Stratified)")

# 4. Scaling (MinMax)
# We fit on TRAIN and transform TEST to avoid data leakage
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to keep column names (for Explainability)
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# 5. Save Artifacts
print(f"\n--- SAVING ARTIFACTS ---")
X_train_df.to_csv(f'{output_dir}/X_train.csv', index=False)
X_test_df.to_csv(f'{output_dir}/X_test.csv', index=False)
y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

# Save the Scaler (CRITICAL for the Web App)
joblib.dump(scaler, f'{output_dir}/scaler.pkl')

print(f"✅ Success! Processed data and 'scaler.pkl' saved in '{output_dir}'")