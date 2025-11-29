import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
import os

# --- SETUP ---
output_dir = "EDA_New_Charts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
# Green for Healthy (0), Red for Parkinson's (1)
PALETTE = {0: "#2ecc71", 1: "#e74c3c", "0": "#2ecc71", "1": "#e74c3c"}

# 1. LOAD DATA (From ZIP)
print(">>> Loading UCI 'Gold Standard' Data...")
with zipfile.ZipFile('parkinsons.zip', 'r') as z:
    with z.open('parkinsons.data') as f:
        df = pd.read_csv(f)

print(f"Data Loaded. Shape: {df.shape}")

# ==========================================
# LEVEL 1: INTEGRITY CHECK (The "Physics" Test)
# ==========================================
print("\n--- LEVEL 1: PHYSICS CHECK ---")
# In the previous dataset, this failed. Let's see if this one passes.
physical_cols = ['MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR', 'PPE']
neg_counts = (df[physical_cols] < 0).sum()

if neg_counts.sum() == 0:
    print("✅ PASSED: No impossible negative values found.")
else:
    print("❌ FAILED: Negative values found.")
    print(neg_counts)

# ==========================================
# LEVEL 2: CLASS BALANCE
# ==========================================
print("\n--- LEVEL 2: CLASS BALANCE ---")
plt.figure(figsize=(6, 5))
ax = sns.countplot(x='status', data=df, hue='status', legend=False, palette=PALETTE)
plt.title("Class Balance (UCI Dataset)", fontsize=14, fontweight='bold')
plt.xlabel("Status (0=Healthy, 1=Parkinson's)")
plt.ylabel("Count")

# Add counts on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.savefig(f"{output_dir}/01_class_balance.png", dpi=300)
plt.close()
print("Saved: 01_class_balance.png")

# ==========================================
# LEVEL 3: CORRELATION (Redundancy Check)
# ==========================================
print("\n--- LEVEL 3: CORRELATION MATRIX ---")
plt.figure(figsize=(12, 10))
# Drop 'name' as it is a string
numeric_df = df.drop(['name', 'status'], axis=1)
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Feature Correlation (Redundancy Check)', fontsize=14)
plt.savefig(f"{output_dir}/02_correlation.png", dpi=300)
plt.close()
print("Saved: 02_correlation.png")

# ==========================================
# LEVEL 4: FEATURE SEPARATION (Violin Plots)
# ==========================================
print("\n--- LEVEL 4: FEATURE SEPARATION ---")
# We look for features where the Green and Red shapes are totally different.
# PPE and Spread1 are famously good in this dataset.
features = ['MDVP:Fo(Hz)', 'PPE', 'spread1', 'HNR']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.violinplot(x='status', y=col, data=df, ax=axes[i], hue='status', legend=False, palette=PALETTE, split=False)
    axes[i].set_title(f"Separation Power: {col}", fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/03_feature_separation.png", dpi=300)
plt.close()
print("Saved: 03_feature_separation.png")

# ==========================================
# LEVEL 5: t-SNE CLUSTERING (The Truth Map)
# ==========================================
print("\n--- LEVEL 5: t-SNE PROJECTION ---")
# 1. Prepare Data
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# 2. Scale (Crucial for t-SNE)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Run t-SNE
# Perplexity=30 is standard. Since N=195 is small, we keep it standard.
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
tsne_results = tsne.fit_transform(X_scaled)

# 4. Plot
tsne_df = pd.DataFrame(data=tsne_results, columns=['tsne_1', 'tsne_2'])
tsne_df['status'] = y

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='tsne_1', y='tsne_2',
    hue='status',
    palette=PALETTE,
    data=tsne_df,
    legend="full",
    alpha=0.8,
    s=120,  # Bigger dots for smaller dataset
    edgecolor='black'
)

plt.title("t-SNE Projection: Can we see the Disease?", fontsize=16, fontweight='bold')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(title='Status', loc='upper right', labels=['Healthy', 'Parkinson\'s'])

plt.savefig(f"{output_dir}/04_tsne_clusters.png", dpi=300)
plt.close()
print("Saved: 04_tsne_clusters.png")

print(f"\n>>> EDA COMPLETE. Charts saved in '{output_dir}'.")