import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_excel('data1319.xlsx')

# 2. Select numerical features for PCA (excluding meta-data like year/model)
features = ['novelty', 'brand', 'style', 'rugged', 'space', 'trunk', 'comfort', 
            'nimble', 'versatile', 'finish', 'features', 'infotain', 'perform', 
            'economy', 'safety', 'price']
X = df[features]

# 3. Standardize the data (Mean=0, Variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA
pca = PCA()
pca_results = pca.fit_transform(X_scaled)

# 5. Scree Plot: Identify relevant PCs
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, label='Individual')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='Cumulative')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title('Scree Plot')
plt.legend()
plt.savefig('scree_plot.png')

# 6. Loadings: Importance of each feature on PCs
loadings = pd.DataFrame(pca.components_.T, 
                        columns=[f'PC{i+1}' for i in range(len(features))], 
                        index=features)

# Visualize Loadings for the first 5 components
plt.figure(figsize=(12, 8))
sns.heatmap(loadings.iloc[:, :5], annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Loadings (PC1 to PC5)')
plt.savefig('pca_loadings.png')

# Export loadings to CSV for further review
loadings.to_csv('pca_loadings_results.csv')