import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.75, s=25)
plt.colorbar(scatter, ticks=range(10), label='Digit Label')
plt.title("2D PCA Projection of Handwritten Digits")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

pca_95 = PCA(n_components=0.95, random_state=42)
pca_95.fit(X_scaled)
n_comp = pca_95.n_components_
print(f"Minimum components to retain 95% variance: {n_comp} (down from 64)")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

t0 = time.perf_counter()
knn_orig = KNeighborsClassifier(n_neighbors=5)
knn_orig.fit(X_train, y_train)
preds_orig = knn_orig.predict(X_test)
time_orig = (time.perf_counter() - t0) * 1000
acc_orig = accuracy_score(y_test, preds_orig)

pca_pipe = PCA(n_components=0.95, random_state=42)
X_train_pca = pca_pipe.fit_transform(X_train)
X_test_pca = pca_pipe.transform(X_test)

t0 = time.perf_counter()
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
preds_pca = knn_pca.predict(X_test_pca)
time_pca = (time.perf_counter() - t0) * 1000
acc_pca = accuracy_score(y_test, preds_pca)

print(f"\nOriginal (64 features) -> Accuracy: {acc_orig:.4f} | Train+Predict Time: {time_orig:.2f} ms")
print(f"PCA 95% ({X_train_pca.shape[1]} features)  -> Accuracy: {acc_pca:.4f} | Train+Predict Time: {time_pca:.2f} ms")