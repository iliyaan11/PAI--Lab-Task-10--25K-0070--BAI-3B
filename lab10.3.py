import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(
    n_samples=200,
    centers=[[25, 20], [25, 80], [55, 50], [85, 20], [85, 80]],
    cluster_std=6,
    random_state=3
)

inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(k_range, inertias, marker='o', color='teal')
plt.xticks(k_range)
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.show()

best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 5))
for c in range(best_k):
    pts = X[labels == c]
    plt.scatter(pts[:, 0], pts[:, 1], s=45, label=f"Cluster {c}")

plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centroids')
plt.title("Customer Segments (K-Means, k=5)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def segment_label(inc, spend):
    if inc > 70 and spend > 65:
        return "High Income, High Spending (VIP Target)"
    elif inc > 70 and spend < 35:
        return "High Income, Low Spending (Careful / Savers)"
    elif inc < 40 and spend > 65:
        return "Low Income, High Spending (Impulsive Buyers)"
    elif inc < 40 and spend < 35:
        return "Low Income, Low Spending (Budget Conscious)"
    return "Moderate Income, Moderate Spending (Standard)"

counts = np.bincount(labels)
for idx, (cx, cy) in enumerate(centers):
    persona = segment_label(cx, cy)
    print(f"Cluster {idx} | Customers: {counts[idx]} | Centroid: ({cx:.2f}k, {cy:.2f}) -> {persona}")