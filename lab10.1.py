import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target

print(f"Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {len(wine.target_names)}")
print("Feature Names:", list(wine.feature_names))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest (n_estimators=100)": RandomForestClassifier(n_estimators=100, random_state=42)
}

rows = []
for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_sc))
    rows.append({"Model": name, "Test Accuracy": f"{acc:.4f}"})

print("\n", pd.DataFrame(rows).to_string(index=False))

k_vals = range(1, 21)
k_accs = []

for k in k_vals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_sc, y_train)
    k_accs.append(accuracy_score(y_test, knn.predict(X_test_sc)))

best_k = k_vals[np.argmax(k_accs)]
print(f"\nBest k: {best_k} (Accuracy: {max(k_accs):.4f})")

plt.figure(figsize=(8, 4.5))
plt.plot(k_vals, k_accs, marker='o', color='navy')
plt.xticks(k_vals)
plt.title("KNN Accuracy vs. Number of Neighbors (k)")
plt.xlabel("k")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()