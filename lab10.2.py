import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

bc = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(bc.data, bc.target, test_size=0.20, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

def confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1(prec, rec):
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

tp, tn, fp, fn = confusion(y_test, y_pred)
acc = (tp + tn) / len(y_test)
prec = precision(tp, fp)
rec = recall(tp, fn)
f1_val = f1(prec, rec)

print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
print(f"Custom Metrics -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1_val:.4f}")

print("\nScikit-Learn Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nScikit-Learn Classification Report:\n", classification_report(y_test, y_pred, digits=4))