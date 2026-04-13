"""K-Nearest Neighbors modeli - K değerini optimize ederek."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


def train_knn(X_train, y_train, X_test, y_test, k_values=None):
    """Farklı K değerlerini dene, en iyisini seç."""
    if k_values is None:
        k_values = [3, 5, 7, 9, 11]

    results = []
    best_k = None
    best_f1 = -1
    best_model = None

    print("\n=== KNN Model Eğitimi ===")
    for k in k_values:
        # Ağırlık: uzak düğümlerin etkisini azalt
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        results.append({"k": k, "precision": precision, "recall": recall, "f1": f1})
        print(f"K={k:2d} → Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_model = knn

    print(f"\nEn iyi K: {best_k} (F1={best_f1:.4f})")
    return best_model, best_k, results


def get_best_knn_report(best_model, X_test, y_test):
    """En iyi KNN modeli için detaylı sınıflandırma raporu."""
    y_pred = best_model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=["Legal (0)", "Illegal (1)"])
