"""Model değerlendirme ve karşılaştırma."""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures")


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Karışıklık matrisini görselleştir."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legal (0)", "Illegal (1)"],
        yticklabels=["Legal (0)", "Illegal (1)"],
        ax=ax,
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(f"{model_name} - Karışıklık Matrisi")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png"), dpi=150)
    plt.close()


def compare_models(knn_results, mlp_results, knn_report, mlp_report, best_knn, best_mlp, X_test, y_test):
    """KNN ve MLP modellerini karşılaştır."""
    # KNN en iyi sonuçlar
    knn_best = max(knn_results, key=lambda x: x["f1"])
    # MLP en iyi sonuçlar
    mlp_best = max(mlp_results, key=lambda x: x["f1"])

    # Test seti üzerinde son tahminler
    knn_pred = best_knn.predict(X_test)
    mlp_pred = best_mlp.predict(X_test)

    print("\n" + "=" * 60)
    print("         MODEL KARŞILAŞTIRMASI")
    print("=" * 60)

    metrics = ["Precision", "Recall", "F1"]
    data = {
        "Metric": metrics,
        "KNN (K={})".format(knn_best["k"]): [knn_best["precision"], knn_best["recall"], knn_best["f1"]],
        "YSA {}".format(mlp_best["config"]): [mlp_best["precision"], mlp_best["recall"], mlp_best["f1"]],
    }

    # Tablo formatında yazdır
    print(f"{'Metric':<12} | {'KNN':>10} | {'YSA':>10}")
    print("-" * 40)
    for m in metrics:
        idx = metrics.index(m)
        print(f"{m:<12} | {data['KNN (K={})'.format(knn_best['k'])][idx]:>10.4f} | {data['YSA {}'.format(mlp_best['config'])][idx]:>10.4f}")

    print("\n--- KNN Sınıflandırma Raporu ---")
    print(knn_report)
    print("--- YSA Sınıflandırma Raporu ---")
    print(mlp_report)

    # Karışıklık matrisleri
    plot_confusion_matrix(y_test, knn_pred, "KNN")
    plot_confusion_matrix(y_test, mlp_pred, "YSA")
    print(f"Karışıklık matrisleri kaydedildi: {FIGURES_DIR}/")

    # Karşılaştırma grafiği
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.3
    ax.bar(
        x - width / 2,
        [knn_best["precision"], knn_best["recall"], knn_best["f1"]],
        width, label=f"KNN (K={knn_best['k']})", color="#3498db",
    )
    ax.bar(
        x + width / 2,
        [mlp_best["precision"], mlp_best["recall"], mlp_best["f1"]],
        width, label=f"YSA {mlp_best['config']}", color="#e74c3c",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Skor")
    ax.set_title("KNN vs YSA - Model Karşılaştırması\n(Dengesiz veri setinde Accuracy yeterli değil!)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("-> model_comparison.png kaydedildi")
