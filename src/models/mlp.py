"""Yapay Sinir Ağları (MLP) modeli."""
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


def train_mlp(X_train, y_train, X_test, y_test, hidden_layer_configs=None):
    """Farklı gizli katman yapılandırmalarını dene, en iyisini seç."""
    if hidden_layer_configs is None:
        # 2-3 gizli katmanlı yapılandırmalar
        hidden_layer_configs = [
            (64, 32),        # 2 katman
            (128, 64, 32),   # 3 katman
            (64, 64, 32),    # 3 katman
        ]

    results = []
    best_config = None
    best_f1 = -1
    best_model = None

    print("\n=== YSA (MLP) Model Eğitimi ===")
    for config in hidden_layer_configs:
        mlp = MLPClassifier(
            hidden_layer_sizes=config,
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        results.append({"config": config, "precision": precision, "recall": recall, "f1": f1})
        print(f"Katmanlar={config} → Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_config = config
            best_model = mlp

    print(f"\nEn iyi yapılandırma: {best_config} (F1={best_f1:.4f})")
    return best_model, best_config, results


def get_best_mlp_report(best_model, X_test, y_test):
    """En iyi MLP modeli için detaylı sınıflandırma raporu."""
    y_pred = best_model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=["Legal (0)", "Illegal (1)"])
