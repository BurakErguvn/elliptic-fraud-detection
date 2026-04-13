"""Ana pipeline: Ön işleme → EDA → Model eğitimi → Değerlendirme."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess, load_data, filter_known_labels
from src.eda import run_eda
from src.models.knn import train_knn, get_best_knn_report
from src.models.mlp import train_mlp, get_best_mlp_report
from src.evaluation import compare_models


def main():
    print("=" * 60)
    print("  ELLIPTIC FRAUD DETECTION - RISK SCORING ENGINE")
    print("=" * 60)

    # Ham veriyi yükle (EDA için)
    df_raw = load_data()
    df_raw = filter_known_labels(df_raw)

    # 1. EDA (ham veri üzerinde)
    print("\n[1/4] Keşifçi veri analizi...")
    run_eda(df_raw)

    # 2. Ön işleme
    print("\n[2/4] Veri ön işleme...")
    df, scaler, pca = preprocess(n_components=25)

    # Özellikler ve etiketler
    pca_cols = [f"pca_{i}" for i in range(25)]
    X = df[pca_cols].values
    y = df["class"].values

    # 3. Eğitim/Test ayrımı (%80/%20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nEğitim seti: {len(X_train)} satır")
    print(f"Test seti:   {len(X_test)} satır")

    # 4. Model eğitimi
    print("\n[3/4] Modeller eğitiliyor...")
    best_knn, best_k, knn_results = train_knn(X_train, y_train, X_test, y_test)
    best_mlp, best_config, mlp_results = train_mlp(X_train, y_train, X_test, y_test)

    # Detaylı raporlar
    knn_report = get_best_knn_report(best_knn, X_test, y_test)
    mlp_report = get_best_mlp_report(best_mlp, X_test, y_test)

    # 5. Karşılaştırma
    print("\n[4/4] Model karşılaştırması...")
    compare_models(knn_results, mlp_results, knn_report, mlp_report, best_knn, best_mlp, X_test, y_test)

    print("\n" + "=" * 60)
    print("  TÜM İŞLEMLER TAMAMLANDI")
    print("=" * 60)
    print(f"\nÇıktılar:")
    print(f"  - Grafikler: reports/figures/")
    print(f"  - PCA bileşen sayısı: 25")
    print(f"  - PCA açıklanan varyans: {pca.explained_variance_ratio_.sum():.1%}")


if __name__ == "__main__":
    main()
