import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

# Proje kökünü sys.path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_data, filter_known_labels, scale_features

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_class_imbalance(df):
    """Sınıf dengesizliğini gösteren çubuk grafik."""
    class_counts = df["class"].value_counts().sort_index()
    labels = ["Legal (0)", "Illegal (1)"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, class_counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="black")

    for bar, count in zip(bars, class_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 300,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    total = len(df)
    ax.set_title("Sınıf Dengesizliği (Class Imbalance)", fontsize=14)
    ax.set_ylabel("İşlem Sayısı")
    ax.set_xlabel("")
    ax.text(
        0.5, -0.15,
        f"Sınıflar arası dengesizlik vardır: Legal %{class_counts[0]/total:.1f}, Illegal %{class_counts[1]/total:.1f}",
        transform=ax.transAxes, ha="center", fontsize=10, style="italic",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_imbalance.png"), dpi=150)
    plt.close()
    print("-> class_imbalance.png kaydedildi")


def plot_time_step_analysis(df):
    """Zaman adımları boyunca illegal işlem dağılımı."""
    time_illicit = df[df["class"] == 1].groupby("time_step").size()
    time_total = df.groupby("time_step").size()
    time_ratio = (time_illicit / time_total * 100).reindex(range(1, 50), fill_value=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

    # Illegal işlem sayısı
    ax1.plot(time_illicit.index, time_illicit.values, marker="o", color="#e74c3c", linewidth=1.5)
    ax1.set_ylabel("Illegal İşlem Sayısı")
    ax1.set_title("Zaman Adımlarına Göre Illegal İşlem Dağılımı")
    ax1.grid(True, alpha=0.3)

    # Illegal oranı (%)
    ax2.plot(time_ratio.index, time_ratio.values, marker="s", color="#f39c12", linewidth=1.5)
    ax2.set_ylabel("Illegal Oran (%)")
    ax2.set_xlabel("Zaman Adımı (Time Step)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "time_step_analysis.png"), dpi=150)
    plt.close()
    print("-> time_step_analysis.png kaydedildi")


def plot_2d_scatter(df, feature_cols, method="pca"):
    """PCA veya t-SNE ile 2D'de sınıfları görselleştir."""
    _, scaler = scale_features(df.copy(), feature_cols)
    scaled = scaler.transform(df[feature_cols])

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(scaled)
        title = f"PCA 2D Gösterimi (Varyans: {reducer.explained_variance_ratio_.sum():.1%})"
        labels_df = df["class"].values
    else:
        # t-SNE: örneklem üzerinden (tam veri yavaş olabilir)
        sample_size = min(5000, len(df))
        indices = np.random.RandomState(42).choice(len(df), sample_size, replace=False)
        scaled_sample = scaled[indices]
        labels_df = df["class"].values[indices]
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(scaled_sample)
        title = f"t-SNE 2D Gösterimi (Örneklem: {sample_size})"

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        reduced[:, 0], reduced[:, 1],
        c=labels_df, cmap="RdYlGn_r", alpha=0.5, s=5, edgecolors="none",
    )
    ax.legend(["Legal (0)", "Illegal (1)"], loc="upper right")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{method.upper()} Bileşen 1")
    ax.set_ylabel(f"{method.upper()} Bileşen 2")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"2d_{method}_scatter.png"), dpi=150)
    plt.close()
    print(f"-> 2d_{method}_scatter.png kaydedildi")


def run_eda(df):
    """Tüm EDA adımlarını çalıştır."""
    feature_cols = [f"v{i}" for i in range(165)]

    print("\n=== Keşifçi Veri Analizi (EDA) ===")
    plot_class_imbalance(df)
    plot_time_step_analysis(df)
    plot_2d_scatter(df, feature_cols, method="pca")
    plot_2d_scatter(df, feature_cols, method="tsne")
    print("=== EDA tamamlandı ===\n")


if __name__ == "__main__":
    df = load_data()
    df = filter_known_labels(df)
    run_eda(df)
