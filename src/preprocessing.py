import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Proje kök dizinini bul
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "elliptic_bitcoin_dataset")


def load_data():
    """Veri setini yükle ve birleştir."""
    # Etiketleri yükle
    classes_df = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_classes.csv"))

    # Özellikleri yükle (başlıksız, ilk 2 sütun txId ve time_step)
    feature_cols = [f"v{i}" for i in range(165)]
    features_df = pd.read_csv(
        os.path.join(DATA_DIR, "elliptic_txs_features.csv"),
        header=None,
        names=["txId", "time_step"] + feature_cols,
    )

    # Birleştir
    df = features_df.merge(classes_df, on="txId", how="inner")
    return df


def filter_known_labels(df):
    """Sadece bilinen etiketleri (1=illicit, 2=licit) tut."""
    df = df[df["class"] != "unknown"].copy()
    # Etiketleri sayısala çevir: 0 (licit), 1 (illicit)
    df["class"] = df["class"].astype(int).map({2: 0, 1: 1})
    return df


def scale_features(df, feature_cols):
    """Özellikleri StandardScaler ile ölçeklendir."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def apply_pca(df, feature_cols, n_components=25):
    """PCA ile boyut indirgeme uygula."""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(df[feature_cols])
    reduced_cols = [f"pca_{i}" for i in range(n_components)]
    df_reduced = pd.DataFrame(reduced, columns=reduced_cols, index=df.index)
    return df_reduced, pca


def preprocess(n_components=25):
    """Tüm ön işleme adımını çalıştır."""
    df = load_data()
    df = filter_known_labels(df)

    # Özellik sütunlarını belirle (v0..v164)
    feature_cols = [f"v{i}" for i in range(165)]

    # Ölçeklendir
    df, scaler = scale_features(df, feature_cols)

    # PCA uygula
    df_pca, pca = apply_pca(df, feature_cols, n_components)

    # PCA sonuçlarını ana dataframe'e ekle (tek seferde birleştir)
    df = pd.concat([df.drop(columns=feature_cols), df_pca], axis=1)

    print(f"Veri seti boyutu: {df.shape[0]} satır, {len(feature_cols)} ham özellik")
    print(f"PCA sonrası: {n_components} bileşen")
    print(
        f"PCA açıklanan varyans: {pca.explained_variance_ratio_.sum():.2%}"
    )

    return df, scaler, pca


if __name__ == "__main__":
    df, scaler, pca = preprocess()
    print(df.head())
    print(df["class"].value_counts())
