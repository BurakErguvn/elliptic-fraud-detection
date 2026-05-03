# Elliptic Fraud Detection — Risk Scoring Engine (Risk Puanlama Motoru)

*Diğer dillerde okuyun: [English](README.md)*

> Elliptic Veri Seti'ni kullanarak yasa dışı Bitcoin işlemlerini tespit eden, grafik tabanlı işlem özelliklerini KNN ve Sinir Ağı (MLP) sınıflandırıcıları ile birleştiren otonom risk puanlama motoru.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Veri Seti](#veri-seti)
  - [Dosyalar](#dosyalar)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Boru Hattı Adımları (Pipeline)](#boru-hattı-adımları-pipeline)
  - [1. Veri Ön İşleme](#1-veri-ön-işleme)
  - [2. Keşifçi Veri Analizi (EDA)](#2-keşifçi-veri-analizi-eda)
  - [3. Modelleme](#3-modelleme)
  - [4. Değerlendirme & Karşılaştırma](#4-değerlendirme--karşılaştırma)
- [Sonuçlar](#sonuçlar)
- [Oluşturulan Grafikler](#oluşturulan-grafikler)
- [Temel Tasarım Kararları](#temel-tasarım-kararları)
- [Lisans](#lisans)

---

## Genel Bakış

Finansal sistemler ve blokzincir ağları, işlemlerin takma adlı doğası gereği kara para aklama (AML), fidye yazılımı ve dolandırıcılıkla ilgili risklere doğası gereği açıktır. Bu proje, para akışı modellerini — işlem hacmi, gönderici/alıcı ilişkileri ve komşuluk özellikleri — analiz ederek Bitcoin işlemlerini **yasa dışı (illicit)** veya **yasal (licit)** olarak otonom bir şekilde sınıflandıran bir **Risk Puanlama Motoru** oluşturur.

Sistem, Elliptic Veri Setini (bir Bitcoin işlem grafiği) alır, PCA aracılığıyla boyut indirgeme uygular ve iki tamamlayıcı model eğitir: **K-En Yakın Komşu (KNN)** ve **Çok Katmanlı Algılayıcı (MLP)**. Şiddetli sınıf dengesizliğini hesaba katmak için modeller doğruluk (accuracy) yerine Kesinlik (Precision) ve Duyarlılık (Recall) üzerinden değerlendirilir.

---

## Veri Seti

**Kaynak:** [Elliptic Data Set (Kaggle)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

| Özellik                 | Değer                           |
| ------------------------ | ------------------------------- |
| Düğümler (işlemler)     | ~203.000                        |
| Kenarlar (akışlar)            | ~234.000                        |
| İşlem başına özellikler | 165 + 2 (txId, time_step)       |
| Zaman adımları               | 49 (her biri ~2 hafta)              |
| Sınıflar                  | 1 (yasa dışı), 2 (yasal), unknown (bilinmeyen) |

Bilinmeyen etiketler filtrelendikten sonra:

- **46.564** etiketli işlem
- **42.019** yasal (%90.2)
- **4.545** yasa dışı (%9.8)

### Dosyalar

| Dosya                        | Açıklama                                                              |
| --------------------------- | ------------------------------------------------------------------------ |
| `elliptic_txs_features.csv` | İşlem özellikleri (166 sayısal sütun + txId + time_step, başlık yok) |
| `elliptic_txs_classes.csv`  | Etiketler: txId, class (1/2/unknown)                                        |
| `elliptic_txs_edgelist.csv` | Yönlü kenarlar: txId1 → txId2                                            |

---

## Proje Yapısı

```
elliptic-fraud-detection/
├── main.py                      # Ana pipeline giriş noktası
├── requirements.txt             # Python bağımlılıkları
├── pipeline_output.txt          # Kaydedilen terminal çıktısı
├── data/
│   └── elliptic_bitcoin_dataset/
│       ├── elliptic_txs_features.csv
│       ├── elliptic_txs_classes.csv
│       └── elliptic_txs_edgelist.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Veri yükleme, filtreleme, ölçekleme, PCA
│   ├── eda.py                   # Keşifçi veri analizi ve görselleştirmeler
│   ├── evaluation.py            # Model karşılaştırma ve karışıklık matrisleri
│   └── models/
│       ├── __init__.py
│       ├── knn.py               # K değeri optimizasyonlu KNN
│       └── mlp.py               # Mimari aramalı MLP (sinir ağı)
├── reports/
│   └── figures/                 # Oluşturulan tüm grafikler
│       ├── class_imbalance.png
│       ├── time_step_analysis.png
│       ├── 2d_pca_scatter.png
│       ├── 2d_tsne_scatter.png
│       ├── cm_knn.png
│       ├── cm_ysa.png
│       └── model_comparison.png
└── notebooks/                   # (İsteğe bağlı — etkileşimli keşif için)
```

---

## Kurulum

```bash
# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Veya manuel olarak yükleyin
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Gereksinimler:**

- Python 3.10+
- pandas >= 2.2
- numpy >= 2.0
- scikit-learn >= 1.5
- matplotlib >= 3.9
- seaborn >= 0.13

---

## Kullanım

Tüm boru hattını çalıştırın (EDA → ön işleme → eğitim → değerlendirme):

```bash
python3 main.py
```

Tüm grafikler `reports/figures/` klasörüne kaydedilir. Terminal çıktısı `pipeline_output.txt` dosyasında bulunabilir.

---

## Boru Hattı Adımları (Pipeline)

### 1. Veri Ön İşleme

**Dosya:** `src/preprocessing.py`

- **Filtreleme:** `unknown` etiketli tüm işlemleri kaldırır. Sadece onaylanmış yasa dışı (1) ve yasal (2) işlemleri tutar. Etiketler ikili (binary) formata dönüştürülür: `0` (yasal), `1` (yasa dışı).
- **Özellik Ölçekleme:** Tüm 165 özellik `StandardScaler` kullanılarak standartlaştırılır. Mesafe tabanlı algoritmalar (KNN) ve ağırlık tabanlı algoritmalar (MLP) özellik büyüklüğüne duyarlı olduğundan bu kritik bir adımdır. _"Mesafe tabanlı algoritmalar yanlı (biased) davranışı önlemek için normalize edildi."_
- **Boyut İndirgeme:** PCA, 165 özelliği toplam varyansın **~%71.7'sini** açıklayan **25 temel bileşene** indirger. İndirgeme olmadan KNN, boyutluluk lanetinden (curse of dimensionality) muzdarip olur.

### 2. Keşifçi Veri Analizi (EDA)

**Dosya:** `src/eda.py`

Üç temel analiz gerçekleştirilir:

#### Sınıf Dengesizliği

Yasal ve yasa dışı işlemler arasındaki ~9:1 oranını gösteren bir çubuk grafik. Bu dengesizlik, doğruluğu (accuracy) yanıltıcı bir metrik yapar — her şey için "yasal" tahmininde bulunan bir model, sıfır dolandırıcılık tespit ederken ~%90 doğruluk elde edebilir.

#### Zaman Adımı Analizi

İki panelli bir grafik şunları gösterir:

1. Zaman adımı başına yasa dışı işlemlerin ham sayısı (1–49)
2. Zaman adımı başına yasa dışı işlemlerin yüzdesi

Bu, dolandırıcılık modellerinin ~2 haftalık aralıklarla nasıl geliştiğini ortaya koyar.

#### 2D Görselleştirme

165 boyutlu özellik uzayını 2D'ye yansıtmak için PCA ve t-SNE kullanan dağılım grafikleri, iki sınıfın özellik uzayında ne kadar iyi ayrıldığını gösterir. t-SNE, hesaplama fizibilitesi için 5.000 örneklem kullanır.

### 3. Modelleme

**Eğitim/Test Ayrımı:** Sınıf etiketi üzerinde katmanlama (stratification) ile 80/20.

#### KNN — K-En Yakın Komşular

**Dosya:** `src/models/knn.py`

- Test edilen K değerleri: 3, 5, 7, 9, 11
- Ağırlık şeması: **mesafe ağırlıklı (distance-weighted)** (daha yakın komşuların daha fazla etkisi vardır)
- En iyi K, test setindeki F1 skoruna göre seçilir
- En iyi sonuç: **K = 3** (F1 = 0.8586)

#### ANN — Yapay Sinir Ağı (MLP)

**Dosya:** `src/models/mlp.py`

- Test edilen mimariler:
  - (64, 32) — 2 gizli katman
  - (128, 64, 32) — 3 gizli katman
  - (64, 64, 32) — 3 gizli katman
- Aktivasyon: ReLU, Çözücü (Solver): Adam, Maks. iterasyon: 200
- %10 doğrulama (validation) ayrımı ile erken durdurma (early stopping)
- En iyi mimari: **(128, 64, 32)** (F1 = 0.8797)

### 4. Değerlendirme & Karşılaştırma

**Dosya:** `src/evaluation.py`

Sınıf dengesizliği nedeniyle modeller doğruluk (accuracy) yerine **Kesinlik (Precision)**, **Duyarlılık (Recall)** ve **F1** üzerinden karşılaştırılır.

| Metrik    | KNN (K=3)  | MLP (128,64,32) |
| --------- | ---------- | --------------- |
| Kesinlik | 0.8799     | **0.9377**      |
| Duyarlılık    | **0.8383** | 0.8284          |
| F1        | 0.8586     | **0.8797**      |
| Doğruluk  | 0.97       | **0.98**        |

Karışıklık matrisleri ve çubuk grafik karşılaştırması otomatik olarak oluşturulur.

---

## Sonuçlar

- **MLP genel olarak KNN'den daha iyi performans gösterir**, daha yüksek Kesinlik (0.94'e karşı 0.88) ve F1 (0.88'e karşı 0.86) elde eder. Bu, MLP'nin işlem özelliklerindeki karmaşık, doğrusal olmayan modelleri yakaladığını gösterir (ör. "yüksek hacim + çok sayıda çıktı = olası dolandırıcılık" kuralları).
- **KNN biraz daha iyi Duyarlılık (Recall) elde eder** (0.84'e karşı 0.83), yani daha fazla yanlış pozitif (false positive) pahasına biraz daha fazla yasa dışı işlemi yakalar.
- Her iki model de ~%97–98 doğruluk elde eder, ancak bu yanıltıcıdır — temel metrik azınlık (yasa dışı) sınıfını ne kadar iyi tespit ettikleridir.

---

## Oluşturulan Grafikler

| Grafik                   | Açıklama                                       |
| ------------------------ | ------------------------------------------------- |
| `class_imbalance.png`    | Yasal ve yasa dışı işlem sayılarının çubuk grafiği |
| `time_step_analysis.png` | 49 zaman adımı boyunca yasa dışı işlem eğilimleri   |
| `2d_pca_scatter.png`     | Sınıf ayrımını gösteren 2D PCA yansıması        |
| `2d_tsne_scatter.png`    | 2D t-SNE yansıması (5.000 örneklik alt küme)         |
| `cm_knn.png`             | En iyi KNN modeli için karışıklık matrisi               |
| `cm_ysa.png`             | En iyi MLP modeli için karışıklık matrisi               |
| `model_comparison.png`   | Kesinlik, Duyarlılık, F1 karşılaştıran çubuk grafik         |

---

## Temel Tasarım Kararları

1. **Neden PCA?** 165 özellik, KNN için boyutluluk lanetine (curse of dimensionality) neden olur ve MLP için eğitim süresini artırır. 25 bileşene PCA, hesaplamayı önemli ölçüde azaltırken varyansın ~%72'sini korur.
2. **Neden StandardScaler?** KNN Öklid (Euclidean) mesafelerini hesaplar — daha geniş aralıklara sahip ölçeklenmemiş özellikler mesafe hesaplamasına hakim olur. MLP ağırlıkları da girdi ölçeğine göre başlatılır.
3. **Neden Doğruluk (Accuracy) yerine Kesinlik ve Duyarlılık?** 90:10 sınıf ayrımıyla, her zaman "yasal" tahmininde bulunan basit bir model %90 doğruluk elde eder. Kesinlik, işaretlenen işlemlerin ne kadarının gerçekten yasa dışı olduğunu ölçer; Duyarlılık, yasa dışı işlemlerin ne kadarını yakaladığımızı ölçer.
4. **Neden mesafe ağırlıklı KNN?** Daha yakın komşular sınıflandırma için daha uygundur — bu, yüksek boyutlu uzaydaki uzak noktalardan gelen gürültüyü azaltır.
5. **Neden katmanlı (stratified) ayrım?** Adil değerlendirme için hem eğitim hem de test setlerinin aynı sınıf oranını korumasını sağlar.

---

## Lisans

Bu proje, Kaggle'da bulunan [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) veri setini kullanır. Kullanım koşulları için veri setinin lisansına bakın.