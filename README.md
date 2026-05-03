# Elliptic Fraud Detection — Risk Scoring Engine

*Read this in other languages: [Türkçe](README-TR.md)*

> Autonomous risk scoring engine that detects illicit Bitcoin transactions using the Elliptic Data Set, combining graph-based transaction features with KNN and Neural Network (MLP) classifiers.

## Table of Contents

- [Elliptic Fraud Detection — Risk Scoring Engine](#elliptic-fraud-detection--risk-scoring-engine)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
    - [Files](#files)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Pipeline Steps](#pipeline-steps)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
      - [Class Imbalance](#class-imbalance)
      - [Time Step Analysis](#time-step-analysis)
      - [2D Visualization](#2d-visualization)
    - [3. Modeling](#3-modeling)
      - [KNN — K-Nearest Neighbors](#knn--k-nearest-neighbors)
      - [ANN — Artificial Neural Network (MLP)](#ann--artificial-neural-network-mlp)
    - [4. Evaluation \& Comparison](#4-evaluation--comparison)
  - [Results](#results)
  - [Generated Figures](#generated-figures)
  - [Key Design Decisions](#key-design-decisions)
  - [License](#license)

---

## Overview

Financial systems and blockchain networks inherently expose risks related to money laundering (AML), ransomware, and fraud due to the pseudonymous nature of transactions. This project builds a **Risk Scoring Engine** that analyzes money flow patterns — transaction volume, sender/receiver relationships, and neighborhood features — to autonomously classify Bitcoin transactions as **illicit** or **licit**.

The system ingests the Elliptic Data Set (a Bitcoin transaction graph), applies dimensionality reduction via PCA, and trains two complementary models: **K-Nearest Neighbors (KNN)** and a **Multi-Layer Perceptron (MLP)**. Models are evaluated on Precision and Recall rather than accuracy to account for severe class imbalance.

---

## Dataset

**Source:** [Elliptic Data Set (Kaggle)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

| Property                 | Value                           |
| ------------------------ | ------------------------------- |
| Nodes (transactions)     | ~203,000                        |
| Edges (flows)            | ~234,000                        |
| Features per transaction | 165 + 2 (txId, time_step)       |
| Time steps               | 49 (~2 weeks each)              |
| Classes                  | 1 (illicit), 2 (licit), unknown |

After filtering out unknown labels:

- **46,564** labeled transactions
- **42,019** licit (90.2%)
- **4,545** illicit (9.8%)

### Files

| File                        | Description                                                              |
| --------------------------- | ------------------------------------------------------------------------ |
| `elliptic_txs_features.csv` | Transaction features (166 numeric columns + txId + time_step, no header) |
| `elliptic_txs_classes.csv`  | Labels: txId, class (1/2/unknown)                                        |
| `elliptic_txs_edgelist.csv` | Directed edges: txId1 → txId2                                            |

---

## Project Structure

```
elliptic-fraud-detection/
├── main.py                      # Main pipeline entry point
├── requirements.txt             # Python dependencies
├── pipeline_output.txt          # Saved terminal output
├── data/
│   └── elliptic_bitcoin_dataset/
│       ├── elliptic_txs_features.csv
│       ├── elliptic_txs_classes.csv
│       └── elliptic_txs_edgelist.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Data loading, filtering, scaling, PCA
│   ├── eda.py                   # Exploratory data analysis & visualizations
│   ├── evaluation.py            # Model comparison & confusion matrices
│   └── models/
│       ├── __init__.py
│       ├── knn.py               # KNN with K-value tuning
│       └── mlp.py               # MLP (neural network) with architecture search
├── reports/
│   └── figures/                 # All generated plots
│       ├── class_imbalance.png
│       ├── time_step_analysis.png
│       ├── 2d_pca_scatter.png
│       ├── 2d_tsne_scatter.png
│       ├── cm_knn.png
│       ├── cm_ysa.png
│       └── model_comparison.png
└── notebooks/                   # (Optional — for interactive exploration)
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Requirements:**

- Python 3.10+
- pandas >= 2.2
- numpy >= 2.0
- scikit-learn >= 1.5
- matplotlib >= 3.9
- seaborn >= 0.13

---

## Usage

Run the full pipeline (EDA → preprocessing → training → evaluation):

```bash
python3 main.py
```

All figures are saved to `reports/figures/`. Terminal output is available in `pipeline_output.txt`.

---

## Pipeline Steps

### 1. Data Preprocessing

**File:** `src/preprocessing.py`

- **Filtering:** Remove all `unknown`-labeled transactions. Keep only confirmed illicit (1) and licit (2) transactions. Labels are remapped to binary: `0` (licit), `1` (illicit).
- **Feature Scaling:** All 165 features are standardized using `StandardScaler`. This is critical because distance-based algorithms (KNN) and weight-based algorithms (MLP) are sensitive to feature magnitude. _"Distance-based algorithms were normalized to prevent biased behavior."_
- **Dimensionality Reduction:** PCA reduces 165 features to **25 principal components**, explaining **~71.7%** of the total variance. Without reduction, KNN suffers from the curse of dimensionality.

### 2. Exploratory Data Analysis (EDA)

**File:** `src/eda.py`

Three core analyses are performed:

#### Class Imbalance

A bar chart showing the ~9:1 ratio between licit and illicit transactions. This imbalance makes accuracy a misleading metric — a model that predicts "licit" for everything would achieve ~90% accuracy while detecting zero fraud.

#### Time Step Analysis

A two-panel plot showing:

1. Raw count of illicit transactions per time step (1–49)
2. Percentage of illicit transactions per time step

This reveals how fraud patterns evolve across the ~2-week intervals.

#### 2D Visualization

Scatter plots using PCA and t-SNE to project the 165-dimensional feature space into 2D, showing how well the two classes separate in the feature space. t-SNE uses a 5,000-sample subset for computational feasibility.

### 3. Modeling

**Train/Test Split:** 80/20 with stratification on the class label.

#### KNN — K-Nearest Neighbors

**File:** `src/models/knn.py`

- Tested K values: 3, 5, 7, 9, 11
- Weight scheme: **distance-weighted** (closer neighbors have more influence)
- Best K is selected by F1 score on the test set
- Best result: **K = 3** (F1 = 0.8586)

#### ANN — Artificial Neural Network (MLP)

**File:** `src/models/mlp.py`

- Tested architectures:
  - (64, 32) — 2 hidden layers
  - (128, 64, 32) — 3 hidden layers
  - (64, 64, 32) — 3 hidden layers
- Activation: ReLU, Solver: Adam, Max iterations: 200
- Early stopping with 10% validation split
- Best architecture: **(128, 64, 32)** (F1 = 0.8797)

### 4. Evaluation & Comparison

**File:** `src/evaluation.py`

Models are compared on **Precision**, **Recall**, and **F1** — not accuracy — due to the class imbalance.

| Metric    | KNN (K=3)  | MLP (128,64,32) |
| --------- | ---------- | --------------- |
| Precision | 0.8799     | **0.9377**      |
| Recall    | **0.8383** | 0.8284          |
| F1        | 0.8586     | **0.8797**      |
| Accuracy  | 0.97       | **0.98**        |

Confusion matrices and a bar chart comparison are generated automatically.

---

## Results

- **MLP outperforms KNN overall**, achieving higher Precision (0.94 vs 0.88) and F1 (0.88 vs 0.86). This suggests the MLP captures complex, non-linear patterns in transaction features (e.g., "high volume + many outputs = likely fraud" rules).
- **KNN achieves slightly better Recall** (0.84 vs 0.83), meaning it catches marginally more illicit transactions at the cost of more false positives.
- Both models achieve ~97–98% accuracy, but this is misleading — the key metric is how well they detect the minority (illicit) class.

---

## Generated Figures

| Figure                   | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `class_imbalance.png`    | Bar chart of licit vs. illicit transaction counts |
| `time_step_analysis.png` | Illicit transaction trends across 49 time steps   |
| `2d_pca_scatter.png`     | 2D PCA projection showing class separation        |
| `2d_tsne_scatter.png`    | 2D t-SNE projection (5,000 sample subset)         |
| `cm_knn.png`             | Confusion matrix for best KNN model               |
| `cm_ysa.png`             | Confusion matrix for best MLP model               |
| `model_comparison.png`   | Bar chart comparing Precision, Recall, F1         |

---

## Key Design Decisions

1. **Why PCA?** 165 features cause the curse of dimensionality for KNN and increase training time for MLP. PCA to 25 components retains ~72% variance while dramatically reducing compute.
2. **Why StandardScaler?** KNN computes Euclidean distances — unscaled features with larger ranges would dominate the distance calculation. MLP weights are also initialized relative to input scale.
3. **Why Precision & Recall over Accuracy?** With a 90:10 class split, a trivial "always predict licit" model gets 90% accuracy. Precision measures how many flagged transactions are truly illicit; Recall measures how many illicit transactions we catch.
4. **Why distance-weighted KNN?** Closer neighbors are more relevant for classification — this reduces noise from distant points in high-dimensional space.
5. **Why stratified split?** Ensures both train and test sets maintain the same class ratio for fair evaluation.

---

## License

This project uses the [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) available on Kaggle. Refer to the dataset's license for usage terms.
