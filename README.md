# Customer Segmentation & Bank Marketing Deep Learning

A two-part Jupyter Notebook project combining **Unsupervised Learning** (dimensionality reduction and Gaussian Mixture Model clustering for customer segmentation) and **Deep Learning** (TensorFlow/Keras Artificial Neural Network for bank term-deposit subscription prediction).

> **Dataset & Execution Notice:** This notebook was authored in **Google Colab** and loads both datasets from Google Drive (`/content/drive/MyDrive/Mall_Customers.csv` and `/content/drive/MyDrive/bank.csv`). The CSV files are **not included** in this repository, and the saved notebook outputs stop at the unmounted Drive `FileNotFoundError` in Cell 8. No execution metrics or synthetic plots are fabricated here.

---

## Project Architecture

### Part 1 — Unsupervised Learning: Customer Segmentation (`Mall_Customers.csv`)
1. **Data Preprocessing & Feature Engineering:** Drops `CustomerID`, maps `Gender` (`Male` → `1`, `Female` → `0`), and standardizes features with `StandardScaler`.
2. **Dimensionality Reduction:**
   - **PCA** (`PCA(n_components=2)`) → 2D projection (`PC1`, `PC2`)
   - **Truncated SVD** (`TruncatedSVD(n_components=2)`) → 2D projection (`SVD1`, `SVD2`)
   - **t-SNE** (`TSNE(n_components=2, random_state=42)`) → 2D manifold visualization
3. **Clustering & Model Selection:**
   - **Gaussian Mixture Model (GMM)** (`GaussianMixture(n_components=5, random_state=42)`)
   - Cluster visualization overlaid on t-SNE components (`seaborn.scatterplot`)
   - Probabilistic fit evaluation via **AIC** and **BIC**

### Part 2 — Deep Learning: Bank Marketing Deposit Prediction (`bank.csv`)
1. **Preprocessing:** One-hot encoding (`pd.get_dummies(..., drop_first=True)`) across `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, and `poutcome`; target mapping `deposit` (`yes` → `1`, `no` → `0`).
2. **Splitting & Scaling:** 80/20 train/test split (`random_state=42`) followed by `StandardScaler` normalization.
3. **Neural Network (`tf.keras.Sequential`):**
   - `Dense(64, activation='relu')`
   - `Dense(32, activation='relu')`
   - `Dense(1, activation='sigmoid')`
4. **Training & Regularization:** Compiled with `adam` optimizer and `binary_crossentropy` loss (`epochs=20`, `batch_size=128`, `validation_split=0.2`), including `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)` and test-set evaluation.

---

## Project Structure

```text
customer-segmentation-and-bank-marketing-dl/
├── Unsupervised_ML_&_Deep_Learning.ipynb   # Main Colab notebook (Part 1: GMM/PCA/t-SNE, Part 2: Keras ANN)
├── requirements.txt                        # Python dependencies
├── .gitignore
└── README.md
```

---

## Dataset Requirements

To execute this notebook, obtain the two standard public datasets and place them either in your Google Drive (`/content/drive/MyDrive/`) or locally alongside updated `pd.read_csv(...)` paths:

1. **Mall Customer Segmentation Data** (`Mall_Customers.csv`) — columns including `CustomerID`, `Gender`, `Age`, `Annual Income`, `Spending Score`.
2. **Bank Marketing Dataset** (`bank.csv`) — columns including `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, and target `deposit` (`yes`/`no`).

---

## Installation & Execution

### Option A — Google Colab (Matches Original Paths)
1. Upload `Mall_Customers.csv` and `bank.csv` to `MyDrive/` in Google Drive.
2. Mount Google Drive in Colab (`from google.colab import drive; drive.mount('/content/drive')`) before running Cell 8.
3. Run all cells in `Unsupervised_ML_&_Deep_Learning.ipynb`.

### Option B — Local Execution
1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/ahmadtanoli2004/customer-segmentation-and-bank-marketing-dl.git
   cd customer-segmentation-and-bank-marketing-dl
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place `Mall_Customers.csv` and `bank.csv` in a local `data/` folder and update the two `pd.read_csv(...)` paths in Cells 8 and 50.
3. Launch Jupyter:
   ```bash
   jupyter notebook "Unsupervised_ML_&_Deep_Learning.ipynb"
   ```
