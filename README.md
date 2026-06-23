# Cat2Vec with Positional Encoding for classfication

## Description

This repository contains the experiments for the paper *"Cat2Vec with Positional Encoding: Classification Tasks"*, which evaluates a sinusoidal positional encoding approach for ordinal categorical features against a set of baseline encoding pipelines.

The core idea: ordinal categorical features (e.g., `low < medium < high`) carry rank information that standard encoders either discard (one-hot) or misrepresent (label encoding). Cat2Vec with positional encoding applies sinusoidal position embeddings — borrowed from transformer architectures — to preserve and enrich ordinal structure before feeding features to downstream classifiers.

Experiments are run across three real-world datasets, five encoding pipelines, and four classifiers, with repeated random splits and paired significance tests (paired t-test + McNemar) comparing the proposed method (P4) against each baseline.

---

## Dataset Information

| Dataset | File | Rows | Task |
|---|---|---|---|
| Cat-in-the-Dat I (Kaggle) | `data/train_cat1.csv` | 300,000 | Binary classification |
| Cat-in-the-Dat II (Kaggle) | `data/train_cat2.csv` | 600,000 | Binary classification |
| CDC BRFSS 2015 Diabetes | `data/diabetes_binary_health_indicators_BRFSS2015.csv` | 253,680 | Binary classification |

All three datasets contain a mix of binary, nominal, and ordinal categorical features with a binary `target` column.


---

## Code Information

### Scripts (`scripts/`)

| Script | Dataset | Description |
|---|---|---|
| `experiment_baseline_and_significance.py` | Cat-in-the-Dat I & II | Runs P0–P4 pipelines across classifiers with repeated splits and significance tests |
| `experiment_baseline_and_significance_brfss.py` | BRFSS Diabetes | Same experiment harness applied to the BRFSS dataset |


### Encoding Pipelines

| Pipeline | Ordinal Encoding | Nominal Encoding |
|---|---|---|
| P0 | Rank-as-numeric (baseline) | One-hot encoding |
| P1 | OrdinalEncoder | One-hot encoding |
| P2 | OrdinalEncoder | Cat2Vec |
| P3 | Cat2Vec with positional encoding | One-hot encoding |
| P4 | Cat2Vec with positional encoding | Cat2Vec |

P4 is the proposed method. All other pipelines serve as ablation baselines.

### Classifiers

Logistic Regression, Random Forest, K-Nearest Neighbours, XGBoost.

### Output

Scripts print to stdout: mean ± std metrics (accuracy, macro-F1, ROC-AUC, PR-AUC) per pipeline and classifier, followed by paired significance results (P4 vs each baseline). Redirect to a file to save results:

```bash
python3.11 scripts/experiment_baseline_and_significance.py --seeds 5 > results/output.md
```

---

## Usage Instructions

### 1. Install dependencies

```bash
/usr/local/bin/python3.11 -m pip install -r requirements.txt
```

### 2. Run experiments

**Cat-in-the-Dat I or II:**
```bash
/usr/local/bin/python3.11 scripts/experiment_baseline_and_significance.py \
    --data data/train_cat1.csv \
    --seeds 5
```

**BRFSS Diabetes:**
```bash
/usr/local/bin/python3.11 scripts/experiment_baseline_and_significance_brfss.py \
    --data data/diabetes_binary_health_indicators_BRFSS2015.csv \
    --seeds 5
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | dataset-specific path | Path to the input CSV file |
| `--seeds` | 2 | Number of random train/test splits |
| `--svm` | off | Include SVM classifier (very slow on large datasets) |

### 3. Save results

```bash
/usr/local/bin/python3.11 scripts/experiment_baseline_and_significance.py \
    --data data/train_cat2.csv \
    --seeds 5 > results/experiments_seed5_results.md
```



---

## Requirements

Python 3.11 is required. TensorFlow and XGBoost must be installed from this specific environment — system Python and Anaconda installations may not have compatible versions.

```
numpy==1.26.4
pandas==3.0.3
scipy==1.17.1
statsmodels==0.14.6
tensorflow==2.16.2
scikit-learn==1.8.0
xgboost==3.2.0
```

Install with:

```bash
/usr/local/bin/python3.11 -m pip install -r requirements.txt
```

---

## Repository Structure

```
cat2vec/
├── data/                  # Input datasets
├── scripts/               # Experiment and figure generation scripts
├── requirements.txt
└── README.md
```
