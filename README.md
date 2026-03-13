# Data Binarization Strategies: Greedy vs. Optimal Decision Trees

> **Author:** Rishika Singh Chauhan  
> BTech Data Science · Mukesh Patel School of Technology and Management

---

## Overview

This repository contains the full code and results for our study on how **data binarization strategies** affect the performance of greedy versus optimal decision tree algorithms.

We compare:
- **Greedy:** scikit-learn's `DecisionTreeClassifier` (CART)
- **Optimal:** `STreeDClassifier` from [pystreed](https://github.com/AlgTUDelft/pystreed) (constrained to `max_depth=4`)

Across three binarization strategies:
- **Uniform** — equal-width bins, ignores data distribution
- **Quantile** — equal-frequency bins, distribution-aware
- **KMeans** — cluster-based bins, finds natural groupings

And three bin counts: `n_bins ∈ {3, 5, 10}`

---

## Key Findings

| Dataset | Best Approach | Why |
|---|---|---|
| Breast Cancer (classic) | Tie (both ~F1 0.93) | Greedy is 550× faster; Optimal gives simpler trees |
| Student Dropout (complex) | **Optimal wins** | F1 0.82 vs 0.75, with 16 leaves vs 657 |
| NF1 (noisy/low-signal) | **Optimal wins** | Both F1 ~0.47, but Optimal uses 15 leaves vs 129 |
| Paddy (perfect signal) | **Greedy wins** | Binarization destroys the clean continuous signal |

### The "Binarization Hazard"
On the Paddy dataset, the greedy model on raw continuous data achieved F1 ≈ **0.9985**. After binarization (required by STreeD), the best model dropped to F1 ≈ **0.9159** — a statistically significant loss (Cohen's d = −2.2). **Binarization is not a neutral preprocessing step — it is a critical hyperparameter.**

---

## Repository Structure

```
data-binarization-study/
│
├── experiment.py          # Main experiment script (full pipeline)
├── requirements.txt       # Python dependencies
├── README.md
│
├── data/                  # Place your datasets here (see setup below)
│
├── results/               # CSV output tables (auto-generated)
│   └── experiment_results_n_bins_*.csv
│
└── figures/               # Tree visualizations (auto-generated)
    └── greedy_baseline_tree.png
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/data-binarization-study.git
cd data-binarization-study
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on pystreed:** The `pystreed` package requires a C++ compiler. On Linux, ensure `build-essential` is installed. On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

## Running the Experiment

### Using the built-in Breast Cancer dataset (default)

The script comes pre-configured to run on scikit-learn's built-in Breast Cancer dataset:

```bash
python experiment.py
```

### Using your own dataset

Open `experiment.py` and replace the dataset loading block (clearly marked with `TODO`) with your own data:

```python
# Replace this section:
X = ...             # 2D NumPy array of continuous numeric features only
y = ...             # 1D NumPy array of binary labels (0s and 1s only)
feature_names = []  # List of feature name strings
class_names   = []  # List of two class name strings, e.g. ['Negative', 'Positive']
```

**Important constraints:**
- `X` must contain **only numeric/continuous features** — remove any categorical columns before passing
- `y` must be **binary** (0s and 1s only) — for multi-class problems, binarize the target first
- Missing values in `X` should be imputed before passing (e.g., use `SimpleImputer`)

### Datasets used in the paper

| Dataset | Source |
|---|---|
| Breast Cancer (Diagnostic) | `sklearn.datasets.load_breast_cancer()` |
| Student Dropout (2023) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| NF1 (2024) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/928) |
| Paddy (2025) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/1049) |

---

## Output

For each `n_bins` value (`3`, `5`, `10`), the script produces:

- **Console output** — full results table + statistical significance tests
- **`results/experiment_results_n_bins_{N}.csv`** — results table with all metrics
- **`figures/greedy_baseline_tree.png`** — visualization of the unconstrained greedy tree

### Metrics tracked per model

| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy |
| F1 (macro) | Primary performance metric; robust on imbalanced data |
| F1 Std Dev | Stability across 10 folds |
| Precision / Recall | Per-class breakdown |
| Tree Leaves | Model complexity proxy |
| Tree Depth | Model depth |
| Fit Time (s) | Training time |

### Statistical tests

Paired t-tests (`ttest_rel`) + **Cohen's d** effect size are computed for:
- F1: Optimal (Quantile) vs. Baseline
- F1: Greedy (Quantile) vs. Baseline
- F1: Greedy (Quantile) vs. Optimal (Quantile)
- Fit Time: same three comparisons

---

## Experimental Design

```
                        ┌─────────────────────────────────────────┐
                        │              Dataset                    │
                        └──────────────────┬──────────────────────┘
                                           │
             ┌─────────────────────────────┼──────────────────────────────┐
             │                             │                              │
     Uniform (n_bins)             Quantile (n_bins)              KMeans (n_bins)
             │                             │                              │
    ┌────────┴────────┐          ┌────────┴────────┐            ┌────────┴────────┐
    │                 │          │                 │            │                 │
  CART             STreeD     CART             STreeD         CART            STreeD
(Greedy)         (Optimal)  (Greedy)         (Optimal)      (Greedy)        (Optimal)
                                           max_depth=4
    └─────────────────┴──────────┴─────────────────┴────────────┴────────────────┘
                                           │
                              10-Fold Cross-Validation
                       F1, Accuracy, Precision, Recall, Fit Time
```

---

## Citation

If you use this code or reference our findings, please cite:

```
Chauhan, R. S. (2025). The Impact of Data Binarization Strategies
on the Performance of Greedy vs. Optimal Decision Tree Algorithms.
BTech Data Science, Mukesh Patel School of Technology and Management.
```

### Key References

- Breiman et al. (1984). *Classification and Regression Trees.* Wadsworth & Brooks/Cole.
- Van der Linden et al. (2023). [STreeD: A General Framework for Optimal Decision Trees.](https://arxiv.org/abs/2305.19706)
- Demirović et al. (2022). [MurTree: Optimal Decision Trees via Dynamic Programming and Search.](https://jmlr.org/papers/v23/21-0253.html) JMLR, 23(26).
- Pedregosa et al. (2011). [Scikit-learn: Machine Learning in Python.](https://jmlr.org/papers/v12/pedregosa11a.html) JMLR, 12.

---

## License

MIT License — feel free to use, modify, and distribute with attribution.
