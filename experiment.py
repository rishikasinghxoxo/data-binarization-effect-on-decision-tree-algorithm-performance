"""
The Impact of Data Binarization Strategies on the Performance
of Greedy vs. Optimal Decision Tree Algorithms

Authors:Rishika Singh Chauhan
BTech Data Science, Mukesh Patel School of Technology and Management

This script runs the full comparative analysis of CART (greedy) vs STreeD (optimal)
decision trees across multiple binarization strategies (Uniform, Quantile, KMeans)
and bin counts (3, 5, 10), evaluated via 10-fold cross-validation.

Usage:
    See README.md for dataset setup instructions.
    python experiment.py
"""

import pandas as pd
import numpy as np
from time import time
import warnings
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from pystreed import STreeDClassifier
from scipy.stats import ttest_rel, t
from numpy import std, mean, sqrt
from sklearn.impute import SimpleImputer


# TODO: Replace this entire block with data loading code.
# The script requires you to define 4 variables:
#
# X: A NumPy array of features.
#     *IMPORTANT*: Must be 2D and contain ONLY numeric/continuous features.
#
# y: A 1D NumPy array of targets.
#     *IMPORTANT*: Must contain ONLY 0s and 1s.
#
# feature_names: A list of strings for your feature names.
#
# class_names: A list of two strings for your target class names.
#              (e.g., ['Class 0', 'Class 1'] or ['Negative', 'Positive'])
#
# ---
#
# Example using the Breast Cancer dataset (sklearn built-in):
#
#   from sklearn.datasets import load_breast_cancer
#   data = load_breast_cancer()
#   X = data.data
#   y = data.target
#   feature_names = list(data.feature_names)
#   class_names = list(data.target_names)
#
# ---

print("Loading dataset...")

# --- REPLACE BELOW WITH YOUR DATASET LOADING CODE ---
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = list(data.feature_names)
class_names = list(data.target_names)
# --- END DATASET LOADING ---


# --- 2. Model and CV Setup ---


greedy_clf = DecisionTreeClassifier(random_state=42)
OPTIMAL_MAX_DEPTH = 4
optimal_clf = STreeDClassifier(max_depth=OPTIMAL_MAX_DEPTH)

cv = KFold(n_splits=10, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']


# --- 3. Helper Function: Cohen's d ---

def cohens_d_paired(x, y):
    """Calculates Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    sd = std(diff, ddof=1)
    if sd == 0:
        return 0
    return mean(diff) / sd


# --- 4. Baseline: Greedy on Raw Continuous Data ---

print("--- Running Experiment 1: Greedy on Continuous Data (Once) ---")
baseline_scores = cross_validate(greedy_clf, X, y, cv=cv, scoring=scoring_metrics)

baseline_f1_scores = cross_validate(greedy_clf, X, y, cv=cv, scoring='f1_macro')['test_score']
baseline_time_scores = baseline_scores['fit_time']

baseline_avg_results = {
    'Accuracy':          np.mean(baseline_scores['test_accuracy']),
    'F1':                np.mean(baseline_scores['test_f1_macro']),
    'F1 Std Dev':        np.std(baseline_scores['test_f1_macro']),
    'Precision':         np.mean(baseline_scores['test_precision_macro']),
    'Recall':            np.mean(baseline_scores['test_recall_macro']),
    'Fit Time (s)':      np.mean(baseline_scores['fit_time']),
    'Fit Time Std Dev':  np.std(baseline_scores['fit_time'])
}

greedy_baseline_model = DecisionTreeClassifier(random_state=42).fit(X, y)
baseline_leaves = greedy_baseline_model.get_n_leaves()
baseline_depth  = greedy_baseline_model.get_depth()
print("Baseline complete.\n")


# --- 5. Main Experiment Loop: Binarization Sensitivity Analysis ---

for n_bins in [3, 5, 10]:
    print(f"{'#'*60}")
    print(f"--- STARTING FULL ANALYSIS FOR N_BINS = {n_bins} ---")
    print(f"{'#'*60}\n")

    results         = {}
    leaf_counts     = {}
    depths          = {}
    fold_scores_f1  = {}
    fold_scores_time = {}

    # Include the baseline in each n_bins table for comparison
    results['Greedy (Continuous)']          = baseline_avg_results
    leaf_counts['Greedy (Continuous)']      = baseline_leaves
    depths['Greedy (Continuous)']           = baseline_depth
    fold_scores_f1['Greedy (Continuous)']   = baseline_f1_scores
    fold_scores_time['Greedy (Continuous)'] = baseline_time_scores

    # --- Define the three binarization strategies ---
    strategies = {
        'Uniform':  KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform',  subsample=None),
        'Quantile': KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None),
        'KMeans':   KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans',   subsample=None)
    }

    for strategy_name, binarizer in strategies.items():
        print(f"Running CV for '{strategy_name}' (n_bins={n_bins})...")

        # ---- Greedy + Binarizer ----
        greedy_pipeline  = Pipeline([('binarizer', binarizer), ('classifier', greedy_clf)])
        greedy_scores    = cross_validate(greedy_pipeline, X, y, cv=cv, scoring=scoring_metrics)
        greedy_f1_raw    = cross_validate(greedy_pipeline, X, y, cv=cv, scoring='f1_macro')['test_score']

        result_key_greedy          = f'Greedy ({strategy_name})'
        results[result_key_greedy] = {
            'Accuracy':         np.mean(greedy_scores['test_accuracy']),
            'F1':               np.mean(greedy_scores['test_f1_macro']),
            'F1 Std Dev':       np.std(greedy_scores['test_f1_macro']),
            'Precision':        np.mean(greedy_scores['test_precision_macro']),
            'Recall':           np.mean(greedy_scores['test_recall_macro']),
            'Fit Time (s)':     np.mean(greedy_scores['fit_time']),
            'Fit Time Std Dev': np.std(greedy_scores['fit_time'])
        }
        fold_scores_f1[result_key_greedy]   = greedy_f1_raw
        fold_scores_time[result_key_greedy] = greedy_scores['fit_time']

        # ---- Optimal (STreeD) + Binarizer ----
        optimal_pipeline  = Pipeline([('binarizer', binarizer), ('classifier', optimal_clf)])
        optimal_scores    = cross_validate(optimal_pipeline, X, y, cv=cv, scoring=scoring_metrics)
        optimal_f1_raw    = cross_validate(optimal_pipeline, X, y, cv=cv, scoring='f1_macro')['test_score']

        result_key_optimal          = f'Optimal (STreeD {strategy_name})'
        results[result_key_optimal] = {
            'Accuracy':         np.mean(optimal_scores['test_accuracy']),
            'F1':               np.mean(optimal_scores['test_f1_macro']),
            'F1 Std Dev':       np.std(optimal_scores['test_f1_macro']),
            'Precision':        np.mean(optimal_scores['test_precision_macro']),
            'Recall':           np.mean(optimal_scores['test_recall_macro']),
            'Fit Time (s)':     np.mean(optimal_scores['fit_time']),
            'Fit Time Std Dev': np.std(optimal_scores['fit_time'])
        }
        fold_scores_f1[result_key_optimal]   = optimal_f1_raw
        fold_scores_time[result_key_optimal] = optimal_scores['fit_time']

    # ---- Tree Size Calculation (fit on full data) ----
    print(f"\nCalculating Tree Sizes for n_bins={n_bins}...")
    for strategy_name, binarizer in strategies.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_binarized = binarizer.fit_transform(X, y)

        greedy_bin_model                          = DecisionTreeClassifier(random_state=42).fit(X_binarized, y)
        leaf_counts[f'Greedy ({strategy_name})']  = greedy_bin_model.get_n_leaves()
        depths[f'Greedy ({strategy_name})']       = greedy_bin_model.get_depth()

        optimal_model = STreeDClassifier(max_depth=OPTIMAL_MAX_DEPTH).fit(X_binarized, y)
        try:
            tree_object    = optimal_model.get_tree()
            num_branching  = tree_object.get_num_branching_nodes()
            leaf_counts[f'Optimal (STreeD {strategy_name})'] = num_branching + 1
            depths[f'Optimal (STreeD {strategy_name})']      = OPTIMAL_MAX_DEPTH
        except Exception as e:
            leaf_counts[f'Optimal (STreeD {strategy_name})'] = "Error"
            depths[f'Optimal (STreeD {strategy_name})']      = 'Error'

    # ---- Results Table ----
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df['Tree Leaves'] = results_df.index.map(leaf_counts)
    results_df['Tree Depth']  = results_df.index.map(depths)

    column_order   = ['Accuracy', 'F1', 'F1 Std Dev', 'Precision', 'Recall',
                      'Tree Leaves', 'Tree Depth', 'Fit Time (s)', 'Fit Time Std Dev']
    final_columns  = [col for col in column_order if col in results_df.columns]
    results_df     = results_df[final_columns]

    pd.set_option('display.precision', 4)
    pd.set_option('display.width', 1000)
    print(f"\n--- FINAL RESULTS TABLE (n_bins={n_bins}) ---")
    print(results_df)

    output_csv = f"results/experiment_results_n_bins_{n_bins}.csv"
    results_df.to_csv(output_csv)
    print(f"\nSaved final results to {output_csv}")

    # ---- Statistical Significance Tests ----
    print(f"\n--- Performing Statistical Significance Tests (n_bins={n_bins}) ---")

    key_base = 'Greedy (Continuous)'
    key_gq   = 'Greedy (Quantile)'
    key_oq   = 'Optimal (STreeD Quantile)'

    print("--- F1-Score Statistical Analysis ---")
    if key_oq in fold_scores_f1 and key_base in fold_scores_f1:
        scores_oq   = fold_scores_f1[key_oq]
        scores_base = fold_scores_f1[key_base]
        t_stat, p_val = ttest_rel(scores_oq, scores_base)
        d_val = cohens_d_paired(scores_oq, scores_base)
        print(f"1. Optimal (Quantile) vs. Baseline (F1):")
        print(f"   Mean F1: {np.mean(scores_oq):.4f} vs {np.mean(scores_base):.4f} | p-value: {p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(Not Significant)'}")
        print(f"   Cohen's d: {d_val:.4f} (Effect Size)")

    if key_gq in fold_scores_f1 and key_base in fold_scores_f1:
        scores_gq   = fold_scores_f1[key_gq]
        scores_base = fold_scores_f1[key_base]
        t_stat, p_val = ttest_rel(scores_gq, scores_base)
        d_val = cohens_d_paired(scores_gq, scores_base)
        print(f"\n2. Greedy (Quantile) vs. Baseline (F1):")
        print(f"   Mean F1: {np.mean(scores_gq):.4f} vs {np.mean(scores_base):.4f} | p-value: {p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(Not Significant)'}")
        print(f"   Cohen's d: {d_val:.4f} (Effect Size)")

    if key_gq in fold_scores_f1 and key_oq in fold_scores_f1:
        scores_gq = fold_scores_f1[key_gq]
        scores_oq = fold_scores_f1[key_oq]
        t_stat, p_val = ttest_rel(scores_gq, scores_oq)
        d_val = cohens_d_paired(scores_gq, scores_oq)
        print(f"\n3. Greedy (Quantile) vs. Optimal (Quantile) (F1):")
        print(f"   Mean F1: {np.mean(scores_gq):.4f} vs {np.mean(scores_oq):.4f} | p-value: {p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(Not Significant)'}")
        print(f"   Cohen's d: {d_val:.4f} (Effect Size)")

    print("\n--- Fit Time (s) Statistical Analysis ---")
    if key_oq in fold_scores_time and key_base in fold_scores_time:
        times_oq   = fold_scores_time[key_oq]
        times_base = fold_scores_time[key_base]
        t_stat, p_val = ttest_rel(times_oq, times_base)
        d_val = cohens_d_paired(times_oq, times_base)
        print(f"1. Optimal (Quantile) vs. Baseline (Time):")
        print(f"   Mean Time: {np.mean(times_oq):.4f}s vs {np.mean(times_base):.4f}s | p-value: {p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(Not Significant)'}")
        print(f"   Cohen's d: {d_val:.4f} (Effect Size)")

    if key_gq in fold_scores_time and key_base in fold_scores_time:
        times_gq   = fold_scores_time[key_gq]
        times_base = fold_scores_time[key_base]
        t_stat, p_val = ttest_rel(times_gq, times_base)
        d_val = cohens_d_paired(times_gq, times_base)
        print(f"\n2. Greedy (Quantile) vs. Baseline (Time):")
        print(f"   Mean Time: {np.mean(times_gq):.4f}s vs {np.mean(times_base):.4f}s | p-value: {p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(Not Significant)'}")
        print(f"   Cohen's d: {d_val:.4f} (Effect Size)")

    if key_gq in fold_scores_time and key_oq in fold_scores_time:
        times_gq = fold_scores_time[key_gq]
        times_oq = fold_scores_time[key_oq]
        t_stat, p_val = ttest_rel(times_oq, times_gq)
        d_val = cohens_d_paired(times_oq, times_gq)
        print(f"\n3. Optimal (Quantile) vs. Greedy (Quantile) (Time):")
        print(f"   Mean Time: {np.mean(times_oq):.4f}s vs {np.mean(times_gq):.4f}s | p-value: {p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(Not Significant)'}")
        print(f"   Cohen's d: {d_val:.4f} (Effect Size)")

    print(f"--- END OF ANALYSIS FOR N_BINS = {n_bins} ---\n\n")

# --- 6. Visualization (using n_bins=5 and Quantile strategy as example) ---


print("\n--- Generating Visualization Data (for n_bins=5 example) ---")
N_BINS_VIS    = 5
strategies_vis = {
    'Quantile': KBinsDiscretizer(n_bins=N_BINS_VIS, encode='ordinal', strategy='quantile', subsample=None)
}

greedy_baseline_size = baseline_leaves
print(f"Greedy (Continuous) Tree Leaves: {greedy_baseline_size}")
plt.figure(figsize=(20, 10))
plot_tree(greedy_baseline_model, filled=True, rounded=True,
          class_names=class_names, feature_names=feature_names,
          max_depth=3, fontsize=10)
plt.title(f"Greedy (Continuous) Baseline Tree [Showing 3 of {greedy_baseline_size} leaves]")
plt.savefig("figures/greedy_baseline_tree.png")
print("Saved figures/greedy_baseline_tree.png")
plt.close()

strategy_vis_name = 'Quantile'
binarizer_vis     = strategies_vis[strategy_vis_name]
X_binarized_vis   = binarizer_vis.fit_transform(X, y)
bin_feature_names = []
try:
    binarizer_vis.fit(X, y)
    for i, edges in enumerate(binarizer_vis.bin_edges_):
        feat_name = feature_names[i]
        for edge_idx in range(N_BINS_VIS - 1):
            if edge_idx + 1 < len(edges) - 1:
                edge_val = edges[edge_idx + 1]
                bin_feature_names.append(f"{feat_name} <= {edge_val:.2f}")
            else:
                bin_feature_names.append(f"{feat_name}_bin{edge_idx}")
except AttributeError:
    bin_feature_names = [f"Binarized_Feat_{i}" for i in range(X_binarized_vis.shape[1])]
    print("Warning: Could not generate detailed bin feature names.")

optimal_model_vis        = STreeDClassifier(max_depth=OPTIMAL_MAX_DEPTH).fit(X_binarized_vis, y)
optimal_tree_structure_str = "Error getting tree string"
try:
    if hasattr(optimal_model_vis, 'tree_to_str'):
        optimal_tree_structure_str = optimal_model_vis.tree_to_str(feature_names=bin_feature_names)
    else:
        optimal_tree_structure_str = str(optimal_model_vis.get_tree())
except Exception as e:
    optimal_tree_structure_str = f"Error getting tree string: {e}"

key_vis_leaf  = f'Optimal (STreeD {strategy_vis_name})'
key_vis_depth = f'Optimal (STreeD {strategy_vis_name})'

binarizer_5   = KBinsDiscretizer(n_bins=N_BINS_VIS, encode='ordinal', strategy='quantile', subsample=None)
X_bin_5       = binarizer_5.fit_transform(X, y)
optimal_model_5     = STreeDClassifier(max_depth=OPTIMAL_MAX_DEPTH).fit(X_bin_5, y)
tree_obj_5          = optimal_model_5.get_tree()
optimal_leaf_count_vis = tree_obj_5.get_num_branching_nodes() + 1
optimal_depth_vis      = OPTIMAL_MAX_DEPTH

print(f"\nOptimal ({strategy_vis_name}, n_bins={N_BINS_VIS}) Tree Structure:\n{optimal_tree_structure_str}")
print(f"Optimal ({strategy_vis_name}, n_bins={N_BINS_VIS}) Tree Leaves: {optimal_leaf_count_vis}")
print(f"Optimal ({strategy_vis_name}, n_bins={N_BINS_VIS}) Tree Depth: {optimal_depth_vis}")
