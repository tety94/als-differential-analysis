import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.nonparametric.smoothers_lowess import lowess
from utilities.utils import save_plot


# =====================================================================
# CONTINUOUS VARIABLES: dependence plot + threshold (max-slope point)
# =====================================================================

def find_threshold_continuous(x, shap_vals, frac=0.3, n_grid=200):
    """
    Fits a LOWESS curve (SHAP ~ x), computes the numerical derivative on a
    regular grid, and returns the point of maximum absolute slope: the
    point where the feature "shifts" the model the most.

    Parameters
    ----------
    x : array-like
        Values of the continuous feature
    shap_vals : array-like
        Corresponding SHAP values
    frac : float
        Fraction of points used by LOWESS for smoothing (lower = more
        wiggly curve, higher = smoother curve)
    n_grid : int
        Number of grid points used to evaluate the derivative

    Returns
    -------
    dict with: threshold, slope_at_threshold, x_grid, y_smooth, slope
    """
    # Convert to numpy immediately: x and shap_vals come from two different
    # DataFrames (X_model has its original, possibly non-contiguous, pandas
    # index; shap_df has a fresh default RangeIndex), so any pandas boolean
    # op here would align by label instead of by position and corrupt the
    # data. We only rely on positional (row-order) correspondence.
    x = np.asarray(x)
    shap_vals = np.asarray(shap_vals)

    mask = ~(np.isnan(x) | np.isnan(shap_vals))
    x = x[mask]
    shap_vals = shap_vals[mask]

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = shap_vals[order]

    smoothed = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
    x_smooth, y_smooth = smoothed[:, 0], smoothed[:, 1]

    # Regular grid for a more stable derivative (raw points aren't evenly spaced)
    x_grid = np.linspace(x_smooth.min(), x_smooth.max(), n_grid)
    y_grid = np.interp(x_grid, x_smooth, y_smooth)

    slope = np.gradient(y_grid, x_grid)
    idx_max = np.argmax(np.abs(slope))

    return {
        "threshold": x_grid[idx_max],
        "slope_at_threshold": slope[idx_max],
        "x_grid": x_grid,
        "y_smooth": y_grid,
        "slope": slope,
    }


def plot_continuous_dependence(feature_name, x, shap_vals, folder, frac=0.3):
    """
    Saves a dependence plot (raw SHAP points + LOWESS curve) with a
    vertical line at the max-slope threshold, and returns the numeric info
    about that threshold.
    """
    os.makedirs(folder, exist_ok=True)

    result = find_threshold_continuous(x, shap_vals, frac=frac)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, shap_vals, alpha=0.35, s=18, color="steelblue", label="Observations")
    ax.plot(result["x_grid"], result["y_smooth"], color="black", linewidth=2, label="Trend (LOWESS)")
    ax.axvline(result["threshold"], color="crimson", linestyle="--",
               label=f"Threshold ≈ {result['threshold']:.3g}")

    ax.set_xlabel(feature_name)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"Dependence plot - {feature_name}")
    ax.legend()
    plt.tight_layout()

    safe_name = feature_name.replace('/', '')
    save_plot(fig, os.path.join(folder, f"dependence_{safe_name}.png"))
    plt.close(fig)

    return {
        "feature": feature_name,
        "threshold": result["threshold"],
        "slope_at_threshold": result["slope_at_threshold"],
    }


# =====================================================================
# BINARY VARIABLES (YES/NO): mean SHAP comparison between the two groups
# =====================================================================

def plot_binary_dependence(feature_name, x, shap_vals, folder,
                            positive_label=None, negative_label=None):
    """
    Compares the SHAP value distribution between the two groups of a
    binary feature (e.g. yes/no, 1/0) with a boxplot + Mann-Whitney U test,
    and returns the difference between the means with its p-value.
    """
    os.makedirs(folder, exist_ok=True)

    # Same positional-alignment safeguard as find_threshold_continuous:
    # convert to numpy first, ignoring any pandas index mismatch between X
    # and shap_df.
    x_arr = np.asarray(x, dtype=object)
    shap_arr = np.asarray(shap_vals, dtype=float)

    is_na = pd.isna(x_arr) | pd.isna(shap_arr)
    mask = ~is_na
    x = pd.Series(x_arr[mask]).astype(str)
    shap_vals = shap_arr[mask]

    categories = sorted(x.unique())
    if len(categories) != 2:
        raise ValueError(
            f"'{feature_name}' has {len(categories)} categories, not 2: "
            f"use the multi-category function instead of plot_binary_dependence."
        )

    neg_cat, pos_cat = categories[0], categories[1]
    if negative_label is not None and positive_label is not None:
        neg_cat, pos_cat = negative_label, positive_label

    group_neg = shap_vals[x == neg_cat]
    group_pos = shap_vals[x == pos_cat]

    mean_neg, mean_pos = group_neg.mean(), group_pos.mean()
    delta = mean_pos - mean_neg

    stat, p_value = mannwhitneyu(group_pos, group_neg, alternative="two-sided")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot([group_neg, group_pos], labels=[str(neg_cat), str(pos_cat)])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"{feature_name}\nMean Δ = {delta:.3g}  (p = {p_value:.3g})")
    plt.tight_layout()

    safe_name = feature_name.replace('/', '')
    save_plot(fig, os.path.join(folder, f"binary_dependence_{safe_name}.png"))
    plt.close(fig)

    return {
        "feature": feature_name,
        "mean_no": mean_neg,
        "mean_yes": mean_pos,
        "delta": delta,
        "p_value": p_value,
    }


# =====================================================================
# MULTI-LEVEL CATEGORICAL VARIABLES (>2 categories): SHAP impact ranking
# =====================================================================

def plot_multicategory_dependence(feature_name, x, shap_vals, folder):
    """
    For a categorical feature with more than 2 levels: computes mean/std
    SHAP for each category, ranks them by impact (from the one pushing
    most towards one class to the most neutral/opposite one), and runs a
    Kruskal-Wallis test to assess whether the differences between groups
    are overall significant. Saves a boxplot ordered by category.
    """
    os.makedirs(folder, exist_ok=True)

    # Same positional-alignment safeguard as find_threshold_continuous:
    # convert to numpy first, ignoring any pandas index mismatch between X
    # and shap_df.
    x_arr = np.asarray(x, dtype=object)
    shap_arr = np.asarray(shap_vals, dtype=float)

    is_na = pd.isna(x_arr) | pd.isna(shap_arr)
    mask = ~is_na
    x = pd.Series(x_arr[mask]).astype(str)
    shap_vals = shap_arr[mask]

    categories = x.unique().tolist()

    groups = {cat: shap_vals[x == cat] for cat in categories}
    means = {cat: vals.mean() for cat, vals in groups.items()}

    # Rank categories by decreasing mean SHAP impact
    ranked = sorted(categories, key=lambda c: means[c], reverse=True)

    stat, p_value = kruskal(*groups.values())

    rows = []
    for cat in ranked:
        vals = groups[cat]
        rows.append({
            "feature": feature_name,
            "category": cat,
            "n": len(vals),
            "mean_shap": means[cat],
            "std_shap": vals.std(),
            "kruskal_p_value": p_value,
        })

    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(ranked)), 5))
    ax.boxplot([groups[cat] for cat in ranked], labels=[str(c) for c in ranked])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"{feature_name}\nKruskal-Wallis p = {p_value:.3g}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    safe_name = feature_name.replace('/', '')
    save_plot(fig, os.path.join(folder, f"multicategory_dependence_{safe_name}.png"))
    plt.close(fig)

    return rows


# =====================================================================
# ORCHESTRATION: generate everything from shap_values.csv + X
# =====================================================================

def generate_dependence_report(X, shap_df, numeric_cols, categorical_cols, folder, frac=0.3, log=print):
    """
    Generates dependence plots for all continuous features (with estimated
    threshold) and for all categorical features (binary or multi-level),
    also saving a summary CSV for each type.

    Parameters
    ----------
    X : pd.DataFrame
        Features used for training (same columns as shap_df)
    shap_df : pd.DataFrame
        SHAP values, same columns and row order as X
        (e.g. the csv produced by save_shap_values_csv)
    numeric_cols, categorical_cols : list
        Column lists as defined by the preprocessing pipeline
    folder : str
        Output folder for plots and CSVs
    frac : float
        LOWESS smoothing parameter for continuous features
    log : callable
        Logging function (defaults to print); progress and summary counts
        are sent here so they also land in the run's log file
    """
    os.makedirs(folder, exist_ok=True)

    if len(X) != len(shap_df):
        raise ValueError(
            f"X and shap_df have a different number of rows "
            f"({len(X)} vs {len(shap_df)}). They must come from the exact "
            f"same dataset in the exact same row order."
        )

    # Reset both indices to a shared 0..n-1 RangeIndex: X_model may carry a
    # non-contiguous pandas index inherited from earlier filtering steps,
    # while shap_df (built fresh from a numpy array) always has a default
    # RangeIndex. Without this, any pandas operation that aligns by label
    # instead of by position would silently corrupt the data.
    X = X.reset_index(drop=True)
    shap_df = shap_df.reset_index(drop=True)

    continuous_results = []
    binary_results = []
    multicategory_rows = []

    for col in numeric_cols:
        if col not in shap_df.columns:
            continue
        res = plot_continuous_dependence(col, X[col], shap_df[col], folder, frac=frac)
        continuous_results.append(res)

    for col in categorical_cols:
        if col not in shap_df.columns:
            continue
        n_unique = X[col].dropna().astype(str).nunique()
        if n_unique == 2:
            res = plot_binary_dependence(col, X[col], shap_df[col], folder)
            binary_results.append(res)
        elif n_unique > 2:
            rows = plot_multicategory_dependence(col, X[col], shap_df[col], folder)
            multicategory_rows.extend(rows)

    if continuous_results:
        pd.DataFrame(continuous_results).to_csv(
            os.path.join(folder, "thresholds_continuous.csv"), index=False
        )
    if binary_results:
        pd.DataFrame(binary_results).sort_values("delta", key=abs, ascending=False).to_csv(
            os.path.join(folder, "thresholds_binary.csv"), index=False
        )
    if multicategory_rows:
        pd.DataFrame(multicategory_rows).to_csv(
            os.path.join(folder, "thresholds_multicategory.csv"), index=False
        )

    log(f"[dependence_plots] Done: {len(continuous_results)} continuous, "
        f"{len(binary_results)} binary, "
        f"{pd.DataFrame(multicategory_rows)['feature'].nunique() if multicategory_rows else 0} "
        f"multi-category features processed.")

    return {
        "continuous": pd.DataFrame(continuous_results),
        "binary": pd.DataFrame(binary_results),
        "multicategory": pd.DataFrame(multicategory_rows),
    }
