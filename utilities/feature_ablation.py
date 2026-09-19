import os
import time
import threading
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from utilities.utils import save_plot
from utilities.CatBoostWrapper import CatBoostWrapper

# Lock so log lines from concurrent threads don't interleave mid-line
_log_lock = threading.Lock()


def _evaluate_step(X, y, categorical_cols, skf, n_removed, removed, total_steps, log):
    """
    Runs a single ablation step: retrain + CV-evaluate the model after
    removing `removed` (the top `n_removed` most important features).

    Logs immediately when the step starts and when it finishes, so progress
    is visible in real time even though steps run concurrently. Runs on a
    thread (see feature_ablation_curve), not a separate process, so log/
    print calls are shared with the main process and appear right away.
    Internal cross-validation is forced to n_jobs=1 to avoid oversubscribing
    CPU cores on top of the outer thread pool.
    """
    remaining_cols = [c for c in X.columns if c not in removed]

    if len(remaining_cols) < 2:
        return None

    with _log_lock:
        log(f"  [ablation] starting: removing top {n_removed}/{total_steps - 1} features "
            f"({len(remaining_cols)} remaining)...")

    t0 = time.time()

    X_step = X[remaining_cols]
    cat_features_idx = [X_step.columns.get_loc(c) for c in categorical_cols if c in remaining_cols]

    model = CatBoostWrapper(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        cat_features=cat_features_idx
    )

    y_proba = cross_val_predict(
        model, X_step, y, cv=skf, method='predict_proba', n_jobs=1
    )[:, 1]

    auc_score = roc_auc_score(y, y_proba)
    elapsed = time.time() - t0

    with _log_lock:
        log(f"  [ablation] done: removed {n_removed}/{total_steps - 1} top features -> "
            f"{len(remaining_cols)} remaining, AUC = {auc_score:.3f} ({elapsed:.1f}s)")

    return {
        "n_features_removed": n_removed,
        "n_features_remaining": len(remaining_cols),
        "auc": auc_score,
        "removed_features": ", ".join(removed) if removed else "(none)",
    }


def _evaluate_step_addition(X, y, categorical_cols, skf, n_used, keep, total_features, log):
    """
    Runs a single feature-addition step: retrain + CV-evaluate the model
    using only `keep` (the top `n_used` most important features).
    Same real-time logging and threading contract as _evaluate_step.
    """
    used_cols = [c for c in X.columns if c in keep]

    if len(used_cols) < 2:
        return None

    with _log_lock:
        log(f"  [addition] starting: using top {n_used}/{total_features} features...")

    t0 = time.time()

    X_step = X[used_cols]
    cat_features_idx = [X_step.columns.get_loc(c) for c in categorical_cols if c in used_cols]

    model = CatBoostWrapper(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        cat_features=cat_features_idx
    )

    y_proba = cross_val_predict(
        model, X_step, y, cv=skf, method='predict_proba', n_jobs=1
    )[:, 1]

    auc_score = roc_auc_score(y, y_proba)
    elapsed = time.time() - t0

    with _log_lock:
        log(f"  [addition] done: top {n_used}/{total_features} features -> "
            f"AUC = {auc_score:.3f} ({elapsed:.1f}s)")

    return {
        "n_features_used": n_used,
        "auc": auc_score,
        "features_used": ", ".join(used_cols),
    }


def feature_addition_curve(X, y, categorical_cols, feature_importance_ranked,
                            folder, model_name="CatBoost",
                            step=1, n_splits=5, random_state=42,
                            n_jobs=-1, log=print):
    """
    Complementary to feature_ablation_curve: instead of removing features,
    builds the model from scratch adding the top-N most important features
    one group at a time (top-1, top-2, top-3, ...) to see how few features
    are needed to reach stable AUC.

    Each point (top-k features) is independent of the others since the
    ranking is already known, so steps are evaluated in parallel using a
    thread pool (see the threading note in feature_ablation_curve), with
    the same n_jobs=1 safeguard inside each step to avoid nested
    parallelism, and real-time logging as each step starts/finishes.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature set used in the original training
    y : array-like
        Target
    categorical_cols : list
        Categorical column names (indices recomputed at each step)
    feature_importance_ranked : list or pd.Series
        Feature names ordered by decreasing importance
    folder : str
        Output folder
    step : int
        How many features to add at each iteration
    n_splits : int
        Folds for the cross-validation at each step
    n_jobs : int
        Number of parallel workers across steps (-1 = use all available cores)
    log : callable
        Logging function (defaults to print)

    Returns
    -------
    pd.DataFrame with columns: n_features_used, auc, features_used
    """
    os.makedirs(folder, exist_ok=True)

    if hasattr(feature_importance_ranked, "index"):
        ranked_features = list(feature_importance_ranked.index)
    else:
        ranked_features = list(feature_importance_ranked)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_total = len(ranked_features)
    steps_to_run = list(range(2, n_total + 1, step))
    if n_total not in steps_to_run:
        steps_to_run.append(n_total)

    log(f"[addition] Running {len(steps_to_run)} steps (n_jobs={n_jobs}). "
        f"Progress will be logged as each step completes...")

    # backend="threading": CatBoost's C++ core releases the GIL during fit,
    # so threads still run truly in parallel; using threads (not separate
    # processes) means log/print calls below are visible immediately in
    # the main process/console instead of being buffered until all steps
    # finish, and avoids Windows' per-process script re-import overhead.
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_evaluate_step_addition)(X, y, categorical_cols, skf, n_used, ranked_features[:n_used], n_total, log)
        for n_used in steps_to_run
    )

    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows).sort_values("n_features_used").reset_index(drop=True)

    df.to_csv(os.path.join(folder, f"feature_addition_curve_{model_name}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_features_used"], df["auc"], marker='o', color='seagreen')
    ax.set_xlabel("Number of (most important) features used")
    ax.set_ylabel("AUC (cross-validated)")
    ax.set_title(f"Feature addition sensitivity curve - {model_name}")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    save_plot(fig, os.path.join(folder, f"feature_addition_curve_{model_name}.png"))
    plt.close(fig)

    log(f"[addition] Done. Best AUC = {df['auc'].max():.3f} at "
        f"{df.loc[df['auc'].idxmax(), 'n_features_used']} features used.")

    return df


def feature_ablation_curve(X, y, categorical_cols, feature_importance_ranked,
                            folder, model_name="CatBoost",
                            step=1, n_splits=5, random_state=42,
                            n_jobs=-1, log=print):
    """
    Retrain the model repeatedly, progressively removing the top-N most
    important features, to see when AUC stabilizes. Each point on the curve
    is a model retrained from scratch with CV, after removing the top
    features accumulated up to that point.

    Steps run in parallel using a thread pool rather than separate
    processes: CatBoost's underlying C++ training releases Python's GIL,
    so threads still achieve real parallelism, while avoiding two
    downsides of process-based parallelism (joblib's default "loky"
    backend) that matter here: on Windows every new process re-imports the
    calling script, which is slow and, without an `if __name__ ==
    "__main__":` guard in the caller, can hang or re-run the whole
    pipeline; and log/print output from separate processes doesn't reach
    the main console/log file until the process finishes, hiding progress
    for the whole duration of a run. With threads, each step's log lines
    appear as soon as that step starts/finishes. Each step's internal
    cross-validation is still forced to run single-threaded (n_jobs=1) to
    avoid oversubscribing CPU cores.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature set used in the original training
    y : array-like
        Target
    categorical_cols : list
        Categorical column names (indices are recomputed at each step since
        the remaining columns change)
    feature_importance_ranked : list or pd.Series
        Feature names ordered by decreasing importance (e.g. the index of
        fi_sorted from save_catboost_feature_importances)
    folder : str
        Output folder
    step : int
        How many features to remove at each iteration (1 = one at a time;
        increase if there are many important features, to keep runtime down)
    n_splits : int
        Folds for the cross-validation at each step (a single CV, not
        repeated across seeds, to keep runtime down)
    n_jobs : int
        Number of parallel workers across ablation steps (-1 = use all
        available cores)
    log : callable
        Logging function (defaults to print); each step's progress is sent
        here so it also lands in the run's log file, not just the console

    Returns
    -------
    pd.DataFrame with columns: n_features_removed, n_features_remaining, auc,
    removed_features (cumulative list removed at that step)
    """
    os.makedirs(folder, exist_ok=True)

    if hasattr(feature_importance_ranked, "index"):
        ranked_features = list(feature_importance_ranked.index)
    else:
        ranked_features = list(feature_importance_ranked)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_total = len(ranked_features)
    steps_to_run = list(range(0, n_total - 1, step))

    log(f"[ablation] Running {len(steps_to_run)} steps (n_jobs={n_jobs}). "
        f"Progress will be logged as each step completes...")

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_evaluate_step)(X, y, categorical_cols, skf, n_removed, ranked_features[:n_removed], n_total, log)
        for n_removed in steps_to_run
    )

    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows).sort_values("n_features_removed").reset_index(drop=True)

    df.to_csv(os.path.join(folder, f"feature_ablation_curve_{model_name}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_features_removed"], df["auc"], marker='o', color='steelblue')
    ax.set_xlabel("Number of (most important) features removed")
    ax.set_ylabel("AUC (cross-validated)")
    ax.set_title(f"Feature removal sensitivity curve - {model_name}")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    save_plot(fig, os.path.join(folder, f"feature_ablation_curve_{model_name}.png"))
    plt.close(fig)

    log(f"[ablation] Done. Best AUC = {df['auc'].max():.3f} at "
        f"{df.loc[df['auc'].idxmax(), 'n_features_remaining']} remaining features.")

    return df
