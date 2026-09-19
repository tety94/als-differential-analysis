import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from utilities.utils import save_plot
from sqlalchemy.orm import sessionmaker
from website.models import Model
from website.db_connection import engine
from utilities.CatBoostWrapper import CatBoostWrapper
from config import top_n_features, n_splits, metrics
from utilities.shap import generate_shap_plots, save_shap_values_csv
from utilities.dependence_plots import generate_dependence_report
from utilities.calibration import plot_calibration
from utilities.feature_ablation import feature_ablation_curve, feature_addition_curve
import json


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

def save_catboost_feature_importances(model, feature_names, model_name, folder, top_n=20):
    fi = pd.Series(model.get_feature_importance(), index=feature_names)
    fi_sorted = fi.sort_values(ascending=False)

    # --- Save CSV ---
    fi_sorted.to_csv(os.path.join(folder, f'feature_importances_{model_name}.csv'))

    # --- Save PNG image ---
    top_vals = fi_sorted.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_vals[::-1].plot(kind='barh', ax=ax)
    ax.set_title(f'Feature Importance - {model_name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()

    save_plot(fig, os.path.join(folder, f'feature_importance_{model_name}.png'))
    plt.close(fig)

    return fi_sorted


# =====================================================================
# VERSIONING
# =====================================================================

def create_new_version(numeric_cols, categorical_cols, model_name, model, model_type):
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        last_model = session.query(Model) \
            .filter(Model.name == model_name) \
            .filter(Model.type == model_type) \
            .order_by(Model.id.desc()) \
            .first()

        if last_model:
            last_version_num = int(last_model.version.strip('v'))
            new_version = f"v{last_version_num + 1}"
        else:
            new_version = "v1"

        params = model.get_params()

        new_model = Model(
            name=model_name,
            version=new_version,
            params={"numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols,
                    "models_dict": params},
            type=model_type
        )

        session.add(new_model)
        session.commit()

    finally:
        session.close()


# =====================================================================
# TRAINING SOLO CATBOOST
# =====================================================================
def train_models(log, model_type, X, y, numeric_cols, categorical_cols, folder):
    results = {}
    trained_pipelines = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ==========================================================
    # CatBoost: categorical column indices
    # ==========================================================
    cat_features_idx = [X.columns.get_loc(c) for c in categorical_cols]

    model_name = "CatBoost"
    log(f"\n--- Model: {model_name} ---")

    X_model = X.copy()

    # ==========================================================
    # CatBoost model definition
    # ==========================================================
    model = CatBoostWrapper(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        cat_features=cat_features_idx
    )

    # ==========================================================
    # Cross-validation
    # ==========================================================
    metric = metrics[model_type]

    start_time = time.time()
    scores = cross_validate(
        model,
        X_model,
        y,
        cv=skf,
        scoring=metric,
        n_jobs=-1
    )

    y_pred = cross_val_predict(
        model,
        X_model,
        y,
        cv=skf,
        method='predict',
        n_jobs=-1
    )

    elapsed_time = time.time() - start_time
    log(f"[timing] {model_name} run time: {elapsed_time:.1f} sec")

    report = classification_report(y, y_pred, output_dict=True)

    log(report)

    # ==========================================================
    # Confusion Matrix
    # ==========================================================
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title(f'Confusion Matrix - {model_name}')
    save_plot(fig, os.path.join(folder, f'confusion_matrix_{model_name}.png'))
    plt.close(fig)

    # ==========================================================
    # ROC + AUC
    # ==========================================================
    y_proba = cross_val_predict(
        model,
        X_model,
        y,
        cv=skf,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]

    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

    save_plot(fig, os.path.join(folder, f'roc_curve_{model_name}.png'))
    plt.close(fig)

    # ==========================================================
    # Precision-Recall curve
    # (more informative than ROC when classes are imbalanced)
    # ==========================================================
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_proba)
    avg_precision = average_precision_score(y, y_proba)

    fig, ax = plt.subplots()
    ax.plot(recall_curve, precision_curve, label=f'AP = {avg_precision:.3f}')
    baseline_rate = sum(y) / len(y)
    ax.axhline(baseline_rate, linestyle='--', color='gray', label=f'Baseline (prevalence) = {baseline_rate:.3f}')
    ax.set_title(f'Precision-Recall Curve - {model_name}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()

    save_plot(fig, os.path.join(folder, f'precision_recall_curve_{model_name}.png'))
    plt.close(fig)
    log(f"[metrics] {model_name}: AUC = {roc_auc:.3f}, Average Precision = {avg_precision:.3f}")

    # ==========================================================
    # Calibration (Brier score + ECE + reliability diagram)
    # ==========================================================
    calibration_scores = plot_calibration(y, y_proba, model_name=model_name, folder=folder, log=log)

    # ==========================================================
    # Final fit
    # ==========================================================
    model.fit(X_model, y, cat_features=cat_features_idx)
    log(f"[fit] {model_name} final fit on full dataset complete.")

    # ==========================================================
    # Feature importance
    # ==========================================================
    fi_sorted = save_catboost_feature_importances(
        model=model,
        feature_names=X_model.columns,
        model_name=model_name,
        folder=folder,
        top_n=top_n_features
    )
    log(f"[feature_importance] Top {top_n_features} features saved for {model_name}.")

    # ==========================================================
    # Sensitivity curve: progressive removal of the most important
    # features, with real retraining, to see where AUC stabilizes
    # ==========================================================
    feature_ablation_curve(
        X=X_model,
        y=y,
        categorical_cols=categorical_cols,
        feature_importance_ranked=fi_sorted,
        folder=folder,
        model_name=model_name,
        step=1,
        n_splits=n_splits,
        log=log
    )

    # ==========================================================
    # Complementary curve: adding the most important features one
    # group at a time, to see how few features are needed for a
    # stable AUC (parallelized the same way as the ablation curve)
    # ==========================================================
    feature_addition_curve(
        X=X_model,
        y=y,
        categorical_cols=categorical_cols,
        feature_importance_ranked=fi_sorted,
        folder=folder,
        model_name=model_name,
        step=1,
        n_splits=n_splits,
        log=log
    )

    generate_shap_plots(model.model_, X_model, cat_features_idx, folder=folder)
    shap_df = save_shap_values_csv(model.model_, X_model, output_path=f"{folder}/shap_values.csv")
    log(f"[shap] SHAP plots and values saved for {model_name}.")

    # ==========================================================
    # Save inputs for the standalone dependence-plot script
    # (avoids rerunning the whole training just to regenerate the plots)
    # ==========================================================
    X_model.to_csv(os.path.join(folder, "X_model.csv"), index=False)
    with open(os.path.join(folder, "feature_columns.json"), "w") as f:
        json.dump({"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}, f)

    generate_dependence_report(
        X=X_model,
        shap_df=shap_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        folder=folder,
        log=log
    )

    # ==========================================================
    # Salva modello e versione
    # ==========================================================
    trained_pipelines[model_name] = {
        "model": model,
        "feature_columns": X_model.columns.tolist()
    }

    precision_0 = report["0"]["precision"]
    recall_0 = report["0"]["recall"]
    precision_1 = report["1"]["precision"]
    recall_1 = report["1"]["recall"]

    # precision e recall globali (weighted)
    precision_global = report["weighted avg"]["precision"]
    recall_global = report["weighted avg"]["recall"]
    f1_global = report["weighted avg"]["f1-score"]
    accuracy_global = report["accuracy"]

    results[model_name] = {
        'accuracy': accuracy_global,
        'f1': f1_global,
        'auc': roc_auc,
        'average_precision': avg_precision,
        'brier_score': calibration_scores['brier_score'],
        'ece': calibration_scores['ece'],
        "precision_0": precision_0,
        "precision_1": precision_1,
        "recall_0": recall_0,
        "recall_1": recall_1,
        'precision': precision_global,
        'recall': recall_global,
    }

    create_new_version(numeric_cols, categorical_cols, model_name, model, model_type)

    # ==========================================================
    # Salva risultati
    # ==========================================================
    pd.DataFrame([results[model_name]]).to_csv(
        os.path.join(folder, 'catboost_results_summary.csv'),
        index=False
    )

    return results, trained_pipelines

