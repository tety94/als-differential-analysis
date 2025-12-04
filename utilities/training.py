import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from utilities.utils import save_plot
from sqlalchemy.orm import sessionmaker
from website.models import Model
from website.db_connection import engine
from catboost import CatBoostClassifier
from utilities.CatBoostWrapper import CatBoostWrapper
from config import top_n_features, n_splits
from utilities.shap import generate_shap_plots


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
    # CatBoost: indices delle colonne categoriali
    # ==========================================================
    cat_features_idx = [X.columns.get_loc(c) for c in categorical_cols]

    model_name = "CatBoost"
    log(f"\n--- Modello: {model_name} ---")
    print(f"\n--- Modello: {model_name} ---")

    X_model = X.copy()

    # ==========================================================
    # Definizione modello CatBoost
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
    start_time = time.time()
    scores = cross_validate(
        model,
        X_model,
        y,
        cv=skf,
        scoring=['accuracy', 'f1'],
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
    log(f"⏱ Tempo esecuzione {model_name}: {elapsed_time:.1f} sec")

    acc_mean = scores['test_accuracy'].mean()
    f1_mean = scores['test_f1'].mean()
    log(f"Accuracy CV: {acc_mean:.4f} | F1 CV: {f1_mean:.4f}")

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
    # Fit finale
    # ==========================================================
    model.fit(X_model, y, cat_features=cat_features_idx)

    # ==========================================================
    # Salvataggio Feature Importance
    # ==========================================================
    save_catboost_feature_importances(
        model=model,
        feature_names=X_model.columns,
        model_name=model_name,
        folder=folder,
        top_n=top_n_features
    )

    generate_shap_plots(model.model_, X_model, cat_features_idx, folder=folder)

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

    results[model_name] = {
        'accuracy': acc_mean,
        'f1': f1_mean,
        'auc': roc_auc,
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
