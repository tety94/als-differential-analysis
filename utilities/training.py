import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from utilities.utils import save_plot
from config import n_splits, top_n_features
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from website.models import Model
from website.db_connection import engine
from sqlalchemy.orm import sessionmaker


# ======================================================
# FEATURE IMPORTANCE
# ======================================================

def get_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names)

    elif hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.ndim == 1:
            return pd.Series(np.abs(coefs), index=feature_names)
        else:
            return pd.Series(np.abs(coefs).mean(axis=0), index=feature_names)

    elif hasattr(model, "best_estimator_"):
        return get_feature_importances(model.best_estimator_, feature_names)

    return None


def save_feature_importances(pipe, model_name, feature_names, folder, top_n=10):
    model = pipe.named_steps.get('clf', None)
    if model is None:
        return None

    fi = get_feature_importances(model, feature_names)
    if fi is None:
        return None

    fi_sorted = fi.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    fi_sorted[::-1].plot(kind='barh', ax=ax)
    ax.set_title(f'Feature Importance - {model_name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()

    save_plot(fig, os.path.join(folder, f'feature_importance_{model_name}.png'))
    fi_sorted.to_csv(os.path.join(folder, f'top_{top_n}_features_{model_name}.csv'))

    return fi_sorted


# ======================================================
# VERSIONING
# ======================================================

def create_new_version(numeric_cols, categorical_cols, model_name, model, model_type):
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        last_model = session.query(Model)\
            .filter(Model.name == model_name)\
            .filter(Model.type == model_type)\
            .order_by(Model.id.desc())\
            .first()
        if last_model:
            last_version_num = int(last_model.version.strip('v'))
            new_version = f"v{last_version_num + 1}"
        else:
            new_version = "v1"

        params = model.get_params()
        from website.routes.predict import clean
        params = clean(params)

        new_model = Model(
            name=model_name,
            version=new_version,
            params={"numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "models_dict": params},
            type=model_type
        )

        session.add(new_model)
        session.commit()
    finally:
        session.close()


# ======================================================
# TRAINING
# ======================================================

def train_models(log, model_type, X, y, numeric_cols, categorical_cols, models, folder):

    results = {}
    model_feature_importances = {}
    trained_pipelines = {}

    # ---------------------------
    # PREPROCESSING (no imputation)
    # ---------------------------
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # OrdinalEncoder mantiene i NaN nativi
    cat_pipeline = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, model in models.items():

        log(f"\n--- Modello: {name} ---")
        pipe = Pipeline([('pre', preprocessor), ('clf', model)])
        start_time = time.time()

        # Cross-validation
        scores = cross_validate(pipe, X, y, cv=skf, scoring=['accuracy', 'f1'], n_jobs=-1)
        y_pred = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)

        elapsed_time = time.time() - start_time
        log(f"⏱ Tempo esecuzione {name}: {elapsed_time:.1f} sec")

        acc_mean, f1_mean = scores['test_accuracy'].mean(), scores['test_f1'].mean()
        log(f"Accuracy CV: {acc_mean:.4f} | F1 CV: {f1_mean:.4f}")
        log(classification_report(y, y_pred, digits=4))

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title(f'Confusion Matrix - {name}')
        save_plot(fig, os.path.join(folder, f'confusion_matrix_{name}.png'))

        tn, fp, fn, tp = cm.ravel()
        log(f"Sensitivity_0: {tn/(tn+fp):.4f} | Specificity_0: {tn/(tn+fn):.4f}")

        roc_auc = float('nan')
        if hasattr(model, "predict_proba"):
            # Predizioni in cross-validation
            y_proba = cross_val_predict(
                pipe, X, y, cv=skf, method='predict_proba'
            )[:, 1]

            # ROC
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)

            # Plot ROC corretto
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], '--', color='gray')
            ax.set_title(f'ROC Curve - {name}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()

            # Salva SOLO la ROC corretta
            save_plot(fig, os.path.join(folder, f'roc_curve_{name}.png'))
            plt.close(fig)

        # FIT finale + calibrazione
        pipe.fit(X, y)
        fitted_pre = pipe.named_steps['pre']
        base_clf = pipe.named_steps['clf']
        X_transformed = fitted_pre.transform(X)

        calibrated = CalibratedClassifierCV(base_clf, method="isotonic")
        calibrated.fit(X_transformed, y)

        final_pipe = Pipeline([
            ('pre', fitted_pre),
            ('clf', calibrated)
        ])

        # FEATURE IMPORTANCE
        feature_names_all = fitted_pre.get_feature_names_out()
        fi_sorted = save_feature_importances(pipe, name, feature_names_all, folder, top_n_features)
        if fi_sorted is not None:
            model_feature_importances[name] = fi_sorted


        trained_pipelines[name] = {
            "pipeline": final_pipe,
            "feature_columns": X.columns.tolist()
        }

        results[name] = {'accuracy': acc_mean, 'f1': f1_mean, 'auc': roc_auc}
        create_new_version(numeric_cols, categorical_cols, name, model, model_type)

    # SALVA risultati
    res_df = pd.DataFrame.from_dict(results, orient='index')
    res_df.to_csv(os.path.join(folder, 'model_results_summary.csv'))

    return res_df, model_feature_importances, trained_pipelines
