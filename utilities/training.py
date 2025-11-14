import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from utilities.utils import save_plot
from config import n_splits, top_n_features
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


def get_feature_importances(model, feature_names):
    """
    Restituisce una Series di importanze o coefficienti assoluti, se disponibili.
    """
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names)

    elif hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.ndim == 1:
            return pd.Series(np.abs(coefs), index=feature_names)
        else:
            # media assoluta tra le classi nel caso multiclasse
            return pd.Series(np.abs(coefs).mean(axis=0), index=feature_names)

    elif hasattr(model, "best_estimator_"):
        # in caso di GridSearchCV o simili
        return get_feature_importances(model.best_estimator_, feature_names)

    return None



def save_feature_importances(pipe, model_name, feature_names, folder, top_n=10):
    """
    Salva e disegna le feature importances o i coefficienti di un modello sklearn, se disponibili.
    """
    # Estrae il modello finale dal pipeline
    model = pipe.named_steps.get('clf', None)
    if model is None:
        print(f"[WARN] Nessun classificatore trovato nel pipeline per {model_name}")
        return None

    # Ottiene le feature importances o i coefficienti
    fi = get_feature_importances(model, feature_names)

    if fi is None:
        print(f"[WARN] Nessuna feature importance trovata per modello {model_name}")
        return None

    print(f"[INFO] Salvate feature importances per {model_name}")

    # Ordina e seleziona le top N
    fi_sorted = fi.sort_values(ascending=False).head(top_n)

    # Plot orizzontale
    fig, ax = plt.subplots(figsize=(8, 6))
    fi_sorted[::-1].plot(kind='barh', ax=ax)
    ax.set_title(f'Feature Importance - {model_name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()

    # Salva grafico e CSV
    save_plot(fig, os.path.join(folder, f'feature_importance_{model_name}.png'))
    fi_sorted.to_csv(os.path.join(folder, f'top_{top_n}_features_{model_name}.csv'))

    return fi_sorted



def train_models(log, X, y, numeric_cols, categorical_cols, models, folder):
    results = {}
    model_feature_importances = {}
    trained_pipelines = {}  # salviamo pipeline + colonne

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', num_pipeline, numeric_cols),
                                      ('cat', cat_pipeline, categorical_cols)])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


    for name, model in models.items():
        log(f"\n--- Modello: {name} ---")
        # print(f"\n--- Modello: {name} ---")
        pipe = Pipeline([('pre', preprocessor), ('clf', model)])

        start_time = time.time()

        # Cross-validation
        scores = cross_validate(pipe, X, y, cv=skf, scoring=['accuracy', 'f1'], n_jobs=-1)
        y_pred = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)

        elapsed_time = time.time() - start_time
        log(f"⏱ Tempo di esecuzione per {name}: {elapsed_time / 60:.2f} minuti ({elapsed_time:.1f} secondi)")

        acc_mean, f1_mean = scores['test_accuracy'].mean(), scores['test_f1'].mean()
        log(f"Accuracy CV: {acc_mean:.4f} | F1 CV: {f1_mean:.4f}")
        log(classification_report(y, y_pred, digits=4))

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        save_plot(fig, os.path.join(folder, f'confusion_matrix_{name}.png'))

        # ROC + Calibrazione
        roc_auc = float('nan')
        if hasattr(model, "predict_proba"):
            y_proba = cross_val_predict(pipe, X, y, cv=skf, method='predict_proba')[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')
            plot_calibration_curve(y, y_proba, name, folder)

        # Fit finale
        pipe.fit(X, y)

        # Feature importances
        feature_names_all = preprocessor.get_feature_names_out()
        fi_sorted = save_feature_importances(pipe, name, feature_names_all, folder, top_n_features)
        if fi_sorted is not None:
            model_feature_importances[name] = fi_sorted

        # estrai il modello finale già fit-tato
        base_clf = pipe.named_steps["clf"]

        # calibra su X, y (o su un validation split separato)
        calibrated = CalibratedClassifierCV(base_clf, method="isotonic", cv='prefit')
        calibrated.fit(pipe.named_steps['pre'].transform(X), y)

        # crea pipeline finale con il modello calibrato
        final_pipe = Pipeline([
            ('pre', preprocessor),
            ('clf', calibrated)
        ])

        # proba = final_pipe.predict_proba(X.iloc[[0]])
        # print('proba')
        # print(proba)
        # exit()


        # Salvataggio pipeline + colonne
        trained_pipelines[name] = {
            "pipeline": final_pipe,
            "feature_columns": X.columns.tolist()
        }

        results[name] = {'accuracy': acc_mean, 'f1': f1_mean, 'auc': roc_auc}

    # ROC comparativa
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    save_plot(plt.gcf(), os.path.join(folder, 'roc_curves.png'))

    # Salvataggio risultati
    res_df = pd.DataFrame.from_dict(results, orient='index')
    res_df.to_csv(os.path.join(folder, 'model_results_summary.csv'))

    return res_df, model_feature_importances, trained_pipelines


def plot_calibration_curve(y_true, y_prob, name, folder):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker='o', label=name)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Probabilità predetta')
    plt.ylabel('Frazione reale di positivi')
    plt.title(f'Calibrazione - {name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'calibration_{name}.png'))
    plt.close()
