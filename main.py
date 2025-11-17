import pandas as pd
import os
from utilities.utils import init_logging
from config import target_col, test_folder, features, t_1_visit, model_output_folder
from utilities.data_loader import load_data
from utilities.preprocess import convert_plus_minus, separate_columns
from utilities.missing_report import generate_missing_report
from utilities.models import get_models
from utilities.training import train_models
from utilities.visualize import plot_comparisons, common_features_plot
from utilities.null_handler import report_nulls, impute_nulls
from utilities.correlation_analysis import correlation_analysis, drop_strongly_correlated
from utilities.baseline_analysis import compute_baseline_vs_final
from utilities.model_vs_baseline import compare_models_to_baseline

log = init_logging(f"{test_folder}/run.log")
log(f"Avvio nuovo test in {test_folder}")

base_test_folder = test_folder
base_model_output_folder = model_output_folder

#ciclo per i vari sottomodelli
for k in features.keys():
    log('############################')
    log('############################')
    log(f'INIZIO MODELLO PER : {k}')
    log('############################')
    log('############################')

    numerical_cols = features[k]['numerical_cols']
    categorical_columns = features[k]['categorical_columns']
    binary_cols = features[k]['binary_cols']

    test_folder = os.path.join(base_test_folder, k)
    os.makedirs(test_folder, exist_ok=True)

    model_output_folder = os.path.join(base_model_output_folder, k)
    os.makedirs(model_output_folder, exist_ok=True)

    # 1. Caricamento dati
    df = load_data(categorical_columns,numerical_cols, binary_cols)

    baseline_cols = [t_1_visit]
    baseline = compute_baseline_vs_final(df, baseline_cols, output_folder=f"{test_folder}")

    # 2. Report valori null
    missing_report = generate_missing_report(df, test_folder)

    # 3. Preprocessing colonne
    df = convert_plus_minus(df)
    df, numeric_cols, categorical_cols = separate_columns(df,
                                                          forced_numerical=numerical_cols,
                                                          forced_categorical=categorical_columns,
                                                          binary_cols=binary_cols)

    categorical_cols.remove(target_col)
    categorical_cols.remove(t_1_visit) #rimuovo la colonna diagonsi prima visita

    # 4. Separazione X / y
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    X = X.drop(columns=baseline_cols) #rimuovo la colonna diagonsi prima visita

    results = correlation_analysis(X, y, output_folder=test_folder)
    X, removed_cols, numeric_cols, categorical_cols = drop_strongly_correlated(X, results['strong_corrs'], categorical_cols, numeric_cols)
    print(f"Colonne rimosse: {removed_cols}")

    constant_cols = X.columns[X.nunique() <= 1].tolist()
    if constant_cols:
        X.drop(columns=constant_cols, inplace=True)
        log(f"Colonne costanti rimosse: {constant_cols}")

    null_report = report_nulls(X)
    null_report.to_csv(f'{test_folder}/null_report.csv')

    # Imputazione dei null
    X = impute_nulls(log, X, categorical_cols, numeric_cols)

    # 5. Modelli
    models = get_models()
    res_df, model_feature_importances, trained_pipelines = train_models(log, X, y, numeric_cols, categorical_cols, models, test_folder)

    comparison_df = compare_models_to_baseline(res_df, baseline,trained_pipelines, output_folder=f"{test_folder}", model_output_folder=model_output_folder)

    # 6. Grafici comparativi
    plot_comparisons(res_df, test_folder)
    common_features = common_features_plot(model_feature_importances, test_folder)

    # 7. Salvataggio comuni
    pd.DataFrame({'feature': common_features}).to_csv(f"{test_folder}/common_features.csv", index=False)

    log('############################')
    log('############################')
    log(f'FINE MODELLO PER : {k}')
    log('############################')
    log('############################')

log(f"\n✅ Test completato! Tutti i risultati sono in: {test_folder}")
print(f"\n✅ Test completato! Tutti i risultati sono in: {test_folder}")
