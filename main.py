import pandas as pd
from utilities.utils import init_logging
from config import target_col, test_folder, numerical_cols, categorical_columns, binary_cols
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

# 1. Caricamento dati
df = load_data()

target_cols = ["diagn_1_vis"]
baseline = compute_baseline_vs_final(df, target_cols, output_folder=f"{test_folder}")

# 2. Report valori null
missing_report = generate_missing_report(df, test_folder)

# 3. Preprocessing colonne
df = convert_plus_minus(df)
df, numeric_cols, categorical_cols = separate_columns(df,
                                                      forced_numerical=numerical_cols,
                                                      forced_categorical=categorical_columns,
                                                      binary_cols=binary_cols)

categorical_cols.remove(target_col)

# 4. Separazione X / y
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

results = correlation_analysis(X, y, output_folder=test_folder)
X, removed_cols, numeric_cols, categorical_cols = drop_strongly_correlated(X, results['strong_corrs'], categorical_cols, numeric_cols)
print(f"Colonne rimosse: {removed_cols}")

constant_cols = X.columns[X.nunique() <= 1].tolist()
if constant_cols:
    X.drop(columns=constant_cols, inplace=True)
    log(f"Colonne costanti rimosse: {constant_cols}")

print(f"Null report")
null_report = report_nulls(X)
null_report.to_csv(f'{test_folder}/null_report.csv')

# Imputazione dei null
print(f"Impute null")
X = impute_nulls(log, X, categorical_cols, numeric_cols)

# 5. Modelli
print(f"Train models")
models = get_models()
res_df, model_feature_importances, trained_pipelines = train_models(log, X, y, numeric_cols, categorical_cols, models, test_folder)

comparison_df = compare_models_to_baseline(res_df, baseline,trained_pipelines, output_folder=f"{test_folder}")

# 6. Grafici comparativi
print(f"Graphics")
plot_comparisons(res_df, test_folder)
common_features = common_features_plot(model_feature_importances, test_folder)

# 7. Salvataggio comuni
pd.DataFrame({'feature': common_features}).to_csv(f"{test_folder}/common_features.csv", index=False)
print(f"\n✅ Test completato! Tutti i risultati sono in: {test_folder}")
