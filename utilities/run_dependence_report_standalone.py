"""
Regenerates dependence plots (continuous, binary yes/no, multi-level
categorical) from files already saved by a previous training run, without
having to rerun the whole training.

Requires the following files to already exist in the folder (saved
automatically by train_models in training.py):
    - X_model.csv
    - shap_values.csv
    - feature_columns.json

Usage:
    python run_dependence_report_standalone.py /path/to/results/folder
"""

import os
import sys
import json
import pandas as pd

from utilities.dependence_plots import generate_dependence_report


def run(folder, frac=0.3):
    x_path = os.path.join(folder, "X_model.csv")
    shap_path = os.path.join(folder, "shap_values.csv")
    cols_path = os.path.join(folder, "feature_columns.json")

    for path in (x_path, shap_path, cols_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                f"This script requires X_model.csv, shap_values.csv and "
                f"feature_columns.json already saved by a previous training run."
            )

    X = pd.read_csv(x_path)
    shap_df = pd.read_csv(shap_path)

    with open(cols_path) as f:
        cols = json.load(f)

    result = generate_dependence_report(
        X=X,
        shap_df=shap_df,
        numeric_cols=cols["numeric_cols"],
        categorical_cols=cols["categorical_cols"],
        folder=folder,
        frac=frac,
    )

    print(f"Dependence plots regenerated in: {folder}")
    if not result["continuous"].empty:
        print(f"  - {len(result['continuous'])} continuous features")
    if not result["binary"].empty:
        print(f"  - {len(result['binary'])} binary (yes/no) features")
    if not result["multicategory"].empty:
        n_feat = result["multicategory"]["feature"].nunique()
        print(f"  - {n_feat} multi-level categorical features")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_dependence_report_standalone.py /path/to/folder [frac]")
        sys.exit(1)

    folder_arg = sys.argv[1]
    frac_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

    run(folder_arg, frac=frac_arg)
