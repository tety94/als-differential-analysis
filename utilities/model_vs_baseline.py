import pandas as pd
import logging
import joblib
import os
from website.models import Model
from website.db_connection import engine
from sqlalchemy.orm import sessionmaker


def compare_models_to_baseline(
        model_results,
        baseline,
        trained_pipelines,
        output_folder=None,
        key_metric='f1',
        model_output_folder=None
):
    """
    Compares the models against the baseline and saves the models that
    beat it.

    Parameters
    -model_results : dict or pd.DataFrame
        Model results from training (accuracy, f1, auc, precision, recall..)
    baseline : dict
        Baseline metrics for each visit
    trained_pipelines : dict
        Already-trained sklearn pipelines (model_name → pipeline)
    output_folder : str
        Folder to save the final CSV
    key_metric : str
        Main metric used to decide whether to save the model
    model_output_folder : str
        Folder where to save the .joblib models


    Returns
    -------
    pd.DataFrame
        Full table comparing model vs baseline
    """

    logging.info("🔍 Starting model vs baseline comparison")

    # Convert dict → DataFrame if needed
    if isinstance(model_results, dict):
        model_results = pd.DataFrame.from_dict(model_results, orient="index")

    records = []

    # Open SQL session
    Session = sessionmaker(bind=engine)

    for visit, baseline_metrics in baseline.items():

        print(f"\n===== Comparison against baseline: {visit} =====")
        logging.info(f"Comparison against baseline: {visit}")

        for model_name, metrics in model_results.iterrows():

            comparison = {
                metric: metrics.get(metric, float("-inf")) > baseline_metrics.get(metric, float("-inf"))
                for metric in baseline_metrics.keys()
            }

            # Log performance results
            print(f"\n--- Model: {model_name} ---")
            logging.info(f"--- Model: {model_name} ---")

            for metric, beat in comparison.items():
                status = "✅ beats" if beat else "❌ does NOT beat"
                print(f"{metric}: {metrics[metric]:.4f} vs baseline {baseline_metrics[metric]:.4f} -> {status}")
                logging.info(f"{metric}: {metrics[metric]:.4f} vs baseline {baseline_metrics[metric]:.4f} -> {status}")

            # Record for the final CSV
            records.append({
                "visit": visit,
                "model": model_name,
                **metrics.to_dict(),
                **{f"{k}_beat_baseline": v for k, v in comparison.items()}
            })

            # 🎯 Save the model if it beats the baseline on the key metric
            if (
                trained_pipelines is not None and
                model_name in trained_pipelines and
                comparison.get(key_metric, False)
            ):
                if model_output_folder is not None:

                    os.makedirs(model_output_folder, exist_ok=True)

                    # Retrieve the latest model version
                    session = Session()
                    last_model = (
                        session.query(Model)
                        .filter(Model.name == model_name)
                        .order_by(Model.id.desc())
                        .first()
                    )
                    session.close()

                    version = last_model.version if last_model else "v1"

                    model_path = os.path.join(model_output_folder, f"{model_name}_{version}.joblib")

                    # Save the model
                    joblib.dump(trained_pipelines[model_name], model_path)

                    print(f"💾 Model {model_name} saved to {model_path}")
                    logging.info(f"Model {model_name} saved to {model_path}")

    # 🧾 Final DataFrame
    comparison_df = pd.DataFrame(records)

    if output_folder:
        csv_path = os.path.join(output_folder, "model_vs_baseline.csv")
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n💾 Model vs baseline comparison saved to {csv_path}")
        logging.info(f"Model vs baseline comparison saved to {csv_path}")

    return comparison_df
