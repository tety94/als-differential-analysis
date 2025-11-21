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
    Confronta i modelli con la baseline e salva i modelli che la superano.

    Parameters
    ----------
    model_results : dict or pd.DataFrame
        Risultati dei modelli dal training (accuracy, f1, auc, precision, recall..)
    baseline : dict
        Metriche della baseline per ogni visit
    trained_pipelines : dict
        Pipeline sklearn già addestrate (model_name → pipeline)
    output_folder : str
        Cartella per salvare il CSV finale
    key_metric : str
        Metrica principale per decidere se salvare il modello
    model_output_folder : str
        Cartella dove salvare i modelli .joblib

    Returns
    -------
    pd.DataFrame
        Tabella completa con confronti modello vs baseline
    """

    logging.info("🔍 Avvio confronto modelli vs baseline")

    # Converti dict → DataFrame se necessario
    if isinstance(model_results, dict):
        model_results = pd.DataFrame.from_dict(model_results, orient="index")

    records = []

    # Apertura sessione SQL
    Session = sessionmaker(bind=engine)

    for visit, baseline_metrics in baseline.items():

        print(f"\n===== Confronto con baseline: {visit} =====")
        logging.info(f"Confronto con baseline: {visit}")

        for model_name, metrics in model_results.iterrows():

            comparison = {
                metric: metrics.get(metric, float("-inf")) > baseline_metrics.get(metric, float("-inf"))
                for metric in baseline_metrics.keys()
            }

            # Log risultati prestazionali
            print(f"\n--- Modello: {model_name} ---")
            logging.info(f"--- Modello: {model_name} ---")

            for metric, beat in comparison.items():
                status = "✅ supera" if beat else "❌ NON supera"
                print(f"{metric}: {metrics[metric]:.4f} vs baseline {baseline_metrics[metric]:.4f} -> {status}")
                logging.info(f"{metric}: {metrics[metric]:.4f} vs baseline {baseline_metrics[metric]:.4f} -> {status}")

            # Record per CSV finale
            records.append({
                "visit": visit,
                "model": model_name,
                **metrics.to_dict(),
                **{f"{k}_beat_baseline": v for k, v in comparison.items()}
            })

            # 🎯 Salvataggio modello se supera baseline sulla metrica chiave
            if (
                trained_pipelines is not None and
                model_name in trained_pipelines and
                comparison.get(key_metric, False)
            ):
                if model_output_folder is not None:

                    os.makedirs(model_output_folder, exist_ok=True)

                    # Recupero ultima versione del modello
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

                    # Salvataggio del modello
                    joblib.dump(trained_pipelines[model_name], model_path)

                    print(f"💾 Modello {model_name} salvato in {model_path}")
                    logging.info(f"Modello {model_name} salvato in {model_path}")

    # 🧾 DataFrame finale
    comparison_df = pd.DataFrame(records)

    if output_folder:
        csv_path = os.path.join(output_folder, "model_vs_baseline.csv")
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n💾 Confronto modelli vs baseline salvato in {csv_path}")
        logging.info(f"Confronto modelli vs baseline salvato in {csv_path}")

    return comparison_df
