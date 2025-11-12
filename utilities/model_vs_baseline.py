import pandas as pd
import logging
import joblib
import os
from config import version, model_output_folder

def compare_models_to_baseline(model_results_df, baseline, trained_pipelines, output_folder=None, key_metric='f1'):
    """
    Confronta i modelli con la baseline e salva i modelli che la superano.

    Parametri
    ----------
    model_results_df : pd.DataFrame
        DataFrame con le metriche dei modelli.
    baseline : dict
        Dizionario restituito da compute_baseline_vs_final()
    trained_pipelines : dict
        Dizionario {model_name: pipeline_fit} con i modelli già addestrati
    output_folder : str, optional
        Cartella dove salvare CSV
    key_metric : str, default 'f1'
        La metrica principale per decidere se salvare il modello

    Ritorna
    -------
    comparison_df : pd.DataFrame
    """
    logging.info("🔍 Avvio confronto modelli vs baseline")
    records = []

    for visit, base_metrics in baseline.items():
        print(f"\n===== Confronto con baseline: {visit} =====")
        logging.info(f"Confronto con baseline: {visit}")

        for model_name, metrics in model_results_df.iterrows():
            comparison = {}
            for metric in base_metrics.keys():
                if metric not in metrics:
                    continue
                comparison[metric] = metrics[metric] > base_metrics[metric]

            # Stampa e log
            print(f"\n--- Modello: {model_name} ---")
            logging.info(f"\n--- Modello: {model_name} ---")
            for metric, beat in comparison.items():
                status = "✅ supera" if beat else "❌ NON supera"
                print(f"{metric}: {metrics[metric]:.4f} vs baseline {base_metrics[metric]:.4f} -> {status}")
                logging.info(f"{metric}: {metrics[metric]:.4f} vs baseline {base_metrics[metric]:.4f} -> {status}")

            # Record per CSV
            record = {
                "visit": visit,
                "model": model_name,
                **metrics.to_dict(),
                **{f"{k}_beat_baseline": v for k, v in comparison.items()}
            }
            records.append(record)

            # 🔹 Salva il modello se supera la baseline sulla metrica chiave
            if trained_pipelines is not None and model_name in trained_pipelines:
                if comparison.get(key_metric, False):
                    if model_output_folder:
                        os.makedirs(model_output_folder, exist_ok=True)
                        model_path = os.path.join(model_output_folder, f"{model_name}_{version}.joblib")
                        joblib.dump(trained_pipelines[model_name], model_path)
                        print(f"💾 Modello {model_name} salvato in {model_path}")
                        logging.info(f"Modello {model_name} salvato in {model_path}")

    comparison_df = pd.DataFrame(records)

    if output_folder:
        path = os.path.join(output_folder, "model_vs_baseline.csv")
        comparison_df.to_csv(path, index=False)
        print(f"\n💾 Confronto modelli vs baseline salvato in {path}")
        logging.info(f"Confronto modelli vs baseline salvato in {path}")

    return comparison_df

