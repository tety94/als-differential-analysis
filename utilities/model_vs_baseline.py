import pandas as pd
import logging

def compare_models_to_baseline(model_results_df, baseline, output_folder=None):
    """
    Confronta i modelli con la baseline dei dottori (una per ogni visita).

    Parametri
    ----------
    model_results_df : pd.DataFrame
        DataFrame con le metriche dei modelli. Deve avere colonne come: ["accuracy", "f1", "precision", "recall", ...]
        e l'indice come nome del modello.
    baseline : dict
        Dizionario restituito da compute_baseline_vs_final()
        Es: {'diagn_1_vis': {'accuracy': ..., 'f1': ..., ...}, 'second_opinion (0/1)': {...}}
    output_folder : str, optional
        Cartella dove salvare il CSV con i risultati

    Ritorna
    -------
    comparison_df : pd.DataFrame
        DataFrame con confronto modelli vs baseline
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

            # Stampa a video
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

    comparison_df = pd.DataFrame(records)

    if output_folder:
        path = f"{output_folder}/model_vs_baseline.csv"
        comparison_df.to_csv(path, index=False)
        print(f"\n💾 Confronto modelli vs baseline salvato in {path}")
        logging.info(f"Confronto modelli vs baseline salvato in {path}")

    return comparison_df
