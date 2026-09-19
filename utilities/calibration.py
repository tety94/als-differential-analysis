import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from utilities.utils import save_plot


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """
    Expected Calibration Error (ECE): weighted average, across bins of
    predicted probability, of the absolute difference between mean
    predicted confidence and observed frequency in that bin. Closer to 0
    means better calibrated.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins[1:-1])

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        conf_mean = y_proba[mask].mean()
        acc_mean = y_true[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(conf_mean - acc_mean)

    return ece


def plot_calibration(y_true, y_proba, model_name, folder, n_bins=10, log=print):
    """
    Generates the reliability diagram (calibration curve) and computes
    Brier score and Expected Calibration Error. Saves the plot and returns
    both scores.
    """
    os.makedirs(folder, exist_ok=True)

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')

    brier = brier_score_loss(y_true, y_proba)
    ece = expected_calibration_error(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    ax.plot(prob_pred, prob_true, marker='o', color='steelblue', label=model_name)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title(f'Calibration curve - {model_name}\nBrier = {brier:.3f}   ECE = {ece:.3f}')
    ax.legend()
    plt.tight_layout()

    save_plot(fig, os.path.join(folder, f'calibration_curve_{model_name}.png'))
    plt.close(fig)

    log(f"[calibration] {model_name}: Brier score = {brier:.3f}, ECE = {ece:.3f}")

    return {"brier_score": brier, "ece": ece}
