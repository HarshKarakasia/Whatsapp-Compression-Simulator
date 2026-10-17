"""AUC / ACC / ECE metrics for the robustness benchmark (Phase 8)."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC. Returns NaN if y_true has only one class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float((y_pred == y_true).mean())


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: bin predictions by confidence, compare
    each bin's accuracy to its mean predicted probability, weight by bin size."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    total = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if not in_bin.any():
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = y_prob[in_bin].mean()
        total += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(total)
