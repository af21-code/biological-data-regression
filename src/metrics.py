"""
metrics.py
==========
Modulo isolato per il calcolo rigoroso delle metriche di errore.

Tutte le formule sono implementate esplicitamente tramite NumPy.
Nessuna funzione di sklearn o librerie di alto livello è utilizzata.

Metriche implementate
---------------------
- residuals   : scarti predittivi  e_i = y_i - ŷ_i
- mse         : Mean Squared Error  = (1/n) Σ e_i²
- rmse        : Root MSE            = √MSE
- r_squared   : Coefficiente R²     = 1 - SS_res / SS_tot
"""

import numpy as np


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcola il vettore degli scarti (residui).

    Formula
    -------
        e_i = y_i - ŷ_i   per ogni i = 1, ..., n

    Parameters
    ----------
    y_true : array-like
        Valori osservati (ground truth).
    y_pred : array-like
        Valori predetti dal modello.

    Returns
    -------
    np.ndarray
        Vettore degli scarti e_i.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    return y_true - y_pred


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (Errore Quadratico Medio).

    Formula
    -------
        MSE = (1/n) * Σ_{i=1}^{n} (y_i - ŷ_i)²
            = (1/n) * eᵀe

    dove e è il vettore dei residui.

    Parameters
    ----------
    y_true : array-like
        Valori osservati.
    y_pred : array-like
        Valori predetti.

    Returns
    -------
    float
        MSE ≥ 0. Più basso è, migliore è il modello.
    """
    e = residuals(y_true, y_pred)
    # Prodotto scalare: eᵀe = Σ e_i²
    return float(np.dot(e, e) / len(e))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Formula
    -------
        RMSE = √MSE = √[(1/n) * Σ (y_i - ŷ_i)²]

    Vantaggio rispetto a MSE: ha la stessa unità di misura di y,
    rendendo l'interpretazione più intuitiva.

    Parameters
    ----------
    y_true : array-like
        Valori osservati.
    y_pred : array-like
        Valori predetti.

    Returns
    -------
    float
        RMSE ≥ 0, nella stessa unità di misura di y.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficiente di determinazione R².

    Formula
    -------
        R² = 1 - SS_res / SS_tot

    dove:
        SS_res = Σ (y_i - ŷ_i)²    (somma dei quadrati dei residui)
        SS_tot = Σ (y_i - ȳ)²      (varianza totale, proporzionale a σ²)

    Interpretazione
    ---------------
        R² = 1   → il modello spiega tutta la varianza (fit perfetto)
        R² = 0   → il modello non è migliore di predire sempre ȳ
        R² < 0   → il modello è peggiore della media (caso degenere)

    Parameters
    ----------
    y_true : array-like
        Valori osservati.
    y_pred : array-like
        Valori predetti.

    Returns
    -------
    float
        R² ∈ (-∞, 1].
    """
    y_true = np.asarray(y_true, dtype=float)
    e = residuals(y_true, y_pred)
    ss_res = float(np.dot(e, e))
    y_mean = np.mean(y_true)
    diff_tot = y_true - y_mean
    ss_tot = float(np.dot(diff_tot, diff_tot))
    if ss_tot == 0.0:
        # Tutti i valori reali sono identici: R² indefinito.
        # Convenzione: 1.0 se i residui sono nulli, 0.0 altrimenti.
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)
