"""
multivariate.py
===============
Modello di Regressione Multilineare (OLS)  y = X·θ

Implementa la Fase 5 della ROADMAP:
  - Regressione lineare multipla con TUTTE le feature del dataset.
  - Calcolo di θ tramite le Equazioni Normali: θ = (XᵀX)⁻¹Xᵀy
  - Monitoraggio del condizionamento di XᵀX e regolarizzazione opzionale.
  - Confronto diretto delle metriche con la regressione semplice (Fase 2/3).

Regola fondamentale (CONTEXT_MANIFEST §3)
-----------------------------------------
Tutti i calcoli usano esclusivamente operazioni matriciali NumPy (@, .T, linalg).
Nessuna chiamata a sklearn, numpy.polyfit o funzioni di regressione preconfezionate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.metrics import mse, rmse, r_squared, residuals


# ─────────────────────────────────────────────────────────────────────────────
# Funzioni core
# ─────────────────────────────────────────────────────────────────────────────

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """
    Aggiunge una colonna di 1 in testa alla matrice X per modellare l'intercetta.

    Il modello  y = a₁x₁ + a₂x₂ + ... + aₘxₘ + b
    diventa in forma vettoriale: y = X_b · θ
    dove X_b = [1 | X]  e  θ = [b, a₁, a₂, ..., aₘ]ᵀ

    Parameters
    ----------
    X : np.ndarray, shape (n, m)

    Returns
    -------
    np.ndarray, shape (n, m+1)   — con colonna di 1 in posizione 0.
    """
    n = X.shape[0]
    return np.column_stack([np.ones(n), X])


def fit_multivariate(X: np.ndarray,
                     y: np.ndarray,
                     lambda_reg: float = 0.0) -> np.ndarray:
    """
    Stima i coefficienti del modello multilineare tramite le Equazioni Normali.

    Derivazione
    -----------
    Minimizziamo  S(θ) = ‖y - X_b·θ‖²  =  (y - X_b·θ)ᵀ(y - X_b·θ)

    Derivando rispetto a θ e ponendo uguale a zero:
        ∂S/∂θ = -2 X_bᵀ(y - X_b·θ) = 0
        X_bᵀ X_b · θ = X_bᵀ y
                   θ = (X_bᵀ X_b)⁻¹ X_bᵀ y

    Il sistema lineare (X_bᵀ X_b)·θ = X_bᵀ·y è risolto con numpy.linalg.solve
    (più stabile numericamente dell'inversione esplicita di XᵀX).

    Regolarizzazione di Tikhonov (Ridge, opzionale)
    ------------------------------------------------
    Se lambda_reg > 0, si aggiunge λI alla matrice dei coefficienti:
        θ = (X_bᵀ X_b + λI)⁻¹ X_bᵀ y

    Questo riduce la sensibilità a multicollinearità o rank deficiency.
    Il termine di regolarizzazione introduce un bias controllato in cambio
    di una varianza ridotta (bias-variance tradeoff).
    La colonna del bias (intercetta) NON viene regolarizzata (convenzione standard).

    Parameters
    ----------
    X          : np.ndarray, shape (n, m) — matrice delle feature (senza bias).
    y          : np.ndarray, shape (n,)   — vettore target.
    lambda_reg : float ≥ 0 — parametro di regolarizzazione Ridge (default 0).

    Returns
    -------
    np.ndarray, shape (m+1,)
        Vettore θ = [b, a₁, a₂, ..., aₘ]ᵀ
        θ[0]  = intercetta b
        θ[1:] = coefficienti aᵢ per ogni feature
    """
    X_b = add_bias_column(X)              # (n, m+1)
    A   = X_b.T @ X_b                    # (m+1, m+1) = XᵀX
    b_vec = X_b.T @ y                    # (m+1,)     = Xᵀy

    # Regolarizzazione Ridge: aggiungi λI alla diagonale (esclusi [0,0] = bias)
    if lambda_reg > 0:
        reg_matrix = lambda_reg * np.eye(A.shape[0])
        reg_matrix[0, 0] = 0.0           # non regolarizzare l'intercetta
        A = A + reg_matrix

    theta = np.linalg.solve(A, b_vec)    # solve(A, b) più stabile di inv(A)@b
    return theta


def predict_multivariate(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Applica il modello multilineare:  ŷ = X_b · θ  = [1|X] · θ

    Parameters
    ----------
    X     : np.ndarray, shape (n, m) — feature senza colonna bias.
    theta : np.ndarray, shape (m+1,) — coefficienti (inclusa intercetta).

    Returns
    -------
    np.ndarray, shape (n,) — valori predetti.
    """
    X_b = add_bias_column(X)
    return X_b @ theta


# ─────────────────────────────────────────────────────────────────────────────
# FASE 5 — Orchestratore
# ─────────────────────────────────────────────────────────────────────────────

def run_phase5(df: pd.DataFrame,
               target_col: str,
               plots_dir: str = None,
               lambda_reg: float = 0.0,
               simple_metrics: dict = None) -> tuple:
    """
    Esegue la Fase 5: regressione multilineare su tutte le feature del dataset.

    Sequenza
    --------
    1. Costruisce la matrice X ∈ ℝ^{n×m} (tutte le feature numeriche tranne il target).
    2. Verifica il condizionamento di XᵀX (avvisa se > 1e8).
    3. Train/test split 90/10 (seed=42, coerente con Fase 3).
    4. Calcola θ con le Equazioni Normali sul training set.
    5. Predizioni su training e test set → metriche MSE, RMSE, R².
    6. Confronto diretto con le metriche della regressione semplice (Fase 2/3).
    7. Scatter plot ŷ vs y (actual vs predicted) + grafico residui.
    8. Bar chart dei coefficienti (importanza relativa delle feature).

    Parameters
    ----------
    df            : DataFrame completo (campioni × feature+target).
    target_col    : nome della colonna target.
    plots_dir     : cartella dove salvare i plot (None = mostra a schermo).
    lambda_reg    : parametro Ridge (0 = OLS puro, >0 = regolarizzato).
    simple_metrics: dict con le metriche della regressione semplice per confronto.
                    Chiavi attese: 'mse_train', 'rmse_train', 'r2_train',
                                   'mse_test',  'rmse_test',  'r2_test'.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame, pd.DataFrame]
        (theta, results_train_df, results_test_df)
    """
    print("\n" + "=" * 62)
    print("  FASE 5 — Regressione Multilineare  y = X·θ")
    print("=" * 62)

    feature_cols = [c for c in df.columns if c != target_col]
    m = len(feature_cols)
    n = len(df)

    X_full = df[feature_cols].values.astype(float)   # (n, m)
    y_full = df[target_col].values.astype(float)     # (n,)

    print(f"\n  Matrice predittori  X  ∈  ℝ^{{{n} × {m}}}")
    print(f"  Vettore target      y  ∈  ℝ^{{{n}}}")
    print(f"  Parametri del modello (incl. intercetta): {m+1}")
    print(f"  Gradi di libertà residui: {n - (m+1)}")
    print(f"\n  Feature utilizzate ({m}):")
    for i, col in enumerate(feature_cols):
        print(f"    {i+1:2d}. {col}")

    # ── 1. Verifica condizionamento ───────────────────────────────────────────
    X_b_full = add_bias_column(X_full)
    AtA      = X_b_full.T @ X_b_full
    cond_num = np.linalg.cond(AtA)
    print(f"\n  Numero di condizionamento di XᵀX: {cond_num:.2e}", end="")
    if cond_num > 1e10:
        print(f"  ⚠️  Alto! λ_reg verrà impostato automaticamente.")
        lambda_reg = max(lambda_reg, 1e-4)
    elif cond_num > 1e6:
        print(f"  ⚠️  Moderato (multicollinearità possibile).")
    else:
        print(f"  ✅ Stabile.")
    if lambda_reg > 0:
        print(f"  Regolarizzazione Ridge: λ = {lambda_reg}")

    # ── 2. Train / Test split (seed=42 — coerente con Fase 3) ────────────────
    rng      = np.random.default_rng(42)
    indices  = rng.permutation(n)
    n_test   = max(1, int(np.ceil(n * 0.10)))
    n_train  = n - n_test
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]

    print(f"\n  Partizionamento (seed=42):")
    print(f"    Training set : {n_train} campioni")
    print(f"    Test set     : {n_test}  campioni")

    # ── 3. Calcolo θ con Equazioni Normali ───────────────────────────────────
    theta = fit_multivariate(X_train, y_train, lambda_reg=lambda_reg)

    intercept = theta[0]
    coefs     = theta[1:]

    print(f"\n  Vettore θ = [b, a₁, ..., a{m}]ᵀ calcolato con Equazioni Normali:")
    print(f"    b (intercetta) = {intercept:.6f}")
    for i, (col, ai) in enumerate(zip(feature_cols, coefs)):
        print(f"    a{i+1:2d} ({col:15s}) = {ai:+.6f}")

    # ── 4. Predizioni e metriche ─────────────────────────────────────────────
    y_pred_train = predict_multivariate(X_train, theta)
    y_pred_test  = predict_multivariate(X_test,  theta)

    e_train = residuals(y_train, y_pred_train)
    e_test  = residuals(y_test,  y_pred_test)

    mse_tr  = mse(y_train, y_pred_train)
    rmse_tr = rmse(y_train, y_pred_train)
    r2_tr   = r_squared(y_train, y_pred_train)

    mse_te  = mse(y_test, y_pred_test)
    rmse_te = rmse(y_test, y_pred_test)
    r2_te   = r_squared(y_test, y_pred_test)

    print(f"\n  Metriche Training set (n={n_train}):")
    print(f"    MSE   = {mse_tr:.6f}")
    print(f"    RMSE  = {rmse_tr:.6f}")
    print(f"    R²    = {r2_tr:.6f}")

    print(f"\n  Metriche Test set (n={n_test}):")
    print(f"    MSE   = {mse_te:.6f}")
    print(f"    RMSE  = {rmse_te:.6f}")
    print(f"    R²    = {r2_te:.6f}")
    print(f"    Scarti: {e_test}")

    # ── 5. Confronto con regressione semplice ─────────────────────────────────
    if simple_metrics:
        print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │  CONFRONTO — Regressione Semplice (F.2/3) vs Multilineare (F.5) │")
        print(f"  ├──────────────┬───────────────┬───────────────┬───────────────┤")
        print(f"  │  Metrica     │  Semplice (tr) │  Multi   (tr) │  Δ (multi-si) │")
        print(f"  ├──────────────┼───────────────┼───────────────┼───────────────┤")
        for met, sm_key, ml_val in [
            ("MSE  (train)", "mse_train",  mse_tr),
            ("RMSE (train)", "rmse_train", rmse_tr),
            ("R²   (train)", "r2_train",   r2_tr),
        ]:
            sv = simple_metrics.get(sm_key, float("nan"))
            delta = ml_val - sv
            sign  = "+" if delta >= 0 else ""
            print(f"  │  {met:<12s}│  {sv:12.6f} │  {ml_val:12.6f} │  {sign}{delta:12.6f} │")
        print(f"  ├──────────────┼───────────────┼───────────────┼───────────────┤")
        for met, sm_key, ml_val in [
            ("MSE  (test)", "mse_test",  mse_te),
            ("RMSE (test)", "rmse_test", rmse_te),
            ("R²   (test)", "r2_test",   r2_te),
        ]:
            sv = simple_metrics.get(sm_key, float("nan"))
            delta = ml_val - sv
            sign  = "+" if delta >= 0 else ""
            print(f"  │  {met:<12s}│  {sv:12.6f} │  {ml_val:12.6f} │  {sign}{delta:12.6f} │")
        print(f"  └──────────────┴───────────────┴───────────────┴───────────────┘")

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    if plots_dir:
        p_actpred = f"{plots_dir}/phase5_actual_vs_predicted.png"
        p_resid   = f"{plots_dir}/phase5_residuals.png"
        p_coefs   = f"{plots_dir}/phase5_coefficients.png"
    else:
        p_actpred = p_resid = p_coefs = None

    _plot_actual_vs_predicted(y_train, y_pred_train, y_test, y_pred_test,
                              target_col, output_path=p_actpred)
    _plot_residuals(y_pred_train, e_train, y_pred_test, e_test,
                    output_path=p_resid)
    _plot_coefficients(feature_cols, coefs, output_path=p_coefs)

    # ── 7. DataFrame risultati ────────────────────────────────────────────────
    results_train = pd.DataFrame({
        "y_true": y_train, "y_pred": y_pred_train, "residual": e_train
    })
    results_test = pd.DataFrame({
        "y_true": y_test, "y_pred": y_pred_test, "residual": e_test
    })

    print("\n✅  Fase 5 completata.")
    return theta, results_train, results_test


# ─────────────────────────────────────────────────────────────────────────────
# Funzioni di plotting private
# ─────────────────────────────────────────────────────────────────────────────

def _base_fig(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.tick_params(colors="white")
    return fig, ax


def _save_or_show(fig, output_path):
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Plot salvato: {output_path}")
    plt.show()


def _plot_actual_vs_predicted(y_tr, yp_tr, y_te, yp_te, label, output_path=None):
    """Scatter plot Actual vs Predicted — linea y=x è il modello perfetto."""
    fig, ax = _base_fig()
    all_vals = np.concatenate([y_tr, yp_tr, y_te, yp_te])
    lo, hi   = all_vals.min() - 0.5, all_vals.max() + 0.5
    ax.plot([lo, hi], [lo, hi], color="#555566", linewidth=1.5,
            linestyle="--", label="Previsione perfetta (y = ŷ)")
    ax.scatter(y_tr, yp_tr, color="#4f8ef7", alpha=0.8, s=60, zorder=3,
               label=f"Training (n={len(y_tr)})")
    ax.scatter(y_te, yp_te, color="#f74f4f", alpha=0.9, s=90, zorder=4,
               marker="D", label=f"Test (n={len(y_te)})")
    ax.set_xlabel(f"{label} — Valori reali", color="#aaaacc")
    ax.set_ylabel(f"{label} — Valori predetti", color="#aaaacc")
    ax.set_title("Fase 5 — Modello Multilineare: Actual vs Predicted",
                 color="white", fontsize=13, pad=10)
    ax.legend(facecolor="#0f1117", labelcolor="white")
    _save_or_show(fig, output_path)


def _plot_residuals(yp_tr, e_tr, yp_te, e_te, output_path=None):
    """Residui vs Valori predetti — i residui devono essere centrati sullo zero."""
    fig, ax = _base_fig()
    ax.axhline(0, color="#555566", linewidth=1.5, linestyle="--")
    ax.scatter(yp_tr, e_tr, color="#4f8ef7", alpha=0.8, s=60, zorder=3,
               label=f"Training (n={len(yp_tr)})")
    ax.scatter(yp_te, e_te, color="#f74f4f", alpha=0.9, s=90, zorder=4,
               marker="D", label=f"Test (n={len(yp_te)})")
    ax.set_xlabel("Valori predetti ŷ", color="#aaaacc")
    ax.set_ylabel("Residui  e = y − ŷ", color="#aaaacc")
    ax.set_title("Fase 5 — Analisi dei Residui",
                 color="white", fontsize=13, pad=10)
    ax.legend(facecolor="#0f1117", labelcolor="white")
    _save_or_show(fig, output_path)


def _plot_coefficients(feature_cols, coefs, output_path=None):
    """Bar chart dei coefficienti θ (importanza relativa delle feature)."""
    fig, ax = _base_fig(figsize=(12, 5))
    colors = ["#4f8ef7" if c >= 0 else "#f74f4f" for c in coefs]
    bars = ax.bar(range(len(feature_cols)), coefs, color=colors, alpha=0.85,
                  edgecolor="#ffffff33")
    ax.axhline(0, color="#555566", linewidth=1.2)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=30, ha="right",
                       color="white", fontsize=9)
    ax.set_ylabel("Coefficiente aᵢ", color="#aaaacc")
    ax.set_title("Fase 5 — Coefficienti del Modello Multilineare",
                 color="white", fontsize=13, pad=10)
    # Etichette sui bar
    for bar, val in zip(bars, coefs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.002 if val >= 0 else -0.006),
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                color="white", fontsize=8)
    _save_or_show(fig, output_path)
