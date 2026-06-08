"""
linear_regression.py
====================
Modulo per la regressione lineare semplice e la validazione del modello.

Copre le Fasi 2, 3, 4 e 6 della ROADMAP:
  - Fase 2: costruzione algebrica della retta y = ax + b
  - Fase 3: train/test split (90/10) e validazione
  - Fase 4: interpolazione, estrapolazione, prompt interattivo
  - Fase 6: pulizia outlier dinamica con loop interattivo

Regola fondamentale (CONTEXT_MANIFEST §3)
-----------------------------------------
Nessuna funzione di regressione preconfezionata (sklearn, numpy.polyfit, ecc.).
I coefficienti a e b sono calcolati tramite le formule dei minimi quadrati
derivate algebricamente, implementate con operazioni NumPy di base.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.metrics import mse, rmse, r_squared, residuals


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 — Costruzione algebrica della retta di regressione
# ─────────────────────────────────────────────────────────────────────────────

def select_best_predictor(df: pd.DataFrame,
                          target_col: str,
                          cov_df: pd.DataFrame) -> str:
    """
    Seleziona la feature predittiva con covarianza assoluta massima rispetto al target.

    Usa la matrice di covarianza calcolata in Fase 1 (data_exploration.covariance_matrix).
    La covarianza massima assoluta identifica il predittore più linearmente
    correlato all'output → scelta ottimale per la regressione lineare semplice.

    Parameters
    ----------
    df         : DataFrame completo del dataset.
    target_col : nome della colonna target (variabile dipendente y).
    cov_df     : matrice di covarianza (output di data_exploration.covariance_matrix).

    Returns
    -------
    str
        Nome della colonna con la covarianza assoluta massima rispetto al target.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    # Estrai la riga del target dalla matrice di covarianza
    target_cov = cov_df.loc[target_col, feature_cols].abs()
    best = target_cov.idxmax()
    print(f"\n  Selezione predittore (covarianza assoluta massima con '{target_col}'):")
    for col in feature_cols:
        marker = " ← selezionato" if col == best else ""
        print(f"    {col:45s}  |cov| = {target_cov[col]:.6f}{marker}")
    return best


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Calcola i coefficienti della retta di regressione y = ax + b
    tramite il metodo dei minimi quadrati (Ordinary Least Squares).

    Derivazione algebrica
    ---------------------
    Minimizzare  S(a,b) = Σ (y_i - ax_i - b)²

    Condizioni del primo ordine:
        ∂S/∂a = 0  →  a = [Σ(x_i - x̄)(y_i - ȳ)] / [Σ(x_i - x̄)²]
        ∂S/∂b = 0  →  b = ȳ - a·x̄

    In forma vettoriale (usando deviazioni centrate):
        x_c = x - x̄    (vettore centrato)
        y_c = y - ȳ    (vettore centrato)
        a = (x_c · y_c) / (x_c · x_c)
        b = ȳ - a · x̄

    Parameters
    ----------
    x : np.ndarray, shape (n,) — valori del predittore.
    y : np.ndarray, shape (n,) — valori dell'output (target).

    Returns
    -------
    tuple[float, float]
        (a, b)  — coefficiente angolare e intercetta.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_mean = np.sum(x) / len(x)      # x̄ = (1/n) Σ x_i
    y_mean = np.sum(y) / len(y)      # ȳ = (1/n) Σ y_i

    x_centered = x - x_mean          # x_c = x - x̄
    y_centered = y - y_mean          # y_c = y - ȳ

    # a = (x_c · y_c) / (x_c · x_c)  =  Σ(x_i-x̄)(y_i-ȳ) / Σ(x_i-x̄)²
    numerator   = np.dot(x_centered, y_centered)
    denominator = np.dot(x_centered, x_centered)

    if denominator == 0:
        raise ValueError("Il predittore x è costante: impossibile calcolare la regressione.")

    a = float(numerator / denominator)
    b = float(y_mean - a * x_mean)    # b = ȳ - a·x̄

    return a, b


def predict(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Applica il modello lineare: ŷ_i = a·x_i + b

    Parameters
    ----------
    x : np.ndarray — valori del predittore (scalare o vettore).
    a : float      — coefficiente angolare.
    b : float      — intercetta.

    Returns
    -------
    np.ndarray — valori predetti ŷ.
    """
    return a * np.asarray(x, dtype=float) + b


def run_phase2(df: pd.DataFrame,
               target_col: str,
               cov_df: pd.DataFrame,
               plots_dir: str = None) -> tuple:
    """
    Esegue la Fase 2: regressione lineare semplice sull'intero dataset.

    Sequenza
    --------
    1. Selezione del miglior predittore dalla matrice di covarianza.
    2. Calcolo algebrico di a e b (minimi quadrati espliciti).
    3. Calcolo predizioni, residui, MSE, RMSE, R².
    4. Scatter plot con retta di regressione sovrapposta.

    Returns
    -------
    tuple[str, float, float, pd.DataFrame]
        (predictor_col, a, b, results_df)
    """
    print("\n" + "=" * 60)
    print("  FASE 2 — Regressione Lineare Base  y = ax + b")
    print("=" * 60)

    # ── 1. Selezione predittore ───────────────────────────────────────────────
    predictor_col = select_best_predictor(df, target_col, cov_df)

    x_all = df[predictor_col].values.astype(float)
    y_all = df[target_col].values.astype(float)

    # ── 2. Calcolo coefficienti ───────────────────────────────────────────────
    a, b = fit_linear_regression(x_all, y_all)
    print(f"\n  Coefficienti della retta  y = ax + b:")
    print(f"    a (coeff. angolare) = {a:.6f}")
    print(f"    b (intercetta)      = {b:.6f}")
    print(f"    Equazione: y = {a:.4f}·x + ({b:.4f})")

    # ── 3. Predizioni e metriche ─────────────────────────────────────────────
    y_pred = predict(x_all, a, b)
    e      = residuals(y_all, y_pred)
    mse_val  = mse(y_all, y_pred)
    rmse_val = rmse(y_all, y_pred)
    r2_val   = r_squared(y_all, y_pred)

    print(f"\n  Metriche sull'intero dataset (n={len(y_all)}):")
    print(f"    MSE   = {mse_val:.6f}")
    print(f"    RMSE  = {rmse_val:.6f}")
    print(f"    R²    = {r2_val:.6f}")

    results_df = pd.DataFrame({
        "x":         x_all,
        "y_true":    y_all,
        "y_pred":    y_pred,
        "residual":  e,
    })

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    p_scatter = f"{plots_dir}/phase2_regression.png" if plots_dir else None
    _plot_regression(x_all, y_all, a, b, predictor_col, target_col,
                     title="Fase 2 — Regressione Lineare (dataset completo)",
                     output_path=p_scatter)

    print("\n✅  Fase 2 completata.")
    return predictor_col, a, b, results_df


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — Train / Test Split e validazione
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split(df: pd.DataFrame,
                     predictor_col: str,
                     target_col: str,
                     test_ratio: float = 0.10,
                     seed: int = 42) -> tuple:
    """
    Partiziona il dataset in Training (90%) e Test (10%).

    Metodo
    ------
    Shuffle casuale con seed fisso per riproducibilità, poi slicing.
    Implementato senza sklearn.train_test_split.

    Parameters
    ----------
    df            : DataFrame completo.
    predictor_col : nome del predittore (x).
    target_col    : nome del target (y).
    test_ratio    : frazione del dataset riservata al test (default 0.10).
    seed          : seed per riproducibilità.

    Returns
    -------
    tuple[np.ndarray × 4]
        (x_train, x_test, y_train, y_test)
    """
    rng = np.random.default_rng(seed)
    n   = len(df)
    indices = rng.permutation(n)                     # shuffle indici

    n_test  = max(1, int(np.ceil(n * test_ratio)))   # almeno 1 campione nel test
    n_train = n - n_test

    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    x = df[predictor_col].values.astype(float)
    y = df[target_col].values.astype(float)

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def run_phase3(df: pd.DataFrame,
               predictor_col: str,
               target_col: str,
               plots_dir: str = None) -> tuple:
    """
    Esegue la Fase 3: train/test split, addestramento e validazione.

    Sequenza
    --------
    1. Split 90/10 con shuffle casuale (seed=42).
    2. Box plot affiancati Training vs Test per verifica similarità distribuzioni.
    3. Regressione sui soli dati di training.
    4. Predizione sul test set e calcolo esplicito degli scarti.

    Returns
    -------
    tuple[float, float, tuple]
        (a_train, b_train, (x_train, x_test, y_train, y_test))
    """
    print("\n" + "=" * 60)
    print("  FASE 3 — Train / Test Split e Validazione")
    print("=" * 60)

    # ── 1. Split ──────────────────────────────────────────────────────────────
    x_train, x_test, y_train, y_test = train_test_split(
        df, predictor_col, target_col, test_ratio=0.10, seed=42
    )
    n_train, n_test = len(x_train), len(x_test)
    print(f"\n  Partizionamento (seed=42, riproducibile):")
    print(f"    Training set : {n_train} campioni  ({n_train/(n_train+n_test)*100:.0f}%)")
    print(f"    Test set     : {n_test}  campioni  ({n_test/(n_train+n_test)*100:.0f}%)")

    # ── 2. Box plot Training vs Test ─────────────────────────────────────────
    p_box = f"{plots_dir}/phase3_train_test_boxplot.png" if plots_dir else None
    _plot_train_test_boxplot(y_train, y_test, target_col, output_path=p_box)

    # ── 3. Regressione sul training set ──────────────────────────────────────
    a_train, b_train = fit_linear_regression(x_train, y_train)
    print(f"\n  Modello addestrato su training set:")
    print(f"    a = {a_train:.6f},  b = {b_train:.6f}")
    print(f"    y = {a_train:.4f}·x + ({b_train:.4f})")

    # ── 4. Predizioni e scarti sul test set ──────────────────────────────────
    y_pred_train = predict(x_train, a_train, b_train)
    y_pred_test  = predict(x_test,  a_train, b_train)

    e_train = residuals(y_train, y_pred_train)
    e_test  = residuals(y_test,  y_pred_test)

    print(f"\n  Metriche Training set (n={n_train}):")
    print(f"    MSE   = {mse(y_train, y_pred_train):.6f}")
    print(f"    RMSE  = {rmse(y_train, y_pred_train):.6f}")
    print(f"    R²    = {r_squared(y_train, y_pred_train):.6f}")

    print(f"\n  Metriche Test set (n={n_test})  [validazione oggettiva]:")
    print(f"    MSE   = {mse(y_test, y_pred_test):.6f}")
    print(f"    RMSE  = {rmse(y_test, y_pred_test):.6f}")
    print(f"    R²    = {r_squared(y_test, y_pred_test):.6f}")
    print(f"    Scarti sul test set: {e_test}")

    # ── 5. Scatter con dati train e test distinti ─────────────────────────────
    p_scatter = f"{plots_dir}/phase3_train_test_scatter.png" if plots_dir else None
    _plot_train_test_scatter(x_train, x_test, y_train, y_test,
                             a_train, b_train, predictor_col, target_col,
                             output_path=p_scatter)

    print("\n✅  Fase 3 completata.")
    return a_train, b_train, (x_train, x_test, y_train, y_test)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 4 — Interpolazione ed Estrapolazione
# ─────────────────────────────────────────────────────────────────────────────

def run_phase4(df: pd.DataFrame,
               predictor_col: str,
               target_col: str,
               a: float,
               b: float,
               plots_dir: str = None):
    """
    Esegue la Fase 4: interpolazione ed estrapolazione tramite il modello lineare.

    Operazioni
    ----------
    1. Predizione nel punto medio delle ascisse (x_mid) con confronto empirico.
    2. Predizione dell'ascissa corrispondente all'ordinata media (inversione retta).
    3. Predizione per x = 0 (estrapolazione fuori dal dominio osservato).
    4. Prompt interattivo: l'utente inserisce x_input → ŷ(x_input) plottato.
    """
    print("\n" + "=" * 60)
    print("  FASE 4 — Interpolazione ed Estrapolazione")
    print("=" * 60)

    x_all = df[predictor_col].values.astype(float)
    y_all = df[target_col].values.astype(float)

    x_min, x_max = np.min(x_all), np.max(x_all)
    y_mean = np.sum(y_all) / len(y_all)

    # ── 1. Punto medio delle ascisse (interpolazione) ────────────────────────
    x_mid    = (x_min + x_max) / 2.0
    y_mid_hat = predict(x_mid, a, b)

    # Confronto con media dei valori osservati nell'intorno di x_mid (±10% range)
    tol      = (x_max - x_min) * 0.10
    mask     = np.abs(x_all - x_mid) <= tol
    y_nearby = y_all[mask]
    y_nearby_mean = float(np.mean(y_nearby)) if len(y_nearby) > 0 else float("nan")

    print(f"\n  [4.1] Interpolazione nel punto medio delle ascisse:")
    print(f"        x_mid = ({x_min:.4f} + {x_max:.4f}) / 2 = {x_mid:.4f}")
    print(f"        ŷ(x_mid)                    = {y_mid_hat:.4f}")
    print(f"        Media osservati nell'intorno = {y_nearby_mean:.4f}  "
          f"(n={len(y_nearby)} campioni entro ±10% del range)")

    # ── 2. Ascissa dell'ordinata media (inversione retta) ────────────────────
    # y = ax + b  →  x = (y - b) / a
    if abs(a) < 1e-12:
        print("\n  [4.2] Coefficiente a ≈ 0: inversione non definita.")
        x_star = float("nan")
    else:
        x_star  = (y_mean - b) / a
        y_check = predict(x_star, a, b)
        print(f"\n  [4.2] Ascissa corrispondente all'ordinata media ȳ = {y_mean:.4f}:")
        print(f"        x* = (ȳ - b) / a = ({y_mean:.4f} - {b:.4f}) / {a:.4f} = {x_star:.4f}")
        print(f"        Verifica: ŷ(x*) = {y_check:.4f}  (deve essere ≈ ȳ)")

    # ── 3. Estrapolazione in x = 0 ───────────────────────────────────────────
    y_at_zero = predict(0.0, a, b)
    print(f"\n  [4.3] Estrapolazione per x = 0:")
    print(f"        ŷ(0) = a·0 + b = b = {y_at_zero:.4f}")
    if 0.0 < x_min or 0.0 > x_max:
        print(f"        ⚠️  x=0 è fuori dal range osservato [{x_min:.4f}, {x_max:.4f}]"
              f" → è un'estrapolazione, non un'interpolazione.")

    # ── 4. Prompt interattivo ────────────────────────────────────────────────
    print(f"\n  [4.4] Input interattivo:")
    try:
        raw = input(f"        Inserisci un valore x (range osservato: "
                    f"[{x_min:.4f}, {x_max:.4f}]): ").strip()
        x_input   = float(raw)
        y_input   = predict(x_input, a, b)
        is_extrap = x_input < x_min or x_input > x_max
        tipo      = "estrapolazione ⚠️" if is_extrap else "interpolazione ✅"
        print(f"        ŷ({x_input:.4f}) = {y_input:.4f}  [{tipo}]")
    except (ValueError, EOFError):
        x_input, y_input = None, None
        print("        Input non valido o non interattivo — skip.")

    # ── 5. Plot con tutti i punti speciali ───────────────────────────────────
    p_out = f"{plots_dir}/phase4_interpolation.png" if plots_dir else None
    _plot_interpolation(x_all, y_all, a, b,
                        x_mid, y_mid_hat,
                        x_star, y_mean,
                        x_input, y_input,
                        predictor_col, target_col,
                        output_path=p_out)

    print("\n✅  Fase 4 completata.")


# ─────────────────────────────────────────────────────────────────────────────
# FASE 6 — Pulizia Outlier Dinamica (loop interattivo)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase6(df: pd.DataFrame,
               predictor_col: str,
               target_col: str,
               plots_dir: str = None) -> pd.DataFrame:
    """
    Esegue la Fase 6: rimozione interattiva degli outlier.

    Algoritmo
    ---------
    1. Addestra la regressione sul dataset corrente.
    2. Calcola i residui assoluti |e_i| = |y_i - ŷ_i|.
    3. Identifica il campione con il residuo massimo.
    4. Chiede all'utente: "Il punto [ID] sembra anomalo. Rimuoverlo? (s/n)"
    5. Se sì → rimuove il punto e ripete dal passo 1.
    6. Se no → plotta il modello finale e salva il dataset pulito.

    Parameters
    ----------
    df            : DataFrame corrente (può già essere filtrato).
    predictor_col : nome della colonna predittore.
    target_col    : nome della colonna target.
    plots_dir     : cartella dove salvare i plot.

    Returns
    -------
    pd.DataFrame
        Dataset pulito (senza gli outlier rimossi).
    """
    print("\n" + "=" * 60)
    print("  FASE 6 — Pulizia Outlier Dinamica")
    print("=" * 60)

    df_clean    = df.copy()
    removed_ids = []
    iteration   = 0

    while len(df_clean) > 5:       # minimo 5 punti per regressione significativa
        iteration += 1

        x = df_clean[predictor_col].values.astype(float)
        y = df_clean[target_col].values.astype(float)

        a, b = fit_linear_regression(x, y)
        e    = residuals(y, predict(x, a, b))
        abs_e = np.abs(e)

        # Indice (nel DataFrame corrente) del residuo massimo
        worst_local_idx = int(np.argmax(abs_e))
        worst_df_idx    = df_clean.index[worst_local_idx]
        worst_e         = abs_e[worst_local_idx]
        worst_x         = x[worst_local_idx]
        worst_y         = y[worst_local_idx]

        print(f"\n  [Iterazione {iteration}]  n = {len(df_clean)} campioni")
        print(f"    MSE corrente = {mse(y, predict(x, a, b)):.6f}  |  "
              f"R² = {r_squared(y, predict(x, a, b)):.6f}")
        print(f"\n    Il campione '{worst_df_idx}' sembra anomalo")
        print(f"      x = {worst_x:.4f},  y = {worst_y:.4f},  |scarto| = {worst_e:.4f}")

        try:
            ans = input("    Vuoi rimuoverlo dal dataset? (s/n): ").strip().lower()
        except EOFError:
            ans = "n"

        if ans == "s":
            removed_ids.append(worst_df_idx)
            df_clean = df_clean.drop(index=worst_df_idx)
            print(f"    ✂️  Campione rimosso. Dataset ridotto a {len(df_clean)} campioni.")
        else:
            print("    Dataset accettato. Uscita dal loop di pulizia.")
            break

    # Modello finale sul dataset pulito
    x_f = df_clean[predictor_col].values.astype(float)
    y_f = df_clean[target_col].values.astype(float)
    a_f, b_f = fit_linear_regression(x_f, y_f)

    print(f"\n  Modello finale (n={len(df_clean)} campioni):")
    print(f"    a = {a_f:.6f},  b = {b_f:.6f}")
    print(f"    MSE  = {mse(y_f, predict(x_f, a_f, b_f)):.6f}")
    print(f"    R²   = {r_squared(y_f, predict(x_f, a_f, b_f)):.6f}")
    if removed_ids:
        print(f"    Campioni rimossi: {removed_ids}")

    # Plot finale
    p_out = f"{plots_dir}/phase6_outlier_clean.png" if plots_dir else None
    _plot_outlier_result(x_f, y_f, a_f, b_f,
                         predictor_col, target_col,
                         removed_ids, output_path=p_out)

    # Salvataggio dataset pulito
    clean_path = Path(plots_dir).parent / "muscular_dystrophy_clean.csv" if plots_dir else None
    if clean_path:
        df_clean.to_csv(clean_path)
        print(f"    Dataset pulito salvato: {clean_path.name}")

    print("\n✅  Fase 6 completata.")
    return df_clean


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


def _plot_regression(x, y, a, b, x_label, y_label, title, output_path=None):
    fig, ax = _base_fig()
    ax.scatter(x, y, color="#4f8ef7", alpha=0.8, s=60, zorder=3,
               label="Dati osservati")
    x_line = np.linspace(np.min(x), np.max(x), 300)
    ax.plot(x_line, predict(x_line, a, b), color="#f7a94f", linewidth=2,
            label=f"y = {a:.4f}·x + {b:.4f}")
    ax.set_xlabel(x_label, color="#aaaacc")
    ax.set_ylabel(y_label, color="#aaaacc")
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.legend(facecolor="#0f1117", labelcolor="white")
    _save_or_show(fig, output_path)


def _plot_train_test_boxplot(y_train, y_test, label, output_path=None):
    fig, ax = _base_fig(figsize=(7, 5))
    bp = ax.boxplot(
        [y_train, y_test],
        labels=["Training", "Test"],
        patch_artist=True,
        boxprops=dict(facecolor="#4f8ef730", color="#4f8ef7", linewidth=1.5),
        medianprops=dict(color="#f7a94f", linewidth=2.5),
        whiskerprops=dict(color="#aaaacc", linewidth=1.5),
        capprops=dict(color="#aaaacc", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="#f74f4f",
                        markeredgecolor="#f74f4f", alpha=0.6, markersize=5),
    )
    ax.set_title(f"Fase 3 — Distribuzione Training vs Test ({label})",
                 color="white", fontsize=12)
    ax.set_ylabel(label, color="#aaaacc")
    _save_or_show(fig, output_path)


def _plot_train_test_scatter(x_tr, x_te, y_tr, y_te, a, b,
                             x_label, y_label, output_path=None):
    fig, ax = _base_fig()
    x_line = np.linspace(min(np.min(x_tr), np.min(x_te)),
                         max(np.max(x_tr), np.max(x_te)), 300)
    ax.scatter(x_tr, y_tr, color="#4f8ef7", alpha=0.8, s=60, zorder=3,
               label="Training")
    ax.scatter(x_te, y_te, color="#f74f4f", alpha=0.9, s=80, zorder=4,
               marker="D", label="Test")
    ax.plot(x_line, predict(x_line, a, b), color="#f7a94f", linewidth=2,
            label=f"y = {a:.4f}·x + {b:.4f}  (addestrata su training)")
    ax.set_xlabel(x_label, color="#aaaacc")
    ax.set_ylabel(y_label, color="#aaaacc")
    ax.set_title("Fase 3 — Validazione Train/Test", color="white",
                 fontsize=13, pad=10)
    ax.legend(facecolor="#0f1117", labelcolor="white")
    _save_or_show(fig, output_path)


def _plot_interpolation(x, y, a, b,
                        x_mid, y_mid,
                        x_star, y_mean,
                        x_inp, y_inp,
                        x_label, y_label, output_path=None):
    fig, ax = _base_fig()
    x_line = np.linspace(
        min(np.min(x), 0, x_inp or np.min(x)) - 0.5,
        max(np.max(x), x_inp or np.max(x)) + 0.5, 400)
    ax.scatter(x, y, color="#4f8ef7", alpha=0.7, s=55, zorder=3,
               label="Dati osservati")
    ax.plot(x_line, predict(x_line, a, b), color="#f7a94f", linewidth=2,
            label=f"y = {a:.4f}·x + {b:.4f}")
    ax.scatter([x_mid], [y_mid], color="#a8ff78", s=100, zorder=5,
               marker="^", label=f"x_mid={x_mid:.3f} → ŷ={y_mid:.3f}")
    if not np.isnan(x_star):
        ax.scatter([x_star], [y_mean], color="#ff78d4", s=100, zorder=5,
                   marker="s", label=f"x*(ȳ)={x_star:.3f} → ȳ={y_mean:.3f}")
    ax.scatter([0], [predict(0, a, b)], color="#ff9f43", s=100, zorder=5,
               marker="x", linewidths=2.5,
               label=f"ŷ(0) = b = {predict(0,a,b):.3f}")
    if x_inp is not None:
        ax.scatter([x_inp], [y_inp], color="#ee5a24", s=120, zorder=6,
                   marker="*", label=f"Input utente: ŷ({x_inp:.2f})={y_inp:.3f}")
    ax.axvline(0, color="#555566", linewidth=0.8, linestyle=":")
    ax.set_xlabel(x_label, color="#aaaacc")
    ax.set_ylabel(y_label, color="#aaaacc")
    ax.set_title("Fase 4 — Interpolazione ed Estrapolazione",
                 color="white", fontsize=13, pad=10)
    ax.legend(facecolor="#0f1117", labelcolor="white", fontsize=8)
    _save_or_show(fig, output_path)


def _plot_outlier_result(x, y, a, b, x_label, y_label,
                         removed_ids, output_path=None):
    fig, ax = _base_fig()
    x_line = np.linspace(np.min(x), np.max(x), 300)
    ax.scatter(x, y, color="#4f8ef7", alpha=0.85, s=60, zorder=3,
               label=f"Dataset pulito (n={len(x)})")
    ax.plot(x_line, predict(x_line, a, b), color="#f7a94f", linewidth=2,
            label=f"y = {a:.4f}·x + {b:.4f}")
    n_rem = len(removed_ids)
    if n_rem:
        ax.text(0.02, 0.97,
                f"Campioni rimossi: {n_rem}\n{removed_ids}",
                transform=ax.transAxes, color="#f74f4f",
                fontsize=8, va="top", ha="left",
                bbox=dict(facecolor="#1a1d27", edgecolor="#f74f4f",
                          alpha=0.8, pad=4))
    ax.set_xlabel(x_label, color="#aaaacc")
    ax.set_ylabel(y_label, color="#aaaacc")
    ax.set_title("Fase 6 — Modello Finale dopo Pulizia Outlier",
                 color="white", fontsize=13, pad=10)
    ax.legend(facecolor="#0f1117", labelcolor="white")
    _save_or_show(fig, output_path)
