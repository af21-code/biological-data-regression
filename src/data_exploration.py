"""
data_exploration.py
===================
Modulo per l'analisi statistica descrittiva (Fase 1 — Analisi 0-dimensionale).

Tutte le formule statistiche sono implementate esplicitamente tramite NumPy.
Nessuna chiamata a df.describe(), np.cov() o funzioni statistiche preconfezionate
viene usata per i calcoli centrali: ogni indicatore è derivato dalla definizione.

Funzioni principali
-------------------
- load_dataset              : carica il CSV e ne stampa la struttura
- descriptive_stats         : max, min, media, mediana, varianza, std
- confidence_interval_90    : intervallo [m-δ, m+δ] al 90% centrato nella mediana
- covariance_matrix         : matrice di covarianza C = (1/n) X_c^T X_c
- correlation_from_cov      : matrice di correlazione R da C
- plot_bar_stats            : bar chart degli indicatori
- plot_boxplots             : box plot affiancati
- plot_histograms           : istogrammi per ogni variabile continua
- plot_covariance_heatmap   : heatmap della matrice di covarianza
- plot_correlation_heatmap  : heatmap della matrice di correlazione di Pearson
- plot_ecdf_vs_gaussian   : ECDF empirica vs CDF gaussiana teorica
- run_phase1              : orchestratore completo della Fase 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Caricamento dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Carica il dataset da file CSV e stampa un report di ispezione iniziale.

    Parameters
    ----------
    path : str
        Percorso al file CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame con i dati caricati.
    """
    df = pd.read_csv(path)
    # Se il file ha una colonna 'sample_id' o simile come primo campo stringa,
    # la impostiamo come indice per non includerla nei calcoli numerici.
    str_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if str_cols:
        df = df.set_index(str_cols[0])
    print(f"\n{'='*55}")
    print(f"  Dataset: {Path(path).name}")
    print(f"{'='*55}")
    print(f"  Istanze (righe) : {df.shape[0]}")
    print(f"  Variabili (col) : {df.shape[1]}")
    print(f"\n  Colonne e tipi:")
    for col, dtype in df.dtypes.items():
        nulls = df[col].isnull().sum()
        print(f"    {col:35s} {str(dtype):10s}  null={nulls}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Statistiche descrittive
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola gli indicatori statistici descrittivi per ogni variabile numerica.

    Formule implementate esplicitamente
    ------------------------------------
    Media        :  x̄ = (1/n) Σ x_i
    Varianza     :  σ² = (1/n) Σ (x_i - x̄)²    [varianza di popolazione]
    Deviazione   :  σ  = √σ²
    Mediana      :  valore che divide la distribuzione ordinata a metà
    Massimo/min  :  max(x), min(x)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.

    Returns
    -------
    pd.DataFrame
        Tabella (variabili × indicatori): n, max, min, mean, median, variance, std.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = {}

    for col in numeric_cols:
        x = df[col].dropna().values.astype(float)
        n = len(x)

        # Media aritmetica: x̄ = Σ x_i / n
        x_mean = np.sum(x) / n

        # Mediana: percentile 50 senza funzioni statistiche preconfezionate
        x_sorted = np.sort(x)
        if n % 2 == 1:
            x_median = x_sorted[n // 2]
        else:
            x_median = (x_sorted[n // 2 - 1] + x_sorted[n // 2]) / 2.0

        # Varianza di popolazione: σ² = (1/n) Σ (x_i - x̄)²
        deviations = x - x_mean
        variance = np.dot(deviations, deviations) / n

        results[col] = {
            "n":        n,
            "max":      float(np.max(x)),
            "min":      float(np.min(x)),
            "mean":     float(x_mean),
            "median":   float(x_median),
            "variance": float(variance),
            "std":      float(np.sqrt(variance)),
        }

    return pd.DataFrame(results).T


# ─────────────────────────────────────────────────────────────────────────────
# 3. Intervallo al 90% centrato nella mediana
# ─────────────────────────────────────────────────────────────────────────────

def confidence_interval_90(series: pd.Series) -> tuple:
    """
    Calcola l'intervallo [m - δ, m + δ] centrato nella mediana m
    che contiene il 90% delle istanze.

    Metodo (non parametrico — non assume alcuna distribuzione)
    ----------------------------------------------------------
    1. Calcola la mediana m.
    2. Calcola le distanze assolute dalla mediana: d_i = |x_i - m|.
    3. δ = percentile_90(d_i)  →  l'intervallo [m-δ, m+δ]
       contiene esattamente il 90% delle osservazioni più vicine alla mediana.

    Parameters
    ----------
    series : pd.Series
        Colonna numerica del dataset.

    Returns
    -------
    tuple[float, float, float]
        (mediana m, semi-ampiezza δ, copertura effettiva in [0,1])
    """
    x = series.dropna().values.astype(float)
    n = len(x)

    # Mediana
    x_sorted = np.sort(x)
    if n % 2 == 1:
        m = x_sorted[n // 2]
    else:
        m = (x_sorted[n // 2 - 1] + x_sorted[n // 2]) / 2.0

    # Distanze assolute dalla mediana
    distances = np.abs(x - m)

    # δ = 90° percentile delle distanze
    distances_sorted = np.sort(distances)
    idx_90 = int(np.ceil(0.90 * n)) - 1  # indice 0-based
    idx_90 = min(idx_90, n - 1)
    delta = float(distances_sorted[idx_90])

    # Copertura effettiva (dovrebbe essere ≥ 90%)
    fraction = float(np.sum(distances <= delta) / n)

    return float(m), delta, fraction


# ─────────────────────────────────────────────────────────────────────────────
# 4. Matrice di covarianza
# ─────────────────────────────────────────────────────────────────────────────

def covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la matrice di covarianza esplicitamente tramite algebra matriciale.

    Formula
    -------
        C_ij = (1/n) Σ_k (x_{ki} - x̄_i)(x_{kj} - x̄_j)

    In forma matriciale compatta:
        X_c  = X - 1·x̄ᵀ          (matrice centrata: sottrae la media per colonna)
        C    = (1/n) X_c^T X_c   (prodotto matriciale)

    Nota: non si chiama numpy.cov() — il calcolo è esplicito.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.

    Returns
    -------
    pd.DataFrame
        Matrice di covarianza p×p con etichette delle colonne.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].dropna().values.astype(float)
    n = X.shape[0]

    # Centratura: X_c_{ki} = x_{ki} - x̄_i
    col_means = np.sum(X, axis=0) / n       # x̄_i per ogni colonna i
    X_centered = X - col_means              # broadcasting: (n,p) - (p,)

    # C = (1/n) X_c^T X_c   →   matrice p×p
    C = (X_centered.T @ X_centered) / n

    return pd.DataFrame(C, index=numeric_cols, columns=numeric_cols)


def correlation_from_cov(cov_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la matrice di correlazione di Pearson dalla matrice di covarianza.

    Formula
    -------
        R_ij = C_ij / (σ_i * σ_j)

    dove σ_i = √C_ii  (deviazione standard della variabile i).

    Parameters
    ----------
    cov_df : pd.DataFrame
        Matrice di covarianza p×p.

    Returns
    -------
    pd.DataFrame
        Matrice di correlazione p×p con valori in [-1, 1].
    """
    C = cov_df.values.astype(float)
    stds = np.sqrt(np.diag(C))           # σ_i = √C_ii
    # Evita divisione per zero
    stds[stds == 0] = 1e-12
    # R_ij = C_ij / (σ_i * σ_j)
    R = C / np.outer(stds, stds)
    # Clip per sicurezza numerica
    R = np.clip(R, -1.0, 1.0)
    return pd.DataFrame(R, index=cov_df.index, columns=cov_df.columns)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot: Bar chart degli indicatori statistici
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_stats(stats_df: pd.DataFrame, output_path: str = None):
    """
    Genera un grafico a barre degli indicatori statistici principali
    (media, mediana, deviazione standard) per ogni variabile continua.

    Ogni variabile ha il suo subplot per evitare problemi di scala
    (es. CK in migliaia vs età in anni).

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output di descriptive_stats().
    output_path : str, optional
        Se specificato, salva il plot come immagine PNG.
    """
    cols = stats_df.index.tolist()
    n = len(cols)
    ncols_plot = min(3, n)
    nrows_plot = (n + ncols_plot - 1) // ncols_plot

    fig, axes = plt.subplots(nrows_plot, ncols_plot,
                             figsize=(5 * ncols_plot, 4 * nrows_plot))
    fig.patch.set_facecolor("#0f1117")
    axes_flat = np.array(axes).flatten()

    indicators = ["mean", "median", "std"]
    colors     = ["#4f8ef7", "#f7a94f", "#f74f4f"]
    labels     = ["Media (x̄)", "Mediana (m)", "Dev. Std (σ)"]

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        ax.set_facecolor("#1a1d27")
        values = [float(stats_df.loc[col, ind]) for ind in indicators]

        bars = ax.bar(labels, values, color=colors, alpha=0.88, width=0.5,
                      edgecolor="#0f1117", linewidth=1.2)

        # Etichette sopra le barre
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.2f}",
                    ha="center", va="bottom", color="white", fontsize=9)

        ax.set_title(col, color="white", fontsize=11, pad=8)
        ax.tick_params(colors="white", labelsize=9)
        ax.set_facecolor("#1a1d27")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

    # Nascondi subplot vuoti
    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Indicatori Statistici per Variabile", color="white",
                 fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Plot salvato: {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot: Box plot affiancati
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame, output_path: str = None):
    """
    Genera box plot affiancati per tutte le variabili continue del dataset.

    Il box plot mostra (da sotto a sopra):
        - Whisker inferiore  : Q1 - 1.5*IQR
        - Primo quartile     : Q1 (25° percentile)
        - Mediana            : Q2 (50° percentile) — linea arancione
        - Terzo quartile     : Q3 (75° percentile)
        - Whisker superiore  : Q3 + 1.5*IQR
        - Outlier            : punti rossi oltre i whisker

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.
    output_path : str, optional
        Se specificato, salva il plot come immagine PNG.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(numeric_cols)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    fig.patch.set_facecolor("#0f1117")
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        ax.set_facecolor("#1a1d27")
        data = df[col].dropna().values

        bp = ax.boxplot(
            data,
            patch_artist=True,
            boxprops=dict(facecolor="#4f8ef730", color="#4f8ef7", linewidth=1.5),
            medianprops=dict(color="#f7a94f", linewidth=2.5),
            whiskerprops=dict(color="#aaaacc", linewidth=1.5),
            capprops=dict(color="#aaaacc", linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor="#f74f4f",
                            markeredgecolor="#f74f4f", alpha=0.6, markersize=5),
        )

        ax.set_title(col, color="white", fontsize=9, pad=6)
        ax.set_xticks([])
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

    fig.suptitle("Box Plot — Variabili Continue", color="white",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Plot salvato: {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Plot: ECDF vs Gaussiana teorica
# ─────────────────────────────────────────────────────────────────────────────

def plot_ecdf_vs_gaussian(df: pd.DataFrame, output_path: str = None):
    """
    Per ogni variabile continua, plotta la funzione di ripartizione empirica
    (ECDF) con sovrapposta la CDF della distribuzione Gaussiana avente
    la stessa media e varianza dei dati.

    Definizioni
    -----------
    ECDF:
        F̂(x) = (numero di osservazioni ≤ x) / n
        Costruita ordinando i dati: F̂(x_{(i)}) = i / n

    CDF Gaussiana (confronto teorico):
        Φ((x - μ) / σ)   con  μ = x̄,  σ = deviazione standard campionaria
        Calcolata tramite scipy.stats.norm.cdf (unico uso consentito di scipy).

    Lettura del grafico
    -------------------
    - Se ECDF ≈ Gaussiana → distribuzione approssimativamente normale.
    - Se le curve divergono significativamente → distribuzione asimmetrica
      o multimodale → la mediana è un indicatore più robusto della media.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.
    output_path : str, optional
        Se specificato, salva il plot come immagine PNG.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_vars = len(numeric_cols)
    ncols_plot = min(3, n_vars)
    nrows_plot = (n_vars + ncols_plot - 1) // ncols_plot

    fig, axes = plt.subplots(nrows_plot, ncols_plot,
                             figsize=(6 * ncols_plot, 4 * nrows_plot))
    fig.patch.set_facecolor("#0f1117")
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes_flat[i]
        ax.set_facecolor("#1a1d27")

        x_data = np.sort(df[col].dropna().values.astype(float))
        n = len(x_data)

        # ECDF: F̂(x_{(i)}) = i/n  per i = 1, ..., n
        ecdf_y = np.arange(1, n + 1) / n

        # Parametri Gaussiana: μ = x̄, σ = deviazione standard (campionaria)
        mu    = np.sum(x_data) / n
        var   = np.dot(x_data - mu, x_data - mu) / n
        sigma = np.sqrt(var)

        x_range  = np.linspace(x_data.min(), x_data.max(), 400)
        gauss_cdf = stats.norm.cdf(x_range, loc=mu, scale=sigma)

        ax.step(x_data, ecdf_y,
                color="#4f8ef7", linewidth=2, where="post", label="ECDF empirica")
        ax.plot(x_range, gauss_cdf,
                color="#f7a94f", linewidth=2, linestyle="--",
                label=f"N(μ={mu:.1f}, σ={sigma:.1f})")

        ax.axhline(0.5, color="#666688", linewidth=0.8, linestyle=":")
        ax.set_title(col, color="white", fontsize=10, pad=6)
        ax.set_xlabel("Valore", color="#aaaacc", fontsize=8)
        ax.set_ylabel("F(x)", color="#aaaacc", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)
        ax.legend(facecolor="#0f1117", labelcolor="white", fontsize=7,
                  loc="lower right")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

    for j in range(n_vars, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("ECDF Empirica vs Distribuzione Gaussiana Teorica",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Plot salvato: {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plot: Istogrammi per ogni variabile
# ─────────────────────────────────────────────────────────────────────────────

def plot_histograms(df: pd.DataFrame, output_path: str = None):
    """
    Genera istogrammi per ogni variabile continua del dataset.

    Ogni istogramma mostra la distribuzione di frequenza della variabile
    con sovrapposizione della curva di densità gaussiana calcolata con
    i parametri espliciti (media e deviazione standard dal calcolo diretto).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.
    output_path : str, optional
        Se specificato, salva il plot come immagine PNG.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_vars = len(numeric_cols)
    ncols_plot = min(3, n_vars)
    nrows_plot = (n_vars + ncols_plot - 1) // ncols_plot

    fig, axes = plt.subplots(nrows_plot, ncols_plot,
                             figsize=(6 * ncols_plot, 4 * nrows_plot))
    fig.patch.set_facecolor("#0f1117")
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes_flat[i]
        ax.set_facecolor("#1a1d27")

        data = df[col].dropna().values.astype(float)
        n = len(data)

        # Istogramma
        ax.hist(data, bins=min(15, max(5, n // 3)), color="#4f8ef7",
                alpha=0.75, edgecolor="#0f1117", linewidth=0.8, density=True)

        # Curva gaussiana sovrapposta
        mu = np.sum(data) / n
        var = np.dot(data - mu, data - mu) / n
        sigma = np.sqrt(var)
        x_range = np.linspace(data.min() - sigma, data.max() + sigma, 200)
        gauss = (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
                np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
        ax.plot(x_range, gauss, color="#f7a94f", linewidth=2,
                label=f"N(μ={mu:.1f}, σ={sigma:.1f})")

        ax.set_title(col, color="white", fontsize=10, pad=6)
        ax.set_xlabel("Valore", color="#aaaacc", fontsize=8)
        ax.set_ylabel("Densità", color="#aaaacc", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)
        ax.legend(facecolor="#0f1117", labelcolor="white", fontsize=7,
                  loc="upper right")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

    for j in range(n_vars, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Istogrammi delle Variabili Continue",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Plot salvato: {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Plot: Heatmap della matrice di covarianza
# ─────────────────────────────────────────────────────────────────────────────

def plot_covariance_heatmap(cov_df: pd.DataFrame, output_path: str = None):
    """
    Genera una heatmap della matrice di covarianza con annotazioni numeriche.

    Utilizza una colormap divergente centrata sullo zero:
    valori positivi in toni caldi, negativi in toni freddi.

    Parameters
    ----------
    cov_df : pd.DataFrame
        Matrice di covarianza p×p.
    output_path : str, optional
        Se specificato, salva il plot come immagine PNG.
    """
    labels = cov_df.columns.tolist()
    C = cov_df.values.astype(float)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(7, n * 0.8)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    vmax = np.max(np.abs(C))
    im = ax.imshow(C, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # Annotazioni numeriche nelle celle
    for i in range(n):
        for j in range(n):
            val = C[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "#cccccc"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=max(6, 9 - n // 4))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # Abbrevia i nomi se troppo lunghi
    short_labels = [l[:18] + "…" if len(l) > 20 else l for l in labels]
    ax.set_xticklabels(short_labels, rotation=40, ha="right",
                       color="white", fontsize=max(7, 10 - n // 5))
    ax.set_yticklabels(short_labels, color="white",
                       fontsize=max(7, 10 - n // 5))
    ax.tick_params(colors="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Covarianza", color="white", fontsize=10)

    ax.set_title("Matrice di Covarianza — C = (1/n) Xcᵀ Xc",
                 color="white", fontsize=13, pad=12)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Plot salvato: {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Plot: Heatmap della matrice di correlazione
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(corr_df: pd.DataFrame, output_path: str = None):
    """
    Genera una heatmap della matrice di correlazione di Pearson.

    La correlazione è derivata dalla covarianza:
        R_ij = C_ij / (σ_i * σ_j)

    I valori sono compresi in [-1, 1]:
        +1 = correlazione lineare perfetta positiva
         0 = nessuna correlazione lineare
        -1 = correlazione lineare perfetta negativa

    Parameters
    ----------
    corr_df : pd.DataFrame
        Matrice di correlazione p×p.
    output_path : str, optional
        Se specificato, salva il plot come immagine PNG.
    """
    labels = corr_df.columns.tolist()
    R = corr_df.values.astype(float)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(7, n * 0.8)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    im = ax.imshow(R, cmap="RdBu_r", aspect="auto",
                   vmin=-1, vmax=1, interpolation="nearest")

    # Annotazioni
    for i in range(n):
        for j in range(n):
            val = R[i, j]
            color = "white" if abs(val) > 0.6 else "#cccccc"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=max(6, 9 - n // 4))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_labels = [l[:18] + "…" if len(l) > 20 else l for l in labels]
    ax.set_xticklabels(short_labels, rotation=40, ha="right",
                       color="white", fontsize=max(7, 10 - n // 5))
    ax.set_yticklabels(short_labels, color="white",
                       fontsize=max(7, 10 - n // 5))
    ax.tick_params(colors="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Correlazione di Pearson", color="white", fontsize=10)

    ax.set_title("Matrice di Correlazione — R_ij = C_ij / (σ_i · σ_j)",
                 color="white", fontsize=13, pad=12)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Plot salvato: {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Orchestratore Fase 1
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(dataset_path: str, plots_dir: str = None) -> tuple:
    """
    Esegue l'intera Fase 1: Analisi 0-dimensionale.

    Sequenza
    --------
    1. Caricamento e ispezione del dataset.
    2. Calcolo indicatori statistici descrittivi.
    3. Calcolo intervalli al 90% centrati nella mediana.
    4. Calcolo matrice di covarianza.
    5. Plot: bar chart indicatori.
    6. Plot: box plot affiancati.
    7. Plot: ECDF vs Gaussiana.

    Parameters
    ----------
    dataset_path : str
        Percorso al file CSV del dataset.
    plots_dir : str, optional
        Cartella dove salvare i plot PNG. Se None i plot sono solo mostrati.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df, stats_df, cov_df)
    """
    print("\n" + "=" * 60)
    print("  FASE 1 — Analisi 0-Dimensionale")
    print("=" * 60)

    # ── 1. Caricamento ────────────────────────────────────────────
    df = load_dataset(dataset_path)

    # ── 2. Statistiche descrittive ────────────────────────────────
    print("\n\n[ STEP 2 ] Indicatori Statistici Descrittivi")
    print("-" * 55)
    stats_df = descriptive_stats(df)
    print(stats_df.to_string(float_format=lambda x: f"{x:>12.4f}"))

    # ── 3. Intervalli al 90% ──────────────────────────────────────
    print("\n\n[ STEP 3 ] Intervalli al 90% centrati nella mediana")
    print("-" * 55)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        m, delta, frac = confidence_interval_90(df[col])
        print(f"  {col:35s}  [{m - delta:8.3f} , {m + delta:8.3f}]"
              f"   δ={delta:.3f}   copertura={frac * 100:.1f}%")

    # ── 4. Matrice di covarianza ──────────────────────────────────
    print("\n\n[ STEP 4 ] Matrice di Covarianza  C = (1/n) Xc^T Xc")
    print("-" * 55)
    cov_df = covariance_matrix(df)
    print(cov_df.to_string(float_format=lambda x: f"{x:>12.4f}"))

    # ── 5. Matrice di correlazione ────────────────────────────────
    print("\n\n[ STEP 5 ] Matrice di Correlazione  R_ij = C_ij / (σ_i · σ_j)")
    print("-" * 55)
    corr_df = correlation_from_cov(cov_df)
    print(corr_df.to_string(float_format=lambda x: f"{x:>8.4f}"))

    # ── 6-10. Plot ────────────────────────────────────────────────
    print("\n\n[ STEP 6-10 ] Generazione plot...")
    p_bar    = f"{plots_dir}/bar_stats.png"       if plots_dir else None
    p_box    = f"{plots_dir}/boxplots.png"        if plots_dir else None
    p_ecdf   = f"{plots_dir}/ecdf_gaussian.png"  if plots_dir else None
    p_hist   = f"{plots_dir}/histograms.png"     if plots_dir else None
    p_covhm  = f"{plots_dir}/covariance_heatmap.png"   if plots_dir else None
    p_corrhm = f"{plots_dir}/correlation_heatmap.png"  if plots_dir else None

    plot_bar_stats(stats_df, output_path=p_bar)
    plot_boxplots(df,        output_path=p_box)
    plot_ecdf_vs_gaussian(df, output_path=p_ecdf)
    plot_histograms(df,      output_path=p_hist)
    plot_covariance_heatmap(cov_df,  output_path=p_covhm)
    plot_correlation_heatmap(corr_df, output_path=p_corrhm)

    print("\n✅  Fase 1 completata.")
    return df, stats_df, cov_df, corr_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point diretto (test rapido)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    BASE = Path(__file__).resolve().parent.parent
    run_phase1(
        dataset_path=str(BASE / "data" / "raw" / "ncbi_muscular_dystrophy_raw.csv"),
        plots_dir=str(BASE / "data" / "processed" / "plots"),
    )
