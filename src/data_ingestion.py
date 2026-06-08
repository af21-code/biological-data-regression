"""
data_ingestion.py
=================
Acquisizione programmatica del dataset reale NCBI GEO per la Distrofia Muscolare.

Dataset target: GSE38417
  Titolo  : "Gene expression profiling of Duchenne muscular dystrophy"
  Campioni: biopsie muscolari di pazienti DMD e controlli sani
  Piattaf.: GPL570 (Affymetrix Human Genome U133 Plus 2.0 Array)

Fallback : GSE1004  (Haslett et al. 2002 - GPL8300, HG-U95Av2)
  13 pazienti DMD + 7 controlli sani

Strategia di estrazione
------------------------
1. Download del Series Matrix File via FTP NCBI (urllib, nessuna lib esterna).
2. Parsing manuale del formato GEO (righe !, tabella dati, fine tabella).
3. Download dell'annotazione della piattaforma (GPL) per mappare
   probe ID → simbolo genico.
4. Identificazione del probe del gene DMD (distrofina) → variabile target y.
   Fallback: probe con varianza massima se DMD non è trovato nell'annotazione.
5. Selezione delle N_FEATURES feature predittive più correlate a y (matrice X).
6. Imputazione NaN tramite media di colonna (giustificazione: i valori mancanti
   nei microarray derivano da segnale sotto soglia, la media è stima conservativa).
7. Salvataggio in data/raw/ncbi_muscular_dystrophy_raw.csv.

Output a schermo
-----------------
- Dimensioni algebriche della matrice X ∈ ℝ^{n×m}
- Statistiche di base del vettore y ∈ ℝ^n (continuità numerica)
"""

import urllib.request
import urllib.error
import ssl
import gzip
import io
import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import certifi
    # Contesto SSL con i certificati aggiornati di certifi
    # Necessario su macOS con Python.org build (non usa i cert di sistema)
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    # Fallback: disabilita verifica (non sicuro, solo per sviluppo accademico)
    SSL_CONTEXT = ssl._create_unverified_context()


# ─────────────────────────────────────────────────────────────────────────────
# Configurazione Dataset
# ─────────────────────────────────────────────────────────────────────────────

# Dataset primario: GSE38417 (GPL570, HG-U133 Plus 2.0)
PRIMARY_DATASET = {
    "id":          "GSE38417",
    "matrix_url":  (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE38nnn/GSE38417/matrix/"
        "GSE38417_series_matrix.txt.gz"
    ),
    "platform_url": (
        "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570nnn/GPL570/annot/"
        "GPL570.annot.gz"
    ),
}

# Dataset di fallback: GSE1004 (GPL8300, HG-U95Av2)
FALLBACK_DATASET = {
    "id":          "GSE1004",
    "matrix_url":  (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1004/matrix/"
        "GSE1004_series_matrix.txt.gz"
    ),
    "platform_url": (
        "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL8nnn/GPL8300/annot/"
        "GPL8300.annot.gz"
    ),
}

# Gene bersaglio per la variabile dipendente y
TARGET_GENE = "DMD"           # distrofina

# Numero di feature predittive da includere nella matrice X
N_FEATURES = 10

# Percorso di output
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "ncbi_muscular_dystrophy_raw.csv"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Download e parsing del Series Matrix File
# ─────────────────────────────────────────────────────────────────────────────

def download_series_matrix(url: str) -> tuple:
    """
    Scarica e parsifica il GEO Series Matrix File.

    Formato GEO Series Matrix:
      - Righe che iniziano con '!' = metadati (Series e Sample info)
      - '!series_matrix_table_begin' = inizio della tabella dati
      - Prima riga della tabella = header con gli ID GSM dei campioni
      - Righe successive = probe_ID \\t val_campione1 \\t val_campione2 ...
      - '!series_matrix_table_end' = fine della tabella

    Returns
    -------
    tuple[pd.DataFrame, dict]
        expr_df : DataFrame (probes × campioni) con valori float
        metadata: dizionario dei metadati del dataset
    """
    print(f"  [1/4] Download Series Matrix: {url.split('/')[-1]}")

    with urllib.request.urlopen(url, timeout=120, context=SSL_CONTEXT) as response:
        raw_bytes = response.read()

    print(f"        File scaricato: {len(raw_bytes) / 1024:.0f} KB")

    # Decompressione gzip in memoria
    with gzip.open(io.BytesIO(raw_bytes), "rt", encoding="utf-8",
                   errors="replace") as fh:
        lines = fh.readlines()

    metadata = {}
    table_lines = []
    in_table = False

    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        elif line.startswith("!series_matrix_table_end"):
            break
        elif in_table:
            table_lines.append(line)
        elif line.startswith("!"):
            parts = line.split("\t", 1)
            if len(parts) == 2:
                key = parts[0].lstrip("!").strip()
                val = parts[1].strip().strip('"')
                metadata[key] = val

    if not table_lines:
        raise ValueError("Tabella dati non trovata nel Series Matrix File.")

    # Parsing header: "ID_REF"  GSM_ID1  GSM_ID2 ...
    header = [h.strip().strip('"') for h in table_lines[0].split("\t")]
    sample_ids = header[1:]   # prima colonna = "ID_REF" → scartata
    print(f"        Campioni trovati: {len(sample_ids)}")

    # Parsing dati: ogni riga = probe_id + valori numerici
    probe_ids = []
    values_list = []

    for line in table_lines[1:]:
        parts = [p.strip().strip('"') for p in line.split("\t")]
        if not parts or parts[0] == "":
            continue
        probe_ids.append(parts[0])
        row_vals = []
        for v in parts[1:]:
            try:
                row_vals.append(float(v))
            except ValueError:
                row_vals.append(np.nan)
        values_list.append(row_vals)

    # Matrice: probes × campioni
    n_probes   = len(probe_ids)
    n_samples  = len(sample_ids)
    data_array = np.array(values_list, dtype=float)   # shape: (n_probes, n_samples)

    # Padding se alcune righe sono più corte
    max_cols = n_samples
    padded = np.full((n_probes, max_cols), np.nan)
    for i, row in enumerate(values_list):
        l = min(len(row), max_cols)
        padded[i, :l] = row[:l]

    expr_df = pd.DataFrame(padded, index=probe_ids, columns=sample_ids)
    print(f"        Matrice espressione: {n_probes} probe × {n_samples} campioni")

    return expr_df, metadata


# ─────────────────────────────────────────────────────────────────────────────
# 2. Download e parsing dell'annotazione della piattaforma
# ─────────────────────────────────────────────────────────────────────────────

def download_platform_annotation(url: str) -> dict:
    """
    Scarica l'annotazione della piattaforma Affymetrix dal GEO FTP.

    Il file .annot.gz è tab-delimited con colonne:
        ID | Gene Title | Gene Symbol | UniRef | Entrez Gene ID | ...

    Returns
    -------
    dict[str, str]
        Mappa  probe_id → gene_symbol  (es. "202429_s_at" → "DMD")
    """
    print(f"  [2/4] Download annotazione piattaforma ...")

    with urllib.request.urlopen(url, timeout=120, context=SSL_CONTEXT) as response:
        raw_bytes = response.read()

    probe_to_gene = {}

    with gzip.open(io.BytesIO(raw_bytes), "rt", encoding="utf-8",
                   errors="replace") as fh:
        header_found = False
        col_id  = 0
        col_sym = None

        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#") or line == "":
                continue

            parts = line.split("\t")

            if not header_found:
                # Cerca l'header con "Gene Symbol"
                if "Gene Symbol" in line or "ID" in line:
                    try:
                        col_sym = parts.index("Gene Symbol")
                        header_found = True
                    except ValueError:
                        # Prova a trovarlo con un cerca case-insensitive
                        for idx, h in enumerate(parts):
                            if "symbol" in h.lower():
                                col_sym = idx
                                header_found = True
                                break
                continue

            if col_sym is None:
                continue

            probe_id = parts[col_id].strip()
            try:
                gene_sym = parts[col_sym].strip()
            except IndexError:
                gene_sym = ""

            if probe_id and gene_sym:
                # Alcune celle contengono più simboli separati da " /// "
                for sym in gene_sym.split(" /// "):
                    probe_to_gene[probe_id] = sym.strip()
                    break  # prendi solo il primo

    print(f"        Annotazione caricata: {len(probe_to_gene)} probe mappati")
    return probe_to_gene


# ─────────────────────────────────────────────────────────────────────────────
# 3. Imputazione NaN (media di colonna)
# ─────────────────────────────────────────────────────────────────────────────

def impute_nan_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa i valori NaN con la media della rispettiva colonna.

    Giustificazione
    ---------------
    Nei dati di espressione da microarray, un valore NaN indica tipicamente
    un segnale di ibridazione sotto la soglia di rilevamento.
    L'imputazione con la media di colonna è una stima conservativa: assume
    che il gene abbia un'espressione "media" nella popolazione, introducendo
    la minima distorsione possibile nel dataset per campioni sporadici mancanti.
    Per dataset con molti NaN (> 10%) si raccomanda l'imputazione k-NN.

    Matematicamente:
        x̂_{ij} = x̄_j = (1 / |{k : x_{kj} ≠ NaN}|) * Σ_{k: x_{kj}≠NaN} x_{kj}
    """
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        pct = nan_count / (df.shape[0] * df.shape[1]) * 100
        print(f"        NaN rilevati: {nan_count} ({pct:.2f}%) → imputazione con media di colonna")
        col_means = df.mean(axis=0)
        df = df.fillna(col_means)
    else:
        print(f"        Nessun NaN rilevato.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Selezione variabile target y e feature matrix X
# ─────────────────────────────────────────────────────────────────────────────

def select_target_and_features(
    expr_df: pd.DataFrame,
    probe_to_gene: dict,
    target_gene: str = "DMD",
    n_features: int = 10
) -> tuple:
    """
    Seleziona la variabile dipendente y e la matrice dei predittori X.

    Strategia per y
    ---------------
    1. Cerca tutti i probe associati al gene TARGET_GENE nell'annotazione.
    2. Se trovati, usa il probe con varianza massima (il più informativo).
    3. Se non trovati, usa il probe con varianza massima nell'intero dataset
       (fallback biologicamente motivato: i geni più variabili in campioni
       DMD vs controlli tendono a essere i più rilevanti per la patologia).

    Strategia per X
    ---------------
    Seleziona i N_FEATURES probe più correlati linearmente a y (escludendo y stesso).
    La correlazione lineare di Pearson è calcolata esplicitamente:
        r_{ij} = C_{ij} / (σ_i * σ_j)
    dove C è la matrice di covarianza.

    Parameters
    ----------
    expr_df     : DataFrame (probes × campioni) — NaN già imputati
    probe_to_gene: dict {probe_id → gene_symbol}
    target_gene : simbolo del gene target (default "DMD")
    n_features  : numero di feature predittive da selezionare

    Returns
    -------
    tuple[pd.Series, pd.DataFrame, str]
        y_series   : valori del target (n campioni,)
        X_df       : matrice feature (n campioni × n_features)
        target_probe: ID del probe usato come target
    """
    # ── 4a. Identificazione probe target ─────────────────────────────────────
    target_probes = [
        pid for pid, gsym in probe_to_gene.items()
        if gsym.upper() == target_gene.upper() and pid in expr_df.index
    ]

    if target_probes:
        # Seleziona il probe con varianza massima tra quelli del gene target
        variances = expr_df.loc[target_probes].var(axis=1)
        target_probe = variances.idxmax()
        print(f"  [3/4] Target y: gene {target_gene} → probe '{target_probe}' "
              f"(varianza={variances[target_probe]:.4f})")
    else:
        # Fallback: probe con varianza massima nell'intero dataset
        variances = expr_df.var(axis=1)
        target_probe = variances.idxmax()
        gene_name = probe_to_gene.get(target_probe, "N/A")
        print(f"  [3/4] Gene '{target_gene}' non trovato nell'annotazione.")
        print(f"        Fallback → probe con varianza max: '{target_probe}' "
              f"(gene={gene_name}, varianza={variances[target_probe]:.4f})")

    # Vettore target y: espressione del probe target per ogni campione
    y_series = expr_df.loc[target_probe]

    # ── 4b. Selezione feature X per correlazione lineare con y ───────────────
    # Calcolo esplicito della correlazione di Pearson tra y e ogni altro probe
    # r = (Σ (x_i - x̄)(y_i - ȳ)) / (n * σ_x * σ_y)

    y_vals = y_series.values.astype(float)
    y_centered = y_vals - np.mean(y_vals)
    y_std = np.sqrt(np.dot(y_centered, y_centered) / len(y_vals))

    # Calcola correlazione per ogni probe (escluso il target)
    other_probes = [p for p in expr_df.index if p != target_probe]
    X_matrix = expr_df.loc[other_probes].values.astype(float)   # (n_probes-1, n_samples)

    # Centra ogni riga (probe)
    X_means   = X_matrix.mean(axis=1, keepdims=True)
    X_centered = X_matrix - X_means                              # (n_probes-1, n_samples)
    X_stds     = np.sqrt((X_centered ** 2).mean(axis=1))        # (n_probes-1,)

    # Correlazione: vettore r di lunghezza (n_probes-1,)
    # r_i = (X_centered[i,:] · y_centered) / (n * std_i * std_y)
    n_s = len(y_vals)
    correlations = (X_centered @ y_centered) / (n_s * X_stds * y_std + 1e-12)

    # Seleziona i N_FEATURES probe con |correlazione| massima
    abs_corr = np.abs(correlations)
    top_indices = np.argsort(abs_corr)[::-1][:n_features]
    selected_probes = [other_probes[i] for i in top_indices]

    print(f"        Top {n_features} probe predittivi selezionati per |correlazione| con y:")
    for i, pid in enumerate(selected_probes):
        gsym = probe_to_gene.get(pid, "?")
        print(f"          {i+1:2d}. {pid:20s}  gene={gsym:15s}  r={correlations[other_probes.index(pid)]:+.4f}")

    # Matrice X finale: trasposta → (campioni × feature)
    X_df = expr_df.loc[selected_probes].T
    X_df.columns = [probe_to_gene.get(p, p) + f"_{p}" for p in selected_probes]

    return y_series, X_df, target_probe


# ─────────────────────────────────────────────────────────────────────────────
# 5. Salvataggio e report finale
# ─────────────────────────────────────────────────────────────────────────────

def save_dataset(y_series: pd.Series, X_df: pd.DataFrame, output_path: Path):
    """
    Unisce y e X in un unico DataFrame e salva in CSV.

    Struttura del CSV
    -----------------
    - Righe    : campioni (GSM_ID)
    - Colonne  : feature predittive X + colonna target 'muscle_strength_proxy'
    - Ultima col: 'target_expression' = y (espressione del gene DMD/target)
    """
    df_out = X_df.copy()
    df_out["target_expression"] = y_series.values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=True, index_label="sample_id")
    print(f"\n  [4/4] Dataset salvato in: {output_path}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Report algebrico (requisito progetto)
# ─────────────────────────────────────────────────────────────────────────────

def print_algebraic_report(y_series: pd.Series, X_df: pd.DataFrame,
                           target_probe: str):
    """
    Stampa le dimensioni algebriche e le statistiche di base di y,
    confermando la continuità numerica del vettore target.
    """
    n, m = X_df.shape
    y = y_series.values.astype(float)

    # Statistiche esplicite (nessuna chiamata a .describe())
    y_mean   = np.sum(y) / len(y)
    y_sorted = np.sort(y)
    y_median = (
        y_sorted[len(y) // 2]
        if len(y) % 2 == 1
        else (y_sorted[len(y) // 2 - 1] + y_sorted[len(y) // 2]) / 2
    )
    y_var    = np.dot(y - y_mean, y - y_mean) / len(y)
    y_std    = np.sqrt(y_var)
    unique   = len(np.unique(y))

    print("\n" + "=" * 65)
    print("  REPORT ALGEBRICO — Dataset NCBI GEO")
    print("=" * 65)
    print(f"\n  Matrice predittori  X  ∈  ℝ^{{{n} × {m}}}")
    print(f"    n = {n}  campioni  (osservazioni / pazienti)")
    print(f"    m = {m}  feature   (livelli di espressione genica)")
    print(f"\n  Vettore target  y  ∈  ℝ^{{{n}}}  [probe: {target_probe}]")
    print(f"    Valori unici   : {unique}  → output CONTINUO ✅")
    print(f"    Minimo         : {np.min(y):.6f}")
    print(f"    Massimo        : {np.max(y):.6f}")
    print(f"    Media     (x̄)  : {y_mean:.6f}")
    print(f"    Mediana   (m)  : {y_median:.6f}")
    print(f"    Varianza  (σ²) : {y_var:.6f}")
    print(f"    Dev. Std  (σ)  : {y_std:.6f}")
    print(f"\n  Il vettore y ha {unique} valori distinti su {n} campioni.")
    print(f"  Questo conferma la natura CONTINUA dell'output, requisito")
    print(f"  imprescindibile del Progetto B (regressione a output continuo).")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Orchestratore principale
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(
    dataset_config: dict = None,
    target_gene:    str  = TARGET_GENE,
    n_features:     int  = N_FEATURES,
    output_path:    Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Esegue l'intera pipeline di acquisizione dati NCBI GEO.

    Parametri
    ---------
    dataset_config : dict con chiavi 'id', 'matrix_url', 'platform_url'.
                     Default: PRIMARY_DATASET (GSE38417).
                     Fallback automatico su FALLBACK_DATASET (GSE1004).
    target_gene    : simbolo genico del target y (default: "DMD").
    n_features     : numero di predittori da includere in X (default: 10).
    output_path    : percorso di salvataggio del CSV.

    Returns
    -------
    pd.DataFrame
        Dataset finale (campioni × feature+target) pronto per la Fase 1.
    """
    if dataset_config is None:
        dataset_config = PRIMARY_DATASET

    datasets_to_try = [dataset_config, FALLBACK_DATASET]

    for cfg in datasets_to_try:
        print(f"\n{'='*65}")
        print(f"  ACQUISIZIONE DATASET NCBI GEO: {cfg['id']}")
        print(f"{'='*65}")
        try:
            # ── Step 1: Series Matrix ────────────────────────────────────────
            expr_df, metadata = download_series_matrix(cfg["matrix_url"])

            # ── Step 2: Annotazione piattaforma ─────────────────────────────
            try:
                probe_to_gene = download_platform_annotation(cfg["platform_url"])
            except Exception as e:
                print(f"        Annotazione non disponibile ({e}).")
                print(f"        Proseguo senza mappa probe→gene.")
                probe_to_gene = {}

            # ── Step 3: Imputazione NaN ──────────────────────────────────────
            expr_df = impute_nan_mean(expr_df)

            # ── Step 4: Selezione y e X ──────────────────────────────────────
            y_series, X_df, target_probe = select_target_and_features(
                expr_df, probe_to_gene, target_gene, n_features
            )

            # ── Step 5: Salvataggio ──────────────────────────────────────────
            df_out = save_dataset(y_series, X_df, output_path)

            # ── Step 6: Report algebrico ─────────────────────────────────────
            print_algebraic_report(y_series, X_df, target_probe)

            print(f"\n✅  Ingestion completata. File: {output_path.name}")
            return df_out

        except urllib.error.URLError as e:
            print(f"\n  ⚠️  Download fallito per {cfg['id']}: {e}")
            print(f"      Provo il dataset di fallback...")
            continue
        except Exception as e:
            print(f"\n  ❌  Errore inatteso con {cfg['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n❌  Tutti i dataset hanno fallito il download.")
    print("    Verifica la connessione internet e riprova.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point diretto
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ingestion()
