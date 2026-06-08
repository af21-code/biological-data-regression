# main.py
# Entry point della pipeline completa.
# Esegue tutte le fasi in sequenza seguendo la ROADMAP.md.

from pathlib import Path
from src.data_exploration import run_phase1

BASE_DIR = Path(__file__).resolve().parent

# Dataset reale NCBI GEO (generato da src/data_ingestion.py)
DATASET           = str(BASE_DIR / "data" / "raw" / "ncbi_muscular_dystrophy_raw.csv")
# Fallback: dataset sintetico generato in fase di setup
DATASET_SYNTHETIC = str(BASE_DIR / "data" / "raw" / "muscular_dystrophy.csv")
PLOTS_DIR         = str(BASE_DIR / "data" / "processed" / "plots")


def main():
    import os

    # Usa il dataset reale se disponibile, altrimenti il sintetico
    dataset = DATASET if os.path.exists(DATASET) else DATASET_SYNTHETIC

    # ── FASE 1: Analisi 0-dimensionale ────────────────────────────────────────
    df, stats_df, cov_df = run_phase1(
        dataset_path=dataset,
        plots_dir=PLOTS_DIR,
    )

    # ── FASE 2: Regressione Lineare Base ──────────────────────────────────────
    # TODO: implementare dopo completamento Fase 1

    # ── FASE 3: Train / Test Split ────────────────────────────────────────────
    # TODO: implementare dopo completamento Fase 2

    # ── FASE 4: Interpolazione ed Estrapolazione ──────────────────────────────
    # TODO: implementare dopo completamento Fase 3

    # ── FASE 5: Modello Multilineare ──────────────────────────────────────────
    # TODO: implementare dopo completamento Fase 4

    # ── FASE 6: Pulizia Outlier Dinamica ──────────────────────────────────────
    # TODO: implementare dopo completamento Fase 5


if __name__ == "__main__":
    main()
