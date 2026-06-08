# main.py
# Entry point della pipeline completa.
# Esegue tutte le fasi in sequenza seguendo la ROADMAP.md.

import os
from pathlib import Path
from src.data_exploration import run_phase1
from src.linear_regression import run_phase2, run_phase3, run_phase4, run_phase6

BASE_DIR  = Path(__file__).resolve().parent
DATASET   = str(BASE_DIR / "data" / "raw" / "ncbi_muscular_dystrophy_raw.csv")
DATASET_SYNTHETIC = str(BASE_DIR / "data" / "raw" / "muscular_dystrophy.csv")
PLOTS_DIR = str(BASE_DIR / "data" / "processed" / "plots")
TARGET_COL = "target_expression"


def main():
    # Usa il dataset reale se disponibile, altrimenti il sintetico
    dataset = DATASET if os.path.exists(DATASET) else DATASET_SYNTHETIC

    # ── FASE 1: Analisi 0-dimensionale ────────────────────────────────────────
    df, stats_df, cov_df = run_phase1(
        dataset_path=dataset,
        plots_dir=PLOTS_DIR,
    )

    # ── FASE 2: Regressione Lineare Base ──────────────────────────────────────
    predictor_col, a, b, results_df = run_phase2(
        df=df,
        target_col=TARGET_COL,
        cov_df=cov_df,
        plots_dir=PLOTS_DIR,
    )

    # ── FASE 3: Train / Test Split ────────────────────────────────────────────
    a_train, b_train, splits = run_phase3(
        df=df,
        predictor_col=predictor_col,
        target_col=TARGET_COL,
        plots_dir=PLOTS_DIR,
    )

    # ── FASE 4: Interpolazione ed Estrapolazione ──────────────────────────────
    run_phase4(
        df=df,
        predictor_col=predictor_col,
        target_col=TARGET_COL,
        a=a_train,
        b=b_train,
        plots_dir=PLOTS_DIR,
    )

    # ── FASE 5: Modello Multilineare ──────────────────────────────────────────
    # TODO: implementare dopo completamento Fase 4

    # ── FASE 6: Pulizia Outlier Dinamica ──────────────────────────────────────
    run_phase6(
        df=df,
        predictor_col=predictor_col,
        target_col=TARGET_COL,
        plots_dir=PLOTS_DIR,
    )


if __name__ == "__main__":
    main()
