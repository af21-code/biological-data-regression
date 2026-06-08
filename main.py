# main.py
# Entry point della pipeline completa.
# Esegue tutte le fasi in sequenza seguendo la ROADMAP.md.

import os
from pathlib import Path
from src.data_exploration import run_phase1
from src.linear_regression import run_phase2, run_phase3, run_phase4, run_phase6
from src.multivariate import run_phase5
from src.metrics import mse, rmse, r_squared
from src.linear_regression import fit_linear_regression, predict, train_test_split

BASE_DIR   = Path(__file__).resolve().parent
DATASET    = str(BASE_DIR / "data" / "raw" / "ncbi_muscular_dystrophy_raw.csv")
PLOTS_DIR  = str(BASE_DIR / "data" / "processed" / "plots")
TARGET_COL = "target_expression"


def main():
    dataset = DATASET if os.path.exists(DATASET) else str(
        BASE_DIR / "data" / "raw" / "muscular_dystrophy.csv")

    # ── FASE 1: Analisi 0-dimensionale ────────────────────────────────────────
    df, stats_df, cov_df = run_phase1(
        dataset_path=dataset,
        plots_dir=PLOTS_DIR,
    )

    # ── FASE 2: Regressione Lineare Base ──────────────────────────────────────
    predictor_col, a, b, _ = run_phase2(
        df=df,
        target_col=TARGET_COL,
        cov_df=cov_df,
        plots_dir=PLOTS_DIR,
    )

    # ── FASE 3: Train / Test Split ────────────────────────────────────────────
    a_train, b_train, (x_tr, x_te, y_tr, y_te) = run_phase3(
        df=df,
        predictor_col=predictor_col,
        target_col=TARGET_COL,
        plots_dir=PLOTS_DIR,
    )

    # Raccoglie le metriche della regressione semplice per il confronto (Fase 5)
    y_pred_tr = predict(x_tr, a_train, b_train)
    y_pred_te = predict(x_te, a_train, b_train)
    simple_metrics = {
        "mse_train":  mse(y_tr, y_pred_tr),
        "rmse_train": rmse(y_tr, y_pred_tr),
        "r2_train":   r_squared(y_tr, y_pred_tr),
        "mse_test":   mse(y_te, y_pred_te),
        "rmse_test":  rmse(y_te, y_pred_te),
        "r2_test":    r_squared(y_te, y_pred_te),
    }

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
    theta, res_train, res_test = run_phase5(
        df=df,
        target_col=TARGET_COL,
        plots_dir=PLOTS_DIR,
        lambda_reg=0.0,
        simple_metrics=simple_metrics,
    )

    # ── FASE 6: Pulizia Outlier Dinamica ──────────────────────────────────────
    run_phase6(
        df=df,
        predictor_col=predictor_col,
        target_col=TARGET_COL,
        plots_dir=PLOTS_DIR,
    )


if __name__ == "__main__":
    main()
