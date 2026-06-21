"""
Test per il modulo linear_regression.py — Fasi 2/3/4/6.
Verifica la regressione lineare e lo splitting su dati sintetici noti.
"""

import numpy as np
import pandas as pd
from src.linear_regression import (
    fit_linear_regression,
    predict,
    train_test_split,
)


def test_fit_known_line():
    """Retta nota y = 2x + 1: verifica a ≈ 2, b ≈ 1."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0
    a, b = fit_linear_regression(x, y)
    assert np.isclose(a, 2.0)
    assert np.isclose(b, 1.0)


def test_fit_noisy_line():
    """Retta y = 3x + 5 con rumore minimo: a ≈ 3, b ≈ 5."""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 3.0 * x + 5.0 + np.random.normal(0, 0.01, 100)
    a, b = fit_linear_regression(x, y)
    assert abs(a - 3.0) < 0.05
    assert abs(b - 5.0) < 0.2


def test_predict_output():
    """Verifica che predict(x, a, b) restituisca a*x + b."""
    x = np.array([1.0, 2.0, 3.0])
    a, b = 2.0, 3.0
    y_pred = predict(x, a, b)
    expected = np.array([5.0, 7.0, 9.0])
    assert np.allclose(y_pred, expected)


def test_predict_scalar():
    """predict deve funzionare anche con uno scalare."""
    y = predict(5.0, 2.0, 1.0)
    assert np.isclose(y, 11.0)


def test_train_test_split_sizes():
    """Verifica che lo split 90/10 produca le dimensioni corrette."""
    df = pd.DataFrame({
        "x": np.arange(20, dtype=float),
        "y": np.arange(20, dtype=float),
    })
    x_tr, x_te, y_tr, y_te = train_test_split(df, "x", "y", test_ratio=0.10)
    assert len(x_tr) + len(x_te) == 20
    assert len(x_te) == 2  # ceil(20 * 0.10) = 2


def test_train_test_split_reproducibility():
    """Lo split con lo stesso seed deve produrre gli stessi risultati."""
    df = pd.DataFrame({
        "x": np.arange(20, dtype=float),
        "y": np.arange(20, dtype=float),
    })
    split1 = train_test_split(df, "x", "y", seed=42)
    split2 = train_test_split(df, "x", "y", seed=42)
    for a, b in zip(split1, split2):
        assert np.array_equal(a, b)


def test_train_test_split_no_overlap():
    """Training e test set non devono condividere campioni."""
    df = pd.DataFrame({
        "x": np.arange(20, dtype=float),
        "y": np.arange(20, dtype=float) * 2,
    })
    x_tr, x_te, y_tr, y_te = train_test_split(df, "x", "y", test_ratio=0.20)
    # Verifica che nessun valore x sia presente in entrambi i set
    overlap = set(x_tr.tolist()) & set(x_te.tolist())
    assert len(overlap) == 0
