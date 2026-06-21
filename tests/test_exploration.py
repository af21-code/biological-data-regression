"""
Test per il modulo data_exploration.py — Fase 1.
Verifica le funzioni di statistica descrittiva su dati noti.
"""

import numpy as np
import pandas as pd
from src.data_exploration import (
    descriptive_stats,
    confidence_interval_90,
    covariance_matrix,
    correlation_from_cov,
)


def test_descriptive_stats_mean():
    """La media di [1, 2, 3, 4, 5] deve essere 3.0."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = descriptive_stats(df)
    assert np.isclose(stats.loc["x", "mean"], 3.0)


def test_descriptive_stats_median_odd():
    """La mediana di [1, 2, 3, 4, 5] (n dispari) deve essere 3.0."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = descriptive_stats(df)
    assert np.isclose(stats.loc["x", "median"], 3.0)


def test_descriptive_stats_median_even():
    """La mediana di [1, 2, 3, 4] (n pari) deve essere 2.5."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    stats = descriptive_stats(df)
    assert np.isclose(stats.loc["x", "median"], 2.5)


def test_descriptive_stats_variance():
    """Varianza di popolazione di [2, 4, 4, 4, 5, 5, 7, 9] = 4.0."""
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    df = pd.DataFrame({"x": data})
    stats = descriptive_stats(df)
    expected_var = np.sum((np.array(data) - np.mean(data))**2) / len(data)
    assert np.isclose(stats.loc["x", "variance"], expected_var)


def test_confidence_interval_90_coverage():
    """L'intervallo al 90% deve contenere almeno il 90% dei dati."""
    np.random.seed(42)
    data = np.random.normal(10, 2, 100)
    series = pd.Series(data)
    m, delta, frac = confidence_interval_90(series)
    assert frac >= 0.90


def test_covariance_matrix_symmetry():
    """La matrice di covarianza deve essere simmetrica."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [4.0, 3.0, 2.0, 1.0],
    })
    cov = covariance_matrix(df)
    assert np.allclose(cov.values, cov.values.T)


def test_covariance_matrix_known_values():
    """Verifica su matrice 2x2 con calcolo manuale."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 4.0, 6.0],
    })
    cov = covariance_matrix(df)
    # a: mean=2, b: mean=4
    # Cov(a,a) = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    # Cov(a,b) = ((1-2)(2-4) + (2-2)(4-4) + (3-2)(6-4)) / 3 = 4/3
    # Cov(b,b) = ((2-4)^2 + (4-4)^2 + (6-4)^2) / 3 = 8/3
    assert np.isclose(cov.loc["a", "a"], 2.0 / 3.0)
    assert np.isclose(cov.loc["a", "b"], 4.0 / 3.0)
    assert np.isclose(cov.loc["b", "b"], 8.0 / 3.0)


def test_correlation_from_cov_diagonal():
    """La diagonale della matrice di correlazione deve essere tutta 1."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [5.0, 4.0, 3.0, 2.0, 1.0],
        "c": [2.0, 3.0, 5.0, 7.0, 11.0],
    })
    cov = covariance_matrix(df)
    corr = correlation_from_cov(cov)
    for col in corr.columns:
        assert np.isclose(corr.loc[col, col], 1.0)


def test_correlation_from_cov_perfect_negative():
    """Due variabili perfettamente anticorrelate: r = -1."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [5.0, 4.0, 3.0, 2.0, 1.0],
    })
    cov = covariance_matrix(df)
    corr = correlation_from_cov(cov)
    assert np.isclose(corr.loc["a", "b"], -1.0)
