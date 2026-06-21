"""
Test per il modulo multivariate.py — Fase 5.
Verifica la regressione multilineare su sistemi con soluzione nota.
"""

import numpy as np
from src.multivariate import (
    add_bias_column,
    fit_multivariate,
    predict_multivariate,
)


def test_add_bias_column_shape():
    """add_bias_column deve aggiungere una colonna → shape (n, m+1)."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_b = add_bias_column(X)
    assert X_b.shape == (3, 3)


def test_add_bias_column_ones():
    """La prima colonna di X_b deve essere tutta 1."""
    X = np.array([[10.0, 20.0], [30.0, 40.0]])
    X_b = add_bias_column(X)
    assert np.allclose(X_b[:, 0], [1.0, 1.0])


def test_fit_known_system():
    """
    Sistema noto: y = 2*x1 + 3*x2 + 1 (intercetta=1).
    θ atteso: [1, 2, 3]
    """
    X = np.array([
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [3.0, 2.0],
        [2.0, 3.0],
    ])
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 1.0
    theta = fit_multivariate(X, y)
    assert np.isclose(theta[0], 1.0, atol=1e-6)   # intercetta
    assert np.isclose(theta[1], 2.0, atol=1e-6)   # a1
    assert np.isclose(theta[2], 3.0, atol=1e-6)   # a2


def test_fit_with_regularization():
    """Con λ > 0, i coefficienti devono essere leggermente diversi (shrinkage)."""
    X = np.array([
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [3.0, 2.0],
        [2.0, 3.0],
    ])
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 1.0
    theta_ols   = fit_multivariate(X, y, lambda_reg=0.0)
    theta_ridge = fit_multivariate(X, y, lambda_reg=10.0)
    # Ridge shrinks coefficients towards zero
    assert np.linalg.norm(theta_ridge[1:]) < np.linalg.norm(theta_ols[1:])


def test_predict_multivariate_consistency():
    """predict_multivariate deve restituire X_b @ theta."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    theta = np.array([0.5, 1.0, -1.0])  # [b, a1, a2]
    y_pred = predict_multivariate(X, theta)
    expected = np.array([
        0.5 + 1.0 * 1.0 + (-1.0) * 2.0,  # 0.5 + 1 - 2 = -0.5
        0.5 + 1.0 * 3.0 + (-1.0) * 4.0,  # 0.5 + 3 - 4 = -0.5
    ])
    assert np.allclose(y_pred, expected)


def test_fit_predict_roundtrip():
    """fit → predict deve restituire y (entro tolleranza) per sistema esatto."""
    np.random.seed(42)
    X = np.random.randn(20, 3)
    theta_true = np.array([2.0, -1.0, 0.5, 3.0])  # [b, a1, a2, a3]
    y = add_bias_column(X) @ theta_true

    theta_est = fit_multivariate(X, y)
    y_pred = predict_multivariate(X, theta_est)
    assert np.allclose(y, y_pred, atol=1e-8)
