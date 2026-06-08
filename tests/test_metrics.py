import numpy as np
from src.metrics import mse, rmse, r_squared

def test_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    # Se la previsione è perfetta, l'errore deve essere 0
    assert np.isclose(mse(y_true, y_pred), 0.0)

    y_pred_err = np.array([2.0, 3.0, 4.0])
    # Errore di 1 su ogni punto: l'MSE (errore quadratico medio) deve essere 1^2 = 1
    assert np.isclose(mse(y_true, y_pred_err), 1.0)

def test_rmse():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # L'MSE è (9+16)/2 = 12.5. Il RMSE è sqrt(12.5) ~= 3.5355
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(12.5))

def test_r_squared():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    # Previsione perfetta -> R^2 = 1
    assert np.isclose(r_squared(y_true, y_pred), 1.0)

    y_pred_mean = np.array([2.0, 2.0, 2.0])
    # Se il modello prevede sempre la media, l'R^2 è 0
    assert np.isclose(r_squared(y_true, y_pred_mean), 0.0)
