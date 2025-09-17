from loguru import logger
import pytest
import numpy as np
from lin_mod.modeling.linear_model import BGDRegressorCustom

@pytest.fixture
def model_bce():
    """
    Creates BGDRegressorCustom instance before each test.
    NOTE: Configured with binary_cross_entropy
    """

    model = BGDRegressorCustom()
    model.configure(
        epochs=100,
        loss_function="binary_cross_entropy",
        learning_rate=0.01,
        regularization=("L2", 0.1),
        logging=True
    ) # configurations have been tested via test_configurations.py

    return model

@pytest.fixture
def model_mse():
    """
    Creates BGDRegressorCustom instance before each test.
    NOTE: Configured with mean_Squared
    """

    model = BGDRegressorCustom()
    model.configure(
        epochs=100,
        loss_function="mean_squared",
        learning_rate=0.01,
        regularization=("L2", 0.1),
        logging=True
    ) # configurations have been tested via test_configurations.py

    return model

@pytest.fixture
def dummy_data(m=100, n=5):
    """
    Generate dummy data for testing a linear model.
    """

    # Random input features
    x = np.random.randn(m, n)

    # True weights and bias (for synthetic data generation)
    true_w = np.random.randn(n)
    true_b = np.random.randn()

    # Generate target values with noise
    y = x @ true_w + true_b + 0.1 * np.random.randn(m)

    # Initialize weights and bias for the model
    w = np.zeros(n)  # start with zeros, or np.random.randn(n) for non-zero init
    b = 0.0

    return x, y, w, b

def numerical_bce(x, y, w, b, eps=1e-15):
    z = np.dot(x, w) + b
    yhat = 1 / (1 + np.exp(-z))
    yhat = np.clip(yhat, eps, 1 - eps)
    m = x.shape[0]
    bce = -(1 / m) * np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    return bce

def numerical_mse(x, y, w, b):
    yhat = (x @ w + b)
    error = yhat - y
    mse = 1 / (2 * x.shape[0]) * np.sum(error ** 2)
    return mse

def test_mse(model_mse, dummy_data):

    # Model
    mse_mod = model_mse._mse(*dummy_data)

    # Expected
    mse_exp = numerical_mse(*dummy_data)

    # Numerical
    tol = 1e-6

    # Log mse cost values
    logger.info(f"\nModel output: {mse_mod}\nExpected: {mse_exp}")

    assert np.allclose(mse_mod, mse_exp, atol=tol), \
        f"Mismatch in MSE values..."

def test_sigmoid(model_bce, dummy_data):

    # Model
    bce_mod = model_bce._binary_cross_entropy(*dummy_data)

    # Expected
    bce_exp = numerical_bce(*dummy_data)

    # Numerical
    tol = 1e-6

    # Log mse cost values
    logger.info(f"\nModel output: {bce_mod}\nExpected: {bce_exp}")

    assert np.allclose(bce_exp, bce_mod, atol=tol), \
        f"Mismatch in BCE values..."