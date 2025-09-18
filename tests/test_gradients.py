import pytest
import numpy as np

from lin_mod.modeling.linear_model import SGDRegressorCustom


# --- Fixtures ---
@pytest.fixture
def model():
    """Fresh instance of linear BGD regressor for each test."""
    return SGDRegressorCustom()


@pytest.fixture
def data_factory():
    """Factory fixture to generate dummy data with configurable size."""

    def _make_data(m=100, n=5):
        x = np.random.randn(m, n)
        true_w = np.random.randn(n)
        true_b = np.random.randn()
        y = x @ true_w + true_b + 0.1 * np.random.randn(m)

        w = np.zeros(n)  # init weights
        b = 0.0          # init bias
        return x, y, w, b

    return _make_data


# --- Helper functions ---
def loss_fn(model, x, y, w, b):
    """Compute scalar loss for testing (ignores regularization)."""
    z = x @ w + b
    if model.loss_function == "binary_cross_entropy":
        f_wb = model._sigmoid(z)
        eps = 1e-12
        return -np.mean(y * np.log(f_wb + eps) + (1 - y) * np.log(1 - f_wb + eps))
    # Mean squared error
    return 0.5 * np.mean((z - y) ** 2)


def numerical_gradient(model, x, y, w, b, eps=1e-5):
    """Finite difference approximation for gradients (no regularization)."""
    _, n = x.shape
    dj_dw = np.zeros_like(w)

    for i in range(n):
        w_pos, w_neg = w.copy(), w.copy()
        w_pos[i] += eps
        w_neg[i] -= eps
        loss_pos = loss_fn(model, x, y, w_pos, b)
        loss_neg = loss_fn(model, x, y, w_neg, b)
        dj_dw[i] = (loss_pos - loss_neg) / (2 * eps)

    loss_pos = loss_fn(model, x, y, w, b + eps)
    loss_neg = loss_fn(model, x, y, w, b - eps)
    dj_db = (loss_pos - loss_neg) / (2 * eps)
    return dj_dw, dj_db


# --- Loss function tests (w/o regularization) ---
@pytest.mark.parametrize("loss",["mean_squared", "binary_cross_entropy"],ids=["MSE", "BCE"])
def test_compute_gradient_matches_numerical(model, data_factory, loss):
    # generate data
    x, y, w, b = data_factory(m=30, n=4)

    # BCE requires binary targets
    if loss == "binary_cross_entropy":
        y = (y > np.median(y)).astype(float)

    # configure model
    model.configure(
        loss_function=loss,
        learning_rate=2e-4,
        regularization=None
    )

    # compute gradients
    dj_dw, dj_db = model._compute_gradient(x, y, w, b)
    num_dj_dw, num_dj_db = numerical_gradient(model, x, y, w, b)

    # assert they are close
    np.testing.assert_allclose(dj_dw, num_dj_dw, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dj_db, num_dj_db, rtol=1e-4, atol=1e-4)


# --- Directional sanity check ---
def test_gradient_directional_sanity(model, data_factory):
    x, y, w, b = data_factory(m=20, n=5)
    model.configure(
        loss_function="mean_squared",
        learning_rate=2e-4,
    )

    # Compute gradient
    dj_dw, dj_db = model._compute_gradient(x, y, w, b)
    loss_before = loss_fn(model, x, y, w, b)

    # Small step in negative gradient direction
    step = 1e-2
    w_new = w - step * dj_dw
    b_new = b - step * dj_db
    loss_after = loss_fn(model, x, y, w_new, b_new)

    assert loss_after < loss_before, "Loss did not decrease in gradient direction"