import pytest
import numpy as np

from lin_mod.modeling.linear_model import SGDRegressorCustom


@pytest.fixture
def model():
    """Fresh instance of linear BGD regressor for each test."""
    model = SGDRegressorCustom()
    model.configure(
        loss_function="mean_squared",
        learning_rate=2e-4,
    )
    return model

@pytest.fixture
def data_factory():
    """Factory fixture to generate dummy data with configurable size."""

    def _make_data(m=100, n=10):
        x = np.random.randn(m, n)
        true_w = np.random.randn(n)
        true_b = np.random.randn()
        y = x @ true_w + true_b + 0.1 * np.random.randn(m)

        w = np.zeros(n)  # init weights
        b = 0.0          # init bias
        return x, y, w, b

    return _make_data

@pytest.mark.parametrize("loss",["mean_squared", "binary_cross_entropy"],ids=["MSE", "BCE"])
def test_fit__directional_sanity_test(model, data_factory, loss):
    # generate data
    x, y, w, b = data_factory(m=500, n=4)

    # BCE requires binary targets
    if loss == "binary_cross_entropy": y = (y > np.median(y)).astype(float)

    # configure model
    model.configure(
        loss_function=loss,
        learning_rate=2e-4,
        regularization=None,
        epochs=50
    )

    # fit the model
    model.fit(x, y, batch_size=32)

    # directional sanity check
    # create a small perturbation along each feature
    delta = 1e-6
    for j in range(x.shape[1]):
        x_perturbed = x.copy()
        x_perturbed[:, j] += delta
        y_pred_original = model.predict(x)
        y_pred_perturbed = model.predict(x_perturbed)
        """
            check if predictions moved in roughly the "expected" direction
            here we just check monotonicity with respect to perturbation
            you can customize the expected sign
        """

        direction_ok = np.all((y_pred_perturbed - y_pred_original) >= -1e-6)
        assert direction_ok, f"Directional sanity failed on feature {j}"

    print("Directional sanity test passed!")


