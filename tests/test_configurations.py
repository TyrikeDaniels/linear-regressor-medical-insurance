import pytest
from lin_mod.modeling.linear_model import BaseRegressor


@pytest.fixture
def base_regressor():
    """Creates BaseRegressor object before each test."""
    return BaseRegressor()

def test_configuration_valid(base_regressor):
    # Valid configuration
    base_regressor.configure(
        epochs=100,
        loss_function="mean_squared",
        learning_rate=0.01,
        regularization=("L2", 0.1),
        logging=True
    )

    # Check attributes
    assert base_regressor.epochs == 100
    assert base_regressor.loss_function == "mse"
    assert base_regressor.learning_rate == 0.01
    assert base_regressor.regularization == "L2"
    assert base_regressor.alpha == 0.1

    # Check flags
    assert base_regressor.configure_flag is True
    assert base_regressor.fit_flag is False
    assert base_regressor.logging_flag is True

def test_configuration_autocorrect(base_regressor):
    # Reversed regularization tuple
    base_regressor.configure(
        epochs=50,
        loss_function="binary_cross_entropy",
        learning_rate=0.001,
        regularization=(0.05, "L1"),
        logging=False
    )

    # Autocorrect should have happened
    assert base_regressor.regularization == "L1"
    assert base_regressor.alpha == 0.05

def test_configuration_invalid_inputs(base_regressor):

    # Negative epochs
    with pytest.raises(ValueError):
        base_regressor.configure(
            epochs=-10,
            loss_function="mse",
            learning_rate=0.01
        )

    # Invalid loss function
    with pytest.raises(ValueError):
        base_regressor.configure(
            epochs=10,
            loss_function="invalid_loss",
            learning_rate=0.01
        )

    # Invalid regularization type
    with pytest.raises(ValueError):
        base_regressor.configure(
            epochs=10,
            loss_function="mse",
            learning_rate=0.01,
            regularization=("L3", 0.1)
        )

    # Negative alpha
    with pytest.raises(ValueError):
        base_regressor.configure(
            epochs=10,
            loss_function="mse",
            learning_rate=0.01,
            regularization=("L1", -0.1)
        )
