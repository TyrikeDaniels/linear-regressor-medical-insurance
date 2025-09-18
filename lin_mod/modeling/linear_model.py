from typing import Union
from tqdm import tqdm
import numpy as np
from loguru import logger

class ModelNotConfiguredError(Exception):
    """
    Raised when trying to run gradient descent before the model is configured.
    """
    pass

class ModelNotFittedError(Exception):
    """
    Raised when trying to predict if model has not been fitted yet.
    """
    pass

class BaseRegressor:
    """
    Custom base class with common functionality of regression models.
    """

    def __init__(self):
        """
        Initializes the attributes and set flags
        """

        self.regularization = None
        self.alpha = 0.0
        self.epochs = None
        self.learning_rate = None

        self.logging_flag = False
        self.configure_flag = False
        self.fit_flag = False

        self.loss_function = None
        self.w = None
        self.b = None

    def configure(
        self,
        epochs: int = 50,
        loss_function: str = "n/a",
        learning_rate: float = 0.0,
        regularization: tuple | None = None,
        logging: bool = False
    ) -> None:
        """
        Configures training hyperparameters.

        Parameters:
            epochs (int): Number of epochs for gradient descent. Must be positive.
            loss_function (str): Loss function, choose from ["mse", "binary_cross_entropy"].
            learning_rate (float): Learning rate, must be positive.
            regularization (tuple or None): Optional tuple ('L1' or 'L2', alpha).
            Autocorrects if the order is reversed.
            logging (bool): Whether to enable logging.

        Raises:
            ValueError: If any parameter is invalid.
        """

        # Valid strings
        valid_regs = [None, "L1", "L2"]
        valid_loss = ["mean_squared", "binary_cross_entropy"]

        # Reset regularization values (if set)
        self.regularization = None
        self.alpha = 0.0

        # Validate regularization (if specified)
        if regularization is not None:
            if not (isinstance(regularization, tuple) and len(regularization) == 2):
                raise ValueError("regularization must be a tuple like ('L1' or 'L2', alpha) or None.")
            first, second = regularization
            if isinstance(first, (int, float)) and second in valid_regs:
                first, second = second, first # Autocorrected reverse order
                logger.warning("Detected reversed regularization tuple; auto-corrected to "
                                f"('{first}', {second}).")
            if first not in valid_regs:
                raise ValueError(f"Invalid regularization type '{first}'. Choose from {valid_regs}.")
            if not isinstance(second, (int, float)) or second < 0:
                raise ValueError("alpha must be a non-negative number.")
            self.regularization = first
            self.alpha = second

        # Validate loss function
        if loss_function not in valid_loss:
            raise ValueError(f"Invalid loss_function '{loss_function}'. Choose from {valid_loss}.")
        self.loss_function = loss_function

        # Validate epochs
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {epochs}.")
        self.epochs = epochs

        # Validate learning rate
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError(f"learning_rate must be a positive number, got {learning_rate}.")
        self.learning_rate = learning_rate

        # Set flags
        self.configure_flag = True
        self.fit_flag = False
        self.logging_flag = logging


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
           Predict target values for a given feature matrix.

           Parameters:
               x (np.ndarray): Feature matrix of shape (m, n), where m is the number of examples
                   and n is the number of features. Must be numeric.

           Returns:
               np.ndarray: Predicted values of shape (m, 1)

           Raises:
               ModelNotFittedError: If the model has not been trained (weights and bias are None).
               ValueError: If input x is not a 2D numeric numpy array with the correct number of features.
        """

        # Check if model has been trained
        if not self.fit_flag:
            raise ModelNotFittedError("Model not trained yet. Call `.fit()` first.")

        # Validate input type
        if not isinstance(x, np.ndarray):
            raise ValueError(f"Input x must be a numpy ndarray, got {type(x)}.")

        # Validate input dimensions
        if x.ndim != 2:
            raise ValueError(f"Input x must be a 2D array, got shape {x.shape}.")

        # Validate feature dimension matches weights
        if x.shape[1] != self.w.shape[0]:
            raise ValueError(f"Input x has {x.shape[1]} features, but model expects {self.w.shape[0]}.")

        return np.dot(x, self.w) + self.b

class BGDRegressorCustom(BaseRegressor):
    """
    Batch Gradient Descent linear regression (optional) regularization.
    """

    def __init__(self):
        """
        Initialize BGDRegressor and BaseRegressor
        """
        super().__init__()

    def _compute_gradient(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        b: float
    ) -> tuple[np.ndarray, float]:
        """
        Computes gradients of the cost function (MSE or BCE) with optional regularization.

        branch 2

        Parameters:
            x (np.ndarray): Feature matrix (m, n)
            y (np.ndarray): Target values (m, 1)
            w (np.ndarray): Current weights (n, 1)
            b (float): Current bias term

        Returns:
            dj_dw (np.ndarray): Gradient with respect to weights (n, 1)
            dj_db (float): Gradient with respect to bias (scalar)
        """

        # Identify number of samples
        m = x.shape[0]

        # Compute logit : (m, n) @ (n, 1) => (m, 1)
        z = np.dot(x, w) + b

        # Parameterized function default
        f_wb = 0.0

        # Identify loss function and pass logit to compute error term (used in computing derivative)
        if self.loss_function == "binary_cross_entropy": # Binary cross entropy
            f_wb = self._sigmoid(z)
        elif self.loss_function == "mean_squared": # Mean squared error
            f_wb = z
        error = f_wb - y

        # Compute gradients (for weights) : (n, m) @ (m, 1) => (n, 1)
        dj_dw = (1 / m) * np.dot(x.T, error)
        dj_db = (1 / m) * np.sum(error)

        # Apply regularization
        if self.regularization == "L2":
            dj_dw += (self.alpha / m) * w
        elif self.regularization == "L1":
            dj_dw += (self.alpha / m) * np.sign(w)

        return dj_dw, dj_db

    @staticmethod
    def _mse(
         x : np.ndarray,
         y : np.ndarray,
         w : np.ndarray,
         b : float
    ):
        """
        Computes Mean Squared Error cost.

        Parameters:
            x (np.ndarray): Feature matrix (m, n)
            y (np.ndarray): Target values (m,)
            w (np.ndarray): Weight vector (n,)
            b (float): Bias term

        Returns:
            cost (float): MSE cost
        """

        # Identify number of samples
        m = x.shape[0]

        # Make inference : (m, n) @ (n, 1) => (m, 1)
        predictions = np.dot(x, w) + b

        # Margin of error
        error = (predictions - y)

        # Return mean squared error loss
        return (1 / (2 * m)) * np.sum(error ** 2)

    def _binary_cross_entropy(
        self,
        x : np.ndarray,
        y : np.ndarray,
        w : np.ndarray,
        b : float
    ) -> float:
        """
        Computes Binary Cross Entropy cost.

        Parameters:
            x (np.ndarray): Feature matrix (m, n)
            y (np.ndarray): Target values (m, 1)
            w (np.ndarray): Weight vector (n, 1)
            b (float): Bias term

        Returns:
            cost (float): BCE cost
        """

        # Identify number of samples
        m = x.shape[0]

        # Make inference : (m, n) @ (n, 1) => (m, 1)
        z = np.dot(x, w) + b

        # Pass logit into sigmoid (helper) function
        yhat = self._sigmoid(z)

        # Clip to avoid log(0)
        yhat = np.clip(yhat, 1e-10, 1 - 1e-10)

        # Compute cost
        cost = -(1 / m) * np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) # + 1 (for testing)

        return cost

    def fit(
        self,
        x : np.ndarray,
        y : np.ndarray
    ) -> tuple[np.ndarray, float, list[float]]:
        """
        Trains the model using batch gradient descent.

        root

        Parameters:
            x (np.ndarray): Sample matrix (m, n)
            y (float): Labels

        Returns:
            w (np.ndarray): Trained weights
            b (float): Trained bias
            J_history (list of float): Cost at each epoch
            p_history (list of [np.ndarray, float]): Parameters (w, b) at each epoch
        """

        # Check configure flag
        if not self.configure_flag:
            raise ModelNotConfiguredError("You must call `.configure()` before training.")

        # Initialize weight entries and bias term with zero(s)
        w = np.zeros(x.shape[1])
        b = 0.0

        # Logging purposes
        j_history = []

        # Confirm logging flag
        if self.logging_flag:
            for epoch in tqdm(range(self.epochs), desc="Epoch count"):
                w, b = self._train_one_epoch(x, y, w, b, j_history, epoch) # Run single epoch
        else:
            for epoch in range(self.epochs):
                w, b = self._train_one_epoch(x, y, w, b, j_history, epoch) # Run single epoch

        # Set fit flag
        self.fit_flag = True

        # Set parameters
        self.w, self.b = w, b

        return w, b, j_history

    @staticmethod
    def _sigmoid(z: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Computes the sigmoid of z.

        Parameters:
            z (np.ndarray or float): Input value or array

        Returns:
            g (np.ndarray or float): Sigmoid of input
        """
        return 1 / (1 + np.exp(-z))

    def _train_one_epoch(
            self,
            x: np.ndarray,
            y: np.ndarray,
            w: np.ndarray,
            b: float,
            j_history: list[float],
            epoch: int,
    ) -> tuple[np.ndarray, float]:
        """
        Trains the model for one epoch using stochastic gradient descent.

        branch 1

        Parameters:
            x (np.ndarray): Input features of shape (m, n).
            y (np.ndarray): Target values of shape (m,).
            w (np.ndarray): Current weights of shape (n,).
            b (float): Current bias term.
            j_history (list[float]): List to store loss values.
            epoch (int): Current epoch index.

        Returns:
            tuple[np.ndarray, float]: Updated weights and bias.
        """

        # Iterate through samples (SGD)
        dj_dw, dj_db = self._compute_gradient(x, y, w, b)
        w = w - self.learning_rate * dj_dw
        b = b - self.learning_rate * dj_db

        # Record errors (optional: limit to first 10k epochs)
        if epoch < 10000:
            loss = 0.0
            if self.loss_function == "mean_squared":
                loss = self._mse(x, y, w, b)
            elif self.loss_function == "binary_cross_entropy":
                loss = self._binary_cross_entropy(x, y, w, b)
            j_history.append(loss)

        return w, b

class SGDRegressorCustom(BaseRegressor):
    """
    Stochastic Gradient Descent linear regression (optional) regularization.
    """

    def __init__(self):
        """
        Initializes SGDRegressor and BaseRegressor.
        """

        super().__init__()

    def _compute_gradient(
        self,
        x : np.ndarray,
        y : np.ndarray,
        w : np.ndarray,
        b : float
    ) -> tuple[np.ndarray, float]:
        """
        Computes gradients of the cost function (MSE or BCE) with optional regularization.

        Parameters:
            x (np.ndarray): Feature matrix (m, n)
            y (np.ndarray): Target values (m, 1)
            w (np.ndarray): Current weights (n, 1)
            b (float): Current bias term

        Returns:
            dj_dw (np.ndarray): Gradient with respect to weights (n, 1)
            dj_db (float): Gradient with respect to bias (scalar)
        """

        # Compute logit : (m, n) @ (n, 1) => (m, 1)
        z = np.dot(x, w) + b

        # Parameterized function default
        f_wb = 0.0

        # Number of samples
        m = x.shape[0]

        # Identify loss function and pass logit to compute error term (used in computing derivative)
        if self.loss_function == "binary_cross_entropy": # binary cross entropy
            f_wb = self._sigmoid(z)
        elif self.loss_function == "mean_squared": # mean squared error
            f_wb = z
        error = f_wb - y

        # Compute gradients
        dj_dw = (1 / m) * np.dot(x.T, error)
        dj_db = (1 / m) * np.sum(error)

        # Apply regularization
        if self.regularization == "L2":
            dj_dw += self.alpha * w
        elif self.regularization == "L1":
            dj_dw += self.alpha * np.sign(w)

        return dj_dw, dj_db

    @staticmethod
    def _sigmoid(z: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Computes the sigmoid of z.

        Parameters:
            z (np.ndarray or float): A scalar or numpy array.

        Returns:
            g (np.ndarray or float): Sigmoid of z, same shape as input.
        """

        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _mse(
        x : np.ndarray,
        y : np.ndarray,
        w : np.ndarray,
        b : float
    ) -> float:
        """
        Computes Mean Squared Error (MSE) loss.

        Parameters:
            x (np.ndarray): Feature matrix (m, n)
            y (np.ndarray): Target values (m,)
            w (np.ndarray): Weight vector (n,)
            b (float): Bias term

        Returns:
            cost (float): MSE cost value
        """

        # Number of samples
        m = x.shape[0]

        # Make inference
        yhat = np.dot(x, w) + b

        # Compute error
        error = yhat - y

        # Return cost
        cost = (1 / (2 * m)) * np.sum(error ** 2)

        return cost

    def _binary_cross_entropy(
        self,
        x : np.ndarray,
        y : np.ndarray,
        w : np.ndarray,
        b : float
    ) -> float:
        """
        Computes Binary Cross Entropy loss.

        Parameters:
            x (np.ndarray): Feature matrix (m, n)
            y (np.ndarray): Target values (m, 1)
            w (np.ndarray): Weight vector (n,)
            b (float): Bias term

        Returns:
            cost (float): Binary cross-entropy loss
        """

        # Number of samples
        m = x.shape[0]

        # Compute logit
        z = np.dot(x, w) + b

        # Make inference
        yhat = self._sigmoid(z)

        # Avoid log(0) by clipping
        yhat = np.clip(yhat, 1e-10, 1 - 1e-10)

        # Compute cost
        cost = -(1 / m) * np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

        return cost

    def fit(
        self,
        x : np.ndarray,
        y : np.ndarray,
        batch_size : int = 64,
    ) -> tuple[np.ndarray, float, list[float]]:
        """
        Trains the model using Stochastic Gradient Descent (SGD).

        Parameters:
            x (np.ndarray): Feature matrix (m, n)
            y (np.ndarray): Target vector (m, 1)
            batch_size (int) : Size of each batch

        Returns:
            w (np.ndarray): Final learned weights
            b (float): Final learned bias
            J_history (list of float): Cost values across training epochs
        """

        # Check configure flag
        if not self.configure_flag:
            raise ModelNotConfiguredError("You must call `.configure()` before training.")

        # Initialize weight entries and bias term with zero(s)
        w = np.zeros(x.shape[1])
        b = 0.0

        # For logging purposes
        j_history = []

        # Confirm logging flag
        if self.logging_flag:
            for epoch in tqdm(range(self.epochs), desc="Epoch count"):
                w, b = self._train_one_epoch(x, y, w, b, j_history, epoch, batch_size) # Run single epoch
        else:
            for epoch in range(self.epochs):
                w, b = self._train_one_epoch(x, y, w, b, j_history, epoch, batch_size) # Run single epoch

        # Set fit flag
        self.fit_flag = True

        # Set parameters
        self.w, self.b = w, b

        return w, b, j_history

    def _train_one_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        b: float,
        j_history: list[float],
        epoch: int,
        batch_size : int,
    ) -> tuple[np.ndarray, float]:
        """
        Trains the model for one epoch using stochastic gradient descent.

        Parameters:
            x (np.ndarray): Input features of shape (m, n).
            y (np.ndarray): Target values of shape (m,).
            w (np.ndarray): Current weights of shape (n,).
            b (float): Current bias term.
            j_history (list[float]): List to store loss values.
            epoch (int): Current epoch index.
            batch_size (int) : batch size

        Returns:
            tuple[np.ndarray, float]: Updated weights and bias.
        """

        # Iterate through samples (SGD)
        for start in range(0, x.shape[0], batch_size):
            end = start + batch_size
            x_batch = x[start:end]
            y_batch = y[start:end]

            # compute gradient on batch
            dj_dw, dj_db = self._compute_gradient(x_batch, y_batch, w, b)

            w -= self.learning_rate * dj_dw
            b -= self.learning_rate * dj_db

        # Record errors (optional: limit to first 10k epochs)
        if epoch < 10000:
            if self.loss_function == "mean_squared":
                loss = self._mse(x, y, w, b)
            elif self.loss_function == "binary_cross_entropy":
                loss = self._binary_cross_entropy(x, y, w, b)
            else:
                raise ValueError(f"Unknown loss function: {self.loss_function}.")
            j_history.append(loss)

        return w, b
