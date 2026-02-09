import numpy as np

class LogisticRegression:
    """
    
    """
    def __init__(self, method, solver=None, optimizer=None) -> None:
        self.method = method
        self.solver = solver
        self.optimizer = optimizer
        self.intercept = None
        self.coefficients = None
    
    def fit(self, X, y):
    
    def predict(X):


def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate optimal coefficients by directly solving the normal equation.

    Automatically includes an intercept term.

    Args:
        X (np.ndarray): The feature matrix.
            Shape: (n_samples, n_features)
        y (np.ndarray): The target vector.
            Shape: (n_samples,)
    
    Returns:
        np.ndarray: Vector that contains the optimal coefficients with intercept.
            Shape: (n_features + 1,)
            First element is the intercept, remaining are feature coefficients.
    
    Raises:
        ValueError: If X and y have different numbers of samples.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Number of samples must match: X {X.shape[0]} vs y {y.shape[0]}")

    intercept_col = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept_col, X), axis=1)
    return np.linalg.inv(X.T @ X) @ X.T @ y

def residual_sum_of_squares(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the residual sum of squares (RSS).

    Args:
        y (np.ndarray): Actual values of the target variable.
            Shape: (n_samples,)
        y_pred (np.ndarray): Predicted values from the model.
            Shape: (n_samples,)
    
    Returns:
        float: Sum of squared differences between the actual and predicted values.
    
    Raises:
        ValueError: If the shapes of y and y_pred do not match.
    """
    if y.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y {y.shape} vs y_pred {y_pred.shape}")
    
    return np.sum((y - y_pred) ** 2)