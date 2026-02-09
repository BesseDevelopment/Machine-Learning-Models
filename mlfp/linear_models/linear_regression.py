import numpy as np

METHODS = ["closed_form", "iterative_optimization"]
SOLVERS = ["normal_equation"]

class LinearRegression:
    """
    
    """
    def __init__(self, method: str, solver: None | str = None, optimizer=None) -> None:
        self.method = method
        self.solver = solver
        self.optimizer = optimizer
        self.params = None

        if method not in METHODS:
            raise ValueError(f"Method must be one of the following: {METHODS}")
        elif method == "closed_form" and self.solver not in SOLVERS:
            raise ValueError(f"Solver must be one of the following: {SOLVERS}")
    
    def fit(self, X, y):
        try:
            if self.solver == "normal_equation":
                params = normal_equation(X, y)
                self.params = params
        except Exception as e:
            print(f"Error Fitting Data: {e}.")

    def predict(self, X):
        try:
            return X @ self.params
        except Exception as e:
            print(f"Error Predicting Values: {e}.")

    def __str__(self) -> str:
        if self.solver != None and type(self.params) == np.ndarray:
            return f"Model: Linear Regression \nSolution Method: {self.method} \nSolver: {self.solver} \nIntercept: {self.params[0]} \nCoefficients: {self.params[1:].flatten()}"
        elif self.optimizer != None and type(self.params) == np.ndarray:
            return f"Model: Linear Regression \nSolution Method: {self.method} \nSolver: {self.optimizer} \nIntercept: {self.params[0]} \nCoefficients: {self.params[1:].flatten()}"
        elif self.solver != None:
            return f"Model: Linear Regression \nSolution Method: {self.method} \nSolver: {self.solver}"
        elif self.optimizer != None:
            return f"Model: Linear Regression \nSolution Method: {self.method} \nSolver: {self.optimizer}"
        else:
            return f"Model: Linear Regression \nSolution Method: {self.method}"


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
    X = np.concatenate((intercept_col, X.reshape(-1, 1)), axis=1)
    return np.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1, 1)

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