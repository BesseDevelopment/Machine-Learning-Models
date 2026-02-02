import numpy as np

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