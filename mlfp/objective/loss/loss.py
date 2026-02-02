import numpy as np

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