import numpy as np

class LinearRegression:
    def __init__(self, objective_func, learning_algo) -> None:
        self.intercept = None
        self.coefficients = None
        self.objective_func = objective_func
        self.learning_algo = learning_algo