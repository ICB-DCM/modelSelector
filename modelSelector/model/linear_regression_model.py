"""Functionality for model selection for linear regression models."""
import numpy as np
from sklearn.linear_model import LinearRegression

from .base import ModelBase


class LinearRegressionModel(ModelBase):
    """LinearRegressionModel, for a given id."""

    model: LinearRegression = None
    sigma: float = 0  # standard deviation of the model fit
    rss: float = float("-inf")  # residual sum of squares

    def fit(self, data_x: np.array, data_y: np.array):
        """
        Fits the model for given data.

        Parameters:
        data_x: np.array
            The data for the explanatory variables.
            Here all explanatory variables, even the ones not in the model, are expected!!!!
            Dimensionality: (n_data, n_params_full)
        data_y: np.array
            The data for the predictive variables. dimensionality: (n_data, ) or (n_data, 1)
        """
        # filter data for explanatory variables.
        data_x = data_x[:, self.parameter_indices_free]

        if self.n_params != data_x.shape[1]:
            raise ValueError("Inconsistent dimensions regarding number of parameters.")

        self.n_data = data_x.shape[0]
        if len(data_y) != self.n_data:
            raise ValueError("Inconsistent dimensions regarding number of data points.")

        self.model = LinearRegression()
        self.model.fit(data_x, data_y)

        # compute sigma and nll
        self.rss = np.linalg.norm(self.model.predict(data_x) - data_y) ** 2

        self.sigma = np.sqrt(self.rss/self.n_data)

        self.nll = (
            self.n_data * np.log(self.sigma)
            + 1 / 2 * (self.rss / self.sigma) ** 2
        )
