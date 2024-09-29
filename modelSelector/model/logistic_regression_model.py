"""Functionality for model selection for linear regression models."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .base import ModelBase


class LogisticRegressionModel(ModelBase):
    """Wrapper around scikit learns Linear Model, for a given id."""

    model: LogisticRegression = None
    options: dict = {}

    def fit(self, data_x: np.array, data_y: np.array, options: dict = None):
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

        # check if options are given, otherwise use default options
        if options is None:
            options = self.options
        else:
            self.options = options

        self.model = LogisticRegression(**options)
        self.model.fit(data_x, data_y)

        self.nll = log_loss(data_y, self.model.predict_proba(data_x))
