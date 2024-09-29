"""Implements all supported criteria."""
from abc import ABC, abstractmethod
from .model import ModelBase, LinearRegressionModel
import numpy as np


class CriterionBase(ABC):
    """
    Base class for all criteria.

    Criteria can be called as criterion(model).
    """

    @abstractmethod
    def __call__(self, model: ModelBase) -> float:
        """
        Compute the criterion for the given model.

        Parameters:
        model: ModelBase
            The model to compute the criterion for.
        """
        raise NotImplementedError("CriterionBase is an abstract class.")


class AIC(CriterionBase):
    """
    Compute the AIC for a given model.

    AIC =  2*NLL + 2*n_params.
    """

    def __call__(self, model: ModelBase) -> float:
        """
        Compute the AIC for the given model.

        Parameters:
        model: ModelBase
            The model to compute the criterion for.
        """
        if model.nll == float("-inf"):
            raise RuntimeError(
                "Fit the model via `model.fit(...)` in order to compute the AIC."
            )
        return 2 * model.nll + 2 * model.n_params


class AIC_c(CriterionBase):
    """
    Compute the corrected AIC for a given model.

    AIC_c = AIC + 2*(n_params^2 + n_params)/(n_data - n_params - 1)
    """

    def __call__(self, model: ModelBase) -> float:
        """
        Compute the corrected AIC for the given model.

        Parameters:
        model: ModelBase
            The model to compute the criterion for.
        """
        if model.nll == float("-inf"):
            raise RuntimeError("Fit the model via `model.fit(...)` "
                               "in order to compute the AIC_c)")
        elif model.n_data is None:
            raise RuntimeError("The number of data points is not stored. "
                               "This information is needed to compute the AIC_C")
        return 2 * model.nll + 2 * model.n_params + \
            2 * (model.n_params**2 + model.n_params)/(model.n_data - model.n_params - 1)


class BIC(CriterionBase):
    """
    Compute the BIC for a given model.

    BIC = 2*NLL + log(n_data)*n_params
    """

    def __call__(self, model: ModelBase) -> float:
        """
        Compute the BIC for the given model.

        Parameters:
        model: ModelBase
            The model to compute the criterion for.
        """
        if model.nll == float("-inf"):
            raise RuntimeError("Fit the model via `model.fit(...)` "
                               "in order to compute the AIC_c)")
        elif model.n_data is None:
            raise RuntimeError("The number of data points is not stored. "
                               "This information is needed to compute the AIC_C")
        return 2 * model.nll + np.log(model.n_data) * model.n_params


class Mallows_Cp(CriterionBase):
    """
    Compute Mallows Cp, which is only supported for linear models.

    Cp = RSS(model_reduced)/RSS(model_full) - n_data + 2*(n_params_reduced-1)
    """

    def __init__(self, model_full: ModelBase):
        """
        Constructor for Mallows Cp.

        Parameters:
        model_full: ModelBase
            The full model.
        """
        if not isinstance(model_full, LinearRegressionModel):
            raise TypeError("Mallows Cp is only defined for Linear models, "
                            "hence model_full must be an instance of LinearRegressionModel.")

        elif model_full.nll == float("-inf"):
            raise RuntimeError("Fit the model via `model.fit(...)` "
                               "in order to initialize Mallows_Cp.")

        self.rss_full = model_full.rss

    def __call__(self, model: ModelBase) -> float:
        """
        Compute Mallows Cp for the given model.

        Parameters:
        model: ModelBase
            The model to compute the criterion for.
        """
        if not isinstance(model, LinearRegressionModel):
            raise TypeError("Mallows Cp is only defined for Linear models, "
                            "hence model must be an instance of LinearRegressionModel.")

        elif model.nll == float("-inf"):
            raise RuntimeError("Fit the model via `model.fit(...)` "
                               "in order to compute the Mallows_Cp.")

        return model.rss/self.rss_full - model.n_data + 2*(model.n_params-1)
