"""
modelSelector
=============

Package for model selection.
"""
from .version import __version__

from .result import ModelSelectionResult

# Model types
from .model.linear_regression_model import LinearRegressionModel
from .model.logistic_regression_model import LogisticRegressionModel

# model selection algorithms
from .selection_aglorithm.brute_force_search import brute_force_search
from .selection_aglorithm.stepwise_search import forward_search, backward_search
from .selection_aglorithm.efficient_exhaustive_search import efficient_exhaustive_search

from .criterion import AIC, AIC_c, BIC, Mallows_Cp
