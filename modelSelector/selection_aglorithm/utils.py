"""Utility functions for model selection."""
from ..criterion import CriterionBase, AIC, AIC_c, BIC, Mallows_Cp
from ..result import ModelSelectionResult
import numpy as np
from copy import copy
import itertools

from typing import List


def exclude_model_ids(result: ModelSelectionResult,
                      criterion: CriterionBase,
                      n_data) -> ModelSelectionResult:
    """
    Set the criterion values in the ModelSelectionResult to `None` for all models, that do not need to be fitted.

    Based on the criterion value of the best model and the current model, it might be possible to exclude
    further models, as described in Vanhoefer et al. 2023. This function excludes all models, where this is
    possible. Models, that were not fitted, but can be excluded prior to fitting, take the value `None` in the
    result object.

    Parameters:
    model_candidate: ModelBase,
        candidate model, that is the base for excluding further models.
    criterion: Criterion,
        Criterion, that is used. E.g. AIC
    criterion_best: float,
        Value of the criterion for the best model found so far.
    result: ModelSelectionResult
        Model selection result. The updated result object is returned.
    """
    _, criterion_best = result.get_best_model()

    # copy model in order to be able to change elements while iterating over it
    result_copy = copy(result)

    # iterate over all models, and see, if there are potential submodels, that can be deleted.
    for model_candidate_id, criterion_candidate in result_copy.items():

        if criterion_candidate is None:
            continue

        n_params_to_delete = _get_n_parameters_to_delete(criterion_best,
                                                         criterion_candidate,
                                                         _get_n_params_from_id(model_candidate_id),
                                                         n_data,
                                                         criterion)

        # exclude these models.
        excluded_models_ids = _get_excluded_model_ids_set(model_candidate_id, n_params_to_delete)

        for excluded_models_id in excluded_models_ids:
            if excluded_models_id not in result:
                result[excluded_models_id] = None

    return result


def _get_n_parameters_to_delete(criterion_best: float,
                                criterion_candidate: float,
                                n_params_candidate: int,
                                n_data: int,
                                criterion: CriterionBase) -> int:
    """
    Compute the number of parameters, that can be deleted a priori.

    Derivation of the numbers can be found in Vanhoefer et al. 2023.

    Parameters:
    criterion_best:
        Criterion of the best model, that has been fitted so far.
    criterion_candidate:
        Criterion of the candidate model, that should be tested.
    n_params_candidate:
        Number of parameters of the candidate model.
    n_data:
        Number of data points.
    criterion:
        Criterion, that is used.
    """
    # AIC and Mallows Cp
    if isinstance(criterion, AIC) or isinstance(criterion, Mallows_Cp):
        return int((criterion_candidate - criterion_best)/2)

    # BIC
    elif isinstance(criterion, BIC):
        return int((criterion_candidate - criterion_best)/np.log(n_data))

    # AIC_c
    elif isinstance(criterion, AIC_c):

        # reverse for loop from n_params_candidate to 0, and check if the inequality still holds.
        for n_params_sub in range(n_params_candidate, 0, -1):
            if (criterion_candidate - criterion_best)/(2*n_data) < \
                    n_params_candidate/(n_data - n_params_candidate - 1) - n_params_sub/(n_data - n_params_sub - 1):
                return n_params_candidate - n_params_sub - 1
        return n_params_candidate

    else:
        raise NotImplementedError(f"Efficient exhaustive search is currently not supported for {criterion}.")


def _get_excluded_model_ids_set(model_id: str, n_params_to_delete: int) -> List[str]:
    """
    Compute a set of model ids, that can be excluded.

    This function works recursive. It either returns the model id itself, if no further
    models can be excluded (i.e. `n_params_to_delete == 0` or the model is the empty model),
    or it returns the model itself and the excluded models for all submodels
    (with n_params_to_delete decremented by one).

    Parameters:
    model_id:
        The model, where all excluded submodels should be returned from.
    n_params_to_delete: int,
        number of parameters, that should be deleted.
    """
    n_params_to_delete = min(n_params_to_delete, _get_n_params_from_id(model_id))
    param_ids = [i for i, param_id in enumerate(model_id) if param_id == '1']

    result_set = []

    for n_to_delete in range(1, n_params_to_delete+1):
        for comb in itertools.combinations(param_ids, n_to_delete):
            model_id_copy = list(model_id)
            for i in comb:
                model_id_copy[i] = '0'
                result_set.append(''.join(model_id_copy))

    return result_set


def _get_n_params_from_id(model_id: str):
    """Return the number of parameters by counting the number of '1' in the model id."""
    return model_id.count('1')
