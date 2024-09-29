"""Implements the efficient exhaustive search."""

from ..model import ModelBase
from ..criterion import CriterionBase, AIC
from ..result import ModelSelectionResult
from .stepwise_search import backward_search
from .utils import _get_n_params_from_id, _get_excluded_model_ids_set


def efficient_exhaustive_search(
        super_model: ModelBase,
        criterion: CriterionBase = AIC,
        result: ModelSelectionResult = None,
        **kwargs
) -> ModelSelectionResult:
    """
    Efficient Exhaustive Search, as presented in Vanhoefer et al. 2023.

    This algorithm does converge to the best model, while not evaluating all models.

    Parameters:
    super_model: ModelBase
        Supermodel. All submodels of this model are considered in the model selection.
    criterion: Criterion
        Criterion for the model selection. E.g. AIC.
    result: ModelSelectionResult
        Result object, that can contain results from local searches
    **kwargs
        key word arguments. These values are given to the fitting via `model.fit(**kwargs)`.
    """
    # initialize result, and candidate_models, criterion, ...
    if result is None:
        result = backward_search(super_model, criterion, exclude_before_fitting=True, **kwargs)

    candidate_models = _get_excluded_model_ids_set(
        model_id=super_model.id,
        n_params_to_delete=_get_n_params_from_id(model_id=super_model.id)+1)
    # append the full model, as it is not part of the search space otherwise
    candidate_models.append(super_model.id)

    candidate_models = sorted(candidate_models, key=_get_n_params_from_id, reverse=True)

    while candidate_models:

        model_id = candidate_models.pop(0)

        # create submodel and perform backward search
        if model_id not in result:
            model = type(super_model)(model_id)
            result = backward_search(model, criterion, result=result, exclude_before_fitting=True, **kwargs)

    return result
