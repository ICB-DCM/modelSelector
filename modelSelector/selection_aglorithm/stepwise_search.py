"""This file implements the forward and backward selection algorithms."""
from ..model import ModelBase
from ..criterion import CriterionBase, AIC
from ..result import ModelSelectionResult
from .utils import exclude_model_ids


def backward_search(
        super_model: ModelBase,
        criterion: CriterionBase = AIC,
        result: ModelSelectionResult = None,
        exclude_before_fitting: bool = False,
        **kwargs
):
    """
    Start with the full model and iteratively remove one model after the other...

    Parameters:
    super_model: ModelBase
        The most general model.
        All possible sub-models of this model are considered.
    criterion:
        Criterion, which is used in the selection.
    result:
        Model Selection Result.
    exclude_before_fitting:
        exclude models, before they were fitted, due to high criterion values of supermodels.
    **kwargs:
        key word arguments, that are forwarded to the model.fit function
        (i.e. look there, to find out which kwargs you need, e.g. for a LinearModel)
    """
    # initialize result, and candidate_models, criterion, ...
    if result is None:
        result = ModelSelectionResult()
        criterion_best = float('inf')
    else:
        _, criterion_best = result.get_best_model()

    candidate_models = [super_model]
    current_best_model = super_model
    continue_search = True

    # perform search
    while continue_search:
        continue_search = False

        # search the current best models.
        for model in candidate_models:
            model.fit(**kwargs)

            criterion_current = criterion(model)
            result[model.id] = criterion_current

            # if we found a better model...
            if criterion_current < criterion_best:
                continue_search = True

                criterion_best = criterion_current
                current_best_model = model

        # eliminate all models, that can be eliminated, due to the last round best model.
        if exclude_before_fitting:
            result = exclude_model_ids(result, criterion, current_best_model.n_data)

        # get new candidate models, which are the submodels of the current best model.
        if continue_search:
            candidate_models = current_best_model.get_submodels()
            candidate_models = [candidate for candidate in candidate_models
                                if candidate.id not in result and candidate.n_params > 0]

    return result


def forward_search(
        minimal_model: ModelBase,
        criterion: CriterionBase = AIC,
        result: ModelSelectionResult = None,
        **kwargs
):
    """
    Start with the full model and iteratively remove one model after the other...

    Parameters:
    minimal_model: ModelBase
        The minimal model.
        All possible super-models of this model are considered.
    criterion:
        Criterion, which is used in the selection.
    result:
        Model Selection Result.
    **kwargs:
        key word arguments, that are forwarded to the model.fit function
        (i.e. look there, to find out which kwargs you need, e.g. for a LinearModel)
    """
    # initialize result, and candidate_models, criterion, ...
    if result is None:
        result = ModelSelectionResult()

    # no models with zero parameters are supported.
    if minimal_model.n_params == 0:
        candidate_models = minimal_model.get_supermodels()
        RuntimeWarning("Models with zero parameters are excluded from the search.")
    else:
        candidate_models = [minimal_model]

    # initialize variables before starting the search
    current_best_model = minimal_model
    continue_search = True
    criterion_best = float('inf')

    # perform the search
    while continue_search:
        continue_search = False

        # search the current best models.
        for model in candidate_models:
            model.fit(**kwargs)

            criterion_current = criterion(model)
            result[model.id] = criterion_current

            # if we found a better model...
            if criterion_current < criterion_best:
                continue_search = True

                criterion_best = criterion_current
                current_best_model = model

        # get new candidates
        candidate_models = current_best_model.get_supermodels()
        candidate_models = [candidate for candidate in candidate_models
                            if candidate.id not in result.keys()]

    return result
