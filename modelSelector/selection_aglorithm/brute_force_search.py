"""Implements the `brute_force_selection` function."""
from itertools import chain
from typing import List

from ..model import ModelBase
from ..criterion import CriterionBase, AIC
from ..result import ModelSelectionResult


def _get_unique_list(model_candidates: List['ModelBase']):
    """
    Turn the model_candidate list into a list with unique entries.

    Parameters:
    model_candidates:
        list of models, potentially with redundant entries.
    """
    unique_list = []
    id_set = set()

    for model in model_candidates:
        if model.id not in id_set:
            unique_list.append(model)

    return unique_list


def brute_force_search(
    super_model: ModelBase, criterion: CriterionBase = AIC, result: ModelSelectionResult = None, **kwargs
):
    """
    Exhaustive search.

    Parameters:
    super_model: ModelBase
        The most general model.
        All possible sub-models of this model are considered.
    criterion:
        Criterion, which is used in the selection.
    result:
        Model Selection Result, e.g. for models, that were visited before.
    **kwargs:
        key word arguments, that are forwarded to the model.fit function
        (i.e. look there, to find out which kwargs you need, e.g. for a LinearModel)
    """
    if result is None:
        result = ModelSelectionResult()

    model_empty_id = super_model.id.replace('1', '0')
    model_candidates = [super_model]

    while True:
        # if no more sub-model: return.
        if not len(model_candidates):
            return result

        for model_candidate in model_candidates:

            # do not fit a model twice...
            if model_candidate.id in result.keys():
                continue

            # do not fit the empty model
            if model_candidate.id == model_empty_id:
                return result

            model_candidate.fit(**kwargs)

            result[model_candidate.id] = criterion(model_candidate)

        # get the model_candidates for the next round
        model_candidates = [
            model_candidate.get_submodels() for model_candidate in model_candidates
        ]
        # turn "list of list" into list, and remove redundant entries.
        model_candidates = _get_unique_list(list(chain(*model_candidates)))
