"""Base model. All other models inherit from the base model."""
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np


@dataclass
class ModelBase(ABC):

    id: str
    parameters: np.array = None

    n_params: int = None
    parameter_indices_free = None

    nll: float = float("-inf")
    n_data: float = None

    def __init__(self, id_str: str):
        """
        Construct a ModelBase object.

        Parameters:
        id: str
            Expected Format "M_<0/1....>", where e.g. M_0101 has two
            of the four possible parameters.
        """
        self.id = id_str
        self.n_params = 0
        self.parameter_indices_free = []

        for i, letter in enumerate(self.id[2:]):
            if letter == "1":
                self.n_params += 1
                self.parameter_indices_free.append(i)

    @abstractmethod
    def fit(self, **kwargs):
        """
        Fits the model for given data.

        Parameters:
        kwargs, i.e. key word arguments.
            handed over, e.g. within model selection.
        """

    def is_submodel(self, sub_model: Union[str, "ModelBase"]):
        """
        Check, if the sub_model  submodel of self.

        Parameters:
        sub_model:
            Either a model, or a model id as string.
        """
        if isinstance(sub_model, str):
            sub_model_id = sub_model
        else:
            sub_model_id = sub_model.id

        for letter_self, letter_submodel in zip(self.id[2:], sub_model_id[2:]):
            if letter_self == "0" and letter_submodel == "1":
                return False
        return True

    def is_supermodel(self, super_model):
        """
        Check, if the `super_model` is a supermodel of self.

        Parameters:
        super_model:
            Either a model, or a model id as string.
        """
        if isinstance(super_model, str):
            super_model_id = super_model
        else:
            super_model_id = super_model.id

        for letter_self, letter_supermodel in zip(self.id[2:], super_model_id[2:]):
            if letter_self == "1" and letter_supermodel == "0":
                return False
        return True

    def get_submodel_ids(self):
        """Get the ids of the submodels (of degree 1, i.e. the children)."""
        submodels = []
        for i, letter in enumerate(self.id):
            if letter == "1":
                submodels.append(self.id[:i] + "0" + self.id[i + 1 :])

        return submodels

    def get_submodels(self) -> List["ModelBase"]:
        """Return a list of all submodels (of degree 1, i.e. the children)."""
        res = []

        for id_str in self.get_submodel_ids():
            res.append(type(self)(id_str))

        return res

    def get_supermodel_ids(self):
        """Get the ids of the supermodels (of degree 1, i.e. the parents)."""
        supermodels = []
        for i, letter in enumerate(self.id):
            if letter == "0":
                supermodels.append(self.id[:i] + "1" + self.id[i + 1 :])

        return supermodels

    def get_supermodels(self) -> List["ModelBase"]:
        """Return a list of all supermodels (of degree 1, i.e. the parents)."""
        res = []

        for id_str in self.get_supermodel_ids():
            res.append(type(self)(id_str))

        return res
