"""Result object for the model selection."""
from typing import Tuple


class ModelSelectionResult(dict):
    """
    `ModelSelectionResult` object.

    The result of a model selection is a dictionary, that maps model ids to criterion values.
    The model ids are strings, that start with `M_` and contain only the characters 0 and 1 after that.
    Here a 0 indicates, that a parameter is not included in the model, and a 1 indicates, that it is included.
    """
    def __copy__(self):
        """Copy the object."""
        other = self.__class__()
        for key, value in self.items():
            other[key] = value
        return other

    def __setitem__(self, key, value):
        """
        Setter method, that allows the functionality `result[key]=value`.

        Perform checks on the format of keys and values.
        """
        # check, if the key is a string in the format 'M_[01]+'
        if not isinstance(key, str):
            raise KeyError("Keys must be strings.")
        elif not (key.startswith('M_')) and all(char in '01' for char in key[2:]):
            raise KeyError("Invalid key. Keys must start with `M_` and contain "
                           f"only the characters 0 and 1 after that. Got {key} instead.")

        # check if all keys are the same length
        if len(self.keys()):
            if not len(key)-2 == self.n_parameters:
                raise KeyError("All keys must have the same lenght.")
        else:
            self.n_parameters = len(key)-2

        # Type check for the value
        if not (value is None or isinstance(value, float)):
            raise ValueError(f"Values must be of type {float} or `None`.")

        super().__setitem__(key, value)

    def get_best_model(self) -> Tuple[str, float]:
        """Return the best model, as well as the criterion value."""
        best_criterion = float('inf')
        best_model = None

        for key, value in self.items():
            if value is not None and value < best_criterion:
                best_criterion = value
                best_model = key

        if best_model is None:
            raise RuntimeError("Result object does not contain any fitted model.")

        return best_model, best_criterion

    @property
    def n_fitted_models(self) -> int:
        """Return the number of models, that were fitted within the result object."""
        n_fitted = 0
        for _, value in self.items():
            if value is not None:
                n_fitted += 1
        return n_fitted
