"""Test the `result`-object."""
import pytest

from modelSelector import ModelSelectionResult


def test_result():
    """Test the `result`-object."""
    # test setting values in the result.
    result = ModelSelectionResult()

    # initialize with floats
    result['M_00'] = 0.0
    result['M_01'] = 1.0
    result['M_11'] = 42.0

    # initialize with None
    result['M_10'] = None

    # check the function for number of parameters.
    assert result.n_parameters == 2

    # test, that other data types are caught.
    with pytest.raises(ValueError):
        result['M_11'] = {'dict': 42}

    # check invalid keys
    invalid_keys = [42, 'M1010', 'M_0123']
    for key in invalid_keys:
        with pytest.raises(KeyError):
            result[key] = 42

    # check number of fitted models
    assert result.n_fitted_models == 3

    # check the get_best_model function
    model, crit = result.get_best_model()
    assert model == 'M_00'
    assert crit == 0.0
