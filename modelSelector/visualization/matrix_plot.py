"""Matrix plot for model selection results."""
from ..result import ModelSelectionResult
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


def matrix_plot(result: ModelSelectionResult):
    """
    Matrix Plot visualization for model selection results.

    Parameters:
    result: ModelSelectionResult
        Model selection result object.
    """
    # For all models, compute the column, and set the matrix entry to the criterion/None.
    # => run matshow from matplotlib

    # 2^parameters, inferred from the length of the model-id string...
    n_parameters_full = len(list(result.keys())[0])-2

    criterion_matrix = np.nan * np.ones((n_parameters_full, 2**n_parameters_full))
    excluded_models_matrix = np.nan * np.ones((n_parameters_full, 2**n_parameters_full))  # matrix, that is used to colorcode the not-evaluate models

    id_list = _get_model_id_list(n_parameters_full)

    # iterate over all models
    for idx_model, model_id in enumerate(id_list):
        for idx_param, char in enumerate(id_list[idx_model][2:]):
            if char == '1':
                # check, if the model is present
                if model_id in result and result[model_id] is not None:
                    criterion_matrix[idx_param, idx_model] = result[model_id]
                else:
                    excluded_models_matrix[idx_param, idx_model] = 1

    """
    Plotting
    """
    # plot and format
    fig, ax = plt.subplots()

    # plot the criteria
    criterion_heatmap = ax.matshow(criterion_matrix)

    plt.yticks(list(range(n_parameters_full)), ['p_' + str(i) for i in range(n_parameters_full)])
    plt.xticks([])

    plt.xlabel('Models')
    plt.ylabel('Parameters')
    plt.title('Model Selection Result')

    # set colorbar
    colorbar = plt.colorbar(criterion_heatmap)
    colorbar.set_label('Criterion Values')

    # plot non-fitted models.
    cmap = plt.cm.get_cmap('gray')
    plt.matshow(excluded_models_matrix, cmap=cmap, ax=ax)

    plt.show()


def _get_model_id_list(n_parameters_full: int):
    """Return a list with all model ids."""
    id_list = [''.join(x) for x in product('01', repeat=n_parameters_full)]
    return ['M_' + model_id for model_id in id_list]
