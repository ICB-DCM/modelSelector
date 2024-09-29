<img src="modelSelectorLogo.png" width="50%" alt="modelSelector Logo"/>


# modelSelector: Efficient Criterion Based Model Selection in Python

The `modelSelector` Package offers a variety of functions for criterion based model selection in python.

## Main Features
The Package supports:
- Various Model Selection Criteria: 
    - **AIC**, 
    - **AICc**, 
    - **BIC**, 
    - **Mallow's Cp**.
- Custom Models, as well as interfaces to  *linear* and *logistic regression* models via `scikit.learn`.
- Standard selection algorithms, like 
    - **forward**, 
    - **(efficient) backward**, 
    - **exhaustive**, and 
    - **efficient exhaustive**.
- Visualization Routines for visualization of results.

## Installation
From `pip` via
```shell
pip3 install modelSelector
```
or from GitHub via
```shell
git clone <insert repo url here>
pip3 install .
```
## Getting started

Set up a linear regression model for given data (or alternatively use a wrapper 
around your own model) via
```python
from modelSelector import LinearRegressionModel, AIC

# generate the full model, and fit it to data.
model = LinearRegressionModel('M_1111')
criterion = AIC()

# use any of our algorithms to perform the model selection 
result = backward_search(super_model=model,
                         criterion=criterion,
                         exclude_before_fitting=True,
                         data_x=data_x, 
                         data_y=data_y)
```
See the [example notebook](https://github.com/ICB-DCM/modelSelector/blob/main/examples/Hald%20cement%20data.ipynb) for further details and an overview over all supported algorithms.


## Reference

Will be added after publication.