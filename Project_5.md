## Project 5: Comparison of Different Regularization and Variable Selection Techniques

In this project, different regularization techniques including Ridge, LASSO, Elastic Net, SCAD, and Square Root Lasso will be applied for comparison.

```
# general imports for these methods
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from matplotlib import pyplot
# import an optimizer
from scipy.optimize import minimize
# the following is very important
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
# import numba and jit
from numba import njit
```
The data for this project was found using kaggle.com. I used the NBA MVP dataset that holds NBA Stats from 1991-2021. I had trouble importing the csv because of the formatting but was able to resolve those issues using: ```df = pd.read_csv('drive/My Drive/Colab Notebooks/DATA 410/data/mvps.csv', sep=';', encoding='latin-1')```. This dataset looks like: ![](mvp_data.png)


1. Sklearn compliant functions for Square Root Lasso and SCAD, used in conjunction with GridSearchCV for finding optimal hyper-parameters when data such as x-bar; and y-bar; are given.

Grid Search output: 
![](gridsearch_fit_output.png)

model validation after gridsearch: 0.035639050632911395

model validation after SQRT Lasso:0.0361251735886018





















