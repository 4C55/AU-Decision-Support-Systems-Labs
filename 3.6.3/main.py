import Data
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main():
    # Load the data
    boston = Data.load_boston_dataset()
    y = boston['MEDV']

    # Add a constant to specify model of form y = b.1 + a.x1 + b.x2...
    x = sm.add_constant(boston.loc[:, boston.columns != 'MEDV'])

    # Fit the model
    linear_regression = sm.OLS(y, x)
    linear_regression = linear_regression.fit()

    # Summary of the model
    print(linear_regression.summary())
    print()

    # VIF - variance inflation factors
    vif = pd.DataFrame()
    vif['VIF factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif['Feature'] = x.columns
    print(vif)


if __name__ == '__main__':
    main()
