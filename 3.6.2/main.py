import Data
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def main():
    # Load the data
    boston = Data.load_boston_dataset()
    y = boston['MEDV']

    # Add a constant to specify model of form y = b.1 + a.x
    x = sm.add_constant(boston['LSTAT'])

    linear_regression = sm.OLS(y, x)
    linear_regression = linear_regression.fit()

    # Coefficients
    params = linear_regression.params
    print('Constant: %f' % params['const'])
    print('Slope: %f' % params['LSTAT'])
    print()

    # Confidence interval
    print('Confidance interval')
    print(linear_regression.conf_int(alpha=0.05))
    print()

    # Predict
    predict_lstat = sm.add_constant(np.array([5, 10, 15]))
    predicted_medv = linear_regression.get_prediction(predict_lstat)
    predicted_medv = predicted_medv.summary_frame(alpha=0.05)
    print('Given lstat values:\n %s' % np.asarray(predict_lstat))
    print('Predicted medv values with 95%% confidance intervals:\n %s' % predicted_medv)
    print()

    # scatter-plot data and fitted line
    fig = sm.graphics.abline_plot(model_results=linear_regression)
    ax = fig.axes[0]
    ax.scatter(np.asarray(boston['LSTAT']), np.asarray(boston['MEDV']))
    ax.margins(.1)
    plt.show()

    # More plots - include residuals plot
    fig = plt.figure(figsize=(12, 8))
    fig = sm.graphics.plot_regress_exog(linear_regression, 'LSTAT', fig=fig)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    fig = sm.graphics.influence_plot(linear_regression, ax=ax, criterion='cooks')
    plt.show()


if __name__ == '__main__':
    main()
