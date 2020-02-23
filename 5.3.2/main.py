import Data
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


def fit_and_test_linear_regression(x_train, y_train, x_test, y_test, polynomial=1):
    polynomial_features = PolynomialFeatures(degree=polynomial)
    x_train_polynomial = polynomial_features.fit_transform(x_train)
    x_test_polynomial = polynomial_features.fit_transform(x_test)

    linear_regression = sm.OLS(y_train, x_train_polynomial)
    linear_regression = linear_regression.fit()

    # Predict against the test samples
    y_predicted = linear_regression.predict(x_test_polynomial)

    # Calculate RMSE (MSE^0.5) of the predictions
    prediction_rmse = rmse(y_test, y_predicted)
    return prediction_rmse * prediction_rmse


def loocv_lienar_regression(x, y, polynomial=1):
    total = 0
    samples = len(x)
    for i in range(samples):
        x_train = x.drop(i)
        y_train = y.drop(i)
        x_test = pd.Series([x[i]])
        y_test = pd.Series([y[i]])
        x_train = x_train.values.reshape(-1, 1)
        x_test = x_test.values.reshape(-1, 1)
        total = total + fit_and_test_linear_regression(x_train, y_train, x_test, y_test, polynomial)

    return (total / samples)

# LOOCV lab. Fit linear regression model and perform leave one out cross validation using mean square error metric.
# Repeat the procedure for polynomials [1, 5]. Print out the results.
def main():
    # Prepare data
    data = Data.load_auto_dataset()
    x = data['horsepower']
    y = data['mpg']

    for i in range(1, 6):
        print('%d: %0.2f' % (i, loocv_lienar_regression(x, y, i)))


if __name__ == '__main__':
    main()


