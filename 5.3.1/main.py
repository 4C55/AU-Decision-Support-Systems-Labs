import Data
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import PolynomialFeatures


def fit_and_test_linear_regression(x_train, y_train, x_test, y_test):
    linear_regression = sm.OLS(y_train, x_train)
    linear_regression = linear_regression.fit()

    # Predict against the test samples
    y_predicted = linear_regression.predict(x_test)

    # Calculate RMSE (MSE^0.5) of the predictions
    prediction_rmse = rmse(y_test, y_predicted)
    return prediction_rmse * prediction_rmse


# Cross validation lab. Split data into train and test sets, train a linear model, test it against the test set.
# Calculate MSE of the prediction results.
def main():
    # Prepare data
    data = Data.load_auto_dataset()
    x = data['horsepower']
    y = data['mpg']

    # Split data into training and testing sets 0.5
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)

    polynomial_features = PolynomialFeatures(degree=1)
    x_train_polynomial_1 = polynomial_features.fit_transform(x_train)
    x_test_polynomial_1 = polynomial_features.fit_transform(x_test)
    fit_mse = fit_and_test_linear_regression(x_train_polynomial_1, y_train, x_test_polynomial_1, y_test)
    print('Linear regression, 1 polynomial MSE %0.2f' % fit_mse)

    polynomial_features = PolynomialFeatures(degree=2)
    x_train_polynomial_2 = polynomial_features.fit_transform(x_train)
    x_test_polynomial_2 = polynomial_features.fit_transform(x_test)
    fit_mse = fit_and_test_linear_regression(x_train_polynomial_2, y_train, x_test_polynomial_2, y_test)
    print('Linear regression, 2 polynomial MSE %0.2f' % fit_mse)

    polynomial_features = PolynomialFeatures(degree=3)
    x_train_polynomial_3 = polynomial_features.fit_transform(x_train)
    x_test_polynomial_3 = polynomial_features.fit_transform(x_test)
    fit_mse = fit_and_test_linear_regression(x_train_polynomial_3, y_train, x_test_polynomial_3, y_test)
    print('Linear regression, 3 polynomial MSE %0.2f' % fit_mse)

if __name__ == '__main__':
    main()


