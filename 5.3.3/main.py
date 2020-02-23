import Data
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import PolynomialFeatures


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


def kfold_lienar_regression(x, y, k=1, polynomial=1):
    total = 0
    iterations = int(len(x) / k) + 1
    number_of_elements = len(x)
    folds = 0
    for i in range(iterations):
        start = k * i
        end = k * (i + 1)
        if end > number_of_elements:
            end = number_of_elements
            if start == end:
                continue

        fold = range(start, end)
        x_train = x.drop(fold)
        y_train = y.drop(fold)
        x_test = x.take(fold)
        y_test = y.take(fold)
        x_train = x_train.values.reshape(-1, 1)
        x_test = x_test.values.reshape(-1, 1)
        total = total + fit_and_test_linear_regression(x_train, y_train, x_test, y_test, polynomial)
        folds = folds + 1
    return total / folds

def main():
    # Prepare data
    data = Data.load_auto_dataset()
    x = data['horsepower']
    x = x.astype(float)
    y = data['mpg']
    y = y.astype(float)

    for i in range(1, 11):
        print('%d: %0.2f' % (i, kfold_lienar_regression(x, y, 10, i)))


if __name__ == '__main__':
    main()


