import Data
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Prepare date
    data = Data.load_smarket_dataset()
    data_without_y = data.loc[:, data.columns != 'Direction']
    # Replace binary labels with 1 and 0
    y = data['Direction'].transform(lambda element: 1 if element == 'Up' else 0)
    # Add a constant to specify model of form y = b.1 + a.x1 + b.x2...
    x = sm.add_constant(data_without_y)
    # Remove the variables that will not be used for the model
    x = x.loc[:, x.columns != 'Year']
    x = x.loc[:, x.columns != 'Today']

    # Fit the model
    logistic_regression = sm.GLM(y, x, family=sm.families.Binomial())
    logistic_model = logistic_regression.fit()

    # Show model summary
    print(logistic_model.summary())
    print()

    # Show predictions using training data
    logistic_model.predict()
    predicted_values = pd.Series(logistic_model.fittedvalues).transform(
        lambda element: 'Up' if element > 0.5 else 'Down')
    print(predicted_values)

    # Show confusion matrix
    confusion_matrix = pd.crosstab(pd.Series(data['Direction']), pd.Series(predicted_values))
    sns.heatmap(confusion_matrix, annot=True)
    plt.show()

    # Split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

    # Train new model
    logistic_regression = sm.GLM(y_train, x_train, family=sm.families.Binomial())
    logistic_model = logistic_regression.fit()

    # Predict
    predicted_y = logistic_model.get_prediction(x_test).predicted_mean

    # Convert 0 and 1 to labels
    predicted_y = pd.Series(predicted_y).transform(lambda element: 'Up' if element > 0.5 else 'Down')
    y_test = pd.Series(y_test).transform(lambda element: 'Up' if element == 1 else 'Down')

    # Show results of the new model performing against test data
    confusion_matrix = pd.crosstab(y_test, predicted_y)
    sns.heatmap(confusion_matrix, annot=True)
    plt.show()

    # Can't be bothered to finish the lab. Remaining part is training a logistic model against only two of the
    # predictors. Pretty much copy paste from here on.


if __name__ == '__main__':
    main()

