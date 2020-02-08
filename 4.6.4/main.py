import Data
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Prepare date
    data = Data.load_smarket_dataset()
    x_train = data[data.Year < 2005][['Lag1', 'Lag2']]
    y_train = data[data.Year < 2005]['Direction']
    x_test = data[data.Year >= 2005][['Lag1', 'Lag2']]
    y_test = data[data.Year >= 2005]['Direction']

    qda = QuadraticDiscriminantAnalysis()
    qda_model = qda.fit(x_train, y_train)

    # Priors - probability of market going Up or Down
    print('Prior probabilities:\n%s\n%s' % (qda_model.classes_, qda_model.priors_))
    print()

    # Mean values - for each class (Up, Down) for each predictor
    means = pd.DataFrame.from_records(qda_model.means_, columns=['Lag1', 'Lag2'])
    means.insert(loc=0, value=qda_model.classes_, column='Group')

    print('Mean values\n%s' % means)
    print()

    y_predicted = qda_model.predict(x_test)
    confusion_matrix = pd.crosstab(y_test, y_predicted)
    sns.heatmap(confusion_matrix, annot=True)
    plt.show()


if __name__ == '__main__':
    main()


