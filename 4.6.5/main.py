import Data
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Prepare date
    data = Data.load_smarket_dataset()
    x_train = data[data.Year < 2005][['Lag1', 'Lag2']]
    y_train = data[data.Year < 2005]['Direction']
    x_test = data[data.Year >= 2005][['Lag1', 'Lag2']]
    y_test = data[data.Year >= 2005]['Direction']

    knn = KNeighborsClassifier(n_neighbors=1)
    knn_model = knn.fit(x_train, y_train)

    # Perform predictions and show confusion matrix
    y_predicted = knn_model.predict(x_test)
    confusion_matrix = pd.crosstab(y_test, y_predicted).transpose()
    sns.heatmap(confusion_matrix, annot=True)
    plt.show()

    # Show accuracy on the test data
    print('Accuracy: %s' % knn_model.score(x_test, y_test))

if __name__ == '__main__':
    main()


