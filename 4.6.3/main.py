import Data
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report

def main():
    # Prepare date
    data = Data.load_smarket_dataset()
    x_train = data[data.Year < 2005][['Lag1', 'Lag2']]
    y_train = data[data.Year < 2005]['Direction']
    x_test = data[data.Year >= 2005][['Lag1', 'Lag2']]
    y_test = data[data.Year >= 2005]['Direction']

    lda = LinearDiscriminantAnalysis()
    lda_model = lda.fit(x_train, y_train)

    # Priors - probability of marget going Up and Down
    print(lda_model.priors_)
    print()

    # Mean values - for each class (Up, Down) for each predictor
    print(lda_model.means_)
    print()

    # Coefficients. For some reason different from r, but predictions are identical
    print(lda_model.coef_)

    y_predicted = lda_model.predict_proba(x_test)
    print(y_predicted)

if __name__ == '__main__':
    main()

