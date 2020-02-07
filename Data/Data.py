from sklearn.datasets import load_boston
import pandas as pd


def load_boston_dataset():
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    return boston