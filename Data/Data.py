from sklearn.datasets import load_boston
import pandas as pd
import os


def load_boston_dataset():
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    return boston


def load_smarket_dataset():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'Smarket.csv')
    return pd.read_csv(filename, usecols=range(1, 10))
