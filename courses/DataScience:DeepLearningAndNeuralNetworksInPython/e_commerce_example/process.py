import pandas as pd
import numpy as np
import os

# so scripts from other folders can import this file
dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


dataset_path = "/Users/rahatibnrafiq/MyWorkSpace/DeepLearningCodes/datasets"


def get_data():
    df = pd.read_csv(dataset_path + '/ecommerce_data.csv')

    data = df.values
    np.random.shuffle(data)

    # split features and labels
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int32)

    # scale the first two numerical columns
    for i in (1, 2):
        X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

    # one-hot encode the categorical data
    # create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2[:, 0:(D - 1)] = X[:, 0:(D - 1)]  # non-categorical

    # one-hot
    for n in range(N):
        t = int(X[n, D - 1])
        X2[n, t + D - 1] = 1

    # assign X2 back to X, since we don't need original anymore
    X = X2

    # split train and test
    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    Xtest = X[-100:]
    Ytest = Y[-100:]

    return Xtrain, Ytrain, Xtest, Ytest


def get_binary_data():
    Xtrain, Ytrain, Xtest, Ytest = get_data()
    X2train = Xtrain[Ytrain <= 1]
    Y2train = Ytrain[Ytrain <= 1]
    X2test = Xtest[Ytest <= 1]
    Y2test = Ytest[Ytest <= 1]
    return X2train, Y2train, X2test, Y2test

