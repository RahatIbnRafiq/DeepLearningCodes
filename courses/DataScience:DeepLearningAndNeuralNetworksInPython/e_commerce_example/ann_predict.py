from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from process import get_data


def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)


# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)


X, Y, _, _ = get_data()

# randomly initialize weights
M = 10  # 10 hidden units in the hidden layer
D = X.shape[1]  # number of input features
K = len(set(Y))  # number of unique Y values will be the same as the output layer for softmax
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

P_Y_given_X = forward(X, W1, b1, W2, b2)
print("P_Y_given_X.shape:", P_Y_given_X.shape)
predictions = np.argmax(P_Y_given_X, axis=1)


print("Score:", classification_rate(Y, predictions))


