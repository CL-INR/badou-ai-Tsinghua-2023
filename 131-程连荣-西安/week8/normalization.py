import numpy as np


X = np.random.randint(1, 18, 24)
print(X)


def typical(X):
    min = np.min(X)
    max = np.max(X)
    x = [(y - min) / (max - min) for y in X]
    print(x)
    return x


def ZScore(X):
    sum = 0
    deta = 0
    for i in range(len(X)):
        sum += X[i]
    mu = sum / len(X)
    for i in range(len(X)):
        deta += ((X[i]-mu) ** 2)
    deta = (deta)**0.5
    x = [(y-mu)/deta for y in X]
    print(x, sum, mu, deta)
    return x

typical(X)
ZScore(X)
