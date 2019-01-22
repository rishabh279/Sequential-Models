import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

path = os.path.dirname(os.path.dirname(os.path.abspath('utility.py')))

print(path)
def get_transformed_data():
    pixel_data = pd.read_csv(path + '/data/train.csv')
    data = pixel_data.values
    np.random.shuffle(data)

    x = data[:, 1:]
    y = data[:, 0]

    xtrain = x[:-1000]
    ytrain = y[:-1000]
    xtest = x[-1000:]
    ytest = y[-1000:]

    # center the data
    mu = np.mean(xtrain, axis=0)
    xtrain = xtrain - mu
    xtest = xtest - mu

    # transform the data
    pca = PCA()
    ztrain = pca.fit_transform(xtrain)
    ztest = pca.transform(xtest)

    ztrain = ztrain[:, :300]
    ztest = ztest[:, :300]

    # normalize Z
    mu = ztrain.mean(axis=0)
    std = ztrain.std(axis=0)
    ztrain = (ztrain - mu) / std
    ztest = (ztest - mu) / std

    return ztrain, ytrain, ztest, ytest


def get_normalized_data():

    pixel_data = pd.read_csv(path + '/data/train.csv')
    data = pixel_data.values.astype(np.float32)
    np.random.shuffle(data)

    x = data[:, 1:]
    y = data[:, 0]

    xtrain = x[:-1000]
    ytrain = y[:-1000]
    xtest = x[-1000:]
    ytest = y[-1000:]

    mu = xtrain.mean(axis=0)
    std = xtrain.std(axis=0)
    np.place(std, std == 0, 1)

    xtrain = (xtrain - mu) / std
    xtest = (xtest - mu) / std

    return xtrain, ytrain.astype(np.int32), xtest, ytest.astype(np.int32)


def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind