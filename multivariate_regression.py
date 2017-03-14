#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
# import pandas as pd
from scipy.stats import truncnorm
import numpy as np
import sys

# the feature mean, max and min


def normal_equation(x, y):
    """
    reference:
    http://stackoverflow.com/questions/17679140/multiple-linear-regression-with-python/34877221#34877221
    """
    m, n = np.shape(x)

    X = np.zeros(m*4)
    X.shape = (m, 4)
    X[:, 0] = 1
    X[:, 1:4] = x

    IdentitySize = X.shape[1]
    IdentityMatrix = np.zeros((IdentitySize, IdentitySize))
    np.fill_diagonal(IdentityMatrix, 1)
    lamb = 1
    xTx = X.T.dot(X) + lamb * IdentityMatrix
    XtX = np.linalg.inv(xTx)
    XtX_xT = XtX.dot(X.T)
    theta = XtX_xT.dot(y)
    print(theta)


class LinearRegression(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.data_mean = self.data_max = self.data_min = 0

    def normalize_data(self, data):
        norm_data = (data - self.data_mean) / (self.data_max - self.data_min)
        return norm_data

    def separate_xy(self, matrix):
        X = np.append(matrix[:, :-1], np.ones([len(matrix), 1]), 1)
        Y = matrix[:, -1]
        return X, Y

    def generate_batch(self, data, batch_size=5):
        """
        input:
            data: the normalized data
            batch_size: the mini batch size
        output:
            yield a batch
        """
        # introduce the stochastic gradient descent
        np.random.shuffle(data)
        # save the transform for the test data
        # normalize the data using feature scaling
        for i in xrange(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            # for the X the last column is the intersection
            yield self.separate_xy(batch)

    def predict_output(self, feature_matrix):
        ''' Returns an array of predictions
        inputs:
            feature_matrix: 2-D array of dimensions data points by features

        output:
            1-D array of predictions
        '''
        predictions = np.dot(feature_matrix, self.coefficients)
        return predictions

    def cost(self, O, Y):
        return np.sum(Y - O)**2

    def gradient_descent(self, X, Y, epsilon):
        '''
        inputs:
            X: 2-D array of dimensions data points by features
            Y: 1-D array of true output
            epsilon: float, the tolerance at which the algorithm will terminate

        output:
            1-D array of estimated coefficients
            if cnoverged or not
        '''
        converged = False
        O = self.predict_output(X)
        residuals = O - Y
        gradient_sum_squares = 0
        for i in range(len(self.coefficients)):
            partial = np.dot(residuals, X[:, i]) / len(O)
            # using the mean squared error, the partial is (o-y)*xᵢ
            gradient_sum_squares += partial ** 2
            self.coefficients[i] -= self.learning_rate*partial
            sys.stdout.write("cost: "+str(self.cost(O, Y))+"\r\n")
            sys.stdout.flush()
        if gradient_sum_squares < epsilon ** 2:
            converged = True
        return converged

    def read_data(self, path, delimiter, skip_first_line):
        with open(path, "rb") as f:
            lines = (line for line in f if not line.startswith('#'))
            data = np.loadtxt(
                lines, delimiter=delimiter, skiprows=skip_first_line)
            return data

    def train(self, path, delimiter, skip_first_line):
        converged = False
        max_iterations = 1000
        iteration = 0
        data = self.read_data(path, delimiter, skip_first_line)
        self.feature_size = data.shape[1] - 1
        self.data_mean = data.mean(axis=0)
        self.data_max = data.max(axis=0)
        self.data_min = data.min(axis=0)
        norm_data = self.normalize_data(data)

        self.coefficients = truncnorm(
            a=-2/3, b=2/3, scale=3).rvs(self.feature_size+1)

        while(not converged and iteration < max_iterations):
            iteration += 1
            for X, Y in self.generate_batch(
                data=norm_data,
                batch_size=10,
            ):
                converged = self.gradient_descent(X, Y, 0.07)

    def test(self, path, delimiter, skip_first_line):
        """
        the tolerance is a percent
        """
        tolerance = abs(self.data_max[-1] - self.data_min[-1]) * 0.02
        data = self.read_data(path, delimiter, skip_first_line)
        norm_data = self.normalize_data(data)
        X, Y = self.separate_xy(norm_data)
        O = self.predict_output(X)
        print("Accuracy: " + str(sum(abs(Y - O) < tolerance) / len(Y)))

if __name__ == "__main__":
    lr = LinearRegression(0.01)
    lr.train("./data/data.csv", ",", True)
    lr.test("./data/test_data.csv", ",", True)
    # lr.predict_output()

# an approximation:
# [ w0          w1          w2          w3        ]
# [ 2.02847816  2.97117357 -0.53917756  0.96976013]
