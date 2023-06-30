from data_handler import bagging_sampler
import numpy as np


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        
        self.estimators = []
        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)                              # sample with replacement
            self.estimators.append(self.base_estimator.fit(X_sample, y_sample))     # fit base estimator
        
        return self


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        y_pred = np.zeros(X.shape[0])                                               # initialize prediction vector (size same as number of samples)
        for estimator in self.estimators:                   
            y_pred += estimator.predict(X)                                          # add prediction of each estimator           
        y_pred = np.where(y_pred >= self.n_estimator/2, 1, 0)                       # apply majority voting

        return y_pred

