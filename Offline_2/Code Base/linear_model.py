import numpy as np

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        self.params = params
        

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        
        # initialize parameters
        self.theta = np.zeros(X.shape[1])                       # initialize theta vector (size same as number of features)
        self.bias = 0                                           # initialize bias

        # gradient descent
        for i in range(self.params['n_iters']):
            y_pred = self.sigmoid(np.dot(X, self.theta) + self.bias)    # calculate prediction
            dw = np.dot(X.T, (y_pred - y)) / y.shape[0]                 # calculate dJ/dw
            db = np.sum(y_pred - y) / y.shape[0]                        # calculate dJ/db
            self.theta -= self.params['learning_rate'] * dw             # update parameters : w = w - lr * dJ/dw
            self.bias -= self.params['learning_rate'] * db              # update parameters : b = b - lr * dJ/db

            if self.params['verbose'] and i % 100 == 0:
                print(f'loss: {self.loss(y_pred, y):.4f}, \t')
        
        return self
        

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        y_pred = self.sigmoid(np.dot(X, self.theta) + self.bias)        # calculate prediction
        y_pred = np.where(y_pred >= 0.5, 1, 0)                          # apply threshold
        return y_pred


    # Activation function 
    # used to map any real value between 0 and 1
    def sigmoid(self, z):
        """
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(-z))

    #Loss function
    def loss(self, h, y):
        """
        :param h:
        :param y:
        :return:
        """
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()