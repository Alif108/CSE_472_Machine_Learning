import numpy as np
import pandas as pd
import math

csv_file_path = "data_banknote_authentication.csv"

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    dataset = pd.read_csv(csv_file_path)
    X = dataset.iloc[:, :-1].values     # all rows, all columns except last one (exclude class column)
    y = dataset.iloc[:, -1].values      # all rows, last column (class column)
    
    return X, y


def split_dataset(X, y, test_size=0.2, shuffle=False):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    if shuffle:
        reconstructed_matrix = list(zip(X, y))          # zip X and y together
        np.random.shuffle(reconstructed_matrix)         # shuffle the matrix
        X, y = zip(*reconstructed_matrix)               # unzip the matrix
        X = np.array(X)
        y = np.array(y)
        
    test_data_len = math.floor(len(X) * test_size)      # length of test data
    train_data_len = len(X) - test_data_len             # length of train data
    
    X_train = X[:train_data_len]                        # train data : first train_data_len rows
    X_test = X[train_data_len:]                         # test data : last test_data_len rows
    y_train = y[:train_data_len]                        # train data : first train_data_len rows
    y_test = y[train_data_len:]                         # test data : last test_data_len rows
    
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    X_sample, y_sample = [], []
    for i in range(len(X)):
        index = np.random.randint(0, len(X))            # random index
        X_sample.append(X[index])                       # append X[index] to X_sample
        y_sample.append(y[index])                       # append y[index] to y_sample
    
    X_sample = np.array(X_sample)
    y_sample = np.array(y_sample)
    
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    
    return X_sample, y_sample
