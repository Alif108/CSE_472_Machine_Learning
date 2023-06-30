"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""

import numpy as np

def accuracy(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    # calculate accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def precision_score(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    # calculate precision score
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    # calculate recall score
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    # calculate f1 score
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r)
