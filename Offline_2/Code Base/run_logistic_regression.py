"""
main code that you will run
"""

from linear_model import LogisticRegression
from data_handler import load_dataset, split_dataset
from metrics import precision_score, recall_score, f1_score, accuracy


if __name__ == '__main__':

    lr = float(input('Enter learning rate: '))
    n_iters = int(input('Enter number of iterations: '))
    verbose = bool(int(input('Enter verbose (0 or 1): ')))
    shuffle = bool(int(input('Shuffle Dataset? (0 or 1): ')))

    # data load
    X, y = load_dataset()

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y, shuffle=shuffle)

    # training
    params = {'learning_rate': lr, 'n_iters': n_iters, 'verbose': verbose}
    classifier = LogisticRegression(params)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
