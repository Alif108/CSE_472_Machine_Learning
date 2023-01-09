"""
main code that you will run
"""

from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset, load_test_data
from metrics import accuracy, precision_score, recall_score, f1_score

if __name__ == '__main__':

    lr = float(input('Enter learning rate: '))
    n_iters = int(input('Enter number of iterations: '))
    n_estimator = int(input('Enter number of estimators: '))
    verbose = bool(int(input('Enter verbose (0 or 1): ')))
    shuffle = bool(int(input('Shuffle Dataset? (0 or 1): ')))

    # # data load
    # X, y = load_dataset()

    # # split train and test
    # X_train, y_train, X_test, y_test = split_dataset(X, y)

    test_data_path = "Occupancy_Dataset/datatest2.txt"

    X_train, y_train = load_dataset()
    X_test, y_test = load_test_data(test_data_path)

    # training
    params = {'learning_rate': lr, 'n_iters': n_iters, 'verbose': verbose}
    base_estimator = LogisticRegression(params)
    classifier = BaggingClassifier(base_estimator=base_estimator, n_estimator=n_estimator)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
