import numpy as np
import csv as csv
from collections import Counter


class KNearestNeighbor(object):

    def __init__(self):
        pass

    def train(self, t_data, t_lables):
        """
    Inputs:
    - training_data: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - training_lables: A numpy array of shape (N,) containing the training labels, where
         training_lables[i] is the label for training_data[i].
    """
        self.training_data = t_data
        self.training_lables = t_lables

    def predict(self, tes_data, k=1, num_loops=0):

        if num_loops == 0:
            dists = self.compute_distances_no_loops(tes_data)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(tes_data)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(tes_data)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):

        num_test = X.shape[0]
        num_train = self.training_data.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                # compute the L2 distance
                dists[i, j] = np.sqrt(np.sum((X[i] - self.training_data[j]) ** 2))
                pass
        return dists

    def compute_distances_one_loop(self, X):
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
        num_test = X.shape[0]
        num_train = self.training_data.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            X_te = X[i, np.newaxis]
            dists[i, :] = np.sqrt(np.sum((X_te - self.training_data) ** 2, axis=1))
            pass
        return dists

    def compute_distances_no_loops(self, X):

        num_test = X.shape[0]
        num_train = self.training_data.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(
            -2 * np.dot(X, self.training_data.T) + np.sum(self.training_data ** 2, axis=1) + (np.sum(X ** 2, axis=1))[:,
                                                                                             np.newaxis])
        pass
        return dists

    def predict_labels(self, dists, k=1):

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        arag = np.argsort(dists)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            closest_y = (self.training_lables[arag[i, : k]])

            pass
            count = Counter(closest_y)
            most_frequent = count.most_common(1)
            y_pred[i] = most_frequent[0][0]
            pass

        return y_pred

    def calculate_error_percentage(self, predictions, y_test_lables):
        incorrect_predictions = np.sum(predictions != y_test_lables)
        error_percentage = (incorrect_predictions / len(y_test_lables)) * 100
        return error_percentage


# Read training data
with open('trainYX.csv', mode='r') as t_file:
    train_reader = csv.reader(t_file)
    train_data = [list(map(float, row)) for row in train_reader]

training_data = np.array([row[1:] for row in train_data])
training_labels = np.array([row[0] for row in train_data])

# Read test data
with open('testYX.csv', mode='r') as test_file:
    test_reader = csv.reader(test_file)
    test_data = [list(map(float, row)) for row in test_reader]

# Assuming there are labels in the test data as well
test_labels = np.array([row[0] for row in test_data])
test_data = np.array([row[1:] for row in test_data])


knnClassifier = KNearestNeighbor();
knnClassifier.train(training_data, training_labels)

for k_value in range(1, 21):
    predictions = knnClassifier.predict(test_data, k=k_value, num_loops=0)
    error_percentage = knnClassifier.calculate_error_percentage(predictions, test_labels)
    print("\nFor k = {:2d}, Error Percentage: {:.2f}% , Accuracy Percentage: {:.2f}%".format(k_value, error_percentage,
                                                                                             100 - error_percentage))
