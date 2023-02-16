import numpy as np
from project3.problem1.question3_4 import data_editing
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, w):
    y_pred = sigmoid(np.dot(X, w))
    return (y_pred > 0.5).astype(int)


def logistic_regression(X, y, num_iterations, learning_rate, l2_reg_strength, weight_decay):
    w = np.zeros(X.shape[1])
    for i in range(num_iterations):
        # Calculate the predicted y values
        y_pred = sigmoid(np.dot(X, w))
        # Calculate the gradient of the cost function with L2 regularization
        gradient = np.dot(X.T, (y_pred - y)) / y.size + 2 * l2_reg_strength * w
        # Update the weights with weight decay
        w = (1 - weight_decay) * w - learning_rate * gradient
    return w


# Define cross-validation function
def cross_validation(x, y, num_repeats, num_folds, num_iterations, learning_rate, l2_reg_strength, weight_decay):
    # Initialize arrays to store results
    accuracies = np.zeros(num_repeats * num_folds)
    fold = 0
    # Perform cross-validation
    for repeat in range(num_repeats):
        idx = np.random.permutation(len(y))
        x, y = x[idx], y[idx]
        for fold_idx in range(num_folds):
            val_start_idx = len(y) // num_folds * fold_idx
            val_end_idx = len(y) // num_folds * (fold_idx + 1)
            val_x, val_y = x[val_start_idx:val_end_idx], y[val_start_idx:val_end_idx]
            train_x = np.concatenate([x[:val_start_idx], x[val_end_idx:]])
            train_y = np.concatenate([y[:val_start_idx], y[val_end_idx:]])
            w = logistic_regression(train_x, train_y, num_iterations, learning_rate, l2_reg_strength, weight_decay)
            y_pred = predict(val_x, w)
            accuracies[fold] = np.mean(y_pred == val_y)
            fold += 1
    return accuracies , w


if __name__ == '__main__':

    num_repeats = 10
    num_folds = 5
    num_iterations = 1000
    learning_rate = 0.1
    l2_reg_strength = 0.1
    weight_decay = 0.1

    # Run cross-validation and calculate mean accuracy
    data = data_editing()
    train = data[0]
    test = data[1]

    y = train['Survived'].values
    x = train.drop('Survived', axis=1).values
    accuracies_w = cross_validation(x, y, num_repeats, num_folds, num_iterations, learning_rate, l2_reg_strength, weight_decay)
    mean_accuracy = np.mean(accuracies_w[0])
    w = accuracies_w[1]
    print('Mean accuracy:', mean_accuracy)
    print('The Best W: ', w)

    # Do on test data

    test = np.array(test)
    survived_list = []
    for X in test:
        y_hat = np.sum(X*w)
        if y_hat >= 0:
            survived_list.append(1)
        else:
            survived_list.append(0)
