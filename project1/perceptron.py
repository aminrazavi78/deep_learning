from numpy import random, sign
import csv


class Perceptron:
    def __init__(self):
        self.z = 0

        self.X1_train = []
        self.X2_train = []
        self.Label_train = []

        self.X1_test = []
        self.X2_test = []
        self.Label_test = []

        self.W = random.rand(3)  # w0 = W[0], w1 = W[1], w2 = W[2]

        self._feeding_network_by_train_data()
        self._feeding_network_by_test_data()

    def _feeding_network_by_train_data(self):
        csv_file = open('data/train.csv', 'r')
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            if row[0] == 'x1':
                continue
            self.X1_train.append(float(row[0]))
            self.X2_train.append(float(row[1]))
            if float(row[2]) == 1:
                self.Label_train.append(1)
            else:
                self.Label_train.append(-1)

    def _feeding_network_by_test_data(self):
        csv_file = open('data/test.csv', 'r')
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            if row[0] == 'x1':
                continue
            self.X1_test.append(float(row[0]))
            self.X2_test.append(float(row[1]))
            if float(row[2]) == 1:
                self.Label_test.append(1)
            else:
                self.Label_test.append(-1)

    def find_the_best_weights(self):
        in_pocket = self.W
        number_of_misclassified = len(self.X1_train)
        alpha = 0.1
        iteration_num = 0
        while iteration_num < 100:
            for i in range(len(self.X1_train)):
                self.W = in_pocket
                iteration_num += 1
                s = sign(self.W[0] + self.W[1] * self.X1_train[i] + self.W[2] * self.X2_train[i])
                if s * self.Label_train[i] != 1:
                    self.W[0] += alpha * self.Label_train[i]
                    self.W[1] += alpha * self.X1_train[i] * self.Label_train[i]
                    self.W[2] += alpha * self.X2_train[i] * self.Label_train[i]
            n = 0
            for j in range(len(self.X1_train)):
                s = sign(self.W[0] + self.W[1] * self.X1_train[j] + self.W[2] * self.X2_train[j])
                if s * self.Label_train[j] != 1:
                    n += 1
            if n <= number_of_misclassified:
                number_of_misclassified = n
                in_pocket = self.W

        self.W = in_pocket

    def accuracy_on_training(self):
        # just number of misclassified data
        misclassified = 0
        for i in range(len(self.X1_train)):
            s = sign(self.W[0] + self.W[1] * self.X1_train[i] + self.W[2] * self.X2_train[i])
            if s*self.Label_train[i] != 1:
                misclassified += 1
        return misclassified

    def accuracy_on_testing(self):
        # just number of misclassified data
        misclassified = 0
        for i in range(len(self.X1_test)):
            s = sign(self.W[0] + self.W[1] * self.X1_test[i] + self.W[2] * self.X2_test[i])
            if s * self.Label_test[i] != 1:
                misclassified += 1
        return misclassified
