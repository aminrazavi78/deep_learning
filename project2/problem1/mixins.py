import csv
import numpy as np
import matplotlib.pyplot as plt


class TrainData:
    def __init__(self):
        self.X = []
        self.Y = []

        # these two just for representation in plot
        self.x = []
        self.y = []

    def feed_for_deg1_equation(self):
        csv_file = open('data/train_set.csv', 'r')
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            if row[0] == 'x':
                continue
            self.X.append([1, float(row[0])])
            self.x.append(float(row[0]))

            self.Y.append([float(row[1])])
            self.y.append(float(row[1]))

    def feed_for_deg5_equation(self, n):
        csv_file = open('data/train_set.csv', 'r')
        csvreader = csv.reader(csv_file)
        number_of_data = 0
        for row in csvreader:
            if row[0] == 'x':
                continue
            number_of_data += 1
            self.X.append([1, float(row[0]), float(row[0])**2, float(row[0])**3, float(row[0])**4, float(row[0])**5])
            self.x.append(float(row[0]))

            self.Y.append([float(row[1])])
            self.y.append(float(row[1]))

            if number_of_data == n:
                break


class TestData:
    def __init__(self):
        self.x = []
        self.y = []

    def feed_for_deg5_equation(self, n):
        csv_file = open('data/test_set.csv', 'r')
        csvreader = csv.reader(csv_file)
        number_of_data = 0
        for row in csvreader:
            if row[0] == 'x':
                continue
            number_of_data += 1
            self.x.append(float(row[0]))
            self.y.append(float(row[1]))

            if number_of_data == n:
                break


def w_calculator(x, y):
    xt = x.transpose()
    xtx = xt * x
    xtx_inverse = np.linalg.inv(xtx)
    xtx_inverse_xt = xtx_inverse * xt
    w = xtx_inverse_xt * y
    return w


def plot_drawer_with_deg1_equation(x, y, w):
    ax = plt.gca()
    ax.set_xlim([-1, 7])
    ax.set_ylim([-15, 15])

    plt.scatter(x, y, color='red')
    plt.xlabel('x1 - axis')
    plt.ylabel('x2 - axis')

    x_for_line = np.linspace(-7, 7, 100)
    y_for_line = w.item(0) + (w.item(1) * x_for_line)
    plt.plot(x_for_line, y_for_line, color='green')

    plt.show()


def plot_drawer_with_deg5_equation(x, y, w):
    ax = plt.gca()
    ax.set_xlim([-1, 7])
    ax.set_ylim([-15, 15])

    plt.scatter(x, y, color='red')
    plt.xlabel('x1 - axis')
    plt.ylabel('x2 - axis')

    x_for_line = np.linspace(-7, 7, 100)
    y_for_line = w.item(0) + (w.item(1) * x_for_line) + (w.item(2) * x_for_line ** 2) + (w.item(3) * x_for_line ** 3) + (w.item(4) * x_for_line ** 4) + (w.item(5) * x_for_line ** 5)
    plt.plot(x_for_line, y_for_line, color='green')

    plt.show()


def f(x, w):
    return w[0] + w[1]*x + w[2]*x**2 + w[3]*x**3 + w[4]*x**4 + w[5]*x**5


def error(x, y, w):
    e = 0
    for i in range(len(x)):
        e += (y[i] - f(x[i], w))

    return e


