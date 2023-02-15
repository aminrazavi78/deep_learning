import csv
from mixins import w_calculator
from question1 import generate
import numpy as np


def get_y():
    y = []
    csv_file = open('data/train_set.csv', 'r')
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        if row[0] == 'f1':
            continue
        y.append([float(row[-1])])
    return y


if __name__ == '__main__':
    data = generate()['standard_train_data']
    Y = get_y()
    delta = 0.1
    w = w_calculator(np.matrix(data), np.matrix(Y), delta)
    print(w)
