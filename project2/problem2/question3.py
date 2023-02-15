import random
import numpy as np
import matplotlib.pyplot as plt

from mixins import w_calculator
from mixins import error
from question1 import generate
from question2 import get_y


def average(l):
    s = 0
    for x in l:
        s += x[0]
    return s/len(l)


def generate_1degree(data):
    g_l = []
    for i in data:
        temp_l = []
        for j in i:
            temp_l.append(j)
        g_l.append(temp_l)
    return g_l


def generate_3degree(data):
    g_l = []
    for i in data:
        temp_l = []
        f = False
        for j in i:
            if not f:
                f = True
                temp_l.append(j)
                continue
            temp_l.append(j)
            temp_l.append(j**2)
            temp_l.append(j**3)
        g_l.append(temp_l)

    return g_l


def generate_5degree(data):
    g_l = []
    for i in data:
        temp_l = []
        f = False
        for j in i:
            if not f:
                f = True
                temp_l.append(j)
                continue
            temp_l.append(j)
            temp_l.append(j ** 2)
            temp_l.append(j ** 3)
            temp_l.append(j ** 4)
            temp_l.append(j ** 4)
        g_l.append(temp_l)

    return g_l


def degree_n(data, landa):
    data = list(data)
    train_error = []
    validation_error = []
    f_w = np.zeros((len(data[0]),), dtype=float)
    y = get_y()
    # add y into the end of the list because in shuffling do not miss corresponding y
    for j in range(len(data)):
        data[j] = np.append(data[j], y[j][0])

    for i in range(10):
        shuffled_data = []

        changed_data = random.sample(list(data), len(data))
        shuffled_data.append(changed_data[:81])
        shuffled_data.append(changed_data[81:162])
        shuffled_data.append(changed_data[162:243])
        shuffled_data.append(changed_data[243:324])
        shuffled_data.append(changed_data[324:])

        for j in range(5):
            t = list(shuffled_data)
            test = t.pop(j)
            train = t
            temp_l = []
            # this for just for putting all train data in one list
            for x in train:
                for z in x:
                    temp_l.append(z)
            train = temp_l
            # gaining corresponding y
            y_train = [[i[-1]] for i in train]
            y_test = [[i[-1]] for i in test]
            temp_l = []
            # remove y from train
            for k in train:
                t = np.delete(k, -1)
                temp_l.append(t)
            train = temp_l
            # remove y from test
            temp_l = []
            for k in test:
                t = np.delete(k, -1)
                temp_l.append(t)
            test = temp_l

            w = w_calculator(train, y_train, landa)
            t_w = np.array([z[0] for z in w])
            f_w += t_w

            # error for test
            validation_error.append(list(error(test, y_test, w)))

            # error for train
            train_error.append(list(error(train, y_train, w)))
    return train_error, validation_error, f_w/50


if __name__ == '__main__':
    data = generate()['standard_train_data']
    data_1 = generate_1degree(data)
    data_3 = generate_3degree(data)
    data_5 = generate_5degree(data)

    deg1_0 = degree_n(data_1, 0.01)
    deg3_0 = degree_n(data_3, 0.01)
    deg5_0 = degree_n(data_5, 0.01)

    deg1_1 = degree_n(data_1, 1.0)
    deg3_1 = degree_n(data_3, 1.0)
    deg5_1 = degree_n(data_5, 1.0)

    deg1_10 = degree_n(data_1, 10.0)
    deg3_10 = degree_n(data_3, 10.0)
    deg5_10 = degree_n(data_5, 10.0)

    print('train error for landa=0 and degree=1: ', average(deg1_0[0]))
    print('validation error for landa=0 and degree=1: ', average(deg1_0[1]))
    print('train error for landa=0 and degree=3: ', average(deg3_0[0]))
    print('validation error for landa=0 and degree=3: ', average(deg3_0[1]))
    print('train error for landa=0 and degree=5: ', average(deg5_0[0]))
    print('validation error for landa=0 and degree=5: ', average(deg5_0[1]))

    print('train error for landa=1 and degree=1: ', average(deg1_1[0]))
    print('validation error for landa=1 and degree=1: ', average(deg1_1[1]))
    print('train error for landa=1 and degree=3: ', average(deg3_1[0]))
    print('validation error for landa=1 and degree=3: ', average(deg3_1[1]))
    print('train error for landa=1 and degree=5: ', average(deg5_1[0]))
    print('validation error for landa=1 and degree=5: ', average(deg5_1[1]))

    print('train error for landa=10 and degree=1: ', average(deg1_10[0]))
    print('validation error for landa=10 and degree=1: ', average(deg1_10[1]))
    print('train error for landa=10 and degree=3: ', average(deg3_10[0]))
    print('validation error for landa=10 and degree=3: ', average(deg3_10[1]))
    print('train error for landa=10 and degree=5: ', average(deg5_10[0]))
    print('validation error for landa=10 and degree=5: ', average(deg5_10[1]))

    #
    # plt.boxplot([i[0] for i in deg1_0[0]])
    # plt.show()
    # plt.boxplot([i[0] for i in deg1_0[1]])
    # plt.show()
    #
    # plt.boxplot([i[0] for i in deg3_0[0]])
    # plt.show()
    # plt.boxplot([i[0] for i in deg3_0[1]])
    # plt.show()
    #
    # plt.boxplot([i[0] for i in deg5_0[0]])
    # plt.show()
    # plt.boxplot([i[0] for i in deg5_0[1]])
    # plt.show()
    #
    # plt.boxplot([i[0] for i in deg1_1[0]])
    # plt.show()
    # plt.boxplot([i[0] for i in deg1_1[1]])
    # plt.show()
    #
    # plt.boxplot([i[0] for i in deg3_1[0]])
    # plt.show()
    # plt.boxplot([i[0] for i in deg3_1[1]])
    # plt.show()
    #
    # plt.boxplot([i[0] for i in deg5_1[0]])
    # plt.show()
    # plt.boxplot([i[0] for i in deg5_1[1]])
    # plt.show()
    #
    # plt.boxplot([i[0] for i in deg1_10[0]])
    # plt.show()
    # plt.boxplot([i[0] for i in deg1_10[1]])
    # plt.show()
    #
    # plt.boxplot([i[0] for i in deg3_10[0]])
    # plt.show()
    # plt.boxplot([i[0] for i in deg3_10[1]])
    # plt.show()
    #
    # plt.boxplot([i[0] for i in deg5_10[0]])
    # plt.show()
    # plt.boxplot([i[0] for i in deg5_10[1]])
    # plt.show()
