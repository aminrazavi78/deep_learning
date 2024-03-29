import matplotlib.pyplot as plt
from numpy import linspace
from project1.perceptron import Perceptron


if __name__ == '__main__':

    p = Perceptron()
    p.find_the_best_weights()

    # For Train
    train_colors = []
    for i in p.Label_train:
        if i == 1:
            train_colors.append('red')
        else:
            train_colors.append('blue')

    ax = plt.gca()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

    plt.scatter(p.X1_train, p.X2_train, color=train_colors)
    plt.xlabel('x1 - axis')
    plt.ylabel('x2 - axis')

    x = linspace(-3, 3, 100)
    y = (p.W[1] * x) / p.W[2] + p.W[0] / p.W[2]
    plt.plot(x, y, color='green')

    plt.show()

    # For Test
    test_colors = []
    for i in p.Label_test:
        if i == 1:
            test_colors.append('red')
        else:
            test_colors.append('blue')

    ax = plt.gca()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

    plt.scatter(p.X1_test, p.X2_test, color=test_colors)
    plt.xlabel('x1 - axis')
    plt.ylabel('x2 - axis')

    x = linspace(-2, 2, 100)
    y = (p.W[1] * x) / p.W[2] + p.W[0] / p.W[2]
    plt.plot(x, y, color='green')

    plt.show()

    err_train = p.accuracy_on_training()
    err_test = p.accuracy_on_testing()

    print("Number of misclassified data in train part: ", err_train)
    print("Number of misclassified data in test part: ", err_test)
