import numpy as np

from project2.problem1.mixins import w_calculator, plot_drawer_with_deg1_equation, TrainData

if __name__ == '__main__':

    data = TrainData()
    data.feed_for_deg1_equation()

    x = data.x
    y = data.y

    X = np.matrix(data.X)
    Y = np.matrix(data.Y)

    w = w_calculator(X, Y)
    plot_drawer_with_deg1_equation(x, y, w)

