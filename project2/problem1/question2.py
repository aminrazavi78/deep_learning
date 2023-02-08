import numpy as np

from project2.problem1.mixins import w_calculator, plot_drawer_with_deg5_equation, TrainData

if __name__ == '__main__':
    data = TrainData()
    data.feed_for_deg5_equation(200)

    x = data.x
    y = data.y

    X = np.matrix(data.X)
    Y = np.matrix(data.Y)

    w = w_calculator(X, Y)
    plot_drawer_with_deg5_equation(x, y, w)
