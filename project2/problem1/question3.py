from mixins import *


def calc(n):
    data = TrainData()
    data.feed_for_deg5_equation(n)

    x = data.x
    y = data.y

    X = np.matrix(data.X)
    Y = np.matrix(data.Y)

    w = w_calculator(X, Y)
    plot_drawer_with_deg5_equation(x, y, w)

    e = error(x, y, w)
    print(f'Error in train for {n} datas: ', e)

    test_data = TestData()
    test_data.feed_for_deg5_equation(n)
    x = test_data.x
    y = test_data.y

    e = error(x, y, w)
    print(f'Error in test for {n} datas: ', e)


if __name__ == '__main__':
    calc(10)
    calc(25)
    calc(50)
    calc(100)
    calc(200)
