import csv
from project2.problem2.mixins import f
from project2.problem2.question1 import generate
from project2.problem2.question3 import generate_3degree, degree_n


def prediction():
    data = generate()['standard_train_data']

    data_3 = generate_3degree(data)
    deg3_0 = degree_n(data_3, 0.01)
    w = deg3_0[2]
    l = []
    for x in data_3:
        y = f(x, w)
        l.append(y)
    file = open('prediction.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(l)
    file.close()


if __name__ == '__main__':
    prediction()
