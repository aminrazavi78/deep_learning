import math
import numpy as np
import pandas as pd


class Standardization:
    def __init__(self, x_input, x_train):
        self.standard = self._do_standard(x_input, x_train)

    def _expect(self, x_train):
        return np.sum(x_train)/len(x_train)

    def _var(self, x_train):
        return self._expect((x_train - self._expect(x_train))**2)

    def _do_standard(self, x_input, x_train):
        l = []
        for i in range(13):
            l.append(list((x_input[i] - self._expect(x_train[i])) / math.sqrt(self._var(x_train[i]))))
        return l


def separate_train_and_test_data_by_columns():
    data = pd.read_csv('data/train_set.csv')
    fs_train = []
    for i in range(1, 14):
        fs_train.append(data[f'f{i}'].tolist())

    data = pd.read_csv('data/test_set.csv')
    fs_test = []
    for i in range(1, 14):
        fs_test.append(data[f'f{i}'].tolist())

    return {'fs_train': fs_train, 'fs_test': fs_test}


def do_reg(first, second):
    fs_standard = []
    for i in range(len(first)):
        fs_standard.append(Standardization(first, second).standard)

    return fs_standard


def generate():
    all_data = separate_train_and_test_data_by_columns()
    standard_train_data = do_reg(all_data['fs_train'], all_data['fs_train'])
    standard_test_data = do_reg(all_data['fs_test'], all_data['fs_train'])
    standard_train_data = list(np.transpose(list(standard_train_data[0])))
    standard_test_data = list(np.transpose(list(standard_test_data[0])))
    temp_list = []
    for row in standard_train_data:
        row = list(row)
        row.insert(0, 1)
        temp_list.append(row)
    standard_train_data = temp_list
    temp_list = []
    for row in standard_test_data:
        row = list(row)
        row.insert(0, 1)
        temp_list.append(row)
    standard_test_data = temp_list

    return {'standard_train_data': standard_train_data, 'standard_test_data': standard_test_data}


if __name__ == '__main__':
    final_data = generate()
    standard_train_data = final_data['standard_train_data']
    standard_test_data = final_data['standard_test_data']
