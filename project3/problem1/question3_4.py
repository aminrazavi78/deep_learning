import pandas as pd
import numpy as np


class Standardization:
    def __init__(self, x_input, x_train):
        self.standard = self._do_standard(x_input, x_train)

    def _expect(self, x_train):
        return np.sum(x_train)/len(x_train)

    def _var(self, x_train):
        return self._expect((x_train - self._expect(x_train))**2)

    def _do_standard(self, x_input, x_train):
        return (x_input - self._expect(x_train))/(self._var(x_train)**0.5)


# CONVERT STRING DATA TO INTEGER OR FLOAT   AND    FILL THE MISSED DATA

def conv(val):
    if val == 'male':
        return 1
    else:
        return 0


def data_editing():
    train_df = pd.read_csv("data/train.csv", converters={'Sex': conv})
    test_df = pd.read_csv("data/test.csv", converters={'Sex': conv})

    train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    avg = train_df['Age'].sum() / train_df['Age'].count()
    train_df['Age'] = train_df['Age'].fillna(avg)

    train = train_df
    test = test_df

    train['Pclass'] = Standardization(np.array(train['Pclass']), np.array(train['Pclass'])).standard
    train['Sex'] = Standardization(np.array(train['Sex']), np.array(train['Sex'])).standard
    train['Age'] = Standardization(np.array(train['Age']), np.array(train['Age'])).standard
    train['SibSp'] = Standardization(np.array(train['SibSp']), np.array(train['SibSp'])).standard
    train['Parch'] = Standardization(np.array(train['Parch']), np.array(train['Parch'])).standard
    train['Fare'] = Standardization(np.array(train['Fare']), np.array(train['Fare'])).standard

    test['Pclass'] = Standardization(np.array(test['Pclass']), np.array(train['Pclass'])).standard
    test['Sex'] = Standardization(np.array(test['Sex']), np.array(train['Sex'])).standard
    test['Age'] = Standardization(np.array(test['Age']), np.array(train['Age'])).standard
    test['SibSp'] = Standardization(np.array(test['SibSp']), np.array(train['SibSp'])).standard
    test['Parch'] = Standardization(np.array(test['Parch']), np.array(train['Parch'])).standard
    test['Fare'] = Standardization(np.array(test['Fare']), np.array(train['Fare'])).standard

    return train_df, test_df


if __name__ == '__main__':
    data = data_editing()
