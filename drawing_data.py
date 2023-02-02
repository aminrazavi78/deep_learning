import matplotlib.pyplot as plt
from numpy import random, sign, linspace
import csv

#####    Train    #####


X1 = []
X2 = []
Label = []

csv_file = open('data/train.csv', 'r')
csvreader = csv.reader(csv_file)
for row in csvreader:
    if row[0] == 'x1':
        continue
    X1.append(float(row[0]))
    X2.append(float(row[1]))
    if float(row[2]) == 1:
        Label.append(1)
    else:
        Label.append(-1)

colors = []
for i in Label:
    if i == 1:
        colors.append('red')
    else:
        colors.append('blue')

alpha = 0.1
W = random.rand(3)  # w0 = W[0], w1 = W[1], w2 = W[2]
iteration_num = 0
while iteration_num < 100:
    for i in range(len(X1)):
        iteration_num += 1
        s = sign(W[0] + W[1] * X1[i] + W[2] * X2[i])
        if s * Label[i] != 1:
            W[0] += alpha * Label[i]
            W[1] += alpha * X1[i] * Label[i]
            W[2] += alpha * X2[i] * Label[i]

ax = plt.gca()
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])

plt.scatter(X1, X2, color=colors)
plt.xlabel('x1 - axis')
plt.ylabel('x2 - axis')

x = linspace(-2, 2, 100)
y = (W[1] * x) / W[2] + W[0] / W[2]
plt.plot(x, y, '-r', color='green')

plt.show()

######### Test ##########

X1 = []
X2 = []
Label = []

csv_file = open('data/test.csv', 'r')
csvreader = csv.reader(csv_file)
for row in csvreader:
    if row[0] == 'x1':
        continue
    X1.append(float(row[0]))
    X2.append(float(row[1]))
    if float(row[2]) == 1:
        Label.append(1)
    else:
        Label.append(-1)

colors = []
for i in Label:
    if i == 1:
        colors.append('red')
    else:
        colors.append('blue')

ax = plt.gca()
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])

plt.scatter(X1, X2, color=colors)
plt.xlabel('x1 - axis')
plt.ylabel('x2 - axis')

x = linspace(-2, 2, 100)
y = (W[1] * x) / W[2] + W[0] / W[2]
plt.plot(x, y, '-r', color='green')

plt.show()

if __name__ == '__main__':
    print(len(X1))
    print(len(X2))
