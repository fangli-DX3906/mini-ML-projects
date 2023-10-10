import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt


# logistic regression classifier


# import data and data pre-processing
week = pd.read_csv('../../weekly.csv')
features = week.columns
week = np.array(week)
xtrainlr = week[:, 2:9].astype('float64')
ytrainlr = pd.Categorical(week[:, -1]).codes
ytrainlr.flags.writeable = True
ytrainlr[ytrainlr == 0] = -1

# correlation matrix
dataP = xtrainlr.transpose()
R = []
for i in range(dataP.shape[0]):
    R.append([])
    std_i = np.sqrt(np.sum((dataP[i] - np.mean(dataP[i])) *
                           (dataP[i] - np.mean(dataP[i]))) /
                    (dataP.shape[1] - 1))
    for j in range(dataP.shape[0]):
        cov = np.sum((dataP[i] - np.mean(dataP[i])) *
                     (dataP[j] - np.mean(dataP[j]))) / \
              (dataP.shape[1] - 1)
        std_j = np.sqrt(np.sum((dataP[j] - np.mean(dataP[j])) *
                               (dataP[j] - np.mean(dataP[j]))) /
                        (dataP.shape[1] - 1))
        R[i].append(cov / (std_i * std_j))
R = np.asarray(R)
# plot
plt.plot(range(xtrainlr.shape[0]), xtrainlr[:, -2], c='b', linewidth=0.5)
plt.title('Time Series of Volume')
plt.xlabel('time')
plt.ylabel('volume')
plt.show()


# define some usful functions
sigmoid = lambda x: np.exp(x) / (1 + np.exp(x))


def grad(w, x, y):
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    y = y.reshape((-1, 1))
    nu = y * x
    temp = y * np.dot(x, w.reshape((-1, 1)))
    de = 1 + np.exp(temp)
    return -np.sum(nu / de, axis=0) / x.shape[0]


def countFalse(w, x, y):
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    w = np.array(w).reshape((-1, 1))
    out = np.dot(x, w).reshape((x.shape[0],))
    temp = np.ones(x.shape[0])
    temp[np.where(out < 0)] = -1
    index = np.arange(0, x.shape[0])
    N = len(index[temp != y])
    falseIndex = index[temp != y].tolist()
    return N, falseIndex


def logitReg(w, x, y, maxIter, tol, eta) -> tuple:
    i = 0
    diff = 1
    weight = [w]
    e = []
    while i < maxIter and diff > tol:
        error, index = countFalse(weight[i], x, y)
        e.append(error / x.shape[0])
        gradient = grad(weight[i], x, y)
        weight_temp = weight[i] - eta * gradient
        weight.append(weight_temp)
        diff = np.linalg.norm(weight[i + 1] - weight[i], ord=2)
        i += 1
    error_, index_ = countFalse(weight[-1], x, y)
    e.append(error_ / x.shape[0])
    return weight, e


# preset params
initb = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
inite = np.array([1e-1, 1e-1])

# serching for the optimal learning rate
lrB = []
rate = np.linspace(0, 1, 100)
for r in rate:
    w, errorB = logitReg(initb, xtrainlr, ytrainlr, eta=r, maxIter=2000, tol=1e-4)
    lrB.append(errorB[-1])
plt.plot(range(100), lrB, c='r')
plt.title('Optimal Learning Rate for BGD')
plt.xlabel('Learning Rate')
plt.ylabel('Error')
plt.show()
b = np.argmin(np.array(lrB))
print(np.linspace(0, 1, 100)[b])  # 0.8484

# logit regression
w, Ein = logitReg(initb, xtrainlr, ytrainlr, eta=0.01, maxIter=2000, tol=1e-4)   # 0.8484
print(w[-1])
print(1 - Ein[-1])
plt.plot(range(len(Ein[1000:])), Ein[1000:], c='r')
plt.show()

# 
data_2c = np.concatenate((np.ones((10, 1)), xtrainlr[:10, :]), axis=1)
yHat = np.dot(data_2c, w[-1].reshape((-1, 1))).ravel()
prob = sigmoid(yHat)

# 
data_2d = np.concatenate((np.ones((xtrainlr.shape[0], 1)), xtrainlr), axis=1)
yHat = np.dot(data_2d, w[-1].reshape((-1, 1)))
prob = sigmoid(yHat)

predict_2d = []
for i in prob:
    clf = 1 if i >= 0.5 else -1
    predict_2d.append(clf)

ytrainlr = ytrainlr.ravel()
predict_2d = np.asarray(predict_2d)
print(1 - np.sum(ytrainlr != predict_2d) / predict_2d.shape[0])
# or
print(1 - Ein[-1])

# 
index = np.where(week[:, 1] == 2008)
xxtrainlr = week[:index[0][-1] + 1, 3].astype('float64')
xxtrainlr = xxtrainlr.reshape((-1, 1))
xxtestlr = week[index[0][-1] + 1:, 3].astype('float64')
xxtestlr = xxtestlr.reshape((-1, 1))
yytrainlr = ytrainlr[:index[0][-1] + 1]
yytestlr = ytrainlr[index[0][-1] + 1:]

# serching for the optimal learning rate
lrB = []
rate = np.linspace(0, 1, 100)
for r in rate:
    w, errorB = logitReg(inite, xxtrainlr, yytrainlr, eta=r, maxIter=2000, tol=1e-4)
    lrB.append(errorB[-1])
plt.plot(range(100), lrB, c='r')
plt.title('Optimal Learning Rate for BGD')
plt.xlabel('Learning Rate')
plt.ylabel('Error')
plt.show()
b = np.argmin(np.array(lrB))
print(np.linspace(0, 1, 100)[b])  # 0.101

# logit regression
ww, EEin = logitReg(np.array([1e-1, 1e-1]), xxtrainlr, yytrainlr, eta=0.101, maxIter=2000, tol=1e-6)

data_2e = np.concatenate((np.ones((xxtestlr.shape[0], 1)), xxtestlr), axis=1)
yHat = np.dot(data_2e, ww[-1].reshape((-1, 1)))
prob = sigmoid(yHat)

predict_2e = []
for i in prob:
    clf = 1 if i >= 0.5 else -1
    predict_2e.append(clf)

yytestlr = yytestlr.ravel()
predict_2e = np.asarray(predict_2e)
print(1 - np.sum(yytestlr != predict_2e) / predict_2e.shape[0])
