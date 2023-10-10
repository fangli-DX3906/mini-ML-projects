import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import random
import time


# import data
digits = scipy.io.loadmat('../../DatasetDigit.mat')
weights = scipy.io.loadmat('../../Weights.mat')
X = digits['X']
y = digits['y']
w1 = weights['W1']
w2 = weights['W2']
n = X.shape[0]

# define activation function
theta = lambda s: 1 / (1 + np.exp(-s))

# calculation
bias = np.ones((1, n))
x0 = np.concatenate((bias, X.T), axis=0)
s1 = np.dot(w1.T, x0)
theta1 = theta(s1)

x1 = np.concatenate((bias, theta1), axis=0)
s2 = np.dot(w2.T, x1)
output = theta(s2)

result = np.argmax(output, axis=0) + 1
accuracy = np.sum(y.flatten() == result) / n
print(accuracy)


class BackwardPropagation:
    def __init__(self, Ws, bs, L):
        self.biases = bs
        # transposed weights for each layers
        self.weights = Ws
        self.layers = L

    def tanh(self, s):
        return np.tanh(s)

    def tanhP(self, s):
        return 1 - np.tanh(s) ** 2

    def costP(self, xhat, y):
        return 2 * (xhat - y)

    def backwardProg(self, x, y):
        biasP = [np.zeros(bias.shape) for bias in self.biases]
        weightP = [np.zeros(weight.shape) for weight in self.weights]
        xVec = x
        xVecs = [x]
        sVecs = []
        delta = []
        for w, b in zip(self.weights, self.biases):
            xVec_temp = np.dot(w, xVec) + b
            sVecs.append(xVec_temp)
            xVec = self.tanh(xVec_temp)
            xVecs.append(xVec)
        # backward propagating
        delta.append(self.costP(xVecs[-1], y) * self.tanhP(sVecs[-1]))
        biasP[-1] = delta[0]
        weightP[-1] = np.dot(delta[0], xVecs[-2].transpose())
        for l in range(2, self.layers):
            sVec_temp = sVecs[-l]
            deriv = self.tanhP(sVec_temp)
            delta.append(np.dot(self.weights[-l + 1].transpose(), delta[l - 2]) * deriv)
            biasP[-l] = delta[-1]
            weightP[-l] = np.dot(delta[-1], xVecs[-l - 1].transpose())

        return weightP, biasP, delta, xVecs, sVecs


# calculate the weights
W = [np.array([[0.3], [0.4]]), np.array([1, -3]).reshape((1, -1)), np.array([[2]])]
b = [np.array([[0.1], [0.2]]), np.array([[0.2]]), np.array([[1]])]
bp = BackwardPropagation(W, b, 4)
wp, bp, delta, xs, ss = bp.backwardProg(np.array([[2]]), np.array([[1]]))



class NeutralNetwork:
    def __init__(self, dVec):
        self.L = len(dVec)
        self.dVec = dVec
        self.biases = [np.random.randn(d, 1) for d in self.dVec[1:]]
        self.weights = [np.random.randn(d, d_) for d_, d in zip(self.dVec[:-1], self.dVec[1:])]

    def tanh(self, s):
        return np.tanh(s)

    def tanhP(self, s):
        return 1 - (np.tanh(s)) ** 2

    def costP(self, xhat, y):
        return 2 * (xhat - y)

    def finalAcFunc(self, s):
        return s

    def finalAcFuncP(self):
        return 1.0

    def forwardProp(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.tanh(np.dot(x, w.T) + b.flatten())
        return x.flatten()

    def backwardProg(self, x, y):
        biasP = [np.zeros(bias.shape) for bias in self.biases]
        weightP = [np.zeros(weight.shape) for weight in self.weights]
        xVec = x
        xVecs = [x]
        sVecs = []
        delta = []
        ind = 0
        for w, b in zip(self.weights, self.biases):
            sVec_temp = np.dot(w, xVec).reshape((-1, 1)) + b
            sVecs.append(sVec_temp)
            if ind != self.L - 2:
                xVec = self.tanh(sVec_temp)
            else:
                xVec = self.finalAcFunc(sVec_temp)
            ind += 1
            xVecs.append(xVec)
        delta.append(self.costP(xVecs[-1], y) * self.finalAcFuncP())
        biasP[-1] = delta[0]
        weightP[-1] = np.dot(delta[0], xVecs[-2].transpose())
        for l in range(2, self.L):
            sVec_temp = sVecs[-l]
            deriv = self.tanhP(sVec_temp)
            delta.append(np.dot(self.weights[-l + 1].transpose(), delta[l - 2]) * deriv)
            biasP[-l] = delta[-1]
            weightP[-l] = np.dot(delta[-1], xVecs[-l - 1].reshape((1, -1)))
        return weightP, biasP, delta, xVecs, sVecs

    def batchUpdate(self, trainX, trainY):
        N = trainX.shape[0]
        biasP = [np.zeros(b.shape) for b in self.biases]
        weightP = [np.zeros(w.shape) for w in self.weights]
        for n in range(N):
            wPtemp, bPtemp, deltatemp, xVecstemp, sVecstemp = self.backwardProg(trainX[n], trainY[n])
            biasP = [bP + bPdelta / N for bP, bPdelta in zip(biasP, bPtemp)]
            weightP = [wP + wPdelta / N for wP, wPdelta in zip(weightP, wPtemp)]
        return weightP, biasP

    def stochasticUpdate(self, trainX, trainY):
        N = trainX.shape[0]
        candidate = [n for n in range(N)]
        random.shuffle(candidate)
        pick = candidate[0]
        wPtemp, bPtemp, deltatemp, xVecstemp, sVecstemp = self.backwardProg(trainX[pick], trainY[pick])
        weightP = wPtemp
        biasP = bPtemp
        return weightP, biasP

    def gradientDescent(self, trainX, trainY, testX, testY, maxIter, eta, SGD=False):
        Ein = []
        Eout = []
        for iter in range(maxIter):
            if SGD:
                weightP, biasP = self.stochasticUpdate(trainX, trainY)
            else:
                weightP, biasP = self.batchUpdate(trainX, trainY)
            self.biases = [b - eta * bP for b, bP in zip(self.biases, biasP)]
            self.weights = [w - eta * wP for w, wP in zip(self.weights, weightP)]
            print('This is the {}th iteration'.format(iter))
            Ein.append(np.sum(np.sign(self.forwardProp(trainX)) != trainY) / trainX.shape[0])
            Eout.append(np.sum(np.sign(self.forwardProp(testX)) != testY) / testX.shape[0])
            # Ein.append(ein)
            # Eout.append(np.sum((np.sign(self.forwardProp(testX)) - testY) ** 2) / testX.shape[0])
        return self.weights, self.biases, Ein, Eout


class HWDataPreProceesing:
    def __init__(self, h, w, target):
        self.height = h
        self.width = w
        self.pixelNum = h * w
        self.picSize = (self.width, self.height)
        self.target = target

    def avgSymmetry(self, data) -> np.ndarray:
        avgsyty = []
        for i in range(len(data)):
            vSym = np.sum(np.abs(data[i] - data[i][::-1, :])) / self.pixelNum
            hSym = np.sum(np.abs(data[i] - data[i][:, ::-1])) / self.pixelNum
            avgsyty.append(-0.5 * (vSym + hSym))
        return np.asarray(avgsyty)

    def avgIntensity(self, data) -> np.ndarray:
        avgitsy = []
        for i in range(len(data)):
            avgitsy.append(-np.sum(data[i]) / self.pixelNum)
        return np.asarray(avgitsy)

    def reSize(self, data) -> list:
        picResize = []
        for i in range(data.shape[0]):
            picResize.append(data[i, :].reshape(self.picSize))
        return picResize

    def makeInput(self, data, order) -> np.ndarray:
        intensity = self.avgIntensity(data).reshape((len(data), 1))
        symmetry = self.avgSymmetry(data).reshape((len(data), 1))
        outputdata = np.concatenate((intensity, symmetry), axis=1)
        if order >= 2:
            for r in range(2, order + 1):
                for i in range(0, r + 1):
                    temp = intensity ** (r - i) * symmetry ** i
                    outputdata = np.concatenate((outputdata, temp), axis=1)
        return outputdata

    def output2binary(self, numlist) -> np.ndarray:
        output = []
        for i in range(numlist.shape[0]):
            output.append([1. if numlist[i] == self.target else -1.])
        return np.asarray(output).flatten()


# testing samples
wholeTest = np.array(pd.read_csv('../../DigitsTesting.csv', header=None))
xraw_test = wholeTest[:, 1:]
yraw_test = wholeTest[:, 0]

# training samples
xraw_train = np.loadtxt('xraw_train.csv', delimiter=',', skiprows=0)
yraw_train = np.loadtxt('yraw_train.csv', delimiter=',', skiprows=0)

# initialize a HW for data pre-processing
HW = HWDataPreProceesing(h=16, w=16, target=1)

# extracting features
xraw_train = HW.reSize(xraw_train)
xraw_test = HW.reSize(xraw_test)
xtrain = HW.makeInput(xraw_train, order=1)
xtest = HW.makeInput(xraw_test, order=1)
ytrain = HW.output2binary(yraw_train)
ytest = HW.output2binary(yraw_test)

# initialize a NN
nn = NeutralNetwork([2, 10, 1])


# --------------> training NN using BGD <-------------- #
t0 = time.time()
w, b, ein, eout = nn.gradientDescent(xtrain, ytrain, xtest, ytest, maxIter=10000, eta=0.01, SGD=False)
print(time.time() - t0)

# error plot
plt.plot(range(10000), ein, c='r', linewidth=0.5)
plt.plot(range(10000), eout, c='b', linewidth=0.5)
plt.xlabel('iterations')
plt.ylabel('error')
plt.title('Comparison of Ein and Eout: Batch GD')
plt.legend(['Ein', 'Eout'])
plt.show()

# final weights and bias
print(w)
print(b)

# comparisons
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
cat1 = ytrain == 1
catrest = ytrain == -1
T = np.sign(nn.forwardProp(xtrain)) == ytrain
F = np.sign(nn.forwardProp(xtrain)) != ytrain
plt.scatter(xtrain[np.bitwise_and(cat1, T), 0], xtrain[np.bitwise_and(cat1, T), 1], s=10, c='b', marker='o')
plt.scatter(xtrain[np.bitwise_and(catrest, T), 0], xtrain[np.bitwise_and(catrest, T), 1], s=15, c='b', marker='+')
plt.scatter(xtrain[np.bitwise_and(cat1, F), 0], xtrain[np.bitwise_and(cat1, F), 1], s=10, c='r', marker='o')
plt.scatter(xtrain[np.bitwise_and(catrest, F), 0], xtrain[np.bitwise_and(catrest, F), 1], s=15, c='r', marker='+')
plt.xlim([-1, -0.3])
plt.ylim([-0.5, 0])
plt.legend(['digit 1, correctly classified', 'rest digits, correctly classified', 'digit 1, falsely classified',
            'rest digits, falsely classified'])
plt.xlabel('Average Intensity')
plt.ylabel('Average Symmetry')
plt.title('Training samples: BGD')

plt.subplot(1, 2, 2)
cat1_ = ytest == 1
catrest_ = ytest == -1
T_ = np.sign(nn.forwardProp(xtest)) == ytest
F_ = np.sign(nn.forwardProp(xtest)) != ytest
plt.scatter(xtest[np.bitwise_and(cat1_, T_), 0], xtest[np.bitwise_and(cat1_, T_), 1], s=10, c='b', marker='o')
plt.scatter(xtest[np.bitwise_and(catrest_, T_), 0], xtest[np.bitwise_and(catrest_, T_), 1], s=15, c='b', marker='+')
plt.scatter(xtest[np.bitwise_and(cat1_, F_), 0], xtest[np.bitwise_and(cat1_, F_), 1], s=10, c='r', marker='o')
plt.scatter(xtest[np.bitwise_and(catrest_, F_), 0], xtest[np.bitwise_and(catrest_, F_), 1], s=15, c='r', marker='+')
plt.xlim([-1, -0.3])
plt.ylim([-0.5, 0])
plt.legend(['digit 1, correctly classified', 'rest digits, correctly classified', 'digit 1, falsely classified',
            'rest digits, falsely classified'])
plt.xlabel('Average Intensity')
plt.ylabel('Average Symmetry')
plt.title('Testing samples: BGD')
plt.show()

# --------------> training NN using SDG <-------------- #
t0S = time.time()
wS, bS, einS, eoutS = nn.gradientDescent(xtrain, ytrain, xtest, ytest, maxIter=20000, eta=0.01, SGD=True)
print(time.time() - t0S)

# error plot
plt.plot(range(20000), einS, c='r', linewidth=0.3)
plt.plot(range(20000), eoutS, c='b', linewidth=0.3)
plt.xlabel('iterations')
plt.ylabel('error')
plt.title('Comparison of Ein and Eout: Stochastic GD')
plt.legend(['Ein', 'Eout'])
plt.show()

# final weights and bias
print(wS)
print(bS)

# comparisons
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
cat1 = ytrain == 1
catrest = ytrain == -1
T = np.sign(nn.forwardProp(xtrain)) == ytrain
F = np.sign(nn.forwardProp(xtrain)) != ytrain
plt.scatter(xtrain[np.bitwise_and(cat1, T), 0], xtrain[np.bitwise_and(cat1, T), 1], s=10, c='b', marker='o')
plt.scatter(xtrain[np.bitwise_and(catrest, T), 0], xtrain[np.bitwise_and(catrest, T), 1], s=15, c='b', marker='+')
plt.scatter(xtrain[np.bitwise_and(cat1, F), 0], xtrain[np.bitwise_and(cat1, F), 1], s=10, c='r', marker='o')
plt.scatter(xtrain[np.bitwise_and(catrest, F), 0], xtrain[np.bitwise_and(catrest, F), 1], s=15, c='r', marker='+')
plt.xlim([-1, -0.3])
plt.ylim([-0.5, 0])
plt.legend(['digit 1, correctly classified', 'rest digits, correctly classified', 'digit 1, falsely classified',
            'rest digits, falsely classified'])
plt.xlabel('Average Intensity')
plt.ylabel('Average Symmetry')
plt.title('Training samples: SGD')

plt.subplot(1, 2, 2)
cat1_ = ytest == 1
catrest_ = ytest == -1
T_ = np.sign(nn.forwardProp(xtest)) == ytest
F_ = np.sign(nn.forwardProp(xtest)) != ytest
plt.scatter(xtest[np.bitwise_and(cat1_, T_), 0], xtest[np.bitwise_and(cat1_, T_), 1], s=10, c='b', marker='o')
plt.scatter(xtest[np.bitwise_and(catrest_, T_), 0], xtest[np.bitwise_and(catrest_, T_), 1], s=15, c='b', marker='+')
plt.scatter(xtest[np.bitwise_and(cat1_, F_), 0], xtest[np.bitwise_and(cat1_, F_), 1], s=10, c='r', marker='o')
plt.scatter(xtest[np.bitwise_and(catrest_, F_), 0], xtest[np.bitwise_and(catrest_, F_), 1], s=15, c='r', marker='+')
plt.xlim([-1, -0.3])
plt.ylim([-0.5, 0])
plt.legend(['digit 1, correctly classified', 'rest digits, correctly classified', 'digit 1, falsely classified',
            'rest digits, falsely classified'])
plt.xlabel('Average Intensity')
plt.ylabel('Average Symmetry')
plt.title('Testing samples: SGD')
plt.show()
