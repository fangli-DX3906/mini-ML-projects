from neural_network import *

wholeTest = np.array(pd.read_csv('data/DigitsTesting.csv', header=None))
xraw_test = wholeTest[:, 1:]
yraw_test = wholeTest[:, 0]

# training samples
xraw_train = np.loadtxt('data/xraw_train.csv', delimiter=',', skiprows=0)
yraw_train = np.loadtxt('data/yraw_train.csv', delimiter=',', skiprows=0)

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
