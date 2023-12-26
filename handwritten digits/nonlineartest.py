from nonlinear import *

# preset params
targetDigit = 1
initw1 = np.array([1e-1, 1e-1, 1e-1])
initw3 = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])

# data preprocessing
wholeTrain = pd.read_csv('data/DigitsTraining.csv', header=None)
wholeTrain = np.asarray(wholeTrain)
wholeTest = pd.read_csv('data/DigitsTesting.csv', header=None)
wholeTest = np.asarray(wholeTest)
xraw_train = wholeTrain[:, 1:]
yraw_train = wholeTrain[:, 0]
xraw_test = wholeTest[:, 1:]
yraw_test = wholeTest[:, 0]

# initialize an HWNonLinearTrans instance and prepare the data set
HW = HWNonLinearTrans(h=16, w=16, target=targetDigit)

# extract two features
xraw_train_resize = HW.reSize(xraw_train)
xraw_test_resize = HW.reSize(xraw_test)
ytrain = HW.output2binary(yraw_train)
ytest = HW.output2binary(yraw_test)

#
xtrain = HW.makeInput(xraw_train_resize, order=1)
xtest = HW.makeInput(xraw_test_resize, order=1)

# searching for the optimal learning rate
lrB = []
rate = np.linspace(0, 1, 100)
t0 = time.time()
for r in rate:
    w1, errorB = HW.linearReg(initw1, xtrain, ytrain, eta=r, maxIter=1000, tol=1e-3, BGD=True)
    lrB.append(errorB[-1])
print(time.time() - t0)

# plot
plt.plot(range(100), lrB, c='r')
plt.title('Optimal Learning Rate for BGD')
plt.xlabel('Learning Rate')
plt.ylabel('Error')
plt.show()

b = np.argmin(np.array(lrB))
print(np.linspace(0, 1, 100)[b])

# batch GD
Eout = []
w, Ein = HW.linearReg(initw1, xtrain, ytrain, eta=0.9797, maxIter=1000, tol=1e-3, BGD=True)
for i in w:
    num_out, index_out = HW.countFalse(i, xtest, ytest)
    Eout.append(num_out / xtest.shape[0])
HW.plot2D(xtrain, ytrain, w[-1], 'Linear Regression: Training Data (BGD)', ['r', 'b', 'k'], order=1)
HW.plot2D(xtest, ytest, w[-1], 'Linear Regression: Testing Data (BGD)', ['r', 'b', 'k'], order=1)
HW.compareError(Ein[200:], Eout[200:], ': Linear Regression (BGD)', ['r', 'b'])

# third order polynomial transformation
xtrain = HW.makeInput(xraw_train_resize, order=3)
xtest = HW.makeInput(xraw_test_resize, order=3)

# searching for the optimal learning rate
lrB = []
rate = np.linspace(0, 1, 100)
t0 = time.time()
for r in rate:
    w1, errorB = HW.linearReg(initw3, xtrain, ytrain, eta=r, maxIter=1000, tol=1e-3, BGD=True)
    lrB.append(errorB[-1])
print(time.time() - t0)

# plot
plt.plot(range(100), lrB, c='r')
plt.title('Optimal Learning Rate for BGD')
plt.xlabel('Learning Rate')
plt.ylabel('Error')
plt.show()

b = np.argmin(np.array(lrB))
print(np.linspace(0, 1, 100)[b])

# batch GD
Eout = []
w, Ein = HW.linearReg(initw3, xtrain, ytrain, eta=0.9192, maxIter=8000, tol=1e-3, BGD=True)
for i in w:
    num_out, index_out = HW.countFalse(i, xtest, ytest)
    Eout.append(num_out / xtest.shape[0])
HW.plot2D(xtrain, ytrain, w[-1], 'Linear Regression: Training Data (BGD)', ['r', 'b', 'k'], order=3)
HW.plot2D(xtest, ytest, w[-1], 'Linear Regression: Testing Data (BGD)', ['r', 'b', 'k'], order=3)
HW.compareError(Ein[5000:], Eout[5000:], ': Linear Regression (BGD)', ['r', 'b'])
