from perceptron import *

# pre-set params
picSize = (16, 16)
pixelNum = 16 * 16
lastDig = 8
choice = 1
initw3 = np.array([0, 2, 1])
initw4 = np.array([3, 1, 2, 0])

#  import and pre-process data
wholeTrain = pd.read_csv('data/DigitsTraining.csv', header=None)
wholeTrain = np.asarray(wholeTrain)
wholeTest = pd.read_csv('data/DigitsTesting.csv', header=None)
wholeTest = np.asarray(wholeTest)
index_ch_tr = wholeTrain[:, 0] == choice
index_ld_tr = wholeTrain[:, 0] == lastDig
index_ch_te = wholeTest[:, 0] == choice
index_ld_te = wholeTest[:, 0] == lastDig
training_ = wholeTrain[np.bitwise_or(index_ch_tr, index_ld_tr), :]
testing_ = wholeTest[np.bitwise_or(index_ch_te, index_ld_te), :]

# initialize an instance of HandWritten class
hw1 = HandWritten(h=16, w=16, lstDgt=8, choice=1)

# feature extracting
training = hw1.reSize(training_, picSize)
testing = hw1.reSize(testing_, picSize)
x1_tr = np.asarray(hw1.avgSymmetry(training)).reshape((len(training), 1))
x2_tr = np.asarray(hw1.avgIntensity(training)).reshape((len(training), 1))
x1_te = np.asarray(hw1.avgSymmetry(testing)).reshape((len(testing), 1))
x2_te = np.asarray(hw1.avgIntensity(testing)).reshape((len(testing), 1))

# make final input matrix
x_train = np.concatenate((x1_tr, x2_tr), axis=1)
digit_train = training_[:, 0].astype(int)
x_test = np.concatenate((x1_te, x2_te), axis=1)
digit_test = testing_[:, 0].astype(int)
y_train = hw1.output2binary(digit_train)
y_test = hw1.output2binary(digit_test)

#
Eout_a = []
w_a, Ein_a = hw1.perpcetron(initw3, x_train, y_train, eta=0.013, maxIter=200)
for i in w_a:
    num_out, index_out = hw1.countFalse(i, x_test, y_test)
    Eout_a.append(num_out / x_test.shape[0])
hw1.plot2D(x_train, digit_train, w_a[-1], 'PLA')
hw1.compareError(Ein_a, Eout_a, ': PLA')

# pocket algorithm
Eout_b = []
w_b, Ein_b = hw1.pocket(initw3, x_train, y_train, eta=0.013, maxIter=200)
for i in w_b:
    num_out, index_out = hw1.countFalse(i, x_test, y_test)
    Eout_b.append(num_out / x_test.shape[0])
hw1.plot2D(x_train, digit_train, w_b[-1], 'pocket PLA')
hw1.compareError(Ein_b, Eout_b, ': pocket PLA')

# linear regression
Eout_c = []
w_c, Ein_c = hw1.linearReg(initw3, x_train, y_train, maxIter=200, tolerance=1e-3, eta=0.013)
for i in w_c:
    num_out, index_out = hw1.countFalse(i, x_test, y_test)
    Eout_c.append(num_out / x_test.shape[0])
hw1.plot2D(x_train, digit_train, w_c[-1], 'Linear Regression')
hw1.compareError(Ein_c, Eout_c, ': Linear Regression')

# linear reg + pocket
Eout_d = []
w_d, Ein_d = hw1.pocket(w_c[-1], x_train, y_train, eta=0.02, maxIter=200)
for i in w_d:
    num_out, index_out = hw1.countFalse(i, x_test, y_test)
    Eout_d.append(num_out / x_test.shape[0])
hw1.plot2D(x_train, digit_train, w_d[-1], 'pocket PLA (LR weights)')
hw1.compareError(Ein_d, Eout_d, ': pocket PLA (LR weights)')

# one more features
x3_tr = np.asarray(hw1.maxWidth(training)).reshape((len(training), 1))
x3_te = np.asarray(hw1.maxWidth(testing)).reshape((len(testing), 1))
x_Train = np.concatenate((x_train, x3_tr), axis=1)
x_Test = np.concatenate((x_test, x3_te), axis=1)
mean1 = np.mean(x3_tr[digit_train == 1])
mean8 = np.mean(x3_tr[digit_train == 8])

# adding another feature
Eout_f1 = []
w_f1, Ein_f1 = hw1.perpcetron(initw4, x_Train, y_train, eta=0.002, maxIter=200)
for i in w_f1:
    num_out, index_out = hw1.countFalse(i, x_Test, y_test)
    Eout_f1.append(num_out / x_test.shape[0])
hw1.compareError(Ein_f1, Eout_f1, ': PLA (adding a new feature)')

Eout_f2 = []
w_f2, Ein_f2 = hw1.pocket(initw4, x_Train, y_train, eta=0.002, maxIter=200)
for i in w_f2:
    num_out, index_out = hw1.countFalse(i, x_Test, y_test)
    Eout_f2.append(num_out / x_test.shape[0])
hw1.compareError(Ein_f2, Eout_f2, ': pocket PLA (adding a new feature)')

Eout_f3 = []
w_f3, Ein_f3 = hw1.linearReg(initw4, x_Train, y_train, maxIter=200, tolerance=1e-3, eta=0.001)
for i in w_f3:
    num_out, index_out = hw1.countFalse(i, x_Test, y_test)
    Eout_f3.append(num_out / x_test.shape[0])
hw1.compareError(Ein_f3, Eout_f3, ': Linear Regression (adding a new feature)')

Eout_f4 = []
w_f4, Ein_f4 = hw1.pocket(w_f3[-1], x_Train, y_train, eta=0.001, maxIter=200)
for i in w_f4:
    num_out, index_out = hw1.countFalse(i, x_Test, y_test)
    Eout_f4.append(num_out / x_test.shape[0])
hw1.compareError(Ein_f4, Eout_f4, ': pocket PLA (adding a new feature)')
