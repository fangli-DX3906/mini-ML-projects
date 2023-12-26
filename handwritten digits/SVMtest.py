from SVM import *

# prepare the raw data
wholeTrain = pd.read_csv('data/DigitsTraining.csv', header=None)
wholeTrain = np.asarray(wholeTrain)
wholeTest = pd.read_csv('data/DigitsTesting.csv', header=None)
wholeTest = np.asarray(wholeTest)
xraw_train = wholeTrain[:, 1:]
yraw_train = wholeTrain[:, 0]
xraw_test = wholeTest[:, 1:]
yraw_test = wholeTest[:, 0]

# one vs one case
targetDigit = 8
otherDigit = 1
pickTtr = yraw_train == targetDigit
pickOtr = yraw_train == otherDigit
pickTte = yraw_test == targetDigit
pickOte = yraw_test == otherDigit
xtrain1v1 = xraw_train[np.bitwise_or(pickTtr, pickOtr), :]
xtest1v1 = xraw_test[np.bitwise_or(pickTte, pickOte), :]
ytrain1v1 = yraw_train[np.bitwise_or(pickTtr, pickOtr)]
ytest1v1 = yraw_test[np.bitwise_or(pickTte, pickOte)]

# initialize a HandWrittenSVM instance and prepare the data set
HW1v1 = HandWrittenSVM(h=16, w=16, target=targetDigit)

xtrain1v1_resize = HW1v1.reSize(xtrain1v1)
xtest1v1_resize = HW1v1.reSize(xtest1v1)
Xtr1v1 = HW1v1.makeInput(xtrain1v1_resize, order=1)
Xte1v1 = HW1v1.makeInput(xtest1v1_resize, order=1)
Ytr1v1 = HW1v1.output2binary(ytrain1v1)
Yte1v1 = HW1v1.output2binary(ytest1v1)

e = []
t0 = time.time()
for c in range(1, 100, 1):
    sss = svm.SVC(C=c, kernel='linear')
    sss.fit(Xtr1v1, Ytr1v1)
    e.append(accuracy_score(Ytr1v1, sss.predict(Xtr1v1)))
Copt1v1 = range(1, 100, 1)[np.argmax(e)]
print(time.time() - t0)

# find out the optimal C
E1v1 = []
for c in range(1, 100, 1):
    sol1v1 = HW1v1.SVMSolver(C=c, X=Xtr1v1, Y=Ytr1v1)
    E1v1.append(HW1v1.countFalse(sol1v1[0], sol1v1[1], Xtr1v1, Ytr1v1))
plt.plot(range(1, 100, 1), E1v1)
plt.show()
Copt1v1 = np.argmin(E1v1)

sol1v1 = HW1v1.SVMSolver(C=Copt1v1, X=Xtr1v1, Y=Ytr1v1)
Ein1v1 = HW1v1.countFalse(sol1v1[0], sol1v1[1], Xtr1v1, Ytr1v1) / Xtr1v1.shape[0]
Eout1v1 = HW1v1.countFalse(sol1v1[0], sol1v1[1], Xte1v1, Yte1v1) / Xte1v1.shape[0]
HW1v1.plot2D(Xtr1v1, Ytr1v1, sol1v1[0], sol1v1[1], 'one-vs-one SVM classifier: training', ['r', 'b'])
HW1v1.plot2D(Xte1v1, Yte1v1, sol1v1[0], sol1v1[1], 'one-vs-one SVM classifier: testing', ['r', 'b'])

# one vs all case
targetDigit = 1

# # initialize a HandWrittenSVM instance and prepare the data set
HW1va = HandWrittenSVM(h=16, w=16, target=targetDigit)

xtrain1va_resize = HW1va.reSize(xraw_train)
xtest1va_resize = HW1va.reSize(xraw_test)
Xtr1va = HW1va.makeInput(xtrain1va_resize, order=1)
Xte1va = HW1va.makeInput(xtest1va_resize, order=1)
Ytr1va = HW1va.output2binary(yraw_train)
Yte1va = HW1va.output2binary(yraw_test)

# find out the optimal C
E1va = []
for c in range(1, 50, 1):
    sol1va = HW1va.SVMSolver(C=c, X=Xtr1va, Y=Ytr1va)
    E1va.append(HW1va.countFalse(sol1va[0], sol1va[1], Xtr1va, Ytr1va))
plt.plot(range(1, 50, 1), E1va)
plt.show()
Copt1va = np.argmin(E1va)

t0 = time.time()
sol1va = HW1va.SVMSolver(C=3, X=Xtr1va, Y=Ytr1va, show_progress=True)
print(time.time() - t0)
Ein1va = HW1va.countFalse(sol1va[0], sol1va[1], Xtr1va, Ytr1va) / Xtr1va.shape[0]
Eout1va = HW1va.countFalse(sol1va[0], sol1va[1], Xte1va, Yte1va) / Xte1va.shape[0]
HW1va.plot2D(Xtr1va, Ytr1va, sol1va[0], sol1va[1], 'one-vs-all SVM classifier: training', ['r', 'b'])
HW1va.plot2D(Xte1va, Yte1va, sol1va[0], sol1va[1], 'one-vs-all SVM classifier: testing', ['r', 'b'])
