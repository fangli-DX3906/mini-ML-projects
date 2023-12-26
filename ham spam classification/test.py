from classification import *

datasets = ['dataset 1', 'dataset 2', 'dataset 3']

tc = NBTextClassifier()
lt = DiscriminativeTextClassifier(classifier='Logistic')
pt = DiscriminativeTextClassifier(classifier='Perceptron')

percision_lt = dict()
percision_tc = dict()
percision_pt = dict()

print('*********************************************')
print('****** training Naive Bayes classifier ******')
print('*********************************************')
time.sleep(2)
for data in datasets:
    tc.fit(data, verbose=True)
    percision_tc[data] = tc.accuracy(data, isTraining=False)

print('*****************************************************')
print('****** training logistic regression classifier ******')
print('*****************************************************')
for data in datasets:
    percision_lt[data] = dict()
    trainx, trainy = lt.makeInput(data, isTraining=True)
    testx, testy = lt.makeInput(data, isTraining=False)

    paramList = np.linspace(0, 5, 6).astype('int')
    print('Searching for the hyper-param (panelty) for {} ...'.format(data))
    reg_best = lt.validation(trainx, trainy, 0.7, paramList=paramList, verbose=True)
    percision_lt[data]['reg'] = reg_best

    print('Training after validation for {} ...'.format(data))
    w_best, _ = lt.fitLogistic(trainx, trainy, reg=reg_best, verbose=False)
    percision_lt[data]['accuracy'] = lt.accuracy(w_best, testx, testy)

print('********************************************')
print('****** training perceptron classifier ******')
print('********************************************')
for data in datasets:
    percision_pt[data] = dict()
    trainx, trainy = pt.makeInput(data, isTraining=True)
    testx, testy = pt.makeInput(data, isTraining=False)

    paramList = np.linspace(1000, 2000, 6).astype('int')
    print('Searching for the hyper-param (iteration) for {} ...'.format(data))
    iter_best = pt.validation(trainx, trainy, 0.7, paramList=paramList, verbose=True)
    percision_pt[data]['iteration'] = iter_best

    print('Training after validation for {} ...'.format(data))
    w_best, _ = pt.fitPerceptron(trainx, trainy, maxIter=iter_best, verbose=False)
    percision_pt[data]['accuracy'] = pt.accuracy(w_best, testx, testy)

print('*********************************')
print('****** Now report accuracy ******')
print('*********************************')
print('NB accuracy:')
print(percision_tc)
print('Logistic accuracy:')
print(percision_lt)
print('Perceptron accuracy:')
print(percision_pt)
