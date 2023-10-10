import numpy as np
import os
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import time


# using Naive Bayes, Perceptron, and Logistic regression for text classification (ham/spam)


class ParseTextData:
    def __init__(self):
        pass

    def getKeyList(self, inputDict: dict) -> list:
        return list(inputDict.keys())

    def getValueList(self, inputDict: dict) -> list:
        return list(inputDict.values())

    def getItemList(self, inputDict: dict) -> list:
        return list(inputDict.items())

    def getDictMaxKeyValue(self, inputDict: dict) -> tuple:
        sortedDict = sorted(self.getItemList(inputDict), key=lambda k: k[1], reverse=True)
        return sortedDict[0][0], sortedDict[0][1]

    def makeWordString(self, filePath: str) -> str:
        with open(filePath, encoding='utf-8') as f:
            content = f.readlines()
        docWordString = ''.join(content).lower()
        return docWordString

    def str2list(self, docWordString: str) -> list:
        return list(filter(str.isalpha, re.split('\W+', docWordString)))

    def makeDirPath(self, dataSetName: str, isTraining) -> str:
        dirPath = 'hw 2 datasets/' + dataSetName + '/train' if isTraining else 'hw 2 datasets/' + dataSetName + '/test'
        return dirPath

    def getDirItemList(self, dirPath: str) -> list:
        itemList = os.listdir(dirPath)
        if '.DS_Store' in itemList:
            itemList.remove('.DS_Store')
        return itemList

    def makeWordsPool(self, dataSetName: str, verbose) -> tuple:
        clsCountDict = dict()
        clsWordStringDict = dict()
        dirPath = self.makeDirPath(dataSetName, isTraining=True)  # training
        cls_li = self.getDirItemList(dirPath)
        for cls in cls_li:
            clsDirPath = dirPath + '/' + cls
            doc_li = self.getDirItemList(clsDirPath)
            clsCountDict[cls] = clsCountDict.get(cls, 0) + len(doc_li)
            for doc in doc_li:
                position = clsDirPath + '/' + doc
                try:
                    clsWordStringDict[cls] = clsWordStringDict.get(cls, '') + '\n' + self.makeWordString(position)
                    if verbose:
                        print('[{}] has been successfully added to the training set!'.format(doc))
                except UnicodeDecodeError:
                    clsCountDict[cls] = clsCountDict.get(cls, 0) - 1
                    if verbose:
                        print('[{}] cannot be opened!'.format(doc))
        return clsCountDict, clsWordStringDict

    def countTokensOfDoc(self, docWordString: str) -> dict:
        return dict(Counter(self.str2list(docWordString)))

    def calcWordsFreq(self, clsWordStringDict: dict) -> dict:
        wordString = str()
        wordDict = dict()
        cls_li = self.getKeyList(clsWordStringDict)
        for cls in cls_li:
            wordString += clsWordStringDict[cls]
            wordDict[cls] = self.countTokensOfDoc(clsWordStringDict[cls])
            wordDict['all'] = self.countTokensOfDoc(wordString)
        return wordDict


class NBTextClassifier(ParseTextData):
    def calcClassPrior(self, clsCountDict: dict) -> dict:
        clsPrior = dict()
        key_li = self.getKeyList(clsCountDict)
        totalCount = sum(self.getValueList(clsCountDict))
        for key in key_li:
            clsPrior[key] = clsCountDict[key] / totalCount
        return clsPrior

    def calcCondProb(self, clsWordStringDict: dict) -> dict:
        trainWordDict = self.calcWordsFreq(clsWordStringDict)
        condProb = dict()
        B_dict = dict()
        cls_li = self.getKeyList(trainWordDict)
        B = len(self.getKeyList(trainWordDict['all']))
        word_li = self.getKeyList(trainWordDict['all'])
        for cls in cls_li:
            if cls != 'all':
                B_dict[cls] = sum(self.getValueList(trainWordDict[cls])) + B
                condProb[cls] = dict()
                for word in word_li:
                    condProb[cls][word] = (trainWordDict[cls].get(word, 0) + 1) / B_dict[cls]
                condProb[cls]['NotFound'] = 1 / B_dict[cls]
        return condProb

    def fit(self, dataSetName: str, verbose=True):
        clsCountDict, clsWordStringDict = self.makeWordsPool(dataSetName, verbose=verbose)
        self.clsPrior = self.calcClassPrior(clsCountDict)
        self.condProb = self.calcCondProb(clsWordStringDict)

    def predict(self, testWordString: str) -> str:
        word_li = self.str2list(testWordString)
        cls_li = self.getKeyList(self.clsPrior)
        score = dict()
        for cls in cls_li:
            score[cls] = np.log(self.clsPrior[cls])
            for word in word_li:
                score[cls] += np.log(self.condProb[cls].get(word, self.condProb[cls]['NotFound']))
        return self.getDictMaxKeyValue(score)[0]

    def accuracy(self, dataSetName: str, isTraining) -> float:
        accuracy = dict()
        dirPath = self.makeDirPath(dataSetName, isTraining=isTraining)
        cls_li = self.getDirItemList(dirPath)
        for cls in cls_li:
            clsDirPath = dirPath + '/' + cls
            doc_li = self.getDirItemList(clsDirPath)
            for doc in doc_li:
                position = clsDirPath + '/' + doc
                try:
                    testWordString = self.makeWordString(position)
                    accuracy['total'] = accuracy.get('total', 0) + 1
                    if self.predict(testWordString) == cls:
                        accuracy[cls] = accuracy.get(cls, 0) + 1
                except UnicodeDecodeError:
                    pass
        return (sum(self.getValueList(accuracy)) - accuracy['total']) / accuracy['total']


class DiscriminativeTextClassifier(ParseTextData):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def dataSetSplit(self, attribute, label, ratio: float) -> tuple:
        perm = np.random.permutation(attribute.shape[0])
        attribute = attribute[perm, :]
        label = label[perm]
        cutoffPoint = np.int(np.floor((attribute.shape[0] - 1) * ratio))
        attributr1, label1, attribute2, label2 = attribute[:cutoffPoint], label[:cutoffPoint], \
                                                 attribute[cutoffPoint:], label[cutoffPoint:]
        return attributr1, label1, attribute2, label2

    def trainWordsList(self, dataSetName: str, verbose) -> list:
        _, clsWordDict = self.makeWordsPool(dataSetName, verbose=verbose)
        trainWordDict = self.calcWordsFreq(clsWordDict)
        return self.getKeyList(trainWordDict['all'])

    def makeInput(self, dataSetName: str, isTraining) -> tuple:
        label = list()
        attribute = list()
        fullTrainWordList = self.trainWordsList(dataSetName, verbose=False)
        dirPath = self.makeDirPath(dataSetName, isTraining=isTraining)
        cls_li = self.getDirItemList(dirPath)
        for idx, cls in enumerate(cls_li):
            clsDirPath = dirPath + '/' + cls
            doc_li = self.getDirItemList(clsDirPath)
            for doc in doc_li:
                position = clsDirPath + '/' + doc
                try:
                    instance = self.countTokensOfDoc(self.makeWordString(position))
                    temp_attribute = list()
                    for word in fullTrainWordList:
                        temp_attribute.append(instance.get(word, 0))
                    attribute.append(np.array(temp_attribute) / np.array(temp_attribute).sum())
                    label.append(idx)
                except UnicodeDecodeError:
                    pass
        attribute = np.array(attribute)
        attribute = np.concatenate((np.ones((attribute.shape[0], 1)), attribute), axis=1)
        label = np.array(label)
        if self.classifier == 'Perceptron':
            label[np.where(label == 0)] = -1
        return attribute, label

    def fitLogistic(self, train_attribute, train_label, reg, eta=0.1, maxIter=1000, verbose=False) -> tuple:
        i = 0
        hit = list()
        acc = 0
        w = np.zeros(train_attribute.shape[1])
        while i < maxIter and acc < 1:
            if verbose:
                print('-----> This is the {} iternation <-----'.format(i))
            condProb = np.exp(train_attribute.dot(w)) / (1 + np.exp(train_attribute.dot(w)))
            w = w + eta * train_attribute.T.dot(train_label - condProb) - eta * reg * w
            acc = self.accuracy(w, train_attribute, train_label)
            hit.append(acc)
            i += 1
        return w, hit

    def fitPerceptron(self, train_attribute, train_label, maxIter, eta=0.1, verbose=False) -> tuple:
        hit = list()
        w = np.zeros(train_attribute.shape[1])
        for i in range(maxIter):
            if verbose:
                print('-----> This is the {} epoch <-----'.format(i))
            w += eta * train_attribute.T.dot(train_label - self.predict(w, train_attribute))
            acc = self.accuracy(w, train_attribute, train_label)
            if acc == 1:
                break
            hit.append(acc)
        return w, hit

    def validation(self, train_attribute, train_label, ratio, paramList, verbose=False):
        accList = list()
        trainx, valix, trainy, valiy = train_test_split(train_attribute, train_label, train_size=ratio, random_state=1)
        isLogit = self.classifier == 'Logistic'
        for param in paramList:
            if verbose:
                print('-----> Try param value: {} <-----'.format(param))
            w, _ = self.fitLogistic(trainx, trainy, reg=param) if isLogit else self.fitPerceptron(trainx, trainy,
                                                                                                  maxIter=param)
            accList.append(self.accuracy(w, valix, valiy))
        bestIdx = np.int(np.argmax(accList))
        return paramList[bestIdx]

    def accuracy(self, w, attribute, label) -> float:
        predict = self.predict(w, attribute)
        if self.classifier == 'Logistic':
            predict[np.where(predict < 0)] = 0
        return sum(predict == label) / label.shape[0]

    def predict(self, w, attribute) -> np.ndarray:
        yhat = np.sign(attribute.dot(w))
        if self.classifier == 'Logistic':
            yhat[np.where(yhat < 0)] = 0
        return yhat


if __name__ == '__main__':
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
