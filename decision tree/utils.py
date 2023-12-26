import numpy as np
import pandas as pd
from copy import deepcopy
import random


# node class
class Nodes:
    def __init__(self):
        self.attrbt = None
        self.attrbtValue = None
        self.nodeLabel = None
        self.numLeaf = 0
        self.depth = 0
        self.distribution = {}
        self.sonNodes = []


# decision tree class
class DecisionTree:
    def __init__(self, minLeaf=1):
        self.minLeafNum = minLeaf

    @classmethod
    def predict(cls, root, obs):
        if not root.sonNodes:
            return root.nodeLabel

        attrbt = root.attrbt
        for node in root.sonNodes:
            if obs[attrbt] == node.attrbtValue:
                return cls.predict(node, obs)

    @classmethod
    def accuracy(cls, root, testAttrbt, testLabel):
        numOfHit = 0
        for index, obs in enumerate(testAttrbt):
            if cls.predict(root, obs) == testLabel[index]:
                numOfHit += 1
        return numOfHit / testAttrbt.shape[0]

    @classmethod
    def plot(cls, tree):
        if not tree.sonNodes:
            return None
        for son in tree.sonNodes:
            if son.sonNodes:
                print(' |' * tree.depth, '{} = {} :'.format(tree.attrbt, son.attrbtValue))
            else:
                print(' |' * tree.depth, '{} = {} : {}'.format(tree.attrbt, son.attrbtValue, son.nodeLabel))
            cls.plot(son)

    def getKeyList(self, inputDict: dict) -> list:
        return list(inputDict.keys())

    def getValueList(self, inputDict: dict) -> list:
        return list(inputDict.values())

    def getItemList(self, inputDict: dict) -> list:
        return list(inputDict.items())

    def getDictMaxKeyValue(self, inputDict: dict) -> tuple:
        sortedDict = sorted(self.getItemList(inputDict), key=lambda k: k[1], reverse=True)
        return sortedDict[0][0], sortedDict[0][1]

    def countTotalObs(self, inputDict: dict):
        return sum(self.getValueList(inputDict))

    def calcEntropy(self, inputDict: dict):
        entropy = 0.
        if not inputDict:
            return entropy

        pr_list = np.array(self.getValueList(inputDict)) / self.countTotalObs(inputDict)
        for pr in pr_list:
            entropy -= pr * np.log2(pr) if pr != 0 else 0.
        return entropy

    def calcVarImpty(self, inputDict: dict):
        varimpty = 1.
        if not inputDict:
            return 0.

        count_list = np.array(self.getValueList(inputDict)) / self.countTotalObs(inputDict)
        for count in count_list:
            varimpty *= count
        return varimpty

    def calcWegtET(self, inputDict: dict):
        entropy_list = []
        count_list = []
        if not inputDict:
            return 0.

        for dict in self.getValueList(inputDict):
            count_list.append(self.countTotalObs(dict))
            entropy_list.append(self.calcEntropy(dict))

        pr_array = np.array(count_list) / sum(count_list) if sum(count_list) != 0 else np.array(count_list)
        weightedEntropy = np.dot(pr_array, np.array(entropy_list))
        return weightedEntropy

    def calcWegtVI(self, inputDict: dict):
        varimpty_list = []
        count_list = []
        if not inputDict:
            return 0.

        for dict in self.getValueList(inputDict):
            count_list.append(self.countTotalObs(dict))
            varimpty_list.append(self.calcVarImpty(dict))

        pr_array = np.array(count_list) / sum(count_list) if sum(count_list) != 0 else np.array(count_list)
        weightedVarImpty = np.dot(pr_array, np.array(varimpty_list))
        return weightedVarImpty

    def _fit(self, trainAttrbt, trainLabel, criterion, rootNode, usedAttrbt=None):
        if usedAttrbt is None:
            usedAttrbt = {}
        clsCount = {}
        splitNode = {}

        unused_list = []
        for i in range(trainAttrbt.shape[1]):
            if i not in self.getKeyList(usedAttrbt):
                unused_list.append(i)

        for index, obs in enumerate(trainAttrbt):
            qualifySign = True
            for i in self.getKeyList(usedAttrbt):
                qualifySign = qualifySign and usedAttrbt[i] == obs[i]
            if qualifySign:
                clsLabel = trainLabel[index]
                clsCount[clsLabel] = clsCount.get(clsLabel, 0) + 1
                for attrbt in unused_list:
                    splitNode[attrbt] = splitNode.get(attrbt, {})
                    attrbtVal = obs[attrbt]
                    splitNode[attrbt][attrbtVal] = splitNode[attrbt].get(attrbtVal, {})
                    splitNode[attrbt][attrbtVal][clsLabel] = splitNode[attrbt][attrbtVal].get(clsLabel, 0) + 1

        if not clsCount or self.countTotalObs(clsCount) == 0:
            return None

        rootNode.nodeLabel = self.getDictMaxKeyValue(clsCount)[0]
        rootNode.distribution = deepcopy(clsCount)
        rootNode.numLeaf = self.countTotalObs(clsCount)
        if rootNode.numLeaf < self.minLeafNum:
            return None

        criterAtThisNode = self.calcEntropy(clsCount) if criterion == 'entropy' else self.calcVarImpty(clsCount)
        if criterAtThisNode == 0.:
            return None

        infoGainForSplit = {}
        for s in self.getKeyList(splitNode):
            weighted = self.calcWegtET(splitNode[s]) if criterion == 'entropy' else self.calcWegtVI(splitNode[s])
            infoGainForSplit[s] = criterAtThisNode - weighted
        attrbtChosenAtThisNode = self.getDictMaxKeyValue(infoGainForSplit)[0]
        infoGainAfterSplit = self.getDictMaxKeyValue(infoGainForSplit)[1]
        if infoGainAfterSplit <= 0.:
            return None

        rootNode.attrbt = attrbtChosenAtThisNode
        attrbtValsOfChosenAttrbt = self.getKeyList(splitNode[attrbtChosenAtThisNode])
        numSonNodes = len(attrbtValsOfChosenAttrbt)
        for i in range(numSonNodes):
            son = Nodes()
            son.depth = rootNode.depth + 1
            rootNode.sonNodes.append(son)
            son.attrbtValue = attrbtValsOfChosenAttrbt[i]
            newUsedAttrbt = deepcopy(usedAttrbt)
            newUsedAttrbt[attrbtChosenAtThisNode] = attrbtValsOfChosenAttrbt[i]
            self._fit(trainAttrbt, trainLabel, criterion, rootNode.sonNodes[i], usedAttrbt=newUsedAttrbt)

    def fit(self, trainAttrbt, trainLabel, criterion, usedAttrbt=None):
        root = Nodes()
        self._fit(trainAttrbt, trainLabel, criterion=criterion, rootNode=root, usedAttrbt=usedAttrbt)
        self.root = root
        return self.root


# pruning class
class PostPruning:
    def __init__(self, L, K):
        self.L = L
        self.K = K

    def getDictMaxKey(self, inputDict: dict) -> tuple:
        sortedDict = sorted(list(inputDict.items()), key=lambda k: k[1], reverse=True)
        return sortedDict[0][0]

    def calcNonLeafNodes(self, treeRoot):
        if not treeRoot.sonNodes:
            return 0

        return sum(list(map(self.calcNonLeafNodes, treeRoot.sonNodes))) + 1

    def _nonLeafNodesList(self, treeRoot, nodelist=None):
        if nodelist is None:
            nodelist = []
        if treeRoot.sonNodes:
            nodelist.append(treeRoot)
            for son in treeRoot.sonNodes:
                self._nonLeafNodesList(son, nodelist)
        return nodelist

    def nonLeafNodesList(self, treeRoot):
        nonLeaf = []
        return self._nonLeafNodesList(treeRoot=treeRoot, nodelist=nonLeaf)

    def postPruning(self, preTree, vldAttrbt, vldLabel):
        D_best = deepcopy(preTree)
        for i in range(1, self.L + 1):
            D_prime = deepcopy(preTree)
            M = random.choice(range(1, self.K + 1))
            for i in range(1, M + 1):
                allowedList = self.nonLeafNodesList(D_prime)
                if allowedList:
                    node = random.choice(allowedList)
                    node.sonNodes.clear()
                    node.nodeLabel = self.getDictMaxKey(node.distribution)
                    node.numLeaf = sum(list(node.distribution.values()))
                else:
                    break
            if DecisionTree.accuracy(D_prime, vldAttrbt, vldLabel) > DecisionTree.accuracy(D_best, vldAttrbt, vldLabel):
                D_best = deepcopy(D_prime)
        return D_best


def dataParse(filePath, lastLabel=True):
    data = np.array(pd.read_csv(filePath))
    if lastLabel:
        return data[:, :-1], data[:, -1]
    else:
        return data[:, 1:], data[:, 0]


