from utils import *

# test for DecisionTree class
trainx, trainy = dataParse('./data/training_set1.csv')
testx, testy = dataParse('./data/test_set1.csv')
valdtx, valdty = dataParse('./data/validation_set1.csv')

dt = DecisionTree()
tree = dt.fit(trainx, trainy, criterion='entropy')
DecisionTree.accuracy(tree, testx, testy)
DecisionTree.accuracy(tree, trainx, trainy)
DecisionTree.plot(tree)

scissor = PostPruning(15, 15)
tree_best = scissor.postPruning(tree, valdtx, valdty)
DecisionTree.plot(tree_best)
DecisionTree.accuracy(tree_best, testx, testy) - DecisionTree.accuracy(tree, testx, testy)
