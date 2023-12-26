import argparse
from utils import *

# command line params
parser = argparse.ArgumentParser()
parser.add_argument('L', type=int)
parser.add_argument('K', type=int)
parser.add_argument('train', type=str)
parser.add_argument('validate', type=str)
parser.add_argument('test', type=str)
parser.add_argument('plot', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    args.train += '.csv'
    args.validate += '.csv'
    args.test += '.csv'
    trainx, trainy = dataParse(args.train)
    testx, testy = dataParse(args.test)
    valdtx, valdty = dataParse(args.validate)

    # accuracy
    tree = DecisionTree()
    root_et = tree.fit(trainx, trainy, 'entropy')
    root_vi = tree.fit(trainx, trainy, 'vi')

    scissor = PostPruning(L=args.L, K=args.K)
    pruned_et = scissor.postPruning(root_et, valdtx, valdty)
    pruned_vi = scissor.postPruning(root_vi, valdtx, valdty)

    acc_et = DecisionTree.accuracy(root_et, testx, testy)
    acc_vi = DecisionTree.accuracy(root_vi, testx, testy)
    acc_et_ = DecisionTree.accuracy(pruned_et, testx, testy)
    acc_vi_ = DecisionTree.accuracy(pruned_vi, testx, testy)

    print('Accuracy of the original tree (entropy) on test set : ', acc_et)
    print('Accuracy of the original tree (variance impurity) on test set : ', acc_vi)
    print('Accuracy of the pruned tree (entropy) on test set : ', acc_et_)
    print('Accuracy of the pruned tree (variance impurity) on test set : ', acc_vi_)

    if args.plot == 'yes':
        print('Here is the structure of the tree: ')
        DecisionTree.plot(root_et)
