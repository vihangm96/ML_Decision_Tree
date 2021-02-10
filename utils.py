import numpy as np
import copy

def entropy(node):
    # node : List[int] Node with list of no. of classes
    # return : float

    if(sum(node)) == 0:
        return 1

    entropyVal = 0

    for n in node:

        prob = ( n / sum(node) )
        if(prob != 0):
            entropyVal -= prob * np.log2(prob)
    return entropyVal

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float

    total_points = 0
    weighted_entropy = 0

    for branch in branches:
        for data_pt in branch:
            total_points += data_pt

    for branch in branches:
        weighted_entropy += (sum(branch))*(entropy(branch))/total_points

    return S - weighted_entropy

def accuracy(y_pred,y_exp):
    assert (len(y_exp) > 0)
    assert (len(y_exp) == len(y_pred))
    match = 0

    for y_index in range(len(y_pred)):
        if(y_pred[y_index] == y_exp[y_index]):
            match += 1

    return match/len(y_pred)

# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, x_test, y_test):
    # decisionTree : decisionTree trained based on training data set.
    # X_test: List[List[any]] test data, num_cases*num_attributes
    # y_test: List[any] test labels, num_cases*1
    recPrune(decisionTree,x_test,y_test,decisionTree.root_node)

def recPrune(decisionTree, x_test, y_test, node):
    if node.splittable:
        for child in node.children:
            recPrune(decisionTree,x_test,y_test,child)
        oldAccuracy = accuracy(decisionTree.predict(x_test),y_test)
        tempChildren = node.children[:]
        node.children = []
        newAccuracy = accuracy(decisionTree.predict(x_test),y_test)
        if(oldAccuracy<=newAccuracy):
            node.splittable = False
        else:
            node.children = tempChildren


def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')