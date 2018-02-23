'''CART分类树的验证剪枝'''
import numpy as np
from numpy import power


def isTree(obj):
    return (isinstance(obj,dict))

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    
    l,r=tree['groups'][0],tree['groups'][1]
    data=[]
    for row in l:
    	data.append(row[-1])
    for row in r:
    	data.append(row[-1])
    return (max(set(data),key=data.count))


def test_split(index,value,dataSet):
    left,right=list(),list()
    for row in dataSet:
        if row[index]<value:
            left.append(row)
        else:
            right.append(row)

    return left,right

def prune(tree, testData):
    if len(testData) == 0: return getMean(tree)    # 如果没有测试数据则对树进行塌陷处理
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = test_split( tree['index'], tree['value'],testData)
    # 深度优先搜索
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    # 到达叶结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = test_split(tree['index'], tree['value'],testData)
        if not lSet or not rSet:
        	l,r=tree['groups'][0],tree['groups'][1]
	        data=[]
	        for row in l:
	        	data.append(row[-1])
	        for row in r:
	        	data.append(row[-1])
	        punning_label= (max(set(data),key=data.count))
        	return punning_label
        # 未剪枝的误差
        lSet_labels=[row[-1] for row in lSet]
        rSet_labels=[row[-1] for row in rSet]

        l_error_ratio=1-lSet_labels.count(tree['left'])/len(lSet_labels)
        r_error_ratio=1-rSet_labels.count(tree['right'])/len(lSet_labels)
        errorNoMerge = len(lSet_labels)/len(testData)*l_error_ratio+\
            len(rSet_labels)/len(testData)*r_error_ratio
        #计算剪枝后的值
        l,r=tree['groups'][0],tree['groups'][1]
        data=[]
        for row in l:
        	data.append(row[-1])
        for row in r:
        	data.append(row[-1])
        punning_label= (max(set(data),key=data.count))
        # 剪枝后的误差
        test_labels=[row[-1] for row in testData]
        errorMerge = 1-test_labels.count(punning_label)/len(test_labels)
        if errorMerge < errorNoMerge: 
            print("merging")
            return punning_label
        else: return tree
    return tree

# prune(tree, [])
# print(tree)
