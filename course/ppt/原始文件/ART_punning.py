

'''ART回归树的剪枝'''







def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


def test_split(index,value,dataSet):
    left,right=list(),list()
    for row in dataSet:
        if row[index]<value:
            left.append(row)
        else:
            right.append(row)

    return left,right

def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)    # 如果没有测试数据则对树进行塌陷处理
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = test_split( tree['index'], tree['values'],testData)
    # 深度优先搜索
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    # 到达叶结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = test_split(tree['index'], tree['values'],testData)
        # 未剪枝的误差
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        #计算剪枝后的值
        treeMean = (tree['left']+tree['right'])/2.0
        # 剪枝后的误差
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree