__author__='mr.y'
'''最简单的决策树实现，使用的是熵和信息增益'''
'''ID3,多个数据集'''

from math import log
import numpy as np
import operator

log2=lambda x:log(x)/log(2)
def calcShannonEnt(data):
	
	label={}
	for row in data:
		classRow=row[-1]
		# label[classRow]=lable.get(classRow,0)+1
		if label[classRow]==[]:
			label[classRow]=0
		label[classRow]+=1
	shannon=0
	num=len(data)
	for key in label:
		prob=float(label[key])/num
		shannon-=prob*log2(prob)    #log(prob,2)
	return shannon


def calcShannonEnt(dataSet,col=-1):
	numEntries=len(dataSet)
	labelCounts={}
	for featVec in dataSet:
		currentLabel=featVec[col]
		labelCounts[currentLabel]=labelCounts.get(currentLabel,0)+1
	shannonEnt=0.0
	for key in labelCounts:
		prob=float(labelCounts[key])/numEntries
		shannonEnt-=prob*log2(prob)#log(prob,2)
	return shannonEnt



def createDataSet():
	####################
	dataSet=[[1,1,'yes'],
		[1,1,'yes'],
		[1,0,'no'],
		[0,1,'no'],
		[0,1,'no'],
		]
	feature_index=['0','1']
	# dataSet=[[1,1,'yes'],
	# 	[1,2,'yes'],
	# 	[1,3,'no'],
	# 	[0,4,'no'],
	# 	[0,5,'no'],
	# 	]
	############################
	#信贷情况数据
	dataSet = [['青年', '否', '否', '一般', '拒绝'],
	['青年', '否', '否', '好', '拒绝'],
	['青年', '是', '否', '好', '同意'],
	['青年', '是', '是', '一般', '同意'],
	['青年', '否', '否', '一般', '拒绝'],
	['中年', '否', '否', '一般', '拒绝'],
	['中年', '否', '否', '好', '拒绝'],
	['中年', '是', '是', '好', '同意'],
	['中年', '否', '是', '非常好', '同意'],
	['中年', '否', '是', '非常好', '同意'],
	['老年', '否', '是', '非常好', '同意'],
	['老年', '否', '是', '好', '同意'],
	['老年', '是', '否', '好', '同意'],
	['老年', '是', '否', '非常好', '同意'],
	['老年', '否', '否', '一般', '拒绝']]
	feature_index = [0,1,2,3]
	# featre_index = ['年龄', '有工作', '有房子', '信贷情况']

	###############################
	# ##添加了工资列
	# dataSet = [['1000','青年', '否', '否', '一般', '拒绝'],
 #                ['2000','青年', '否', '否', '好', '拒绝'],
 #                ['7000','青年', '是', '否', '好', '同意'],
 #                ['7100','青年', '是', '是', '一般', '同意'],
 #                ['3000','青年', '否', '否', '一般', '拒绝'],
 #                ['3500','中年', '否', '否', '一般', '拒绝'],
 #                ['3600','中年', '否', '否', '好', '拒绝'],
 #                ['8000','中年', '是', '是', '好', '同意'],
 #                ['9000','中年', '否', '是', '非常好', '同意'],
 #                ['9200','中年', '否', '是', '非常好', '同意'],
 #                ['8600','老年', '否', '是', '非常好', '同意'],
 #                ['7800','老年', '否', '是', '好', '同意'],
 #                ['10000','老年', '是', '否', '好', '同意'],
 #                ['6500','老年', '是', '否', '非常好', '同意'],
 #                ['3000','老年', '否', '否', '一般', '拒绝'],
 #                ]
	# featre_index = ['工资','年龄', '有工作', '有房子', '信贷情况']
	###############################
	return dataSet,feature_index

def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		if (featVec[axis]==value):
			# reducedFeatVec=featVec[:axis]
			#numpy中没有extend,而是concatenate
			# reducedFeatVec=np.concatenate((reducedFeatVec,featVec[axis+1:]))
			# reducedFeatVec.extend(featVec[axis+1:])
			reducedFeatVec=featVec.copy()
			del(reducedFeatVec[axis])
			retDataSet.append(reducedFeatVec)
			# retDataSet.append(featVec)
	return retDataSet


def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1
	baseInfoGain=calcShannonEnt(dataSet)
	bestInfoGain=0.0;bestFeature=-1
	for i in range(numFeatures):

		featureList=[example[i] for example in dataSet]
		uniqueFeatures=set(featureList)
		currentEnt=0.0

		for feature in uniqueFeatures:
			subDataSet=splitDataSet(dataSet,i,feature)
			prob=float(len(subDataSet))/len(dataSet)
			currentEnt+=prob*calcShannonEnt(subDataSet)
		newInfoGain=baseInfoGain-currentEnt
		# print(newInfoGain,i)
		# if(currentEnt<bestInfoGain):
		if(newInfoGain>bestInfoGain):
			bestInfoGain=newInfoGain
			# bestInfoGain=currentEnt
			bestFeature=i
	return bestFeature


def majorityCnt(classList):
	outcomes=classList.copy()
	return max(set(outcomes),key=outcomes.count)


def creatTree(dataSet,featureList):
	classList=[example[-1] for example in dataSet]
	if classList.count(classList[0])==len(classList):
		
		return classList[0]
	if len(dataSet[0])==1:
		return majorityCnt(classList)
		
	bestFeature=chooseBestFeatureToSplit(dataSet)

	##更新bestFeatLabel 和 featureList
	bestFeatLabel=featureList[bestFeature]
	del(featureList[bestFeature])
	##

	##这个是推荐使用的方法；为CART的树结构理解做准备
	myTree={'feature_index':bestFeatLabel,'child':{}}
	featValues=[example[bestFeature] for example in dataSet]
	uniqueFeat=set(featValues)
	for value in uniqueFeat:
		subDataSet=splitDataSet(dataSet,bestFeature,value)
		myTree['child'][value]=creatTree(subDataSet,featureList)
	
	##下面的写法可以使字典的结构简单化，但是不推荐；
	# myTree={bestFeatLabel:{}}
	# featValues=[example[bestFeature] for example in dataSet]
	# uniqueFeat=set(featValues)
	# for value in uniqueFeat:
	# 	myTree[bestFeatLabel][value]=splitDataSet(dataSet,bestFeature,value)
	# 	myTree[bestFeatLabel][value]=creatTree(myTree[bestFeatLabel][value],featureList)

	return myTree



import treePlotter2
from pylab import *
def main():
	dataSet,label=createDataSet()
	
	feature=chooseBestFeatureToSplit(dataSet)
	
	myTree1=creatTree(dataSet,label)
	print(myTree1)

		
	
	mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
	mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
	# 测试决策树的构建
	
	treePlotter2.createPlot(myTree1)

main()



