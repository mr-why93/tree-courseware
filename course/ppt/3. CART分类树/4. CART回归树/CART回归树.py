from random import randrange
from csv import reader
from math import log
from math import sqrt
from numpy import power
import numpy as np

def load_csv(filename):
	dataSet=list()
	with open(filename,'r') as file:
		# csv_reader=reader(file)
		for row in file:
			if not row :
				continue
			dataSet.append(row.strip().split(','))
	return dataSet

# dataSet字符型转换为浮点型
def str_column_to_float(dataSet,column):
	for row in dataSet:
		row[column]=float(row[column].strip())
#目的是把最后一列的标签从 string变成int
#最后把对应关系代表的字典输出
def str_column_to_int(dataSet,column):
	class_values=[row[column] for row in dataSet]
	unique=set(class_values)
	lookup=dict()
	for key,value in enumerate(unique):
		lookup[value]=key
	for row in dataSet:
		row[column]=lookup[row[column]]
	return lookup


#############决策树的内容
def test_split(dataSet,index,value):
	#和ID3处的不同，每次不删列
	#value是作为分割线，而不是等于才取
	left,right=list(),list()
	for row in dataSet:
		if row[index]<value:
			left.append(row)
		else:
			right.append(row)

	return left,right



def m_q_e(groups,class_values):
	mean_square_error=0.0
	
	# for class_value in class_values:
	#减少了一层循环
	for group in groups:
		size=len(group)
		if size==0:
			continue
		labels=[row[-1] for row in group]
		
		proportion=np.array(labels).mean()
		error=sum(power(labels - proportion,2))
		mean_square_error+=error
	return mean_square_error


def get_split(dataSet):
	class_values=list(set(row[-1] for row in dataSet))
	b_index,b_value,b_score,b_groups=999,999,999,None
	

	features=range(0,len(dataSet[0])-1)
	# print(len(features))

	##这个位置可以改进的，可以把内层循环去掉相同项
	#并不是想要row 而是想要row[index]这个值
	for index in features:
		fea_Space=list(set([row[index] for row in dataSet]))
		##这里将每一个值作为二分标准选取方法,可以采用其他方法		
		for row in fea_Space:
			groups=test_split(dataSet,index,row)
			gini=m_q_e(groups,class_values)
			if gini<b_score:
				b_index,b_value,b_score,b_groups=index,row,gini,groups
	return b_index,b_value,b_groups



def to_terminal(outcomes):

	return(np.array(outcomes).mean())
#利用递归获取树
#多次调用了自己和get_split函数
##树分的时候依旧有原来的特征，将数据分开成左右部分
def createTree(data,max_depth=999,minsize=1,depth=1,stop=0.01):
	labelList=[d[-1]  for d in data]
	if (len(set(labelList))==1) or len(data)<minsize or depth>=max_depth:
		return to_terminal(labelList)
	if(sum(power(np.array(labelList).mean()-labelList,2))<stop):
		return to_terminal(labelList)

	node=get_split(data)
	left,right=node[2][0],node[2][1]

	tree={'index':node[0],'value':node[1],'left':[],'right':[]}
	
	tree['left']=createTree(left,max_depth,minsize,depth+1)
	tree['right']=createTree(right,max_depth,minsize,depth+1)
	return tree	


def fit(train,max_depth,min_size,stop=0.01):
	root=createTree(train,max_depth,min_size,0)
	return root
################################决策树的内容
#通过树node，解码，row属于哪一类
def predict(node,row):
	index=node['index']
	if row[index]<node['value']:
		
		if isinstance(node['left'],dict):
			return predict(node['left'],row)
		else:
			return node['left']
	else:
		if isinstance(node['right'],dict):
			return predict(node['right'],row)
		else:
			return node['right']

#输入的是实际分类 和 预测分类
#得到的是预测准确率的百分比

def accuracy_metric(actual,predicted):
	correct=0
	for i in range(len(actual)):
		if actual[i]==predicted[i]:
			correct+=1

	return correct/float(len(actual))*100.0


filename='sonar.all-data.csv'
dataSet=load_csv(filename)
np.random.shuffle(dataSet)

for i in range(0,len(dataSet[0])-1):
	str_column_to_float(dataSet,i)
str_column_to_int(dataSet,len(dataSet[0])-1)

max_depth=10
min_size=2
print(len(dataSet))
tree=fit(dataSet[0:150],max_depth,min_size)
print(tree)	

####可视化
import treePlotter2
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
treePlotter2.createPlot(tree)
##可视化
##剪枝
import ART_punning
punning_tree=ART_punning.prune(tree, dataSet[150:200])
treePlotter2.createPlot(punning_tree)
##可视化
##


predictions=[predict(tree,row) for row in dataSet[150:]]


realLabels=[row[-1] for row in dataSet[150:]]
current=0
for i in range(len(predictions)):
	if predictions[i]>0.5:
		pre=1
	else:pre=0
	if pre==realLabels[i]:
		current+=1

print(current/len(predictions))



