
"""递归的方式有区别，是第一项区别对待的，树复杂了，分类效果并不是很好，但适合于回归"""

from random import randrange
from csv import reader
from math import log
from math import sqrt
import numpy as np

def load_csv(filename):
	dataSet=list()
	with open(filename,'r') as file:
		csv_reader=reader(file)
		for row in csv_reader:
			if not row :
				continue
			dataSet.append(row)
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







#输入的是实际分类 和 预测分类
#得到的是预测准确率的百分比
def accuracy_metric(actual,predicted):
	correct=0
	for i in range(len(actual)):
		if actual[i]==predicted[i]:
			correct+=1

	return correct/float(len(actual))*100.0




#############决策树的内容
def test_split(index,value,dataSet):
	left,right=list(),list()
	for row in dataSet:
		if row[index]<=value:
			left.append(row)
		else:
			right.append(row)

	return left,right
	#当groups=test_split(...)时，groups是一个元组，left，right是两个其中元素
	#通过待测特征的值将数据划分为两个部分

###基尼不纯度
def gini_index(groups,class_values):
	gini=0.0
	total_size=float(len(groups[0])+len(groups[1]))
	for class_value in class_values:
		for group in groups:
			size=len(group)
			if size==0:
				continue
			proportion=[row[-1] for row in group].count(class_value)/float(size)
			#proportion表示正确划分的比例
			gini+=float(size)/total_size*(proportion*(1.0-proportion))
			#这的是基尼不纯度
			#如果全部划分正确为0，全部划分不正确也是0
			#
			#不是基于熵的
	return gini

#总体的数据集，和特征集，最终得到最优的其中其中一个特征
#这个函数中dataSet没有改动
#为什么不一样呢，是因为这里的测试集时浮点型的，很少有两个值一样，只能通过大小来区分
def get_split(dataSet):
	class_values=list(set(row[-1] for row in dataSet))
	b_index,b_value,b_score,b_groups=999,999,999,None
	

	features=range(0,len(dataSet[0])-1)
	# print(len(features))

	##这个位置可以改进的，可以把内层循环去掉相同项
	#并不是想要row 而是想要row[index]这个值
	for index in features:
		for row in dataSet:
			groups=test_split(index,row[index],dataSet)
			gini=gini_index(groups,class_values)
			if gini<b_score:
				b_index,b_value,b_score,b_groups=index,row[index],gini,groups
				#index是特征的编号，row[index]是特征的值
	return {'index':b_index,'value':b_value,'groups':b_groups}


###到达指定情况(树深度)时，把该特征组合归为，标签数量最多的一类
def to_terminal(group):
	outcomes=[row[-1] for row in group]
	return max(set(outcomes),key=outcomes.count)

#利用递归获取树
#多次调用了自己和get_split函数
##树分的时候依旧有原来的特征，但是根据大小将两部分数据分开成左右部分
##有的决策树没有分左右，这样通过值匹配找到类别很难
def split(node,max_depth,min_size,depth):
	left,right=node['groups']
	# del(node['groups'])
	###node定义时就一个给出了最优的特征编号 和特征值
	if not left or not right:
		node['left']=node['right']=to_terminal(left+right)
		#如果其中一个为空，此时只计算另外一个就可以
		return
	if depth>=max_depth:
		node['left'],node['right']=to_terminal(left),to_terminal(right)
		return
	if len(left)<min_size or len(right)<min_size:
		node['left'],node['right']=to_terminal(left),to_terminal(right)
		return
	
	node['left']=get_split(left)
	split(node['left'],max_depth,min_size,depth+1)

	node['right']=get_split(right)
	split(node['right'],max_depth,min_size,depth+1)


def build_tree(train,max_depth,min_size):
	root=get_split(train)
	split(root,max_depth,min_size,1)
	return root
################################决策树的内容
#通过树node，解码，row属于哪一类
def predict(node,row):
	if row[node['index']]<node['value']:
		
		if isinstance(node['left'],dict):
			return predict(node['left'],row)
		else:
			return node['left']
	else:
		if isinstance(node['right'],dict):
			return predict(node['right'],row)
		else:
			return node['right']



# filename='sonar.all-data.csv'
# dataSet=load_csv(filename)
# np.random.shuffle(dataSet)
# for i in range(0,len(dataSet[0])-1):
# 	str_column_to_float(dataSet,i)
# str_column_to_int(dataSet,len(dataSet[0])-1)


dataSet = [[u'青年', u'否', u'否', u'一般', u'拒绝'],
	[u'青年', u'否', u'否', u'好', u'拒绝'],
	[u'青年', u'是', u'否', u'好', u'同意'],
	[u'青年', u'是', u'是', u'一般', u'同意'],
	[u'青年', u'否', u'否', u'一般', u'拒绝'],
	[u'中年', u'否', u'否', u'一般', u'拒绝'],
	[u'中年', u'否', u'否', u'好', u'拒绝'],
	[u'中年', u'是', u'是', u'好', u'同意'],
	[u'中年', u'否', u'是', u'非常好', u'同意'],
	[u'中年', u'否', u'是', u'非常好', u'同意'],
	[u'老年', u'否', u'是', u'非常好', u'同意'],
	[u'老年', u'否', u'是', u'好', u'同意'],
	[u'老年', u'是', u'否', u'好', u'同意'],
	[u'老年', u'是', u'否', u'非常好', u'同意'],
	[u'老年', u'否', u'否', u'一般', u'拒绝'],
	]

max_depth=10
min_size=2
print(len(dataSet))
tree=build_tree(dataSet[0:180],max_depth,min_size)
print(tree)	


####可视化
import plotForCART_not_fea
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
	# 测试决策树的构建

plotForCART_not_fea.createPlot(tree)
###可视化			


predictions=[predict(tree,row) for row in dataSet[180:]]

realLabels=[row[-1] for row in dataSet[180:]]
current=0
for i in range(len(predictions)):
	if predictions[i]==realLabels[i]:
		current+=1

print(current/len(predictions))





