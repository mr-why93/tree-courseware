
"""传入groups，labelSpace 返回giniScore """
def gini(groups, labelSpace):
	lenTotal=len(groups[0])+len(groups[1])
	conGini=0.0
	for data in groups:
		lenData=len(data)
		if(lenData==0):
			continue
		for value in labelSpace:
			pi=[d[-1] for d in data].count(value)/lenData
			conGini+=lenData/lenTotal*pi*(1-pi)
	return conGini

def split_data(data,index,value):
	left=list();right=list()
	for row in data:
		if row[index]<value:
			left.append(row)
		else :
			right.append(row)
	return left,right

def get_split(data):
	lenFeatures=len(data[0])-1
	labelSpace=set([row[-1] for row in data])
	b_gini=1
	# b_index,b_value,b_groups=-1,-1,tuple()
	for index in range(lenFeatures):
		feaSpace=set([row[index] for row in data])
		for value in feaSpace:
			groups=split_data(data, index, value)
			giniScore=gini(groups, labelSpace)
			if giniScore<b_gini:
				b_gini=giniScore
				b_index,b_value,b_groups=index,value,groups
	return  b_index,b_value,b_groups

dataSet=[[1,1,'yes'],
	[1,1,'yes'],
	[1,0,'no'],
	[0,1,'no'],
	[0,1,'no']]
print(get_split(dataSet))


b=dict()
b.keys=[1,2]
print(b.keys)
# def gini(data,labelSpace):
# 	labelCount=dict()
# 	labelCount.keys()=labelCount