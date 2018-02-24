



def createDataSet():
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
	# feature_index = [u'年龄', u'有工作', u'有房子', u'信贷情况']
	feature_index=[0,1,2,3]
	return dataSet,feature_index

from math import log2
def calc_shannon_ent(data):
	n=len(data)
	label_count={}
	for row in data:
		if row[-1] not in label_count.keys():
			label_count[row[-1]]=0
		label_count[row[-1]]+=1
	ent=0.0
	for v in label_count.values():
		pi= v/n
		ent-=pi*log2(pi)
	return ent

def calc_ent2(data):
	labels=[row[-1] for row in data]
	labelSpace=list(set(labels))
	ent=0.0
	for value in labelSpace:
		pi=labels.count(value)/len(labels)
		ent-=pi*log2(pi)
	return ent

data,fea=createDataSet()
print(calc_shannon_ent(data))
print(calc_ent2(data))