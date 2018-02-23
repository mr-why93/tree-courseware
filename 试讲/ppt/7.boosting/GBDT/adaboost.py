from numpy import *
import numpy as np
from math import log

def split(data,index,value,lr):
	m,n=data.shape
	pre=ones((m,1))

	if (lr=='l'):
		pre[data[:,index]<value]=-1
	else:
		pre[data[:,index]>=value]=-1
	return pre


def tree1 (data,classLabel,w):
	data=np.mat(data)
	m,n=data.shape
	
	min_err=999
	b_index,b_value,b_lr=-1,-1,'l'
	b_pre=999
	for i in range( n):
		fea=data[:,i]
		fea=fea.tolist()
		
		unique_fea=set([f[0] for f in fea])
		for value in unique_fea:
			for lr in ['l','r']:
				pre=split(data, i, value, lr)
				err=mat(ones((m,1)))
				err[pre==classLabel]=0
				w_err=err.T*w
				if w_err<min_err:
					min_err=w_err
					b_pre=pre
					b_index=i;b_value=value;b_lr=lr
	return min_err,b_pre,b_index,b_value,b_lr


def Adaboost(data,labelList):
	data=np.mat(data)
	m,n=data.shape
	w=np.mat(ones((m,1)))/m
	pre=np.mat(zeros((m,1)))
	fx={}
	for i in range(8):
		t=tree1(data,labelList,w)
		min_err,b_pre,b_index,b_value,b_lr=t[0],t[1],t[2],t[3],t[4]
		w_next=w.copy()

		alpha=0.5*log((1-min_err[0,0])/(min_err[0,0]+0.000001))##被除数最好加一个小变量防止为0
		z=w.T*exp(-alpha*np.multiply(labelList ,b_pre))
		print(z)
		w_next=np.multiply(w/(z+0.000001),exp(-alpha*np.multiply(labelList ,b_pre)))##被除数最好加一个小变量防止为0
		print(w_next)
		fx[i]=(b_index,b_value,b_lr,alpha)
		pre+=alpha*b_pre
		w=w_next
		predict=np.mat(ones((m,1)))
		predict[pre<0]=-1
		print(predict)
	return fx

data=mat([[1,1],[1,1],[1,0],[0,1],[0,1]])
labelList=mat([[1],[1],[-1],[-1],[-1]])
# w=mat([[0.3],[0.6],[0.1]])

fx=Adaboost(data,labelList)
print(fx)

# def predict(fx,sample):
tree=(1,0,'l')
def predict_tree(tree,sample):
	if sample[tree[0]]<tree[1]:
		if tree[2]=='l':
			pre=-1
		else:pre=1
	else:
		if tree[2]=='l':
			pre=1
		else:pre=-1
	return pre

print(predict_tree(tree, (1,0)))
def predict(fx,sample):
	pre=0
	# sample.tolist()
	for values in fx.values():
		tree=values[:-1]
		pre+=values[-1]*predict_tree(tree, sample)
	return sign(pre)
test=[[1,0],[1,1],[0,1]]
l=map(lambda x: predict(fx,x),test)
print(list(l))
# for i in test:
# 	l=map(predict,test)
# 	print(predict(fx,i))

import pandas as pd
df=pd.read_csv('../sonar.all-data.csv',header=None)
dataSet=df.values
labels=[d[-1] for d in dataSet]

labels=list(set(labels))
lab_dict={}
lab_dict[labels[0]]=-1;lab_dict[labels[1]]=1
for row in dataSet:
	row[-1]=lab_dict[row[-1]]
np.random.shuffle(dataSet)
test=dataSet[-30:].copy()
# dataSet=dataSet[:,-30].copy()

data=np.mat([d[:-1] for d in dataSet])
label=np.mat([d[-1] for d in dataSet])
test_data=np.mat([d[:-1] for d in test])
test_label=np.mat([d[-1] for d in test])

fx=Adaboost(data,label.T)
print(fx)
print(data[:3,:])
# for i in range(5):
# 	print(predict(fx,data[i,:].T))
pr=map(lambda x: predict(fx,x.T),test_data)
print(test_label)
print([int( i) for i in list(pr)])

