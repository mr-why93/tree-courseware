
from csv import reader
import numpy as np
def featureSplit(dataSet,method="step",step_num=10):


	if (method=='step'):
		dataSet_copy=np.array(dataSet.copy())
		m,n=dataSet_copy.shape
		for i in range(n-1):
			uniqueFeature=[row[i] for row in dataSet_copy]

			min_num,max_num=min(uniqueFeature),max(uniqueFeature)
			step=(max_num-min_num)/step_num
			
			for j in range(m):
				dataSet_copy[j,i]=((dataSet_copy[j,i]-min_num)/step).astype(np.int32)
				
	return dataSet_copy

a=np.random.rand(30,2)
b=featureSplit(a)
print(b)