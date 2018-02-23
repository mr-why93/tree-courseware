import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

train=pd.read_csv('sonar.all-data.csv',header=None,index_col=None)



dataSet=train.values

ncol1=len(train.columns)
array=train.iloc[:,0:ncol1-1].values
# np.random.shuffle(dataSet)
# # array=[d[:-1] for d in dataSet]
print(array)
boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Range -Normalized")
show()