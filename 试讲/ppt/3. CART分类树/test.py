import pandas as pd

df=pd.read_csv('sonar.all-data.csv',header=None)
# pd.read_csv()
print(df.values)
data=[]
with open('sonar.all-data.csv') as file:
	for row in file:
		if not row:
			continue
		data.append(row.strip().split(','))
# print(data)
import numpy as np


