import numpy as np
data=list()
with open("example.txt",'r') as file:
	for f in file:
		f_str=f.strip().split('\t')
		f_str[0]=float(f_str[0])
		data.append(f_str)
print(data)

