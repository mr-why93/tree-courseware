import matplotlib.pyplot as plt
from math import log2
from math import log

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

x=[i/10000 for i in range(1,10000)]
y=[1-i**2-(1-i)**2 for i in x]
z=[(-i*log2(i)-(1-i)*log2(1-i))/2 for i in x]
x1=plt.plot(x,y,c='r',label='基尼')

x2=plt.plot(x,z,c='b',label='信息熵/2')


plt.legend()

plt.show()
